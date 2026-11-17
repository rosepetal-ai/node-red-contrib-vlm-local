/**
 * Detection prompt, response schema and parser: a port of the parts of the
 * production `llm_labeler.py` (rosepetal-external-services) this node needs.
 *
 * Kept byte-for-byte with the Python by `test/detect.test.js`, which replays
 * fixtures generated from it. Change the Python first, regenerate, then port.
 */

'use strict';

const COORD_SCALE = 1000;

// grid formats only: the models in the catalog answer on a 0..1000 grid.
const COORD_FORMATS = {
    grid_xyxy: { order: 'xyxy' },
    grid_yxyx: { order: 'yxyx' },
};

const BOX_KEYS = ['box', 'bbox', 'box_2d', 'bbox_2d'];

function coordInstruction(coordFormat) {
    const spec = COORD_FORMATS[coordFormat];
    if (!spec) throw new Error(`Unknown coord_format '${coordFormat}'.`);
    const order = spec.order === 'yxyx' ? '[ymin, xmin, ymax, xmax]' : '[x1, y1, x2, y2]';
    return `Coordinates are ${order}, integers normalized to 0..${COORD_SCALE} `
        + "regardless of the image's pixel size.";
}

const GEOMETRY_TAIL = '[x1,y1] is the top-left corner and [x2,y2] the bottom-right; the origin (0,0) is '
    + 'the top-left of the image. Each box must enclose the ENTIRE object — from its '
    + 'topmost to bottommost and leftmost to rightmost visible pixel — without cropping '
    + 'any part.';

function buildTagsPrompt(tags, coordFormat) {
    const vocabulary = tags.map((tag) => {
        const description = String(tag.description || '').trim();
        return `- "${tag.name}"` + (description ? `: ${description}` : '');
    }).join('\n');
    const names = tags.map((tag) => `"${tag.name}"`).join(', ');
    return 'You are a precise object detector. Find every distinct instance of each of the '
        + 'following labels in the image:\n'
        + `${vocabulary}\n\n`
        + 'Only include instances that are clearly visible. If there are none, return an '
        + 'empty list.\n\n'
        + 'Return ONLY a JSON object (no prose, no markdown) of the form '
        + '{"detections": [{"label": <label>, "box": [x1, y1, x2, y2], '
        + '"confidence": <number between 0 and 1>}]}, where "label" is EXACTLY one of '
        + `${names} — do not invent labels.\n`
        + `${coordInstruction(coordFormat)}\n`
        + GEOMETRY_TAIL;
}

function buildFreePrompt(request, coordFormat) {
    return 'You are a precise object detector. Find every distinct instance of '
        + `what the user asks for:\n${String(request).trim()}\n\n`
        + 'Only include instances that are clearly visible. If there are none, '
        + 'return an empty list.\n\n'
        + 'Return ONLY a JSON object (no prose, no markdown) of the form '
        + '{"detections": [{"label": <short label>, "box": [x1, y1, x2, y2], '
        + '"confidence": <number between 0 and 1>}]}, where "label" names what '
        + 'you found in a word or two.\n'
        + `${coordInstruction(coordFormat)}\n`
        + GEOMETRY_TAIL;
}

/** Structured-outputs schema; a closed tag set constrains `label` to the names. */
function detectionSchema(tags) {
    const label = { type: 'string' };
    if (tags.length) label.enum = [...new Set(tags.map((tag) => String(tag.name)))];
    const detection = {
        type: 'object',
        additionalProperties: false,
        required: ['label', 'box', 'confidence'],
        properties: {
            label,
            box: { type: 'array', items: { type: 'number' } },
            confidence: { type: 'number' },
        },
    };
    return {
        type: 'json_schema',
        json_schema: {
            name: 'detections',
            strict: true,
            schema: {
                type: 'object',
                additionalProperties: false,
                required: ['detections'],
                properties: { detections: { type: 'array', items: detection } },
            },
        },
    };
}

function tryParse(text) {
    try { return { ok: true, value: JSON.parse(text) }; } catch { return { ok: false }; }
}

/** JSON out of model output, tolerating markdown fences and stray prose; null if nothing parses. */
function extractJson(text) {
    if (text === null || text === undefined) return null;
    let body = String(text).trim();
    const fence = /```(?:json)?\s*([\s\S]*?)```/i.exec(body);
    if (fence) body = fence[1].trim();

    const candidates = [body];
    for (const pattern of [/\{[\s\S]*\}/, /\[[\s\S]*\]/]) {
        const match = pattern.exec(body);
        if (match) candidates.push(match[0]);
    }
    for (const candidate of candidates) {
        const parsed = tryParse(candidate);
        if (parsed.ok) return parsed.value;
    }
    return null;
}

/** The detections list; unparseable is an error, an empty list is a real answer. */
function parseDetections(text) {
    const parsed = extractJson(text);
    if (parsed === null) throw new Error(`Model response was not valid JSON: ${String(text).slice(0, 500)}`);
    if (Array.isArray(parsed)) return parsed;
    if (parsed && typeof parsed === 'object') {
        for (const key of ['detections', 'objects', 'boxes', 'results', 'lines']) {
            if (Array.isArray(parsed[key])) return parsed[key];
        }
        const keys = Object.keys(parsed).sort().map((k) => `'${k}'`).join(', ');
        throw new Error(`No detections array in response; keys were [${keys}]`);
    }
    throw new Error(`Unexpected JSON payload type: ${typeof parsed}`);
}

// Python's float(): numbers and numeric strings; anything else is not a number.
function toFloat(value) {
    if (typeof value === 'number') return value;
    if (typeof value === 'string' && value.trim() !== '') return Number(value);
    return NaN;
}

const clamp01 = (v) => Math.max(0, Math.min(1, v));

/**
 * Model detections -> `[[x, y, w, h], confidence, tag]`, relative 0..1.
 * Drops unknown labels (unless there is exactly one tag) and unusable geometry.
 */
function toBoxes(detections, tags, coordFormat) {
    const spec = COORD_FORMATS[coordFormat];
    if (!spec) throw new Error(`Unknown coord_format '${coordFormat}'.`);
    const byName = new Map(tags.map((tag) => [String(tag.name).trim().toLowerCase(), tag.id]));
    const singleTag = tags.length === 1 ? tags[0].id : null;
    const openVocabulary = tags.length === 0;

    const boxes = [];
    let skippedLabel = 0;
    let skippedGeometry = 0;

    for (const detection of detections) {
        if (!detection || typeof detection !== 'object' || Array.isArray(detection)) {
            skippedGeometry += 1;
            continue;
        }
        const key = BOX_KEYS.find((k) => Array.isArray(detection[k]));
        const raw = key ? detection[key] : null;
        if (!raw || raw.length < 4) {
            skippedGeometry += 1;
            continue;
        }

        const rawLabel = String(detection.label || detection.tag || '').trim();
        let tagId;
        if (openVocabulary) {
            tagId = rawLabel || 'object';
        } else {
            tagId = byName.has(rawLabel.toLowerCase()) ? byName.get(rawLabel.toLowerCase()) : singleTag;
            if (tagId === null) {
                skippedLabel += 1;
                continue;
            }
        }

        const [a, b, c, d] = raw.slice(0, 4).map(toFloat);
        if (![a, b, c, d].every(Number.isFinite)) {
            skippedGeometry += 1;
            continue;
        }
        const [x1, y1, x2, y2] = (spec.order === 'yxyx' ? [b, a, d, c] : [a, b, c, d])
            .map((v) => v / COORD_SCALE);

        // Models routinely emit coordinates outside the image, or corners swapped.
        const left = clamp01(Math.min(x1, x2));
        const right = clamp01(Math.max(x1, x2));
        const top = clamp01(Math.min(y1, y2));
        const bottom = clamp01(Math.max(y1, y2));
        if (right - left <= 0 || bottom - top <= 0) {
            skippedGeometry += 1;
            continue;
        }

        const rawConfidence = detection.confidence !== undefined ? detection.confidence
            : (detection.score !== undefined ? detection.score : 1.0);
        const confidence = toFloat(rawConfidence);
        boxes.push([[left, top, right - left, bottom - top],
            Number.isFinite(confidence) ? clamp01(confidence) : 1.0, tagId]);
    }

    return { boxes, skippedLabel, skippedGeometry };
}

/** `[[x, y, w, h], confidence, tag]` -> the entry the orchestrator sends the online node. */
function toEntry([[x, y, w, h], confidence, tag]) {
    return {
        raw_boxes: [[x, y], [x + w, y], [x + w, y + h], [x, y + h]],
        '2points': { xmin: x, ymin: y, xmax: x + w, ymax: y + h },
        tag,
        confidence,
    };
}

module.exports = {
    COORD_FORMATS,
    buildTagsPrompt,
    buildFreePrompt,
    detectionSchema,
    extractJson,
    parseDetections,
    toBoxes,
    toEntry,
};
