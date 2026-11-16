/**
 * The picture the model saw, with what it answered drawn on it.
 *
 * One renderer for every node in this package that shows a debug image on the
 * editor canvas: `inferencer`, `ocr-inferencer`, `triton-inferencer` and
 * `vlm-online-inferencer`. They all hand over the frame as the flow holds it and
 * the result as `convertResultFromProto` shaped it, and get back a JPEG.
 *
 * It used to be two things. The `inferencer` asked the serving container for the
 * picture, which ran `Results.plot()` in Python and shipped raw pixels back over
 * /dev/shm; the nodes without a container drew boxes here, in sharp/SVG, and
 * only boxes. Two drawings of the same result that agreed on nothing but the
 * colour green, and a debug aid that cost a gRPC field, a shared-memory file and
 * a cleanup path. Now the container is asked for nothing and every task is drawn
 * here, from the same result the message carries.
 *
 * What is drawn, per task, follows rpcommons' `plot()` for each result type
 * unless there was a reason not to:
 *
 *   detection            box outline, caption "tag: 0.94" inside the top-left
 *   segmentation         each polygon filled at half strength and outlined, the
 *                        same caption at the topmost point
 *   anomaly              the mask filled and outlined, "NOK: 0.87" in the corner
 *   classification       "tag: 0.93" in the corner (top-1, as plot() draws)
 *   ocr                  each text box outlined, caption "text: 0.98"
 *   recognition          "text: 0.98" in the corner
 *   document_orientation "90° (label)" in the corner
 *   embedding, unknown   the frame alone, which still answers "what did it see"
 *
 * Colours are decided here from the tag, with one rule for every node, so the
 * same class is the same colour whichever node drew it (see `tagColor` for why
 * the `color` on the result is not that). Green, `plot()`'s default, is what a
 * tagless item gets; an anomaly is red for NOK and green for OK. Text goes
 * black or white by the luminance of what it sits on.
 *
 * Thickness and font size scale with the image the way `rp-add-bbs` scales
 * them (geometric mean of the sides against 500px), so a flow that draws its
 * own boxes downstream gets a preview that matches what it will produce.
 *
 * Coordinates are normalized 0-1. Anomaly polygons used to be the exception —
 * pixels, of the model's mask on the Python backend and of the original image
 * on the Triton one — which is why the same result drew differently in the two
 * nodes. Both backends normalize them now, but a device can be running an
 * older serving container, so `looksAbsolute` still detects pixel coordinates
 * and scales them by the original size, which it has from the frame's
 * metadata. The caller need not know either way.
 *
 * Draw small: the frame is shrunk to twice the display width first and the
 * overlay is rasterised on the thumbnail. Drawing on a 2576px frame and then
 * shrinking cost ten times more and turned a 4px line into a grey smear.
 */

'use strict';

const fs = require('fs');
const { isRawBitmap } = require('./image-input');

// (0, 255, 0) in the BGR tuple `plot()` defaults to, which is plain green.
const BOX_COLOR = '#00ff00';
// (0, 0, 255) BGR, what `AnomaliesResults.plot()` defaults to.
const ANOMALY_COLOR = '#ff0000';
const LABEL_TEXT_COLOR = '#000000';
// The two the panel exposes, with `rp-add-bbs`'s names, defaults and bounds, so
// a number that means something on that node means the same here.
const DEFAULT_THICKNESS = 2;
const MIN_THICKNESS = 1;
const MAX_THICKNESS = 10;
const DEFAULT_FONT = 0.5;
const MIN_FONT = 0.3;
const MAX_FONT = 2;

// `SegmentationResults.plot(alpha=0.5)` and `AnomaliesResults.plot(alpha=0.4)`.
const SEGMENTATION_FILL_OPACITY = 0.5;
const ANOMALY_FILL_OPACITY = 0.4;

// Breathing room inside the label's filled rectangle, as a fraction of the font
// size rather than a pixel count. `plot()` has none, which is why its captions
// read as if they are touching the box; a fixed one looked right at the default
// size and let the last glyph sit outside the green at a large one.
const LABEL_PADDING_X_RATIO = 0.22;
const LABEL_PADDING_Y_RATIO = 0.14;

// A little slack on a width, wherever one had to be estimated rather than
// measured. It is not a fudge factor for an unknown font -- see `measureLabels`
// for why there is no unknown font any more -- only room for rounding.
const TEXT_WIDTH_SAFETY = 1.03;

// FONT_HERSHEY_SIMPLEX, measured with cv2.getTextSize rather than guessed:
// cap height is 27 * fontScale and the baseline drop is ~6 * fontScale for a
// label with descenders.
const CV_CAP_HEIGHT = 27;
const CV_BASELINE = 6;
// Hershey is a stroke font with no real em box; a sans-serif whose cap height
// matches is as close as SVG gets. Arial's cap height is ~0.72 em.
const CAP_HEIGHT_RATIO = 0.72;
const DESCENDERS = /[gjpqy,;]/;

const TASKS = Object.freeze({
    DETECTION: 'detection',
    SEGMENTATION: 'segmentation',
    ANOMALY: 'anomaly',
    CLASSIFICATION: 'classification',
    OCR: 'ocr',
    RECOGNITION: 'recognition',
    DOCUMENT_ORIENTATION: 'document_orientation',
    EMBEDDING: 'embedding',
});

/** A setting from the editor, bounded, with the default for anything unusable. */
function clamp(value, min, max, fallback) {
    const number = Number(value);
    if (!Number.isFinite(number)) return fallback;
    return Math.min(max, Math.max(min, number));
}

// Helvetica/Arial advance widths, units per 1000 em, for printable ASCII.
//
// The label's background has to be as wide as the text that goes in it. SVG
// cannot measure text and librsvg — the renderer sharp uses — silently ignores
// `textLength`, so the width has to be computed here. cv2's own advance is no
// help: a proportional sans-serif is about a quarter wider than the Hershey
// stroke font at the same cap height, which is exactly how the last character
// ended up outside its background.
//
// Checked against a render: "bottle: 0.94" at font-size 30 measures 146px on
// screen and 148px from this table.
const ADVANCE = (() => {
    const widths = {};
    const put = (chars, width) => { for (const c of chars) widths[c] = width; };
    put(' !', 278); put('"', 355); put('#$', 556); put('%', 889); put('&', 667);
    put("'", 191); put('()', 333); put('*', 389); put('+', 584); put(',', 278);
    put('-', 333); put('./', 278); put('0123456789', 556); put(':;', 278);
    put('<=>', 584); put('?', 556); put('@', 1015);
    put('AE', 667); put('B', 667); put('CD', 722); put('F', 611); put('G', 778);
    put('H', 722); put('I', 278); put('J', 500); put('K', 667); put('L', 556);
    put('M', 833); put('N', 722); put('O', 778); put('P', 667); put('Q', 778);
    put('R', 722); put('S', 667); put('T', 611); put('U', 722); put('V', 667);
    put('W', 944); put('XY', 667); put('Z', 611);
    put('[\\]', 278); put('^', 469); put('_', 556); put('`', 333);
    put('abdeghnopqu', 556); put('c', 500); put('f', 278); put('ij', 222);
    put('k', 500); put('l', 222); put('m', 833); put('r', 333); put('s', 500);
    put('t', 278); put('v', 500); put('w', 722); put('x', 500); put('y', 500);
    put('z', 500); put('{}', 334); put('|', 260); put('~', 584);
    return widths;
})();

// An unknown glyph gets a lowercase-average width rather than nothing, so an
// accented tag name still gets a box wide enough to hold it.
const DEFAULT_ADVANCE = 556;

/** How wide `text` renders at `fontSize`, in pixels. */
function textWidthAt(text, fontSize) {
    let units = 0;
    for (const char of String(text)) units += ADVANCE[char] === undefined ? DEFAULT_ADVANCE : ADVANCE[char];
    return (units / 1000) * fontSize;
}

const escapeXml = (value) => String(value == null ? '' : value)
    .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;').replace(/'/g, '&apos;');

// ---------------------------------------------------------------------------
// What kind of result this is
// ---------------------------------------------------------------------------

/**
 * The task a converted result belongs to, from its shape.
 *
 * The callers that have the proto at hand pass the task explicitly (the oneof
 * key on the `ImageResult`), because the shapes are not all distinct: a list
 * of one `{polygons, tag, confidence}` could be either a segmentation or an
 * anomaly from a node old enough to still wrap it. This is the fallback for a
 * caller that only has the converted value, and it guesses anomaly from the
 * object shape, or from the tags that task uses.
 */
function taskOf(result) {
    if (Array.isArray(result)) {
        const first = result.find((item) => item && typeof item === 'object');
        if (!first) return TASKS.DETECTION;
        if (Array.isArray(first.polygons)) {
            const anomalyTag = result.length === 1 && /^(ok|nok)$/i.test(String(first.tag || ''));
            return anomalyTag ? TASKS.ANOMALY : TASKS.SEGMENTATION;
        }
        return TASKS.DETECTION;
    }
    if (!result || typeof result !== 'object') return null;
    // Anomaly is the one object-shaped result that carries geometry; it has to
    // be tested before the tag+confidence test below, which would otherwise
    // read it as a classification.
    if (Array.isArray(result.polygons)) return TASKS.ANOMALY;
    if (Array.isArray(result.texts)) return TASKS.OCR;
    if (Array.isArray(result.all_tags)) return TASKS.CLASSIFICATION;
    if (Array.isArray(result.embedding)) return TASKS.EMBEDDING;
    if (result.orientation_label !== undefined || (result.angle !== undefined && result.text === undefined)) {
        return TASKS.DOCUMENT_ORIENTATION;
    }
    if (typeof result.text === 'string') return TASKS.RECOGNITION;
    if (result.tag !== undefined && result.confidence !== undefined) return TASKS.CLASSIFICATION;
    return null;
}

// ---------------------------------------------------------------------------
// Colour
// ---------------------------------------------------------------------------

/**
 * The colour an item is drawn in, decided here rather than taken from the result.
 *
 * The result does carry a `color`, and the first version of this used it. It
 * turned out to be three different things. The serving container computes one
 * (rpcommons `_hash_color`, a numpy RNG seeded with the tag's MD5) and then
 * drops it on the floor: `rgb_string_to_dict` is written with `\\(` inside a raw
 * string, so its regex wants a literal backslash and never matches, and the
 * proto goes out without a colour. The Triton backend attaches the first three
 * bytes of the MD5 instead, which its own comment says is not the Python rule.
 * `rp-add-bbs` has a fourth. So "the platform's colour" is not one thing, and
 * an `inferencer` and a `triton-inferencer` looking at the same class showed a
 * green box and a navy one side by side.
 *
 * One rule, here, is what makes the same class the same colour on every node.
 * The class index picks a hue along the golden angle, which is what spreads a
 * model's five classes as far apart on the wheel as five things can be --
 * a hash of the tag was tried first and put `bottle` and `cap` on the same
 * colour, as any hash will for a handful of names. An item without a class
 * index (the VLM node's detections) hashes its tag instead, and one with
 * neither is green. The colour on the message is left untouched for whoever
 * consumes it downstream.
 */
const GOLDEN_ANGLE = 137.508;

const hex2 = (n) => Math.max(0, Math.min(255, Math.round(Number(n) || 0))).toString(16).padStart(2, '0');

function hslToHex(hue, saturation, lightness) {
    const h = ((hue % 360) + 360) % 360;
    const c = (1 - Math.abs(2 * lightness - 1)) * saturation;
    const x = c * (1 - Math.abs(((h / 60) % 2) - 1));
    const m = lightness - c / 2;
    const [r1, g1, b1] = h < 60 ? [c, x, 0] : h < 120 ? [x, c, 0] : h < 180 ? [0, c, x]
        : h < 240 ? [0, x, c] : h < 300 ? [x, 0, c] : [c, 0, x];
    return `#${hex2((r1 + m) * 255)}${hex2((g1 + m) * 255)}${hex2((b1 + m) * 255)}`;
}

/** A hue for a name: FNV-1a over the string, spread across the wheel. */
function tagHue(tag) {
    const text = String(tag);
    let hash = 0x811c9dc5;
    for (let i = 0; i < text.length; i += 1) {
        hash ^= text.charCodeAt(i);
        hash = Math.imul(hash, 0x01000193) >>> 0;
    }
    return hash % 360;
}

/** The colour for a class index, or for a name when there is no index. */
function tagColor(tag, classId) {
    const index = Number(classId);
    if (Number.isInteger(index) && index >= 0) return hslToHex(index * GOLDEN_ANGLE, 0.9, 0.5);
    const text = String(tag == null ? '' : tag);
    if (!text) return BOX_COLOR;
    return hslToHex(tagHue(text), 0.9, 0.5);
}

/** The colour an item is drawn in: from its class or tag, or the fallback when it has neither. */
function colorOf(item, fallback) {
    if (!item || typeof item !== 'object') return fallback || BOX_COLOR;
    const hasClass = Number.isInteger(Number(item.classId)) && item.classId !== undefined && item.classId !== null;
    const hasTag = item.tag != null && String(item.tag) !== '';
    if (!hasClass && !hasTag) return fallback || BOX_COLOR;
    return tagColor(item.tag, hasClass ? item.classId : undefined);
}

/** Black on a light background, white on a dark one. */
function textColorOn(hex) {
    const m = /^#([0-9a-f]{2})([0-9a-f]{2})([0-9a-f]{2})$/i.exec(hex);
    if (!m) return LABEL_TEXT_COLOR;
    const [r, g, b] = [m[1], m[2], m[3]].map((h) => parseInt(h, 16) / 255);
    const luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b;
    return luminance < 0.45 ? '#ffffff' : LABEL_TEXT_COLOR;
}

// ---------------------------------------------------------------------------
// Geometry
// ---------------------------------------------------------------------------

/**
 * The four corners of one detection, in absolute pixels.
 *
 * `raw_boxes` is preferred because `plot()` draws a closed polyline over the
 * corners as given, which need not be axis-aligned. `2points` is the fallback
 * for an entry that somehow lacks them.
 */
function cornersOf(detection, width, height) {
    const clampUnit = (value, max) => Math.min(max, Math.max(0, Number(value) || 0));
    const raw = detection && detection.raw_boxes;
    if (Array.isArray(raw) && raw.length >= 3) {
        return raw.map((point) => [clampUnit(point[0], 1) * width, clampUnit(point[1], 1) * height]);
    }
    const box = detection && detection['2points'];
    if (!box) return null;
    const x0 = clampUnit(box.xmin, 1) * width;
    const y0 = clampUnit(box.ymin, 1) * height;
    const x1 = clampUnit(box.xmax, 1) * width;
    const y1 = clampUnit(box.ymax, 1) * height;
    return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]];
}

/**
 * One polygon in absolute pixels of the drawing surface.
 *
 * `scale` maps the polygon's own units to that surface: `[width, height]` for
 * normalized coordinates, `[width / originalWidth, height / originalHeight]`
 * for pixels of the original frame.
 */
function polygonPixels(polygon, scale, width, height) {
    if (!Array.isArray(polygon)) return [];
    const points = [];
    for (const point of polygon) {
        if (!Array.isArray(point) || point.length < 2) continue;
        const x = Math.min(width, Math.max(0, (Number(point[0]) || 0) * scale[0]));
        const y = Math.min(height, Math.max(0, (Number(point[1]) || 0) * scale[1]));
        points.push([x, y]);
    }
    return points;
}

/** True when a polygon list is in pixels rather than 0-1 (anomaly's contract). */
function looksAbsolute(polygons) {
    for (const polygon of polygons || []) {
        for (const point of polygon || []) {
            if (Array.isArray(point) && (Number(point[0]) > 1.5 || Number(point[1]) > 1.5)) return true;
        }
    }
    return false;
}

const pointList = (points) => points.map(([x, y]) => `${x.toFixed(1)},${y.toFixed(1)}`).join(' ');

/** The topmost point of a set, leftmost on a tie: where its caption goes. */
function topmost(points) {
    return points.reduce((best, point) => (
        point[1] < best[1] || (point[1] === best[1] && point[0] < best[0]) ? point : best
    ), points[0]);
}

// ---------------------------------------------------------------------------
// Scale, metrics, captions
// ---------------------------------------------------------------------------

/**
 * Box thickness and font scale, the way `rp-add-bbs` computes them.
 *
 * Both are relative to the image rather than absolute, so the same settings look
 * the same on a 320px thumbnail and a 2576px frame. The reference is the
 * geometric mean of the sides against 500px, which is that node's convention and
 * worth keeping: a flow that draws its own boxes downstream with `rp-add-bbs`
 * then gets a preview here that matches what it will produce.
 *
 * This is a deliberate departure from `DetectionResults.plot()`, whose factor is
 * `width / 600` and which has no settings at all.
 */
function drawingScale(width, height, boxThickness, fontSize) {
    const scaleFactor = Math.sqrt(width * height) / 500;
    return {
        stroke: Math.max(1, Math.round(boxThickness * scaleFactor * 0.5)),
        fontScale: fontSize * Math.min(2, scaleFactor * 0.5),
    };
}

/**
 * The numbers every label on one frame is drawn with.
 *
 * Separated out because `renderDebugImage` needs the font size before the
 * overlay exists, so it can measure the captions at exactly the size they will
 * be drawn at.
 */
function overlayMetrics(width, height, options) {
    const settings = options || {};
    const boxThickness = clamp(settings.boxThickness, MIN_THICKNESS, MAX_THICKNESS,
                               DEFAULT_THICKNESS);
    const requestedFont = clamp(settings.fontSize, MIN_FONT, MAX_FONT, DEFAULT_FONT);
    const { stroke, fontScale } = drawingScale(width, height, boxThickness, requestedFont);
    const capHeight = CV_CAP_HEIGHT * fontScale;
    return { stroke, fontScale, capHeight, fontSize: capHeight / CAP_HEIGHT_RATIO };
}

/** "tag: 0.94", or just the tag, or just the number, or '' when there is nothing. */
function caption(tag, confidence) {
    const number = Number(confidence);
    const text = tag == null ? '' : String(tag);
    if (!text) return Number.isFinite(number) ? number.toFixed(2) : '';
    return text + (Number.isFinite(number) ? `: ${number.toFixed(2)}` : '');
}

/** The caption a detection gets, or '' when it has nothing to say. */
function labelFor(detection) {
    return caption(detection.tag, detection.confidence);
}

/**
 * Where a box's label goes: inside its own box, at the top-left corner.
 *
 * Both `DetectionResults.plot()` and `rp-add-bbs` put it *above* the box, and
 * that was the first thing tried here. It does not survive contact with a real
 * frame. An object photographed head-on runs off the top of the picture, so its
 * box starts at or near y=0 and there is nothing above it; `plot()` then draws
 * at negative y and the caption vanishes, `rp-add-bbs` clamps the background but
 * not the text and draws an inverted rectangle under a floating caption.
 *
 * Requiring clearance before going above only moved the problem. The preview is
 * a 320px thumbnail of a 2576px frame, an eightfold reduction, so the 19px of
 * clearance that looks generous while computing the overlay is two pixels once
 * it is on screen -- still flush against the border, still reported as text
 * spilling out. There is no threshold that fixes this, because the thing being
 * asked for does not fit.
 *
 * Inside the box is one rule with no edge cases. It cannot collide with the
 * frame border unless the box is itself at the border, in which case the label
 * is exactly where it belongs. It costs the top strip of the object, which on a
 * thumbnail meant for checking that the boxes landed is not what you are
 * looking at.
 */
function labelBox(x0, y0, textWidth, labelHeight, width, height, padX) {
    const boxWidth = textWidth * TEXT_WIDTH_SAFETY + padX * 2;

    // Clamped so a box hard against an edge still gets a readable label, and a
    // label wider than its own box runs into the picture rather than out of it.
    let left = x0;
    if (left + boxWidth > width) left = width - boxWidth;
    if (left < 0) left = 0;

    let top = y0;
    if (top + labelHeight > height) top = height - labelHeight;
    if (top < 0) top = 0;

    return { left, top, boxWidth, boxHeight: Math.min(labelHeight, height) };
}

// ---------------------------------------------------------------------------
// The overlay
// ---------------------------------------------------------------------------

/**
 * Collects shapes and captions for one frame and lays the captions out.
 *
 * Shapes are SVG fragments. Captions are placed in two ways: `at(x, y)` puts
 * one inside the thing it names, `corner()` stacks one under the previous
 * corner caption at the top-left of the frame, the way `plot()` writes a
 * classification tag.
 */
class Overlay {
    constructor(width, height, metrics, widths) {
        this.width = width;
        this.height = height;
        this.metrics = metrics;
        this.widths = widths;
        this.parts = [];
        this.cornerTop = 0;
    }

    shape(svg) { this.parts.push(svg); }

    _measure(text) {
        const { fontSize, capHeight, fontScale } = this.metrics;
        const measured = this.widths && this.widths.get ? this.widths.get(text) : undefined;
        const padX = Math.max(2, fontSize * LABEL_PADDING_X_RATIO);
        const padY = Math.max(1, fontSize * LABEL_PADDING_Y_RATIO);
        const baseline = DESCENDERS.test(text) ? CV_BASELINE * fontScale : 0;
        const textWidth = typeof measured === 'number' ? measured : textWidthAt(text, fontSize);
        return { padX, padY, textWidth, labelHeight: capHeight + baseline + padY * 2 };
    }

    _draw(text, place, padX, padY, fill) {
        const { fontSize, capHeight } = this.metrics;
        this.parts.push(`<rect x="${place.left.toFixed(1)}" y="${place.top.toFixed(1)}" `
            + `width="${place.boxWidth.toFixed(1)}" height="${place.boxHeight.toFixed(1)}" `
            + `fill="${fill}"/>`);
        this.parts.push(`<text x="${(place.left + padX).toFixed(1)}" `
            + `y="${(place.top + padY + capHeight).toFixed(1)}" `
            + `font-family="Helvetica, Arial, sans-serif" font-size="${fontSize.toFixed(1)}" `
            + `fill="${textColorOn(fill)}">${escapeXml(text)}</text>`);
    }

    /** A caption inside the shape whose top-left corner is (x, y). */
    at(text, x, y, fill) {
        if (!text) return;
        const { padX, padY, textWidth, labelHeight } = this._measure(text);
        const place = labelBox(x, y, textWidth, labelHeight, this.width, this.height, padX);
        this._draw(text, place, padX, padY, fill);
    }

    /** A caption at the top-left of the frame, under any drawn before it. */
    corner(text, fill) {
        if (!text) return;
        const { padX, padY, textWidth, labelHeight } = this._measure(text);
        const place = labelBox(0, this.cornerTop, textWidth, labelHeight, this.width, this.height, padX);
        this._draw(text, place, padX, padY, fill);
        this.cornerTop = place.top + place.boxHeight;
    }

    toSvg() {
        return `<svg xmlns="http://www.w3.org/2000/svg" width="${this.width}" height="${this.height}">`
            + this.parts.join('') + '</svg>';
    }
}

function drawDetections(overlay, items, stroke) {
    for (const item of items) {
        if (!item || typeof item !== 'object') continue;
        const corners = cornersOf(item, overlay.width, overlay.height);
        if (!corners) continue;
        const color = colorOf(item);
        overlay.shape(`<polygon points="${pointList(corners)}" fill="none" stroke="${color}" `
            + `stroke-width="${stroke}"/>`);
        // The topmost corner, like `rp-add-bbs`. `plot()` uses the first corner
        // as given, which for a rotated box can be the bottom one and puts the
        // caption under the object it names.
        const anchor = topmost(corners);
        overlay.at(labelFor(item), anchor[0], anchor[1], color);
    }
}

/**
 * Filled polygons, outlined, with one caption per item at its topmost point.
 *
 * The outline is a departure from `SegmentationResults.plot()`, which fills and
 * then draws the bounding box. On a thumbnail a half-strength fill on a
 * similar-coloured background disappears, and a box around a mask says less
 * about the mask than its own edge does.
 */
function drawPolygons(overlay, items, stroke, scale, opacity, fallbackColor, withCaptions) {
    for (const item of items) {
        if (!item || !Array.isArray(item.polygons)) continue;
        const color = colorOf(item, fallbackColor);
        let anchor = null;
        for (const polygon of item.polygons) {
            const points = polygonPixels(polygon, scale, overlay.width, overlay.height);
            if (points.length < 3) continue;
            overlay.shape(`<polygon points="${pointList(points)}" fill="${color}" `
                + `fill-opacity="${opacity}" stroke="${color}" stroke-width="${stroke}"/>`);
            const top = topmost(points);
            if (!anchor || top[1] < anchor[1]) anchor = top;
        }
        if (!withCaptions || !anchor) continue;
        overlay.at(labelFor(item), anchor[0], anchor[1], color);
    }
}

/** Red for a NOK verdict, green for OK. */
function anomalyColor(item) {
    if (!item) return ANOMALY_COLOR;
    const nok = Number(item.classId) === 1 || /nok|anomal|defect|bad/i.test(String(item.tag || ''));
    return nok ? ANOMALY_COLOR : BOX_COLOR;
}

function drawOcr(overlay, result, stroke) {
    for (const entry of result.texts || []) {
        if (!entry || !Array.isArray(entry.coords) || entry.coords.length < 3) continue;
        const points = polygonPixels(entry.coords, [overlay.width, overlay.height],
                                     overlay.width, overlay.height);
        if (points.length < 3) continue;
        overlay.shape(`<polygon points="${pointList(points)}" fill="none" stroke="${BOX_COLOR}" `
            + `stroke-width="${stroke}"/>`);
        const anchor = topmost(points);
        overlay.at(caption(entry.text, entry.score), anchor[0], anchor[1], BOX_COLOR);
    }
}

/**
 * Every caption a result will produce, before anything is drawn.
 *
 * `renderDebugImage` measures these in the real font so the backgrounds fit;
 * keeping the list here, next to the drawing that uses it, is what keeps the
 * two in step.
 */
function captionsOf(result, task) {
    const texts = [];
    switch (task) {
        case TASKS.DETECTION:
        case TASKS.SEGMENTATION:
            for (const item of Array.isArray(result) ? result : []) {
                if (item && typeof item === 'object') texts.push(labelFor(item));
            }
            break;
        case TASKS.ANOMALY: {
            const item = Array.isArray(result) ? result[0] : result;
            if (item) texts.push(labelFor(item));
            break;
        }
        case TASKS.CLASSIFICATION:
            if (result && typeof result === 'object') texts.push(labelFor(result));
            break;
        case TASKS.OCR:
            for (const entry of (result && result.texts) || []) {
                if (entry) texts.push(caption(entry.text, entry.score));
            }
            break;
        case TASKS.RECOGNITION:
            if (result) texts.push(caption(result.text, result.confidence));
            break;
        case TASKS.DOCUMENT_ORIENTATION:
            if (result) texts.push(orientationCaption(result));
            break;
        default:
            break;
    }
    return [...new Set(texts.filter(Boolean))];
}

function orientationCaption(result) {
    const angle = Number(result.angle);
    const label = result.orientation_label ? String(result.orientation_label) : '';
    const degrees = Number.isFinite(angle) ? `${Math.round(angle)}°` : '';
    if (degrees && label) return `${degrees} (${label})`;
    return degrees || label;
}

/**
 * The SVG drawn over the frame.
 *
 * @param {*}      result   what `convertResultFromProto` returned for one image
 * @param {number} width    pixels of the surface being drawn on
 * @param {number} height   pixels of the surface being drawn on
 * @param {Object} [options]
 *   `task`         the result's task; guessed from the shape when absent
 *   `boxThickness` and `fontSize`, as the panel offers them, with the same
 *                  meaning and defaults `rp-add-bbs` gives those two names
 *   `original`     `{width, height}` of the frame the result was computed on,
 *                  for the one task whose coordinates are in its pixels
 *   `widths`       measured caption widths, from `measureLabels`
 */
function buildOverlay(result, width, height, options) {
    const settings = options || {};
    const task = settings.task || taskOf(result);
    const metrics = overlayMetrics(width, height, settings);
    const overlay = new Overlay(width, height, metrics, settings.widths);
    const { stroke } = metrics;
    const normalized = [width, height];

    switch (task) {
        case TASKS.DETECTION:
            drawDetections(overlay, Array.isArray(result) ? result : [], stroke);
            break;

        case TASKS.SEGMENTATION:
            drawPolygons(overlay, Array.isArray(result) ? result : [], stroke, normalized,
                         SEGMENTATION_FILL_OPACITY, BOX_COLOR, true);
            break;

        case TASKS.ANOMALY: {
            // One object since the nodes stopped wrapping the single anomaly
            // verdict in a list; still accepts the list an older flow may hold.
            const items = Array.isArray(result) ? result : (result ? [result] : []);
            // Normalized, unless an older serving container sent pixels.
            const original = settings.original || {};
            const absolute = items.some((item) => item && looksAbsolute(item.polygons));
            const scale = absolute && original.width > 0 && original.height > 0
                ? [width / original.width, height / original.height]
                : normalized;
            // The verdict picks the colour: red for NOK, green for OK. A hashed
            // tag colour would make "OK" some arbitrary hue on a frame whose
            // whole point is the verdict.
            const item = items[0];
            const color = anomalyColor(item);
            for (const entry of items) {
                if (!entry || !Array.isArray(entry.polygons)) continue;
                for (const polygon of entry.polygons) {
                    const points = polygonPixels(polygon, scale, width, height);
                    if (points.length < 3) continue;
                    overlay.shape(`<polygon points="${pointList(points)}" fill="${color}" `
                        + `fill-opacity="${ANOMALY_FILL_OPACITY}" stroke="${color}" stroke-width="${stroke}"/>`);
                }
            }
            if (item) overlay.corner(labelFor(item), color);
            break;
        }

        case TASKS.CLASSIFICATION:
            if (result && typeof result === 'object') overlay.corner(labelFor(result), BOX_COLOR);
            break;

        case TASKS.OCR:
            if (result && typeof result === 'object') drawOcr(overlay, result, stroke);
            break;

        case TASKS.RECOGNITION:
            if (result) overlay.corner(caption(result.text, result.confidence), BOX_COLOR);
            break;

        case TASKS.DOCUMENT_ORIENTATION:
            if (result) overlay.corner(orientationCaption(result), BOX_COLOR);
            break;

        default:
            break;
    }

    return overlay.toSvg();
}

// ---------------------------------------------------------------------------
// Measuring text in the font that will draw it
// ---------------------------------------------------------------------------

/**
 * How wide each caption really is, on this machine, in this font.
 *
 * Every estimate here was wrong and the reason took three rounds to find: the
 * font is not the same everywhere. `font-family="Helvetica, Arial, sans-serif"`
 * resolves through fontconfig, and a Debian container has neither of the first
 * two -- it lands on DejaVu Sans, which is about a tenth wider than Helvetica.
 * So a baked advance table is right on a developer's machine and short on the
 * device, which is exactly the shape of "it still cuts the confidence off".
 *
 * There is no table that survives that. The width has to come from the renderer
 * that will draw it, so each caption is rasterised once at the size it will be
 * used at and its ink measured. Cached on text and size: a flow sends the same
 * few tags over and over, so this is paid once and then never again.
 *
 * Falls back to the table if the measurement fails, because a slightly narrow
 * background is a better outcome than no debug image.
 */
const measuredWidths = new Map();

async function measureLabels(sharp, labels, fontSize) {
    const widths = new Map();
    for (const label of labels) {
        const key = `${fontSize.toFixed(2)}|${label}`;
        if (measuredWidths.has(key)) {
            widths.set(label, measuredWidths.get(key));
            continue;
        }
        let width;
        try {
            // Generous canvas: the point is to find where the ink stops, so it
            // must not be the thing that stops it.
            const canvas = Math.ceil(textWidthAt(label, fontSize) * 2 + fontSize * 4);
            const height = Math.ceil(fontSize * 2.5);
            const svg = `<svg xmlns="http://www.w3.org/2000/svg" width="${canvas}" `
                + `height="${height}"><text x="0" y="${(fontSize * 1.5).toFixed(1)}" `
                + `font-family="Helvetica, Arial, sans-serif" font-size="${fontSize.toFixed(1)}" `
                + `fill="#000000">${escapeXml(label)}</text></svg>`;
            const { data, info } = await sharp(Buffer.from(svg))
                .ensureAlpha().raw().toBuffer({ resolveWithObject: true });
            let right = -1;
            for (let y = 0; y < info.height; y += 1) {
                for (let x = info.width - 1; x > right; x -= 1) {
                    if (data[(y * info.width + x) * info.channels + 3] > 16) {
                        right = x;
                        break;
                    }
                }
            }
            width = right >= 0 ? right + 1 : textWidthAt(label, fontSize);
        } catch {
            width = textWidthAt(label, fontSize);
        }
        measuredWidths.set(key, width);
        widths.set(label, width);
    }
    return widths;
}

// ---------------------------------------------------------------------------
// Opening whatever the flow is holding
// ---------------------------------------------------------------------------

/** The shared-memory handle the `inferencer` accepts as an input image. */
function isShmHandle(value) {
    return Boolean(value && typeof value === 'object'
        && value.MemorySegmentName && value.shared_memory_handle
        && Number.isFinite(value.shared_memory_handle.width)
        && Number.isFinite(value.shared_memory_handle.height));
}

/**
 * Whatever the flow is holding, as something sharp can open without a detour.
 *
 * A raw bitmap goes straight in as raw pixels. Encoding it to JPEG first so that
 * this function could decode it again cost 52ms on a 2576px frame -- more than
 * the entire drawing -- and the flows this runs in pass raw bitmaps between the
 * image nodes, so that was the common case rather than the odd one.
 *
 * sharp reads RGB(A). The image nodes also emit BGR(A), and a bitmap handed
 * over unswapped draws a red/blue-inverted frame that nothing ever reports.
 *
 * Returns `{input, options}` for `sharp(input, options)`, so the caller can
 * open the same source twice (once for its size, once to draw on it) without
 * the swap or the file read happening twice.
 */
async function resolveSource(image) {
    if (isShmHandle(image)) {
        const handle = image.shared_memory_handle;
        const channels = Number(image.channels) || 3;
        const data = await fs.promises.readFile('/dev/shm/' + image.MemorySegmentName);
        return { input: data, options: { raw: { width: handle.width, height: handle.height, channels } } };
    }
    if (isRawBitmap(image)) {
        const channels = Number(image.channels) || Math.round(image.data.length / (image.width * image.height));
        const colorSpace = String(image.colorSpace || '').toUpperCase();
        let data = Buffer.isBuffer(image.data) ? image.data : Buffer.from(image.data);
        if ((colorSpace === 'BGR' || colorSpace === 'BGRA') && channels >= 3) {
            const swapped = Buffer.from(data);
            for (let i = 0; i < swapped.length; i += channels) {
                const b = swapped[i];
                swapped[i] = swapped[i + 2];
                swapped[i + 2] = b;
            }
            data = swapped;
        }
        return { input: data, options: { raw: { width: image.width, height: image.height, channels } } };
    }
    if (Buffer.isBuffer(image)) return { input: image, options: {} };
    const text = String(image == null ? '' : image);
    const base64 = text.startsWith('data:') ? text.slice(text.indexOf(',') + 1) : text;
    return { input: Buffer.from(base64, 'base64'), options: {} };
}

/**
 * @param {Buffer|Object|string} image  the frame that was inferred: an encoded
 *                              buffer, a rosepetal raw bitmap, a shared-memory
 *                              handle, or base64
 * @param {*}      result       what `convertResultFromProto` returned for that
 *                              image, with or without the derived formats
 * @param {number} width        the editor's debug width; rendered at twice it
 * @param {Object} [options]    `task`, `boxThickness`, `fontSize`
 * @returns {Promise<string>}   annotated JPEG, base64, ready for RED.comms
 */
async function renderDebugImage(image, result, width, options) {
    const sharp = require('sharp');
    const settings = Object.assign({}, options || {});
    const source = await resolveSource(image);
    const metadata = await sharp(source.input, source.options).metadata();

    // Shrink first, then draw on what is left.
    //
    // This used to go the other way: rasterise the overlay over the full frame,
    // encode that, then throw most of it away in a downscale. For a 2576px
    // camera frame that is a 6 MB intermediate PNG and about half a second, to
    // produce a 320px thumbnail -- ten times the cost of drawing on the
    // thumbnail directly, and slow enough to be visible as a picture that lands
    // well after the results.
    //
    // Drawing small is also the better picture. Stroke and font scale with the
    // image, so a 2px setting on a 2576px frame is a 4px line that an eightfold
    // reduction turns into a grey smear; at the target size it is a solid line,
    // and the text is rasterised at the resolution it will be seen at instead of
    // being shrunk into mush.
    //
    // `max_width = debug_width * 2`: rendered at twice the display size so it
    // stays sharp, and never upscaled past the original.
    const maxWidth = Math.max(32, Math.round(Number(width) || 160)) * 2;

    // Down to raw pixels rather than to an encoded buffer. The overlay has to
    // match the resized frame exactly, and sharp's own rounding is the only
    // thing that knows what that is -- computing it here would be a second
    // opinion that is occasionally a pixel out. Raw also means no encode and
    // decode in the middle just to ask how big it came out.
    const shrink = sharp(source.input, source.options);
    if (metadata.width > maxWidth) shrink.resize({ width: maxWidth });
    const { data: pixels, info } = await shrink
        .toColourspace('srgb')
        .raw()
        .toBuffer({ resolveWithObject: true });

    const task = settings.task || taskOf(result);
    const { fontSize } = overlayMetrics(info.width, info.height, settings);
    const widths = await measureLabels(sharp, captionsOf(result, task), fontSize);
    const overlay = buildOverlay(result, info.width, info.height, Object.assign(settings, {
        task,
        widths,
        original: { width: metadata.width, height: metadata.height },
    }));

    // JPEG, like the `inferencer` has always published. The same thumbnail as a
    // level-9 PNG is 99 KB against 14 KB and takes twice as long to encode, and
    // it crosses a websocket to an editor that is often on the other end of a
    // VPN from the device. 80 rather than the old 70 because this picture is
    // mostly thin coloured lines, which are what JPEG is worst at.
    const annotated = await sharp(pixels, {
        raw: { width: info.width, height: info.height, channels: info.channels },
    })
        .composite([{ input: Buffer.from(overlay), top: 0, left: 0 }])
        .jpeg({ quality: 80 })
        .toBuffer();
    return annotated.toString('base64');
}

module.exports = {
    renderDebugImage, buildOverlay, labelBox, drawingScale, overlayMetrics, labelFor,
    caption, captionsOf, measureLabels, cornersOf, textWidthAt, taskOf, colorOf, tagColor, hslToHex, textColorOn,
    resolveSource,
    TASKS, BOX_COLOR, ANOMALY_COLOR, DEFAULT_THICKNESS, DEFAULT_FONT,
};
