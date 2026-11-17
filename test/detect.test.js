// Replays fixtures generated from production llm_labeler.py (tests/vlm-local/make_fixtures.py).
const test = require('node:test');
const assert = require('node:assert/strict');
const fixtures = require('./fixtures/labeler.json');
const detect = require('../lib/detect');

for (const c of fixtures.prompts) {
    test(`prompt: ${c.mode} ${c.coord_format}`, () => {
        const actual = c.mode === 'tags'
            ? detect.buildTagsPrompt(c.tags, c.coord_format)
            : detect.buildFreePrompt(c.prompt, c.coord_format);
        assert.equal(actual, c.expected);
    });
}

for (const c of fixtures.schemas) {
    test(`schema: ${c.tags.length} tags`, () => {
        assert.deepEqual(detect.detectionSchema(c.tags), c.expected);
    });
}

const close = (a, b) => Math.abs(a - b) < 1e-9;

for (const c of fixtures.responses) {
    test(`response: ${c.name}`, () => {
        if (c.error) {
            assert.throws(() => detect.parseDetections(c.text), (err) => err.message.startsWith(c.error));
            return;
        }
        const { boxes } = detect.toBoxes(detect.parseDetections(c.text), c.tags, c.coord_format);
        assert.equal(boxes.length, c.expected.length);
        boxes.forEach(([xywh, conf, tag], i) => {
            const [exywh, econf, etag] = c.expected[i];
            assert.ok(xywh.every((v, k) => close(v, exywh[k])), `${xywh} != ${exywh}`);
            assert.ok(close(conf, econf));
            assert.equal(tag, etag);
        });
    });
}

test('toEntry matches the orchestrator envelope (corners TL, TR, BR, BL)', () => {
    assert.deepEqual(detect.toEntry([[0.1, 0.2, 0.3, 0.4], 0.9, 'cap']), {
        raw_boxes: [[0.1, 0.2], [0.4, 0.2], [0.4, 0.6000000000000001], [0.1, 0.6000000000000001]],
        '2points': { xmin: 0.1, ymin: 0.2, xmax: 0.4, ymax: 0.6000000000000001 },
        tag: 'cap',
        confidence: 0.9,
    });
});
