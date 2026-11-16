const test = require('node:test');
const assert = require('node:assert/strict');
const { expandResultFormats } = require('../lib/result-conversion');

test('a 2points detection gains every other box format', () => {
    const [item] = expandResultFormats([{ tag: 'can', confidence: 0.9, '2points': { xmin: 0.1, ymin: 0.2, xmax: 0.5, ymax: 0.6 } }]);

    assert.equal(item.tag, 'can');
    assert.deepEqual(item.cwh, { x: 0.3, y: 0.4, w: 0.4, h: 0.4, r: 0 });
    assert.deepEqual(item.xywhr, { x: 0.1, y: 0.2, w: 0.4, h: 0.4, r: 0 });
    assert.deepEqual(item.raw_boxes, [[0.1, 0.2], [0.5, 0.2], [0.5, 0.6], [0.1, 0.6]]);
    assert.deepEqual(item['4points'], { x0: 0.1, y0: 0.2, x1: 0.5, y1: 0.2, x2: 0.5, y2: 0.6, x3: 0.1, y3: 0.6 });
});

test('an empty result and a result without geometry pass through', () => {
    assert.deepEqual(expandResultFormats([]), []);
    const classification = { tag: 'ok', confidence: 0.8 };
    assert.equal(expandResultFormats(classification), classification);
});
