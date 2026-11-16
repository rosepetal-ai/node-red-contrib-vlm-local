const test = require('node:test');
const assert = require('node:assert/strict');
const sharp = require('sharp');
const { renderDebugImage } = require('../lib/debug-image');
const { expandResultFormats } = require('../lib/result-conversion');

test('a detection is drawn on the frame and comes back as a JPEG', async () => {
    const frame = await sharp({ create: { width: 64, height: 48, channels: 3, background: '#000000' } }).png().toBuffer();
    const detections = expandResultFormats([{ tag: 'can', confidence: 0.9, '2points': { xmin: 0.2, ymin: 0.2, xmax: 0.8, ymax: 0.8 } }]);

    const base64 = await renderDebugImage(frame, detections, 64, {});
    const meta = await sharp(Buffer.from(base64, 'base64')).metadata();
    assert.equal(meta.format, 'jpeg');

    // Something was drawn: the black frame is no longer all black.
    const { data } = await sharp(Buffer.from(base64, 'base64')).raw().toBuffer({ resolveWithObject: true });
    assert.ok(data.some((v) => v > 100));
});
