const test = require('node:test');
const assert = require('node:assert/strict');
const sharp = require('sharp');
const { toJpegBase64, sniffFormat } = require('../lib/image-input');

test('an encoded JPEG passes through untouched', async () => {
    const jpeg = await sharp({ create: { width: 8, height: 8, channels: 3, background: '#808080' } }).jpeg().toBuffer();
    const out = await toJpegBase64(jpeg);
    assert.equal(out.passthrough, true);
    assert.equal(out.base64, jpeg.toString('base64'));
});

test('a BGR raw bitmap is swapped to RGB before encoding', async () => {
    // Pure blue in BGR order: B=255, G=0, R=0.
    const data = Buffer.alloc(8 * 8 * 3);
    for (let i = 0; i < data.length; i += 3) data[i] = 255;
    const out = await toJpegBase64({ data, width: 8, height: 8, channels: 3, colorSpace: 'BGR' });

    const decoded = Buffer.from(out.base64, 'base64');
    assert.equal(sniffFormat(decoded), 'jpeg');
    const { data: rgb } = await sharp(decoded).raw().toBuffer({ resolveWithObject: true });
    assert.ok(rgb[2] > 200 && rgb[0] < 50, `expected blue, got rgb(${rgb[0]}, ${rgb[1]}, ${rgb[2]})`);
});

test('a raw buffer without dimensions is rejected', async () => {
    await assert.rejects(toJpegBase64(Buffer.alloc(64)), /not a recognized encoded image/);
});
