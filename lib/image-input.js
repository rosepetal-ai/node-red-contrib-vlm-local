/**
 * Whatever a flow is holding -> JPEG base64, ready for the wire.
 *
 * Two shapes arrive in practice, both produced by node-red-contrib-image-tools:
 * an encoded Buffer straight from a camera or a file, and the rosepetal raw
 * bitmap `{data, width, height, channels, colorSpace, dtype}` that the image
 * nodes pass between themselves.
 *
 * JPEG, always. The same 5 MP frame as PNG is three to four times larger, and
 * this payload crosses the internet and then gets re-encoded by the provider
 * adapter anyway — so PNG's losslessness buys nothing and costs upload time.
 * A JPEG that arrives already encoded is passed through untouched, because
 * re-encoding it would only lose quality.
 */

const JPEG_QUALITY = 85;

// Magic bytes, enough to tell an encoded image from a raw buffer and to know
// whether a re-encode is needed.
function sniffFormat(buffer) {
    if (buffer.length < 12) return null;
    if (buffer[0] === 0xff && buffer[1] === 0xd8 && buffer[2] === 0xff) return 'jpeg';
    if (buffer[0] === 0x89 && buffer.slice(1, 4).toString('latin1') === 'PNG') return 'png';
    if (buffer.slice(0, 4).toString('latin1') === 'RIFF'
        && buffer.slice(8, 12).toString('latin1') === 'WEBP') return 'webp';
    if (buffer[0] === 0x42 && buffer[1] === 0x4d) return 'bmp';
    if (buffer.slice(0, 3).toString('latin1') === 'GIF') return 'gif';
    return null;
}

function toBuffer(value) {
    if (Buffer.isBuffer(value)) return value;
    if (value instanceof Uint8Array) return Buffer.from(value);
    // A Buffer that has been through JSON (context storage, an HTTP hop).
    if (value && value.type === 'Buffer' && Array.isArray(value.data)) return Buffer.from(value.data);
    return null;
}

function isRawBitmap(value) {
    return Boolean(
        value && typeof value === 'object'
        && toBuffer(value.data)
        && Number.isFinite(value.width) && Number.isFinite(value.height)
    );
}

/**
 * @returns {Promise<{base64: string, width: number|null, height: number|null, passthrough: boolean}>}
 */
async function toJpegBase64(value, { quality = JPEG_QUALITY } = {}) {
    // Required lazily: sharp is a heavy native module and this node may never run.
    const sharp = require('sharp');

    const direct = toBuffer(value);
    if (direct) {
        const format = sniffFormat(direct);
        if (!format) {
            throw new Error(
                'Buffer input is not a recognized encoded image (JPEG/PNG/WebP/BMP/GIF). '
                + 'A raw pixel buffer must be wrapped as {data, width, height, channels}.'
            );
        }
        if (format === 'jpeg') {
            return { base64: direct.toString('base64'), width: null, height: null, passthrough: true };
        }
        const jpeg = await sharp(direct).jpeg({ quality }).toBuffer();
        return { base64: jpeg.toString('base64'), width: null, height: null, passthrough: false };
    }

    if (isRawBitmap(value)) {
        const data = toBuffer(value.data);
        const width = Number(value.width);
        const height = Number(value.height);
        const channels = Number(value.channels) || (data.length / (width * height));
        if (![1, 3, 4].includes(channels)) {
            throw new Error(`Unsupported channel count ${channels}; expected 1, 3 or 4.`);
        }
        if (data.length !== width * height * channels) {
            throw new Error(
                `Raw bitmap is ${data.length} bytes but ${width}x${height}x${channels} needs `
                + `${width * height * channels}.`
            );
        }

        const colorSpace = String(value.colorSpace || (channels === 1 ? 'GRAY' : channels === 4 ? 'RGBA' : 'RGB'))
            .toUpperCase();
        let pixels = data;
        // sharp reads RGB(A). The image nodes also emit BGR(A), and handing that
        // over unswapped sends a red/blue-inverted image that degrades the answer
        // without ever failing.
        if (colorSpace === 'BGR' || colorSpace === 'BGRA') {
            pixels = Buffer.from(data);
            for (let i = 0; i < pixels.length; i += channels) {
                const b = pixels[i];
                pixels[i] = pixels[i + 2];
                pixels[i + 2] = b;
            }
        }

        const jpeg = await sharp(pixels, { raw: { width, height, channels } })
            .jpeg({ quality })
            .toBuffer();
        return { base64: jpeg.toString('base64'), width, height, passthrough: false };
    }

    throw new Error(
        'Unsupported image input. Expected an encoded image Buffer or a rosepetal raw '
        + 'bitmap {data, width, height, channels, colorSpace}.'
    );
}

module.exports = { toJpegBase64, sniffFormat, isRawBitmap, toBuffer, JPEG_QUALITY };
