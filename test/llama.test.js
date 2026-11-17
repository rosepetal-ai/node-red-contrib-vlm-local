const test = require('node:test');
const assert = require('node:assert/strict');
const http = require('http');
const { chat, samplingParams } = require('../lib/llama');
const { parseNvidiaSmi } = require('../lib/gpu');

async function fakeServer(handler) {
    const seen = [];
    const server = http.createServer((req, res) => {
        let body = '';
        req.on('data', (c) => { body += c; });
        req.on('end', () => { seen.push(JSON.parse(body)); handler(req, res); });
    });
    await new Promise((r) => server.listen(0, '127.0.0.1', r));
    return { port: server.address().port, seen, close: () => server.close() };
}

const call = (port, extra = {}) => chat({
    port, imageBase64: 'AAAA', prompt: 'find', responseFormat: { type: 'json_schema' },
    parameters: {}, timeoutMs: 2000, ...extra,
});

test('sends image, prompt, schema and defaults; returns text and timings', async () => {
    const s = await fakeServer((req, res) => res.end(JSON.stringify({
        choices: [{ message: { content: '{"detections": []}' }, finish_reason: 'stop' }],
        timings: { prompt_ms: 812.4, predicted_ms: 99.6 },
    })));
    try {
        const out = await call(s.port, { parameters: { max_tokens: 100 } });
        assert.deepEqual(out, { text: '{"detections": []}', finishReason: 'stop', promptMs: 812, predictedMs: 100 });
        const body = s.seen[0];
        assert.equal(body.messages[0].content[0].image_url.url, 'data:image/jpeg;base64,AAAA');
        assert.equal(body.messages[0].content[1].text, 'find');
        assert.deepEqual(body.response_format, { type: 'json_schema' });
        assert.equal(body.temperature, 0);
        assert.equal(body.max_tokens, 100);
    } finally { s.close(); }
});

test('a server error and an empty answer are errors, with the reason', async () => {
    const s = await fakeServer((req, res) => {
        if (s.seen.length === 1) { res.writeHead(400); res.end(JSON.stringify({ error: { message: 'bad schema' } })); return; }
        res.end(JSON.stringify({ choices: [{ message: { content: '' }, finish_reason: 'length' }] }));
    });
    try {
        await assert.rejects(call(s.port), /400: bad schema/);
        await assert.rejects(call(s.port), /no text \(finish_reason=length\)/);
    } finally { s.close(); }
});

test('a slow server times out with a clear message', async () => {
    const s = await fakeServer(() => {});
    try {
        await assert.rejects(call(s.port, { timeoutMs: 200 }), /did not answer within 200 ms/);
    } finally { s.close(); }
});

test('unknown parameters are rejected, not dropped', () => {
    assert.throws(() => samplingParams({ api_base: 'x' }), /Unknown parameter\(s\) api_base/);
    assert.deepEqual(samplingParams({ temperature: 0.2 }), { temperature: 0.2, max_tokens: 4096 });
});

test('nvidia-smi output parses; garbage means no GPU', () => {
    assert.deepEqual(parseNvidiaSmi('NVIDIA GeForce RTX 3060, 12288, 399\n'),
        { name: 'NVIDIA GeForce RTX 3060', totalMB: 12288, usedMB: 399, freeMB: 11889 });
    assert.equal(parseNvidiaSmi(''), null);
});
