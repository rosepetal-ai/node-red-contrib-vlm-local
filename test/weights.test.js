const test = require('node:test');
const assert = require('node:assert/strict');
const crypto = require('crypto');
const fs = require('fs');
const http = require('http');
const os = require('os');
const path = require('path');
const { ensureWeights, weightsStatus } = require('../lib/weights');

const blob = (n, seed) => Buffer.from(Array.from({ length: n }, (_, i) => (i * 31 + seed) % 256));
const sha = (b) => crypto.createHash('sha256').update(b).digest('hex');
const MODEL = blob(200000, 1);
const MMPROJ = blob(50000, 7);
const REV = 'a'.repeat(40);

function entry(overrides = {}) {
    return {
        id: 'fake',
        files: {
            model: { repo: 'org/fake', revision: REV, path: 'm.gguf', bytes: MODEL.length, sha256: sha(MODEL) },
            mmproj: { repo: 'org/fake', revision: REV, path: 'mmproj.gguf', bytes: MMPROJ.length, sha256: sha(MMPROJ) },
        },
        ...overrides,
    };
}

/** HF-like server with Range support; `cutAfter` drops the first response mid-file. */
async function fakeHub({ cutAfter = null, ignoreRange = false } = {}) {
    const files = { [`/org/fake/resolve/${REV}/m.gguf`]: MODEL, [`/org/fake/resolve/${REV}/mmproj.gguf`]: MMPROJ };
    const log = [];
    let cut = cutAfter;
    const server = http.createServer((req, res) => {
        const body = files[req.url];
        log.push({ url: req.url, range: req.headers.range });
        if (!body) { res.writeHead(404).end(); return; }
        const m = !ignoreRange && /bytes=(\d+)-/.exec(req.headers.range || '');
        const start = m ? Number(m[1]) : 0;
        res.writeHead(m ? 206 : 200, { 'Content-Length': body.length - start });
        if (cut !== null && req.url.endsWith('m.gguf')) {
            res.write(body.subarray(start, start + cut));
            cut = null;
            setTimeout(() => res.destroy(), 50); // let the first bytes arrive, then drop
            return;
        }
        res.end(body.subarray(start));
    });
    await new Promise((r) => server.listen(0, '127.0.0.1', r));
    return { baseUrl: `http://127.0.0.1:${server.address().port}`, log, close: () => server.close() };
}

const tmpRoot = () => fs.mkdtempSync(path.join(os.tmpdir(), 'vlm-w-'));

test('downloads both files, verifies them and reports installed', async () => {
    const hub = await fakeHub();
    const root = tmpRoot();
    const progress = [];
    try {
        assert.equal(weightsStatus(entry(), root).missingBytes, MODEL.length + MMPROJ.length);
        await ensureWeights(entry(), { root, baseUrl: hub.baseUrl, onProgress: (p) => progress.push(p) });
        const status = weightsStatus(entry(), root);
        assert.equal(status.installed, true);
        assert.equal(status.missingBytes, 0);
        assert.deepEqual(fs.readFileSync(path.join(root, 'fake', 'm.gguf')), MODEL);
        assert.ok(!fs.existsSync(path.join(root, 'fake', 'm.gguf.part')));
        assert.equal(progress.filter((p) => p.phase === 'download').pop().received, MODEL.length + MMPROJ.length);
    } finally { hub.close(); }
});

test('an interrupted download resumes with a Range request', async () => {
    const hub = await fakeHub({ cutAfter: 70000 });
    const root = tmpRoot();
    try {
        await assert.rejects(ensureWeights(entry(), { root, baseUrl: hub.baseUrl }));
        assert.ok(fs.statSync(path.join(root, 'fake', 'm.gguf.part')).size > 0);
        await ensureWeights(entry(), { root, baseUrl: hub.baseUrl });
        assert.equal(weightsStatus(entry(), root).installed, true);
        assert.match(hub.log.find((l) => l.url.endsWith('m.gguf') && l.range).range, /^bytes=\d+-$/);
    } finally { hub.close(); }
});

test('a server that ignores Range restarts the file instead of appending it', async () => {
    const hub = await fakeHub({ ignoreRange: true });
    const root = tmpRoot();
    try {
        fs.mkdirSync(path.join(root, 'fake'));
        fs.writeFileSync(path.join(root, 'fake', 'm.gguf.part'), MODEL.subarray(0, 1000));
        await ensureWeights(entry(), { root, baseUrl: hub.baseUrl });
        assert.deepEqual(fs.readFileSync(path.join(root, 'fake', 'm.gguf')), MODEL);
    } finally { hub.close(); }
});

test('a hash mismatch discards the download and fails', async () => {
    const hub = await fakeHub();
    const root = tmpRoot();
    const bad = entry();
    bad.files.model.sha256 = '0'.repeat(64);
    try {
        await assert.rejects(ensureWeights(bad, { root, baseUrl: hub.baseUrl }), /SHA-256 mismatch/);
        assert.ok(!fs.existsSync(path.join(root, 'fake', 'm.gguf')));
        assert.ok(!fs.existsSync(path.join(root, 'fake', 'm.gguf.part')));
    } finally { hub.close(); }
});

test('verified files are not downloaded again; hand-copied files are verified in place', async () => {
    const root = tmpRoot();
    fs.mkdirSync(path.join(root, 'fake'));
    fs.writeFileSync(path.join(root, 'fake', 'm.gguf'), MODEL);
    fs.writeFileSync(path.join(root, 'fake', 'mmproj.gguf'), MMPROJ);
    // No server at all: any request would fail.
    await ensureWeights(entry(), { root, baseUrl: 'http://127.0.0.1:9' });
    assert.equal(weightsStatus(entry(), root).installed, true);
    await ensureWeights(entry(), { root, baseUrl: 'http://127.0.0.1:9' });
});

test('two callers for the same model share one download', async () => {
    const hub = await fakeHub();
    const root = tmpRoot();
    try {
        const a = ensureWeights(entry(), { root, baseUrl: hub.baseUrl });
        const b = ensureWeights(entry(), { root, baseUrl: hub.baseUrl });
        assert.equal(a, b);
        await a;
        assert.equal(hub.log.filter((l) => l.url.endsWith('m.gguf')).length, 1);
    } finally { hub.close(); }
});
