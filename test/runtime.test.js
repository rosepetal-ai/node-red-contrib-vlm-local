const test = require('node:test');
const assert = require('node:assert/strict');
const { pinnedRef, serverArgs, configHash } = require('../lib/runtime');
const { loadCatalog } = require('../lib/catalog');

const DIGEST = 'sha256:' + 'd'.repeat(64);

test('pinnedRef drops the tag and keeps the digest', () => {
    assert.equal(pinnedRef(`ghcr.io/ggml-org/llama.cpp:server-cuda-b12429@${DIGEST}`), `ghcr.io/ggml-org/llama.cpp@${DIGEST}`);
    assert.equal(pinnedRef(`localhost:5000/llama@${DIGEST}`), `localhost:5000/llama@${DIGEST}`);
    assert.throws(() => pinnedRef('ghcr.io/ggml-org/llama.cpp:server-cuda'), /not pinned/);
});

test('serverArgs: the phase-0 command', () => {
    const entry = loadCatalog({ local: null }).models[0];
    const args = serverArgs(entry);
    assert.deepEqual(args, [
        '-m', '/models/Qwen3.5-9B-Q4_K_M.gguf', '--mmproj', '/models/mmproj-F16.gguf',
        '--host', '0.0.0.0', '--port', '8080', '--jinja', '-c', '8192', '-np', '1', '-ngl', '99',
        '--reasoning', 'off', '--image-min-tokens', '1024',
    ]);
});

test('configHash changes with anything that needs a new container', () => {
    const base = configHash('img', ['a'], '/d');
    assert.equal(base, configHash('img', ['a'], '/d'));
    assert.notEqual(base, configHash('img2', ['a'], '/d'));
    assert.notEqual(base, configHash('img', ['a', 'b'], '/d'));
    assert.notEqual(base, configHash('img', ['a'], '/e'));
});

test('containerName keeps two Node-RED instances apart even with the same node id', () => {
    const { containerName } = require('../lib/runtime');
    assert.equal(containerName('1880', 'abc'), 'nodered-vlm-local-1880-abc');
    assert.notEqual(containerName('1880', 'abc'), containerName('1890', 'abc'));
});
