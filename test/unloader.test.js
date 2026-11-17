const test = require('node:test');
const assert = require('node:assert/strict');
const { createUnloader } = require('../lib/unloader');

const tick = (ms) => new Promise((r) => setTimeout(r, ms));

function harness({ startMs = 20, failStart = false, failStop = false } = {}) {
    const calls = [];
    const states = [];
    const errors = [];
    const u = createUnloader({
        stop: async () => { calls.push('stop'); await tick(10); if (failStop) throw new Error('stop failed'); },
        start: async () => { calls.push('start'); await tick(startMs); if (failStart) throw new Error('start failed'); },
        onState: (s) => states.push(s),
        onError: (e) => errors.push(e.message),
    });
    return { u, calls, states, errors };
}

test('unloads after N idle seconds and loads again on the next request', async () => {
    const { u, calls, states } = harness();
    assert.deepEqual(await u.acquire(), { wakeMs: 0 });
    u.release(0.05);
    await tick(30);
    assert.deepEqual(calls, []);
    await tick(60);
    assert.deepEqual(calls, ['stop']);
    assert.equal(u.state(), 'unloaded');

    const { wakeMs } = await u.acquire();
    assert.ok(wakeMs >= 15);
    assert.deepEqual(calls, ['stop', 'start']);
    assert.deepEqual(states, ['unloading', 'unloaded', 'loading', 'loaded']);
    u.release(null);
});

test('a request cancels the countdown; null keeps the model loaded', async () => {
    const { u, calls } = harness();
    await u.acquire();
    u.release(0.05);
    await tick(20);
    await u.acquire();            // arrives before the countdown ends
    u.release(null);
    await tick(80);
    assert.deepEqual(calls, []);
});

test('the last request in flight decides, with its own N', async () => {
    const { u, calls } = harness();
    await u.acquire();
    await u.acquire();
    u.release(0);                 // one still in flight: no unload yet
    await tick(20);
    assert.deepEqual(calls, []);
    u.release(0.03);
    await tick(60);
    assert.deepEqual(calls, ['stop']);
});

test('concurrent requests share one load, also while unloading', async () => {
    const { u, calls } = harness();
    await u.acquire();
    u.release(0);
    await tick(2);                // stop in progress
    await Promise.all([u.acquire(), u.acquire()]);
    assert.deepEqual(calls, ['stop', 'start']);
});

test('a failed load fails the request and the next one retries', async () => {
    const h = harness({ failStart: true });
    await h.u.acquire();
    h.u.release(0);
    await tick(30);
    await assert.rejects(h.u.acquire(), /start failed/);
    await assert.rejects(h.u.acquire(), /start failed/);
    assert.deepEqual(h.calls, ['stop', 'start', 'start']);
});

test('a failed unload is reported and the model counts as loaded', async () => {
    const h = harness({ failStop: true });
    await h.u.acquire();
    h.u.release(0);
    await tick(40);
    assert.deepEqual(h.errors, ['stop failed']);
    assert.equal(h.u.state(), 'loaded');
    assert.deepEqual(await h.u.acquire(), { wakeMs: 0 });
});
