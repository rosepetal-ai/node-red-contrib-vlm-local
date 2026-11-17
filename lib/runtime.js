/**
 * The llama.cpp server container, one per node: pulled anonymously by digest,
 * weights mounted read-only, listening on a Docker-assigned 127.0.0.1 port.
 *
 * Node-RED runs on the host network on the controller, so 127.0.0.1:<port> is
 * reachable from it, and /opt/storage is the same path inside and out, so the
 * weights directory can be bind-mounted as is.
 */

'use strict';

const crypto = require('crypto');
const Docker = require('dockerode');
const { fetchWithTimeout } = require('./http');

const CONTAINER_PORT = 8080;
const LABEL = 'nodered.vlm-local';
const HEALTH_TIMEOUT_MS = 10 * 60 * 1000; // a cold page cache reads ~7 GB from disk

const docker = new Docker();

/** `repo:tag@sha256:…` -> `repo@sha256:…`: the digest is what is pulled and run. */
function pinnedRef(runtime) {
    const at = runtime.indexOf('@');
    if (at < 0) throw new Error(`Runtime '${runtime}' is not pinned by digest`);
    const name = runtime.slice(0, at);
    const colon = name.lastIndexOf(':');
    const repo = colon > name.lastIndexOf('/') ? name.slice(0, colon) : name;
    return `${repo}${runtime.slice(at)}`;
}

const containerName = (nodeId) => `nodered-vlm-local-${nodeId}`;

/** llama-server arguments for a catalog entry; the files are mounted at /models. */
function serverArgs(entry, { sleepSeconds = 0 } = {}) {
    const p = entry.parameters || {};
    const args = [
        '-m', `/models/${entry.files.model.path}`,
        '--mmproj', `/models/${entry.files.mmproj.path}`,
        '--host', '0.0.0.0', '--port', String(CONTAINER_PORT),
        '--jinja',
        '-c', String(p.ctx || 8192),
        '-np', '1',      // one slot: the default 4 reserve KV cache for requests that never come
        '-ngl', '99',
    ];
    if (p.reasoning === false) args.push('--reasoning', 'off');
    if (p.image_min_tokens) args.push('--image-min-tokens', String(p.image_min_tokens));
    if (sleepSeconds > 0) args.push('--sleep-idle-seconds', String(sleepSeconds));
    return args;
}

/** Everything that, if changed, needs a new container. */
function configHash(image, args, dir) {
    return crypto.createHash('sha256').update(JSON.stringify({ image, args, dir })).digest('hex').slice(0, 16);
}

async function imageExists(ref) {
    try {
        await docker.getImage(ref).inspect();
        return true;
    } catch (err) {
        if (err.statusCode === 404) return false;
        throw err;
    }
}

/** Pull the runtime if missing, reporting overall layer progress 0..1. */
async function ensureImage(ref, onProgress = () => {}) {
    if (await imageExists(ref)) return;
    let stream;
    try {
        stream = await docker.pull(ref);
    } catch (err) {
        throw new Error(`Runtime not available (${err.message}). Without network, load it with: docker load`);
    }
    const layers = new Map();
    await new Promise((resolve, reject) => {
        docker.modem.followProgress(stream, (err) => (err ? reject(err) : resolve()), (event) => {
            const d = event.progressDetail;
            if (event.id && d && d.total) layers.set(event.id, { current: d.current || 0, total: d.total });
            if (event.status === 'Download complete' && layers.has(event.id)) {
                layers.get(event.id).current = layers.get(event.id).total;
            }
            const sum = [...layers.values()].reduce((a, l) => [a[0] + l.current, a[1] + l.total], [0, 0]);
            if (sum[1]) onProgress(sum[0] / sum[1]);
        });
    });
}

async function findContainer(name) {
    const list = await docker.listContainers({ all: true, filters: { name: [name] } });
    const info = list.find((c) => (c.Names || []).includes(`/${name}`));
    return info ? { container: docker.getContainer(info.Id), info } : null;
}

async function hostPort(container) {
    const details = await container.inspect();
    const bound = details.NetworkSettings.Ports[`${CONTAINER_PORT}/tcp`];
    if (!bound || !bound[0]) throw new Error('llama-server container has no published port');
    return Number(bound[0].HostPort);
}

async function logTail(container, lines = 15) {
    try {
        const buf = await container.logs({ stdout: true, stderr: true, tail: lines });
        // Non-TTY logs are multiplexed with 8-byte frame headers; drop the control bytes.
        return buf.toString('utf8').replace(/[\x00-\x08]/g, '').trim();
    } catch {
        return '';
    }
}

// A redeploy can overlap the old instance's start with the new one's: one at a time per container.
const starting = new Map();

/**
 * Start (or reuse) the node's server and wait for it to answer /health.
 * @returns {Promise<{port: number, reused: boolean}>}
 */
async function startServer(opts) {
    const name = containerName(opts.nodeId);
    const previous = starting.get(name) || Promise.resolve();
    const run = previous.catch(() => {}).then(() => startServerNow(name, opts));
    starting.set(name, run);
    try {
        return await run;
    } finally {
        if (starting.get(name) === run) starting.delete(name);
    }
}

async function startServerNow(name, { image, entry, dir, sleepSeconds = 0, vramMb, nodeId, onStatus = () => {} }) {
    const args = serverArgs(entry, { sleepSeconds });
    const hash = configHash(image, args, dir);

    const existing = await findContainer(name);
    if (existing && existing.info.Labels[`${LABEL}.config`] === hash) {
        if (existing.info.State !== 'running') await existing.container.start();
        const port = await hostPort(existing.container);
        await waitHealthy(existing.container, port, onStatus);
        return { port, reused: true };
    }
    if (existing) await existing.container.remove({ force: true });

    let container;
    try {
        container = await docker.createContainer({
            name,
            Image: image,
            Cmd: args,
            ExposedPorts: { [`${CONTAINER_PORT}/tcp`]: {} },
            Labels: {
                [`${LABEL}.node`]: nodeId,
                [`${LABEL}.model`]: entry.id,
                [`${LABEL}.config`]: hash,
                [`${LABEL}.vram_mb`]: String(vramMb || entry.vram_mb),
                [`${LABEL}.sleep`]: String(sleepSeconds),
            },
            HostConfig: {
                // Empty HostPort: Docker picks a free one, so there is no pick-then-bind race.
                PortBindings: { [`${CONTAINER_PORT}/tcp`]: [{ HostIp: '127.0.0.1', HostPort: '' }] },
                Binds: [`${dir}:/models:ro`],
                DeviceRequests: [{ Count: -1, Capabilities: [['gpu']] }],
            },
        });
        await container.start();
    } catch (err) {
        if (container) await container.remove({ force: true }).catch(() => {});
        throw new Error(/gpu|nvidia|device/i.test(err.message)
            ? `No GPU available to Docker: ${err.message}`
            : `Cannot start llama-server: ${err.message}`);
    }
    const port = await hostPort(container);
    await waitHealthy(container, port, onStatus);
    return { port, reused: false };
}

async function waitHealthy(container, port, onStatus) {
    const deadline = Date.now() + HEALTH_TIMEOUT_MS;
    onStatus('Loading model…');
    while (Date.now() < deadline) {
        try {
            const res = await fetchWithTimeout(`http://127.0.0.1:${port}/health`, {}, 5000);
            if (res.ok) return;
        } catch { /* still starting */ }
        const state = (await container.inspect()).State;
        if (!state.Running) {
            throw new Error(`llama-server exited (code ${state.ExitCode}): ${await logTail(container)}`);
        }
        await new Promise((r) => setTimeout(r, 1000));
    }
    throw new Error(`llama-server not ready after ${HEALTH_TIMEOUT_MS / 1000}s: ${await logTail(container)}`);
}

async function removeServer(nodeId) {
    const existing = await findContainer(containerName(nodeId));
    if (existing) await existing.container.remove({ force: true });
}

/** `/props` answers without waking a sleeping server; null if unreachable. */
async function serverProps(port) {
    try {
        const res = await fetchWithTimeout(`http://127.0.0.1:${port}/props`, {}, 3000);
        return res.ok ? await res.json() : null;
    } catch {
        return null;
    }
}

/** Running vlm-local containers, for the panel's VRAM sum. */
async function listServers() {
    const list = await docker.listContainers({ filters: { label: [`${LABEL}.node`] } });
    return list.map((c) => ({
        nodeId: c.Labels[`${LABEL}.node`],
        model: c.Labels[`${LABEL}.model`],
        vramMb: Number(c.Labels[`${LABEL}.vram_mb`]) || 0,
        sleepSeconds: Number(c.Labels[`${LABEL}.sleep`]) || 0,
    }));
}

module.exports = {
    pinnedRef, containerName, serverArgs, configHash,
    ensureImage, startServer, removeServer, serverProps, listServers,
};
