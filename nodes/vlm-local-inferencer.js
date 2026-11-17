/**
 * vlm-local-inferencer — an open vision-language model on this device.
 *
 * The local twin of vision-platform's `vlm-online-inferencer`: same input, same
 * `msg.inference` array, entry for entry, so a flow can swap one for the other.
 * The model runs in a llama.cpp container this node owns; weights come from
 * public Hugging Face repos and the runtime from a public image, so nothing here
 * needs credentials or the cloud once installed.
 */

'use strict';

const { loadCatalog, findModel, runtimeFor } = require('../lib/catalog');
const { ensureWeights, weightsStatus, modelDir } = require('../lib/weights');
const { pinnedRef, ensureImage, startServer, removeServer, serverProps, listServers } = require('../lib/runtime');
const { chat } = require('../lib/llama');
const { gpuInfo } = require('../lib/gpu');
const detect = require('../lib/detect');
const { toJpegBase64 } = require('../lib/image-input');
const { renderDebugImage } = require('../lib/debug-image');
const { expandResultFormats } = require('../lib/result-conversion');

const DEBUG_IMAGE_TOPIC = 'vlm-local-debug-image';

/** Panel choice -> llama-server --sleep-idle-seconds (0 = never; the server rejects 0). */
function sleepSecondsFor(config) {
    if (config.releaseVram === 'after') return 1;
    if (config.releaseVram === 'idle') return Math.max(1, Math.round(Number(config.idleSeconds) || 60));
    return 0;
}

/** Tags as the online node accepts them: objects, explicit ids, or a comma list. */
function normalizeTags(raw) {
    if (!raw) return null;
    const list = Array.isArray(raw) ? raw : String(raw).split(',').map((s) => s.trim()).filter(Boolean);
    if (!list.length) return null;
    return list.map((entry) => {
        if (entry && typeof entry === 'object') {
            const name = String(entry.name || entry.tag || '').trim();
            return { id: String(entry.id || name), name, description: String(entry.description || '').trim() };
        }
        const name = String(entry).trim();
        return { id: name, name, description: '' };
    }).filter((tag) => tag.name);
}

module.exports = function (RED) {
    function resolveTyped(node, msg, name, fallbackType) {
        const value = node.config[name];
        const type = node.config[`${name}Type`] || fallbackType;
        switch (type) {
            case 'msg': return value ? RED.util.getMessageProperty(msg, value) : undefined;
            case 'flow': return node.context().flow.get(value);
            case 'global': return node.context().global.get(value);
            // `list` is the panel's tag-row editor; it stores the same JSON as `json`.
            case 'list':
            case 'json': try { return JSON.parse(value); } catch { return undefined; }
            default: return value;
        }
    }

    function VlmLocalInferencer(config) {
        RED.nodes.createNode(this, config);
        const node = this;
        node.config = config;
        node.outputPath = config.output || 'inference';
        node.findBy = config.mode || 'prompt';
        node.timeoutMs = Number(config.timeoutMs) || 120000;
        node.sleepSeconds = sleepSecondsFor(config);
        node.debugShowImage = config.debugShowImage === true || config.debugShowImage === 'true';
        node.debugInterval = Math.max(1, Number(config.debugInterval) || 1);
        node.debugWidth = Math.max(1, Number(config.debugWidth) || 160);

        let entry = null;
        let port = null;
        let ready = false;
        let initError = null;
        let closing = false;
        let removed = false;
        let debugCount = 0;
        let sleepTimer = null;
        let revertTimer = null;

        const idleStatus = () => node.status({ fill: 'green', shape: 'dot', text: 'Ready' });

        // Status every whole percent at most: a 6 GB download emits thousands of chunks.
        let lastShown = '';
        const progress = (label, fraction) => {
            const text = `${label} ${Math.floor(fraction * 100)}%`;
            if (text !== lastShown && !closing) node.status({ fill: 'yellow', shape: 'ring', text });
            lastShown = text;
        };

        /** After the idle period, show whether the server really released the VRAM. */
        const scheduleSleepCheck = () => {
            if (!node.sleepSeconds) return;
            clearTimeout(sleepTimer);
            sleepTimer = setTimeout(async () => {
                const props = await serverProps(port);
                if (props && props.is_sleeping && !closing) {
                    node.status({ fill: 'grey', shape: 'ring', text: 'Sleeping (VRAM released)' });
                }
            }, (node.sleepSeconds + 1) * 1000);
        };

        async function init() {
            if (!config.model) throw new Error('No model selected');
            const catalog = loadCatalog();
            entry = findModel(catalog, config.model);
            const image = pinnedRef(runtimeFor(catalog, entry));

            node.status({ fill: 'yellow', shape: 'ring', text: 'Checking runtime…' });
            await ensureImage(image, (f) => progress('Pulling runtime…', f));
            await ensureWeights(entry, {
                onProgress: (p) => (p.phase === 'verify'
                    ? node.status({ fill: 'yellow', shape: 'ring', text: `Verifying ${p.file}…` })
                    : progress('Downloading weights…', p.received / p.total)),
            });
            if (closing) return;
            const server = await startServer({
                nodeId: node.id, image, entry, dir: modelDir(entry), sleepSeconds: node.sleepSeconds,
                onStatus: (text) => !closing && node.status({ fill: 'yellow', shape: 'ring', text }),
            });
            port = server.port;
            // Deleted while we were starting: do not leave its container holding VRAM.
            if (removed) { await removeServer(node.id); return; }
            ready = true;
            idleStatus();
            scheduleSleepCheck();
            node.log(`${entry.name} ready on 127.0.0.1:${port}${server.reused ? ' (reused)' : ''}`);
        }

        const started = init().catch((err) => {
            initError = err.message;
            if (!closing) node.status({ fill: 'red', shape: 'ring', text: err.message.slice(0, 50) });
            node.error(`vlm-local: ${err.message}`);
        });

        const setError = (err) => {
            node.status({ fill: 'red', shape: 'dot', text: String(err.message || err).slice(0, 50) });
            clearTimeout(revertTimer);
            revertTimer = setTimeout(() => ready && idleStatus(), 8000);
        };

        node.on('input', async (msg, send, done) => {
            try {
                // Not queued: a message that arrives during a 6 GB download would wait minutes.
                if (!ready) throw new Error(initError ? `Not ready: ${initError}` : 'Not ready: model still loading');

                const imageValue = resolveTyped(node, msg, 'image', 'msg');
                if (imageValue === undefined || imageValue === null) throw new Error('No image on the configured input property.');

                let tags = [];
                let prompt;
                if (node.findBy === 'prompt') {
                    const value = resolveTyped(node, msg, 'prompt', 'str');
                    const request = value === undefined || value === null ? '' : String(value).trim();
                    if (!request) throw new Error('Find by prompt needs a prompt.');
                    prompt = detect.buildFreePrompt(request, entry.coord_format);
                } else {
                    tags = normalizeTags(resolveTyped(node, msg, 'tags', 'str'));
                    if (!tags) throw new Error('Find by tags needs at least one tag.');
                    prompt = detect.buildTagsPrompt(tags, entry.coord_format);
                }
                const parameters = resolveTyped(node, msg, 'parameters', 'json');

                const startTime = Date.now();
                node.status({ fill: 'yellow', shape: 'dot', text: 'encoding' });
                const { base64 } = await toJpegBase64(imageValue);
                const encodeMs = Date.now() - startTime;

                // /props does not wake the server, so this tells a slow call from a waking one.
                const woke = node.sleepSeconds > 0 && Boolean((await serverProps(port) || {}).is_sleeping);
                node.status({ fill: 'yellow', shape: 'dot', text: woke ? 'waking model…' : 'inferring' });
                const requestStarted = Date.now();
                const answer = await chat({
                    port, imageBase64: base64, prompt,
                    responseFormat: detect.detectionSchema(tags),
                    parameters: parameters && typeof parameters === 'object' ? parameters : {},
                    timeoutMs: node.timeoutMs,
                });
                const requestMs = Date.now() - requestStarted;

                const { boxes } = detect.toBoxes(detect.parseDetections(answer.text), tags, entry.coord_format);
                const entries = boxes.map(detect.toEntry);
                RED.util.setMessageProperty(msg, node.outputPath, expandResultFormats(entries), true);

                const inferenceMs = answer.promptMs !== null && answer.predictedMs !== null
                    ? answer.promptMs + answer.predictedMs : null;
                const totalMs = Date.now() - startTime;
                msg.performance = msg.performance || {};
                msg.performance[config.name || 'VLMLocal'] = {
                    startTime, endTime: startTime + totalMs, encodeMs, requestMs, inferenceMs,
                    // Reloading a sleeping model happens outside the server's own timings.
                    wakeMs: woke && inferenceMs !== null ? requestMs - inferenceMs : 0,
                    totalMs,
                };

                if (node.debugShowImage) {
                    const show = debugCount === 0;
                    debugCount = (debugCount + 1) % node.debugInterval;
                    if (show) {
                        renderDebugImage(base64, entries, node.debugWidth, {
                            task: 'detection', boxThickness: config.debugBoxThickness, fontSize: config.debugFontSize,
                        })
                            .then((data) => RED.comms.publish(DEBUG_IMAGE_TOPIC, { id: node.id, data }))
                            .catch((err) => node.warn(`debug image not drawn: ${err.message}`));
                    }
                }

                node.status({
                    fill: 'green', shape: 'dot',
                    text: `${entries.length} found | Total: ${totalMs}ms | Model: ${inferenceMs ?? '?'}ms`
                        + (woke ? ` | Wake: ${requestMs - (inferenceMs || 0)}ms` : ''),
                });
                scheduleSleepCheck();
                send(msg);
                done();
            } catch (err) {
                setError(err);
                done(err);
            }
        });

        node.on('close', async (wasRemoved, done) => {
            closing = true;
            removed = wasRemoved;
            clearTimeout(sleepTimer);
            clearTimeout(revertTimer);
            // A redeploy keeps the container: the new instance reuses it if nothing changed.
            if (wasRemoved) {
                await started;
                await removeServer(node.id).catch((err) => node.warn(`container not removed: ${err.message}`));
            }
            done();
        });
    }

    RED.nodes.registerType('vlm-local-inferencer', VlmLocalInferencer);

    // Editor: models with their install state, the GPU, and other vlm-local servers.
    RED.httpAdmin.get('/vlm-local-inferencer/info', RED.auth.needsPermission('vlm-local-inferencer.read'),
        async (req, res) => {
            try {
                const catalog = loadCatalog();
                const models = catalog.models.map((m) => {
                    const status = weightsStatus(m);
                    return {
                        id: m.id, name: m.name, vramMb: m.vram_mb, license: m.license,
                        deprecated: Boolean(m.deprecated), source: m.source,
                        installed: status.installed, missingBytes: status.missingBytes,
                    };
                });
                const [gpu, servers] = await Promise.all([gpuInfo(), listServers().catch(() => [])]);
                res.json({ models, gpu, servers });
            } catch (err) {
                res.status(500).json({ error: err.message });
            }
        });
};

module.exports.sleepSecondsFor = sleepSecondsFor;
module.exports.normalizeTags = normalizeTags;
