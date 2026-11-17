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
const { ensureWeights, weightsStatus, deleteWeights, modelDir } = require('../lib/weights');
const { pinnedRef, ensureImage, startServer, stopServer, removeServer, listServers } = require('../lib/runtime');
const { createUnloader } = require('../lib/unloader');
const { chat } = require('../lib/llama');
const { gpuInfo } = require('../lib/gpu');
const detect = require('../lib/detect');
const { toJpegBase64 } = require('../lib/image-input');
const { renderDebugImage } = require('../lib/debug-image');
const { expandResultFormats } = require('../lib/result-conversion');

const DEBUG_IMAGE_TOPIC = 'vlm-local-debug-image';

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

/** Seconds of inactivity before unloading: null keeps the model loaded. Throws on a bad N. */
function unloadSeconds(mode, value) {
    if (mode === 'after') return 0;
    if (mode !== 'idle') return null;
    const seconds = typeof value === 'string' && value.trim() === '' ? NaN : Number(value);
    if (value === undefined || value === null || !Number.isFinite(seconds) || seconds < 0) {
        throw new Error(`Unload after N seconds: N must be a number >= 0, got ${JSON.stringify(value)}`);
    }
    return seconds;
}

module.exports = function (RED) {
    // Tells this Node-RED's containers from another instance's on the same device.
    const INSTANCE = String(RED.settings.uiPort || 1880);

    function resolveTyped(node, msg, name, fallbackType) {
        const value = node.config[name];
        const type = node.config[`${name}Type`] || fallbackType;
        switch (type) {
            case 'msg': return value ? RED.util.getMessageProperty(msg, value) : undefined;
            case 'flow': return node.context().flow.get(value);
            case 'global': return node.context().global.get(value);
            case 'num': return value === '' || value === undefined ? undefined : Number(value);
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
        node.unload = config.unload || 'never';
        node.debugShowImage = config.debugShowImage === true || config.debugShowImage === 'true';
        node.debugInterval = Math.max(1, Number(config.debugInterval) || 1);
        node.debugWidth = Math.max(1, Number(config.debugWidth) || 160);

        let entry = null;
        let image = null;
        let port = null;
        let ready = false;
        let initError = null;
        let closing = false;
        let removed = false;
        let debugCount = 0;
        let revertTimer = null;

        const status = (fill, shape, text) => !closing && node.status({ fill, shape, text });

        // Status every whole percent at most: a 6 GB download emits thousands of chunks.
        let lastShown = '';
        const progress = (label, fraction) => {
            const text = `${label} ${Math.floor(fraction * 100)}%`;
            if (text !== lastShown) status('yellow', 'ring', text);
            lastShown = text;
        };

        const start = async () => {
            const server = await startServer({
                nodeId: node.id, instance: INSTANCE, image, entry, dir: modelDir(entry),
                onStatus: (text) => status('yellow', 'ring', text),
            });
            port = server.port; // Docker may assign a new port when the container restarts
            return server;
        };

        const unloader = createUnloader({
            start,
            stop: () => stopServer(INSTANCE, node.id),
            onState: (s) => {
                if (s === 'unloading') status('grey', 'ring', 'Unloading…');
                if (s === 'unloaded') status('grey', 'ring', 'Unloaded (VRAM released)');
            },
            onError: (err) => node.warn(`vlm-local: unload failed, model stays loaded: ${err.message}`),
        });

        async function init() {
            if (!config.model) throw new Error('No model selected');
            const catalog = loadCatalog();
            entry = findModel(catalog, config.model);
            image = pinnedRef(runtimeFor(catalog, entry));

            status('yellow', 'ring', 'Checking runtime…');
            await ensureImage(image, (f) => progress('Pulling runtime…', f));
            await ensureWeights(entry, {
                onProgress: (p) => (p.phase === 'verify'
                    ? status('yellow', 'ring', `Verifying ${p.file}…`)
                    : progress('Downloading weights…', p.received / p.total)),
            });
            if (closing) return;
            const server = await start();
            // Deleted while we were starting: do not leave its container holding VRAM.
            if (removed) { await removeServer(INSTANCE, node.id); return; }
            ready = true;
            status('green', 'dot', 'Ready');
            node.log(`${entry.name} ready on 127.0.0.1:${port}${server.reused ? ' (reused)' : ''}`);

            // Start the countdown now when N is known without a message.
            if (node.unload !== 'never' && config.unloadSecondsType !== 'msg') {
                try {
                    unloader.schedule(unloadSeconds(node.unload, resolveTyped(node, {}, 'unloadSeconds', 'num')));
                } catch (err) {
                    node.warn(err.message);
                }
            }
        }

        const started = init().catch((err) => {
            initError = err.message;
            status('red', 'ring', err.message.slice(0, 50));
            node.error(`vlm-local: ${err.message}`);
        });

        const setError = (err) => {
            status('red', 'dot', String(err.message || err).slice(0, 50));
            clearTimeout(revertTimer);
            revertTimer = setTimeout(() => {
                if (!ready) return;
                if (unloader.state() === 'unloaded') status('grey', 'ring', 'Unloaded (VRAM released)');
                else status('green', 'dot', 'Ready');
            }, 8000);
        };

        node.on('input', async (msg, send, done) => {
            let acquired = false;
            let seconds = null;
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
                // Read before inferring: a bad N fails the message, not the unload later.
                seconds = unloadSeconds(node.unload, resolveTyped(node, msg, 'unloadSeconds', 'num'));

                const startTime = Date.now();
                status('yellow', 'dot', 'encoding');
                const { base64 } = await toJpegBase64(imageValue);
                const encodeMs = Date.now() - startTime;

                const { wakeMs } = await unloader.acquire();
                acquired = true;
                status('yellow', 'dot', 'inferring');
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
                    startTime, endTime: startTime + totalMs, encodeMs, wakeMs, requestMs, inferenceMs, totalMs,
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

                status('green', 'dot', `${entries.length} found | Total: ${totalMs}ms | Model: ${inferenceMs ?? '?'}ms`
                    + (wakeMs ? ` | Load: ${wakeMs}ms` : ''));
                send(msg);
                done();
            } catch (err) {
                setError(err);
                done(err);
            } finally {
                if (acquired) unloader.release(seconds);
            }
        });

        node.on('close', async (wasRemoved, done) => {
            closing = true;
            removed = wasRemoved;
            unloader.dispose();
            clearTimeout(revertTimer);
            // A redeploy keeps the container: the new instance reuses it if nothing changed.
            if (wasRemoved) {
                await started;
                await removeServer(INSTANCE, node.id).catch((err) => node.warn(`container not removed: ${err.message}`));
            }
            done();
        });
    }

    RED.nodes.registerType('vlm-local-inferencer', VlmLocalInferencer);

    // ---- Editor endpoints ----

    /** Models with their disk state, the GPU, and every vlm-local container. */
    RED.httpAdmin.get('/vlm-local-inferencer/info', RED.auth.needsPermission('vlm-local-inferencer.read'),
        async (req, res) => {
            try {
                const catalog = loadCatalog();
                const models = catalog.models.map((m) => {
                    const s = weightsStatus(m);
                    return {
                        id: m.id, name: m.name, vramMb: m.vram_mb, license: m.license,
                        deprecated: Boolean(m.deprecated), source: m.source,
                        installed: s.installed, missingBytes: s.missingBytes, diskBytes: s.diskBytes,
                    };
                });
                const [gpu, servers] = await Promise.all([gpuInfo(), listServers().catch(() => [])]);
                res.json({ models, gpu, servers, instance: INSTANCE });
            } catch (err) {
                res.status(500).json({ error: err.message });
            }
        });

    /** Delete a model's files. Refused while a deployed node or a container uses it. */
    RED.httpAdmin.delete('/vlm-local-inferencer/models/:id', RED.auth.needsPermission('vlm-local-inferencer.write'),
        async (req, res) => {
            try {
                const entry = findModel(loadCatalog(), req.params.id);
                const users = new Set();
                RED.nodes.eachNode((n) => {
                    if (n.type === 'vlm-local-inferencer' && n.model === entry.id) users.add(n.name || n.id);
                });
                for (const s of await listServers()) {
                    if (s.model === entry.id) users.add(`node ${s.nodeId} (Node-RED on port ${s.instance})`);
                }
                if (users.size) {
                    res.status(409).json({
                        error: `${entry.name} is in use by: ${[...users].join(', ')}. `
                            + 'Change or delete those nodes and deploy first.',
                    });
                    return;
                }
                const { diskBytes } = weightsStatus(entry);
                deleteWeights(entry);
                res.json({ deleted: entry.id, freedBytes: diskBytes });
            } catch (err) {
                res.status(/not in the catalog/.test(err.message) ? 404 : 500).json({ error: err.message });
            }
        });
};

module.exports.unloadSeconds = unloadSeconds;
module.exports.normalizeTags = normalizeTags;
