/**
 * The model catalog: the one packaged with this npm, plus an optional file on
 * the device that adds or replaces entries by `id` without a release.
 */

'use strict';

const fs = require('fs');
const path = require('path');

const PACKAGED = path.join(__dirname, '..', 'catalog.json');
// Same path on the host and in the Node-RED container (/opt/storage is mounted 1:1),
// which matters because the runtime container is created with a host bind mount.
const MODELS_DIR = process.env.VLM_LOCAL_MODELS_DIR || '/opt/storage/models/vlm';

const DIGEST = /@sha256:[0-9a-f]{64}$/;
const COORD_FORMATS = ['grid_xyxy', 'grid_yxyx'];

/** Throws naming the entry and field: a bad row should be obvious, not a later crash. */
function validateEntry(m, source) {
    const where = `${source}: model '${m && m.id}'`;
    const fail = (what) => { throw new Error(`${where}: ${what}`); };
    if (!m || typeof m.id !== 'string' || !m.id) fail('missing id');
    if (!/^[a-z0-9._-]+$/i.test(m.id)) fail('id must be a plain directory name');
    if (!m.name) fail('missing name');
    if (!COORD_FORMATS.includes(m.coord_format)) fail(`coord_format must be one of ${COORD_FORMATS.join(', ')}`);
    if (!Number.isInteger(m.vram_mb) || m.vram_mb <= 0) fail('vram_mb must be a positive integer (measured)');
    if (m.runtime !== null && m.runtime !== undefined && !DIGEST.test(m.runtime)) fail('runtime must be pinned by @sha256 digest');
    for (const role of ['model', 'mmproj']) {
        const f = m.files && m.files[role];
        if (!f) fail(`missing files.${role}`);
        if (!f.repo || !f.path || f.path.includes('/')) fail(`files.${role}: repo and a plain file name are required`);
        if (!/^[0-9a-f]{40}$/.test(f.revision || '')) fail(`files.${role}.revision must be a commit hash`);
        if (!/^[0-9a-f]{64}$/.test(f.sha256 || '')) fail(`files.${role}.sha256 is required`);
        if (!Number.isInteger(f.bytes) || f.bytes <= 0) fail(`files.${role}.bytes is required`);
    }
}

function readJson(file) {
    try {
        return JSON.parse(fs.readFileSync(file, 'utf8'));
    } catch (err) {
        throw new Error(`Cannot read model catalog ${file}: ${err.message}`);
    }
}

/**
 * Packaged entries in order, local ones replacing by id or appended after.
 * @returns {{runtime: string, models: object[]}}
 */
function loadCatalog({ packaged = PACKAGED, local = path.join(MODELS_DIR, 'catalog.json') } = {}) {
    const base = readJson(packaged);
    const models = new Map(base.models.map((m) => [m.id, { ...m, source: 'packaged' }]));
    let runtime = base.runtime;

    if (local && fs.existsSync(local)) {
        const extra = readJson(local);
        if (extra.runtime) runtime = extra.runtime;
        for (const m of extra.models || []) models.set(m.id, { ...m, source: 'local' });
    }

    if (!DIGEST.test(runtime || '')) throw new Error('Catalog runtime must be pinned by @sha256 digest');
    for (const m of models.values()) validateEntry(m, m.source === 'local' ? local : packaged);
    return { runtime, models: [...models.values()] };
}

function findModel(catalog, id) {
    const entry = catalog.models.find((m) => m.id === id);
    if (!entry) throw new Error(`Model '${id}' is not in the catalog.`);
    return entry;
}

/** The image a model runs on: its own pin, or the catalog's default. */
function runtimeFor(catalog, entry) {
    return entry.runtime || catalog.runtime;
}

module.exports = { MODELS_DIR, loadCatalog, findModel, runtimeFor, validateEntry };
