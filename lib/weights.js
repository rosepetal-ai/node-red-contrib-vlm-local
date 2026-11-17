/**
 * Model weights on disk: public Hugging Face downloads at a pinned revision,
 * resumable, SHA-256 verified, renamed into place only when complete.
 *
 * Layout: <MODELS_DIR>/<model id>/<file>, with <file>.part while downloading
 * and <file>.verified (holding the hash) once checked, so a 6 GB file is hashed
 * once rather than on every deploy. A file copied in by hand (offline install)
 * has no marker yet and is verified on first use.
 */

'use strict';

const crypto = require('crypto');
const fs = require('fs');
const path = require('path');
const { Readable, Transform } = require('stream');
const { pipeline } = require('stream/promises');
const { MODELS_DIR } = require('./catalog');

const HF_BASE = 'https://huggingface.co';
const ROLES = ['model', 'mmproj'];

const modelDir = (entry, root = MODELS_DIR) => path.join(root, entry.id);

async function sha256File(file) {
    const hash = crypto.createHash('sha256');
    await pipeline(fs.createReadStream(file), hash);
    return hash.digest('hex');
}

function isVerified(final, sha256) {
    try {
        return fs.readFileSync(`${final}.verified`, 'utf8').trim() === sha256 && fs.existsSync(final);
    } catch {
        return false;
    }
}

const sizeOf = (file) => { try { return fs.statSync(file).size; } catch { return 0; } };

/** What is on disk for a model: per file, the bytes still to download and the bytes used. */
function weightsStatus(entry, root = MODELS_DIR) {
    const dir = modelDir(entry, root);
    let missingBytes = 0;
    let diskBytes = 0;
    const files = ROLES.map((role) => {
        const f = entry.files[role];
        const final = path.join(dir, f.path);
        const verified = isVerified(final, f.sha256);
        const partial = sizeOf(`${final}.part`);
        diskBytes += sizeOf(final) + partial;
        if (!verified && !fs.existsSync(final)) missingBytes += Math.max(0, f.bytes - partial);
        return { role, path: final, verified };
    });
    return { dir, installed: files.every((f) => f.verified), missingBytes, diskBytes, files };
}

/** Delete a model's directory (weights, partial downloads, markers). */
function deleteWeights(entry, root = MODELS_DIR) {
    const dir = modelDir(entry, root);
    if (inFlight.has(dir)) throw new Error(`${entry.name} is downloading; delete it when it finishes`);
    fs.rmSync(dir, { recursive: true, force: true });
}

/** Append the rest of `url` to `part` from byte `start`, counting progress. */
async function fetchInto(url, part, start, onBytes) {
    const res = await fetch(url, { headers: start ? { Range: `bytes=${start}-` } : {} });
    if (res.status === 416) return; // already complete
    if (!res.ok) throw new Error(`HTTP ${res.status} downloading ${url}`);
    // A server that ignores Range sends the whole file: start over rather than append it.
    const resumed = res.status === 206;
    if (start && !resumed) onBytes(-start);
    const counter = new Transform({
        transform(chunk, _enc, cb) { onBytes(chunk.length); cb(null, chunk); },
    });
    await pipeline(Readable.fromWeb(res.body), counter,
        fs.createWriteStream(part, { flags: resumed ? 'a' : 'w' }));
}

async function ensureFile(f, dir, { baseUrl, onBytes, onVerify }) {
    const final = path.join(dir, f.path);
    if (isVerified(final, f.sha256)) return;

    // Copied in by hand: verify it in place, or discard it and download.
    if (fs.existsSync(final)) {
        onVerify(f.path);
        if (await sha256File(final) === f.sha256) {
            fs.writeFileSync(`${final}.verified`, f.sha256);
            return;
        }
        fs.unlinkSync(final);
    }

    const part = `${final}.part`;
    let start = 0;
    try { start = fs.statSync(part).size; } catch { /* fresh download */ }
    if (start > f.bytes) { fs.unlinkSync(part); start = 0; }
    onBytes(start);
    if (start < f.bytes) {
        const url = `${baseUrl}/${f.repo}/resolve/${f.revision}/${encodeURIComponent(f.path)}`;
        await fetchInto(url, part, start, onBytes);
    }

    onVerify(f.path);
    const size = fs.statSync(part).size;
    const actual = size === f.bytes ? await sha256File(part) : null;
    if (actual !== f.sha256) {
        fs.unlinkSync(part);
        throw new Error(size === f.bytes
            ? `${f.path}: SHA-256 mismatch (got ${actual}), download discarded`
            : `${f.path}: got ${size} bytes, expected ${f.bytes}; download discarded`);
    }
    fs.renameSync(part, final);
    fs.writeFileSync(`${final}.verified`, f.sha256);
}

// One download per model per process: two nodes on the same model share it.
const inFlight = new Map();

/**
 * Make sure both files of a model are on disk and verified.
 * @param {object} entry  catalog entry
 * @param {object} [opts] onProgress({phase: 'download'|'verify', received, total, file})
 */
function ensureWeights(entry, { root = MODELS_DIR, baseUrl = HF_BASE, onProgress = () => {} } = {}) {
    const key = modelDir(entry, root);
    if (inFlight.has(key)) return inFlight.get(key);

    const total = ROLES.reduce((sum, role) => sum + entry.files[role].bytes, 0);
    let received = 0;
    const run = (async () => {
        fs.mkdirSync(key, { recursive: true });
        for (const role of ROLES) {
            const before = received;
            await ensureFile(entry.files[role], key, {
                baseUrl,
                onBytes: (n) => { received += n; onProgress({ phase: 'download', received, total }); },
                onVerify: (file) => onProgress({ phase: 'verify', received, total, file }),
            });
            received = before + entry.files[role].bytes;
        }
    })().finally(() => inFlight.delete(key));

    inFlight.set(key, run);
    return run;
}

module.exports = { ensureWeights, weightsStatus, deleteWeights, modelDir, sha256File };
