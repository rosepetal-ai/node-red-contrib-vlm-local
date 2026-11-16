const test = require('node:test');
const assert = require('node:assert/strict');
const catalog = require('../catalog.json');

const DIGEST = /@sha256:[0-9a-f]{64}$/;
const COORD_FORMATS = ['grid_xyxy', 'grid_yxyx'];

test('the default runtime is pinned by digest', () => {
    assert.match(catalog.runtime, DIGEST);
});

test('every model entry is complete and pinned', () => {
    const ids = catalog.models.map((m) => m.id);
    assert.equal(new Set(ids).size, ids.length, 'duplicate model id');

    for (const m of catalog.models) {
        assert.ok(m.name, `${m.id}: name`);
        assert.ok(COORD_FORMATS.includes(m.coord_format), `${m.id}: coord_format`);
        assert.ok(Number.isInteger(m.vram_mb) && m.vram_mb > 0, `${m.id}: vram_mb`);
        assert.ok(m.runtime === null || DIGEST.test(m.runtime), `${m.id}: runtime`);
        assert.equal(m.license.commercial, true, `${m.id}: license`);
        for (const role of ['model', 'mmproj']) {
            const f = m.files[role];
            assert.match(f.revision, /^[0-9a-f]{40}$/, `${m.id}.${role}: revision must be a commit`);
            assert.match(f.sha256, /^[0-9a-f]{64}$/, `${m.id}.${role}: sha256`);
            assert.ok(Number.isInteger(f.bytes) && f.bytes > 0, `${m.id}.${role}: bytes`);
            assert.ok(f.repo && f.path, `${m.id}.${role}: repo/path`);
        }
    }
});
