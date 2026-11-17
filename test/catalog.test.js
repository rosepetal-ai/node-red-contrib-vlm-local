const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('fs');
const os = require('os');
const path = require('path');
const { loadCatalog, findModel, runtimeFor } = require('../lib/catalog');

const tmp = (data) => {
    const file = path.join(fs.mkdtempSync(path.join(os.tmpdir(), 'vlm-cat-')), 'catalog.json');
    fs.writeFileSync(file, typeof data === 'string' ? data : JSON.stringify(data));
    return file;
};

test('the packaged catalog loads and validates, Qwen3.5 9B first', () => {
    const catalog = loadCatalog({ local: null });
    assert.deepEqual(catalog.models.map((m) => m.id), ['qwen3.5-9b-q4', 'qwen3.5-4b-q4']);
    assert.match(runtimeFor(catalog, catalog.models[0]), /@sha256:[0-9a-f]{64}$/);
});

test('a local catalog replaces by id and appends new entries', () => {
    const packaged = loadCatalog({ local: null });
    const changed = { ...packaged.models[1], vram_mb: 5000 };
    delete changed.source;
    const added = { ...changed, id: 'my-model', name: 'Mine' };
    const catalog = loadCatalog({ local: tmp({ models: [changed, added] }) });

    assert.deepEqual(catalog.models.map((m) => m.id), ['qwen3.5-9b-q4', 'qwen3.5-4b-q4', 'my-model']);
    assert.equal(findModel(catalog, 'qwen3.5-4b-q4').vram_mb, 5000);
    assert.equal(findModel(catalog, 'my-model').source, 'local');
});

test('a bad local entry fails naming the file and the field', () => {
    const bad = { ...loadCatalog({ local: null }).models[0], id: 'x', coord_format: 'pixels_xyxy' };
    const file = tmp({ models: [bad] });
    assert.throws(() => loadCatalog({ local: file }), (err) => err.message.includes(file) && err.message.includes('coord_format'));
});

test('an unreadable local catalog fails clearly', () => {
    assert.throws(() => loadCatalog({ local: tmp('{oops') }), /Cannot read model catalog/);
});

test('an unknown model id fails clearly', () => {
    assert.throws(() => findModel(loadCatalog({ local: null }), 'nope'), /not in the catalog/);
});
