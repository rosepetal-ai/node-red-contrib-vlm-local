const test = require('node:test');
const assert = require('node:assert/strict');
const { unloadSeconds, normalizeTags } = require('../nodes/vlm-local-inferencer');

test('unloadSeconds: never, after each inference, or a valid N', () => {
    assert.equal(unloadSeconds('never', 5), null);
    assert.equal(unloadSeconds('after', undefined), 0);
    assert.equal(unloadSeconds('idle', 30), 30);
    assert.equal(unloadSeconds('idle', '2.5'), 2.5);
    assert.equal(unloadSeconds('idle', 0), 0);
});

test('unloadSeconds: a missing or bad N from msg/flow fails clearly', () => {
    for (const bad of [undefined, null, '', 'soon', -1, NaN]) {
        assert.throws(() => unloadSeconds('idle', bad), /N must be a number >= 0/);
    }
});

test('normalizeTags accepts objects, explicit ids and a comma list', () => {
    assert.deepEqual(normalizeTags('cap, barcode'), [
        { id: 'cap', name: 'cap', description: '' }, { id: 'barcode', name: 'barcode', description: '' },
    ]);
    assert.deepEqual(normalizeTags([{ name: 'cap', description: 'neck capsule', id: 7 }]),
        [{ id: '7', name: 'cap', description: 'neck capsule' }]);
    assert.equal(normalizeTags([]), null);
});
