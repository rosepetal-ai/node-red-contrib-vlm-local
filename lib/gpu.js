/** The device's GPU, as nvidia-smi reports it (the Node-RED image ships it). */

'use strict';

const { execFile } = require('child_process');

function parseNvidiaSmi(stdout) {
    const line = String(stdout).trim().split('\n')[0] || '';
    const [name, total, used] = line.split(',').map((s) => s.trim());
    const totalMB = parseInt(total, 10);
    const usedMB = parseInt(used, 10);
    if (!name || !Number.isFinite(totalMB) || !Number.isFinite(usedMB)) return null;
    return { name, totalMB, usedMB, freeMB: totalMB - usedMB };
}

/** @returns {Promise<{name, totalMB, usedMB, freeMB}|null>} null when there is no GPU */
function gpuInfo() {
    return new Promise((resolve) => {
        execFile('nvidia-smi', ['--query-gpu=name,memory.total,memory.used', '--format=csv,noheader,nounits'],
            { timeout: 10000 }, (err, stdout) => resolve(err ? null : parseNvidiaSmi(stdout)));
    });
}

module.exports = { gpuInfo, parseNvidiaSmi };
