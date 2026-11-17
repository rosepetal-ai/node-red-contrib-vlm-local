/**
 * Unload the model (stop its container) after N idle seconds, load it again on
 * the next request. N is given per request, so a flow can set it per message.
 *
 * Node-side rather than llama-server's --sleep-idle-seconds: that one is fixed
 * when the container starts, so a per-message N would mean recreating it.
 * Stopping frees all of the VRAM; starting again costs ~3 s with the weights in
 * the OS file cache (measured on an RTX 3060, as much as the server's own sleep).
 */

'use strict';

/**
 * @param {object} hooks
 * @param {() => Promise<void>} hooks.stop   stop the server (frees VRAM)
 * @param {() => Promise<void>} hooks.start  start it and wait until it answers
 * @param {(state: string) => void} [hooks.onState]  'unloading' | 'unloaded' | 'loading' | 'loaded'
 * @param {(err: Error) => void} [hooks.onError]     a failed unload (the model stays loaded)
 */
function createUnloader({ stop, start, onState = () => {}, onError = () => {} }) {
    let state = 'loaded';
    let inFlight = 0;
    let timer = null;
    let stopping = null;
    let starting = null;

    const clear = () => { clearTimeout(timer); timer = null; };

    function unloadNow() {
        timer = null;
        if (inFlight > 0 || state !== 'loaded') return;
        state = 'unloading';
        onState('unloading');
        stopping = stop()
            .then(() => { state = 'unloaded'; onState('unloaded'); })
            .catch((err) => { state = 'loaded'; onError(err); })
            .finally(() => { stopping = null; });
    }

    /** Unload after `seconds` without requests; null/undefined keeps the model loaded. */
    function schedule(seconds) {
        clear();
        if (seconds === null || seconds === undefined || inFlight > 0) return;
        timer = setTimeout(unloadNow, Math.max(0, seconds) * 1000);
    }

    /**
     * Before a request: cancel any pending unload and load the model if needed.
     * @returns {Promise<{wakeMs: number}>} time spent loading, 0 if it was loaded
     */
    async function acquire() {
        inFlight += 1;
        clear();
        try {
            if (stopping) await stopping;
            if (state === 'loaded') return { wakeMs: 0 };
            const t = Date.now();
            if (!starting) {
                state = 'loading';
                onState('loading');
                starting = start()
                    .then(() => { state = 'loaded'; onState('loaded'); })
                    .catch((err) => { state = 'unloaded'; throw err; })
                    .finally(() => { starting = null; });
            }
            await starting;
            return { wakeMs: Date.now() - t };
        } catch (err) {
            inFlight -= 1;
            throw err;
        }
    }

    /** After a request: the last one in flight starts the countdown. */
    function release(seconds) {
        inFlight = Math.max(0, inFlight - 1);
        if (inFlight === 0) schedule(seconds);
    }

    return { acquire, release, schedule, dispose: clear, state: () => state };
}

module.exports = { createUnloader };
