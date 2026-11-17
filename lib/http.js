'use strict';

/**
 * fetch with a deadline. Not AbortSignal.timeout(): its timer is unref'd, so a
 * request hanging on a half-started container lets the event loop drain.
 */
async function fetchWithTimeout(url, options, timeoutMs) {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), timeoutMs);
    try {
        return await fetch(url, { ...options, signal: controller.signal });
    } catch (err) {
        if (controller.signal.aborted) {
            const timeout = new Error(`no answer from ${url} within ${timeoutMs} ms`);
            timeout.name = 'TimeoutError';
            throw timeout;
        }
        throw err;
    } finally {
        clearTimeout(timer);
    }
}

module.exports = { fetchWithTimeout };
