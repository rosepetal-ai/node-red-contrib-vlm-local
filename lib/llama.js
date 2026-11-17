/**
 * One detection request against a llama-server: image + prompt in, assistant
 * text and the server's own timings out.
 */

'use strict';

const { fetchWithTimeout } = require('./http');

// What a flow may override, like the hosted node's whitelist (minus `reasoning`,
// which is a server flag here). Anything else is rejected, not silently dropped.
const SAMPLING_PARAMS = new Set([
    'max_tokens', 'temperature', 'top_p', 'top_k', 'seed', 'stop',
    'presence_penalty', 'frequency_penalty', 'repeat_penalty',
]);
// ~30 tokens a box: 4096 is ~130 boxes, and bounds a repetition loop to about two minutes.
const DEFAULT_MAX_TOKENS = 4096;

function samplingParams(parameters) {
    const params = parameters || {};
    const unknown = Object.keys(params).filter((k) => !SAMPLING_PARAMS.has(k));
    if (unknown.length) {
        throw new Error(`Unknown parameter(s) ${unknown.join(', ')}; allowed: ${[...SAMPLING_PARAMS].join(', ')}`);
    }
    return { temperature: 0, max_tokens: DEFAULT_MAX_TOKENS, ...params };
}

/**
 * @returns {Promise<{text: string, finishReason: string, promptMs: number|null, predictedMs: number|null}>}
 */
async function chat({ port, imageBase64, prompt, responseFormat, parameters, timeoutMs }) {
    const body = {
        messages: [{
            role: 'user',
            content: [
                { type: 'image_url', image_url: { url: `data:image/jpeg;base64,${imageBase64}` } },
                { type: 'text', text: prompt },
            ],
        }],
        response_format: responseFormat,
        ...samplingParams(parameters),
    };

    let res;
    try {
        res = await fetchWithTimeout(`http://127.0.0.1:${port}/v1/chat/completions`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body),
        }, timeoutMs);
    } catch (err) {
        if (err.name === 'TimeoutError') throw new Error(`llama-server did not answer within ${timeoutMs} ms`);
        throw new Error(`llama-server unreachable: ${err.message}`);
    }
    const data = await res.json().catch(() => ({}));
    if (!res.ok) {
        const reason = (data.error && data.error.message) || res.statusText;
        throw new Error(`llama-server error ${res.status}: ${reason}`);
    }

    const choice = (data.choices || [])[0] || {};
    const text = choice.message && choice.message.content;
    if (!text || !String(text).trim()) {
        throw new Error(`llama-server returned no text (finish_reason=${choice.finish_reason})`);
    }
    const timings = data.timings || {};
    return {
        text,
        finishReason: choice.finish_reason,
        promptMs: Number.isFinite(timings.prompt_ms) ? Math.round(timings.prompt_ms) : null,
        predictedMs: Number.isFinite(timings.predicted_ms) ? Math.round(timings.predicted_ms) : null,
    };
}

module.exports = { chat, samplingParams, SAMPLING_PARAMS, DEFAULT_MAX_TOKENS };
