# node-red-contrib-vlm-local

Node-RED node that runs an open vision-language model on the device itself and
turns an image plus a prompt, or a list of tags, into detections.

> **Status: work in progress.** The node itself is not written yet; this
> repository holds the shared code it will build on.

## What it will do

- `vlm-local-inferencer`: image + prompt or tags → a flat array of detections,
  with coordinates normalized 0–1 in every box format (`raw_boxes`, `xywhr`,
  `cwh`, `4points`, `2points`).
- Fully offline once set up: the runtime is the public
  [llama.cpp](https://github.com/ggml-org/llama.cpp) server image pinned by
  digest, and the weights are downloaded from public Hugging Face repositories
  at a pinned revision, with SHA-256 verification. No credentials on the device.
- Models are data: a catalog JSON lists each model's files, coordinate format,
  parameters and measured VRAM. A device can add its own entries without a
  release.

## Layout

| Path | Contents |
|---|---|
| `lib/result-conversion.js` | `expandResultFormats`: every box format derived from whichever one a result has |
| `lib/image-input.js` | `toJpegBase64`: encoded Buffer or raw bitmap → JPEG base64 |
| `lib/debug-image.js` | `renderDebugImage`: the frame with the result drawn on it |

These three files are copied unchanged from
`@rosepetal/node-red-contrib-vision-platform` v1.4.16, so this node's output
matches the other Rosepetal inference nodes. Keep them in sync by copying, not
by editing them here.

## Development

```bash
npm install
npm test
```

## License

Apache-2.0
