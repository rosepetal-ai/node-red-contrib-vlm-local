# node-red-contrib-vlm-local

Node-RED node that runs an open vision-language model on the device itself and
turns an image plus a prompt, or a list of tags, into detections.

`vlm-local-inferencer` is the local twin of vision-platform's
`vlm-online-inferencer`: same inputs, same `msg.inference` array entry for entry
(`tag`, `confidence`, `raw_boxes`, `2points`, `4points`, `xywhr`, `cwh`, all 0–1),
so a flow can swap one for the other.

- **Offline once installed.** The runtime is the public
  [llama.cpp](https://github.com/ggml-org/llama.cpp) server image, pinned by
  digest; the weights come from public Hugging Face repos at a pinned commit,
  SHA-256 verified. No credentials on the device.
- **One container per node**, `nodered-vlm-local-<Node-RED port>-<node id>`, on a
  Docker-assigned `127.0.0.1` port, removed when the node is deleted.
- **Unload**: keep the model loaded, or stop its container after each inference /
  after N idle seconds (N from the panel, `msg`, `flow` or `global`); ≈3 s to load again.
- **Delete models** from the panel; refused while a node or container uses them.
- **Models are data**: `catalog.json` lists each model's files, coordinate
  format, parameters and measured VRAM. A device can add or override entries in
  `/opt/storage/models/vlm/catalog.json` without a release.

## Installation

```bash
cd ~/.node-red
npm install @rosepetal/node-red-contrib-vlm-local
```

Or search `@rosepetal/node-red-contrib-vlm-local` in the Node-RED Palette Manager.

## Documentation

| Page | Contents |
|------|----------|
| [Overview](docs/index.md) | Installation, requirements, the models of the catalog, what the node downloads and where |
| [vlm-local-inferencer](docs/vlm-local-inferencer.md) | Reference of the node: configuration, prompt or tags, unload, input and output, status and errors, containers, the catalog format |

The same pages are published at <https://docs.rosepetal.ai/flows/vlm-local>.

## Requirements

An NVIDIA GPU available to Docker, the Docker socket, and `/opt/storage` mounted
at the same path in Node-RED and on the host (as on Rosepetal controllers, where
Node-RED runs with `network_mode: host`). Weights go to
`/opt/storage/models/vlm/<model id>/` (override with `VLM_LOCAL_MODELS_DIR`).

## Models

| id | VRAM (measured, RTX 3060) | Notes |
|---|---|---|
| `qwen3.5-9b-q4` | 6970 MB | Default. F1 .55 / IoU .87 on 10 client datasets |
| `qwen3.5-4b-q4` | 4390 MB | Smaller GPUs. F1 .46 / IoU .87 |

## Layout

| Path | Contents |
|---|---|
| `nodes/vlm-local-inferencer.{js,html}` | The node and its edit panel |
| `catalog.json` | The offered models; the first entry is the default |
| `lib/catalog.js` | Packaged + device catalog, merged by id and validated |
| `lib/weights.js` | Resumable HF downloads, SHA-256 verification, one download per model |
| `lib/runtime.js` | The llama-server container: pull, start/reuse, health, removal |
| `lib/unloader.js` | Unload after N idle seconds, load again on the next request |
| `lib/llama.js` | One chat-completions request with the image and the JSON schema |
| `lib/detect.js` | Prompt, schema and parser, ported from the production labeler |
| `lib/gpu.js` | GPU name and memory from `nvidia-smi` |
| `lib/result-conversion.js`, `lib/image-input.js`, `lib/debug-image.js`, `resources/debug-image-canvas.js` | Copied unchanged from `@rosepetal/node-red-contrib-vision-platform` v1.4.16; keep in sync by copying |

`test/fixtures/labeler.json` is generated from the production Python labeler;
`test/detect.test.js` replays it so the port cannot drift.

## Development

```bash
npm install
npm test
```

## License

Apache-2.0
