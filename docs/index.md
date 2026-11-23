---
title: "VLM Local - Vision-Language Models on the Device"
description: "Find objects described by a prompt or a list of tags with an open vision-language model that runs on the device's NVIDIA GPU with llama.cpp. Part of @rosepetal/node-red-contrib-vlm-local."
head:
  - - meta
    - name: "keywords"
      content: "node-red, vlm, vision-language model, llama.cpp, open-vocabulary detection, object detection, qwen, gguf, nvidia gpu, docker, on-device inference, rosepetal"
  - - meta
    - property: "og:title"
      content: "VLM Local - @rosepetal/node-red-contrib-vlm-local"
  - - meta
    - property: "og:description"
      content: "Find objects described in words with an open vision-language model running on the device, with the same output as the VLM online node."
---

# VLM Local

`@rosepetal/node-red-contrib-vlm-local` provides one Node-RED node, `vlm-local-inferencer`, that runs an open vision-language model on the device itself and turns an image plus a prompt, or a list of tags, into detections.

## Installation

```bash
cd ~/.node-red
npm install @rosepetal/node-red-contrib-vlm-local
```

Or search `@rosepetal/node-red-contrib-vlm-local` in the Node-RED Palette Manager. After an install from the command line, restart Node-RED.

The package is on the public npm registry. Installing it downloads no model: the first deploy of a node pulls the llama.cpp runtime image and downloads the weights of its model (see [Files and containers](#files-and-containers)).

## Requirements

- Node.js 18.17 or later (`package.json` declares `>=18`, but `sharp` 0.33, which the package depends on, needs `^18.17.0 || ^20.3.0 || >=21.0.0`)
- Node-RED 3.0.0 or later
- Docker, reached from Node-RED through the Docker socket
- An NVIDIA GPU available to Docker: the node starts its container with every GPU of the device (what `docker run --gpus all` does) and loads the whole model on it. The editor reads the GPU's name and free memory with `nvidia-smi`; where Node-RED cannot run it, the edit dialog reports that there is no GPU
- `/opt/storage` mounted at the same path in Node-RED and on the host, and Node-RED able to reach the host's `127.0.0.1`, as on Rosepetal controllers, where Node-RED runs with `network_mode: host`. The container mounts the weights directory by the path Node-RED sees, and listens on a `127.0.0.1` port of the host
- On the first deploy of a model: access to `ghcr.io` and `huggingface.co`, and to the CDN hosts they redirect the downloads to, to download the runtime image (about 2.6 GB, as the node's help gives it) and the model's files (see [Models](#models)), and the disk space to keep them. After that, deploys and restarts download nothing

## Nodes

The node appears as **VLM local** under the **Rosepetal** category of the palette.

| Node | Purpose |
|------|---------|
| **[vlm-local-inferencer](./vlm-local-inferencer.md)** | Find objects described by a prompt or a list of tags with an open vision-language model on the device's GPU, and unload it when idle |

## Models

The models come from `catalog.json`, in the package. The first one is the model a new node selects.

| Model | Weights (Hugging Face repository and files) | Download | VRAM (measured, RTX 3060) | Notes |
|-------|----------------------------------------------|----------|---------------------------|-------|
| `qwen3.5-9b-q4` (Qwen3.5 9B) | `unsloth/Qwen3.5-9B-GGUF`: `Qwen3.5-9B-Q4_K_M.gguf`, `mmproj-F16.gguf` | 6.60 GB | 6970 MB | Default. F1 .55 / IoU .87 on 10 client datasets |
| `qwen3.5-4b-q4` (Qwen3.5 4B) | `unsloth/Qwen3.5-4B-GGUF`: `Qwen3.5-4B-Q4_K_M.gguf`, `mmproj-F16.gguf` | 3.41 GB | 4390 MB | Smaller GPUs. F1 .46 / IoU .87 |

- Each file is taken at a fixed commit of its repository and checked against the size and SHA-256 written in the catalog.
- The catalog gives both models the license `apache-2.0`, with commercial use allowed. They run with an 8192-token context, reasoning off and at least 1024 tokens per image.
- The F1 and IoU figures are the maintainers' measurements, as the README gives them.
- The editor counts 1 GB as 1024³ bytes: it shows the downloads as 6.1 GB and 3.2 GB, and the VRAM as 6.8 GB and 4.3 GB.

A device can add models, or replace these, in `/opt/storage/models/vlm/catalog.json`, without a new release of the package: see [Model catalog](./vlm-local-inferencer.md#model-catalog).

## Files and containers

- **Runtime**: the public llama.cpp server image, `ghcr.io/ggml-org/llama.cpp:server-cuda-b12429`, pulled without credentials by its digest (`ghcr.io/ggml-org/llama.cpp@sha256:d68ec493…`) when Docker does not have it yet.
- **Weights**: downloaded from the public Hugging Face repositories into `/opt/storage/models/vlm/<model id>/` (the environment variable `VLM_LOCAL_MODELS_DIR` changes the root). A download that is interrupted resumes where it stopped, a file is put in place only once its SHA-256 matches, and two nodes of the same Node-RED on the same model share one download. Files copied into that directory by hand are verified on the first deploy, so a device without network can be installed by copying them, with the runtime image already in Docker.
- **Containers**: one per node and Node-RED instance, `nodered-vlm-local-<Node-RED port>-<node id>`, with the weights mounted read-only and the server on a `127.0.0.1` port that Docker assigns. A redeploy reuses it when neither the model nor its runtime has changed; deleting the node removes it.
- **Unload**: the node can stop its container after each inference or after N seconds without use, which frees all of its VRAM; the next message starts it again, in about 3 seconds while the weights are still in the operating system's file cache.
- **Deleting models**: the edit dialog shows which models are on the device and deletes the files of one, unless a node or a container still uses it. Nothing is deleted automatically.

The [node's page](./vlm-local-inferencer.md#containers-and-files) has the details.

## Same Output as VLM Online

`vlm-local-inferencer` is the local twin of `vlm-online-inferencer`, the node of `@rosepetal/node-red-contrib-vision-platform` that calls a hosted model through the Rosepetal platform ([Vision Platform documentation](https://docs.rosepetal.ai/flows/vision-platform)). Both take an image and a prompt or a list of tags, and write the same `msg.inference` array, entry for entry: `tag`, `confidence` and the box formats `raw_boxes`, `2points`, `4points`, `xywhr` and `cwh`, relative to the image (0 to 1). A flow can swap one for the other.

What changes with the local node: no Credentials node and no tokens; the model list is this package's catalog; the **Unload** setting; a default **Timeout** of 120000 ms instead of 30000; and timings under `msg.performance.VLMLocal` (when the node has no name), with the time spent loading an unloaded model in `wakeMs`.
