---
title: "VLM Local Node - Vision-Language Detection on the Device"
description: "Find objects described by a prompt or a list of tags with an open vision-language model that runs in a llama.cpp container on the device's NVIDIA GPU, with the same output as the VLM online node. Part of @rosepetal/node-red-contrib-vlm-local."
head:
  - - meta
    - name: "keywords"
      content: "vlm, vision-language model, llama.cpp, llama-server, open-vocabulary detection, prompt, tags, qwen, gguf, nvidia gpu, docker, unload, vram, bounding boxes, node-red, @rosepetal/node-red-contrib-vlm-local"
  - - meta
    - property: "og:title"
      content: "VLM Local Node - @rosepetal/node-red-contrib-vlm-local"
  - - meta
    - property: "og:description"
      content: "Find objects described in words with an open vision-language model on the device's GPU, and get boxes in the same shape as the VLM online node."
---

# VLM Local Node

## Overview

The VLM Local node (`vlm-local-inferencer`, `VLM local` in the palette) finds objects in an image with an open vision-language model: you describe what to look for in words, with a prompt or a list of tags, instead of training a model for it. It returns a box and a confidence for each object found.

The model runs **on the device**, on its NVIDIA GPU, in a llama.cpp server container that the node starts and owns. The runtime image is public and the weights come from public Hugging Face repositories, so the node needs no credentials and, once the model is on the device, no network.

It is the local twin of `vlm-online-inferencer`, of `@rosepetal/node-red-contrib-vision-platform`: same inputs and the same `msg.inference` array, entry for entry, so a flow can swap one for the other.

An image takes seconds, more when it holds more objects (see [Speed](#speed)), so the node is meant for inspecting a frame when something happens, not for every frame of a fast line.

## Key Features

- **Open vocabulary**: a prompt such as "every screw head", or a closed list of tags with a description each
- **Same detections as `vlm-online-inferencer`**: coordinates relative to the image (0 to 1), in every box format
- **On the device**: public runtime image and weights, pinned and verified; no cloud and no credentials
- **Model catalog** with the measured VRAM and the license of each model; a device can add its own models
- **Unload**: free the GPU memory after each inference or after N seconds without use, and load again in about 3 seconds
- **Model files from the editor**: what is installed, what is left to download, and a button to delete a model's files
- **Debug image** with the boxes drawn under the node

## Configuration

| Property | Default | Meaning |
|----------|---------|---------|
| **Name** | empty | Label of the node; `VLM local` when empty. Also the key of its timings in `msg.performance` |
| **Model** | the first model of the catalog | The model that answers, from the [catalog](#model-catalog). Required. See [Model list](#model-list) |
| **Unload** | `Never — keep the model in VRAM` | `Never`, `After each inference` or `After N seconds without use`. See [Unload](#unload) |
| **N (seconds)** | `60` | Shown with `After N seconds without use`: a number, or a `msg`, flow or global property |
| **Image** | `msg.payload` | Where the image is: a `msg`, flow or global property |
| **Find by** | `Prompt` | `Prompt — the model names what it finds` or `Tags — a fixed list of labels`. Only the field chosen is shown and read |
| **Prompt** | empty | What to find, in words. A string, or a `msg`, flow or global property |
| **Tags** | empty | The labels the model must choose from (see [Tags](#tags)). Edited as rows (`list`), as JSON, or read from a `msg`, flow or global property |
| **Output** | `inference` | `msg` property that receives the detections |
| **Parameters** | `{}` | Sampling overrides for the request, as a JSON object or from a `msg`, flow or global property (see [Parameters](#parameters)) |
| **Timeout** | `120000` | Milliseconds to wait for the model's answer to one image, counted from the request: it includes the time the image waits behind another one, since the container answers one at a time. Loading an unloaded model is not counted |
| **Debug** | off | Draw the boxes on the image, under the node (see [Debug image](#debug-image)) |
| **Interval** | `1` | With Debug: one message in N is drawn |
| **Width** | `160` | With Debug: width of the image under the node, in pixels |
| **Box thickness** | `2` | With Debug: 1 to 10 |
| **Font size** | `0.5` | With Debug: 0.3 to 2 |

### Model list

**Model** lists the models of the catalog, each as `<name> · ≈<VRAM> VRAM · <state>`, where the state is:

- `installed`: both files of the model are on the device and verified
- `<size> to download`: what is still missing (the part of an interrupted download that is already on disk does not count)
- `on disk, verified on first deploy`: the files are there, copied by hand, and have not been checked yet

The line under the field compares the measured VRAM of the model with the GPU: `Needs ≈6.8 GB · GPU: <name>, <total> (<free> free)`. It adds `0 GB while unloaded` when **Unload** is not `Never`, and the VRAM held by the other loaded VLM local models of the device, of any Node-RED instance. It ends with the model's license, and turns red when the model does not fit in the free memory (the memory held by this node's own loaded model counts as free). Without a GPU, or when Node-RED cannot run `nvidia-smi`, it reads `No NVIDIA GPU found on this device: this node needs one.`; when the catalog cannot be read, `Cannot read the model catalog: <reason>`.

The editor counts 1 GB as 1024³ bytes, and 1 GB of VRAM as 1024 MB.

A model marked `deprecated` in the catalog is listed, with *(deprecated)*, only on a node that already uses it.

The trash button of a model whose files are on the device deletes its directory (weights, partial downloads and markers), after a confirmation. It is refused while a deployed node uses that model, while a container of that model exists (loaded or unloaded, of this or another Node-RED instance), or while this Node-RED is downloading its files: change or delete those nodes and deploy first. A node that uses a deleted model downloads it again on its next deploy. Nothing is ever deleted automatically.

### Prompt or tags

A prompt opens the vocabulary: the model names what it finds in a word or two, and that name comes back in `tag` (`object` when it gives none). Tags close it: the answer may only use the names of the tags. They are matched without regard to case; with a single tag, every detection gets that tag, and with several, a detection whose label is none of them is dropped.

The node inserts the prompt, or the tags with their descriptions, into a fixed detection prompt that asks the model for a JSON list of boxes, so the prompt only has to say what to find.

### Tags

Each tag has a name and an optional description, which becomes that tag's own line in the prompt: the place to say "the capsule on the neck, not the neck" next to the label it applies to. The node's help says the description matters: measured on client datasets, opaque names such as `golpe` or `stacker` without a description found nothing.

```json
[
  { "name": "cap", "description": "the aluminium capsule on the neck" },
  { "name": "barcode" }
]
```

The name is also the `tag` of each detection, unless the tag has an `id`: then the detection's `tag` is that id, for flows that map the answer to their own tag ids. Read from a `msg`, flow or global property, the value can be an array of such objects, an array of strings, or a comma-separated string of names (`"cap, barcode"`).

### Parameters

Optional sampling overrides, as a JSON object:

| Key | Default |
|-----|---------|
| `max_tokens` | `4096`. A box takes about 30 tokens, so this is room for about 130 boxes: raise it for crowded frames |
| `temperature` | `0` |
| `top_p`, `top_k`, `seed`, `stop`, `presence_penalty`, `frequency_penalty`, `repeat_penalty` | Not sent: llama-server's own |

Any other key fails the message with `Unknown parameter(s) <keys>; allowed: …`. `reasoning` is not one of them: reasoning is a setting of the container, taken from the catalog. A value that is not an object, or a property that does not exist, sends the defaults. An array counts as an object: a non-empty one fails with `Unknown parameter(s) 0, …`.

### Unload

| Unload | What happens |
|--------|--------------|
| **Never** | The model stays in VRAM while the node is deployed |
| **After each inference** | The container stops as soon as no message is being answered |
| **After N seconds without use** | The container stops N seconds after the last message was answered; a message that arrives in the meantime cancels the countdown |

Stopping the container frees all of its VRAM, and the status turns grey: `Unloaded (VRAM released)`. The next message starts it again, which takes about 3 seconds while the weights are still in the operating system's file cache (measured on an RTX 3060); after a reboot, the first load reads them from disk. That time is reported in `wakeMs` and, in the status, as `Load: <ms>`. Messages that arrive while the model is loading share that one load.

With **After N seconds without use**, N is read on every message, and the last message decides how long the model stays loaded after it. A missing or negative N, or one that is not a number, fails that message with `Unload after N seconds: N must be a number >= 0, got <value>`.

Unless N's type is `msg`, the countdown also starts at deploy, as soon as the model is ready: with **After each inference**, the model is then unloaded right after a deploy and the first message loads it (N's type is kept when **Unload** changes, even while its field is hidden). When N comes from `msg`, the model stays loaded after a deploy until the first message arrives; when it comes from a flow or global variable that is not set yet, the node warns and the model also stays loaded until then.

If the container cannot be stopped, the node warns `vlm-local: unload failed, model stays loaded: <reason>` and the model stays loaded.

### Debug image

With **Debug** on, the node draws each box with a `tag: confidence` caption on the JPEG it sent to the model, and shows that image under the node on the editor canvas, **Width** pixels wide. It draws the first message and then one in **Interval**. A drawing that fails only warns, `debug image not drawn: <reason>`: the message is sent anyway.

## Input

```javascript
msg.payload = imageBuffer;   // a JPEG, PNG, WebP or GIF Buffer, or a raw bitmap
return msg;
```

**Image** names the property (`msg.payload` by default).

- An encoded image is sent as JPEG: a JPEG as it is, PNG, WebP and GIF converted on the device (quality 85). A `Uint8Array`, or a Buffer that went through JSON (`{ "type": "Buffer", "data": [...] }`), is read as a Buffer.
- A raw bitmap `{ data, width, height, channels, colorSpace }`, as the image nodes of `@rosepetal/node-red-contrib-image-tools` pass it, is converted to JPEG. `channels` (1, 3 or 4) is computed from the length of `data` when missing; `BGR` and `BGRA` are swapped to RGB and RGBA first.
- A Buffer that is not an encoded image fails with `Buffer input is not a recognized encoded image (JPEG/PNG/WebP/BMP/GIF). A raw pixel buffer must be wrapped as {data, width, height, channels}.`
- A raw bitmap whose `data` does not match `width × height × channels`, or with a channel count other than 1, 3 or 4, fails with an error that says which; any other value fails with `Unsupported image input. Expected an encoded image Buffer or a rosepetal raw bitmap {data, width, height, channels, colorSpace}.`

## Output

**Output** (`msg.inference` by default) receives an array, one entry per object found, and an empty array when nothing is found:

```javascript
msg.inference = [
  {
    raw_boxes: [[0.41, 0.08], [0.58, 0.08], [0.58, 0.19], [0.41, 0.19]],
    "2points": { xmin: 0.41, ymin: 0.08, xmax: 0.58, ymax: 0.19 },
    tag: "cap",
    confidence: 0.97,
    xywhr: { x: 0.41, y: 0.08, w: 0.17, h: 0.11, r: 0 },
    cwh: { x: 0.495, y: 0.135, w: 0.17, h: 0.11, r: 0 },
    "4points": { x0: 0.41, y0: 0.08, x1: 0.58, y1: 0.08, x2: 0.58, y2: 0.19, x3: 0.41, y3: 0.19 }
  }
]
```

| Field | Meaning |
|-------|---------|
| `tag` | With a prompt, the name the model gave; with tags, the tag's name, or its `id` when it has one |
| `confidence` | What the model wrote, from 0 to 1 (1 when it wrote none). It is not a calibrated score: the node's help says it is usually 0.95 to 0.99 |
| `raw_boxes` | The four corners: top-left, top-right, bottom-right, bottom-left |
| `2points` | `xmin`, `ymin`, `xmax`, `ymax` |
| `4points` | The same four corners as `x0`, `y0` … `x3`, `y3` |
| `cwh` | Centre `x`, `y`, width `w`, height `h` and rotation `r` |
| `xywhr` | Top-left `x`, `y`, width `w`, height `h` and rotation `r` |

The boxes are not rotated: `r` is 0. The model answers on a 0 to 1000 grid; the node divides by 1000, puts the corners in order, clips the box to the image and drops a box with no area. There is no `classId` or `color`: a prompted model answers with names, not indices into a class list.

Timings go to `msg.performance.<Name>` (`msg.performance.VLMLocal` when the node has no name):

| Property | Meaning |
|----------|---------|
| `startTime`, `endTime` | When the node started on the image and finished, in milliseconds since 1970 |
| `encodeMs` | JPEG encoding on the device |
| `wakeMs` | Loading an unloaded model; 0 when it was loaded |
| `requestMs` | The request to llama-server |
| `inferenceMs` | llama-server's own time for the prompt and the answer; `null` when it reports none |
| `totalMs` | The whole message |

### Speed

The node's help gives these times, on an RTX 3060: seconds per image, growing with the number of objects (about 30 tokens per box): 1 to 2 s with a couple of objects, about 15 s with 15, about 30 s with 30. The default **Timeout**, 120 s, leaves room for that. The container answers one image at a time (llama-server runs with one slot).

## Status and Errors

| Status | Meaning |
|--------|---------|
| yellow ring `Checking runtime…` | Startup: looking for the runtime image in Docker |
| yellow ring `Pulling runtime… N%` | Pulling it |
| yellow ring `Downloading weights… N%` | Downloading the model's two files, counted together |
| yellow ring `Verifying <file>…` | Checking a file's SHA-256 |
| yellow ring `Loading model…` | The container has started; waiting for llama-server to answer. Also when an unloaded model loads again |
| green dot `Ready` | The node takes messages |
| red ring, a message | Startup failed: the first 50 characters of the error |
| yellow dot `encoding`, `inferring` | Converting the image, waiting for the model |
| green dot `N found \| Total: Xms \| Model: Yms` | Last message: detections, total time and llama-server's own time (`?` when it reports none), plus `\| Load: Zms` when the model had to be loaded |
| grey ring `Unloading…`, `Unloaded (VRAM released)` | Stopping the container, and stopped (see [Unload](#unload)) |
| red dot, a message | The last message failed: the first 50 characters of the error. Back to `Ready` or `Unloaded (VRAM released)` after 8 seconds, once the node is ready |

### Startup errors

The node logs `vlm-local: <error>` and shows the error in its status. It does not retry: every message fails with `Not ready: <error>` until the node starts again, at the next deploy that restarts it or when Node-RED restarts.

| Error | Cause |
|-------|-------|
| `No model selected` | **Model** is empty |
| `Model '<id>' is not in the catalog.` | The model of the node is in no catalog of this device |
| `Cannot read model catalog <file>: <reason>` | The packaged or the device catalog is not valid JSON |
| `Catalog runtime must be pinned by @sha256 digest` | The `runtime` of the device catalog has no `@sha256:` digest |
| `<file>: model '<id>': <what is wrong>` | A catalog entry does not pass the checks of the [catalog](#model-catalog) |
| `Runtime not available (<reason>). Without network, load it with: docker load` | The runtime image is not in Docker and could not be pulled |
| `HTTP <status> downloading <url>` | Hugging Face answered the request for a file with an error status |
| `<file>: SHA-256 mismatch (got <hash>), download discarded`, `<file>: got <n> bytes, expected <m>; download discarded` | A downloaded file is not the one in the catalog. It is deleted; the next start downloads it again |
| `No GPU available to Docker: <reason>` | Docker could not give the container a GPU |
| `Cannot start llama-server: <reason>` | Docker could not create or start the container |
| `llama-server exited (code <n>): <last lines of its log>` | The server stopped while loading the model |
| `llama-server not ready after 600s: <last lines of its log>` | The server did not answer within 10 minutes |
| Another message | Docker or the network failed, with their own error |

### Message errors

A message that fails is not sent on: the error reaches the debug sidebar and any *catch* node.

| Error | Cause |
|-------|-------|
| `Not ready: model still loading` | The message arrived before the startup finished. Messages are not queued: a first download can take minutes |
| `Not ready: <error>` | The startup failed with that error |
| `No image on the configured input property.` | Nothing at **Image** |
| `Find by prompt needs a prompt.`, `Find by tags needs at least one tag.` | The field chosen by **Find by** is empty |
| `Unload after N seconds: N must be a number >= 0, got <value>` | A missing or bad **N** (see [Unload](#unload)) |
| The image errors of [Input](#input) | The image cannot be read |
| `Unknown parameter(s) <keys>; allowed: …` | A key of **Parameters** that the node does not accept |
| `llama-server did not answer within <Timeout> ms` | No answer within **Timeout** |
| `llama-server unreachable: <reason>`, `llama-server error <status>: <reason>` | The container could not be reached, or answered with an error |
| `llama-server returned no text (finish_reason=<reason>)` | An empty answer |
| `Model response was not valid JSON: <answer>`, `No detections array in response; keys were [<keys>]`, `Unexpected JSON payload type: <type>` | The answer is not a list of detections |
| A startup error of the container | An unloaded model could not be loaded again; the next message tries again |

## Containers and Files

### Runtime and weights

- **Runtime**: the image named by the catalog (`runtime` of the model, or else of the catalog), pulled without credentials by its digest: `ghcr.io/ggml-org/llama.cpp@sha256:d68ec493…` for the packaged models. It is pulled only when Docker does not have it.
- **Weights**: each model has two files, the model and its multimodal projector (`mmproj`). They are downloaded from `https://huggingface.co/<repo>/resolve/<revision>/<file>` into `/opt/storage/models/vlm/<model id>/` (`$VLM_LOCAL_MODELS_DIR/<model id>/` when that variable is set). A file is written as `<file>.part` and resumes from there when a download is interrupted; it gets its final name only when its size and SHA-256 match the catalog, and a `<file>.verified` marker that holds the hash, so later deploys do not hash it again. Two nodes of the same Node-RED on the same model share one download; two Node-RED instances of the device that download the same model at the same time would both write to the same `<file>.part`.
- **Without network**: copy the two files, with the names of the catalog, into the model's directory. The first deploy verifies them (`Verifying <file>…`) and deletes a file whose hash does not match. The runtime image must already be in Docker; the node's help says to `docker load` it beforehand.

### Container

- **One per node and Node-RED instance**: `nodered-vlm-local-<port>-<node id>`, where the port is Node-RED's `uiPort` (1880 by default), so two Node-RED instances of the same device that hold the same flow get a container each. The containers carry the labels `nodered.vlm-local.node`, `.instance`, `.model`, `.config` and `.vram_mb`.
- The model's directory is mounted read-only; llama-server listens on port 8080 of the container, published on a `127.0.0.1` port of the host that Docker assigns. The container gets every GPU, and llama-server loads all the layers of the model on the GPU, with one slot and the context size of the catalog.
- At startup, the node waits up to 10 minutes for llama-server's `/health` to answer, since a first load can read about 7 GB from disk.
- A redeploy keeps the container: the new instance of the node reuses it (and starts it if it was unloaded) when the runtime, the model files and the server settings are the same, and replaces it otherwise.
- Deleting the node and deploying removes its container. Stopping Node-RED leaves the container as it is (a loaded model keeps its VRAM), and the node reuses it when Node-RED starts again.

## Model Catalog

The models come from `catalog.json`, in the package. A device can add or replace entries without a new release, in `/opt/storage/models/vlm/catalog.json` (`$VLM_LOCAL_MODELS_DIR/catalog.json` when that variable is set), with the same format:

- An entry with the `id` of a packaged one replaces it as a whole: a field it leaves out is not taken from the packaged entry. An entry with a new `id` is added after the packaged ones.
- A `runtime` in the device file replaces the default runtime image.
- The catalogs are read when a node starts and when its edit dialog opens: a change applies at the next deploy of the node.
- An entry that does not pass the checks stops every VLM local node from starting, and the edit dialog from listing the models, with an error that names the file, the model and the field.

The fields of an entry:

| Field | Meaning |
|-------|---------|
| `id` | A plain directory name (letters, digits, `.`, `_`, `-`): the value a node stores, and the name of the model's directory |
| `name` | The name shown in the editor |
| `files.model`, `files.mmproj` | The model and its multimodal projector, each with `repo` (Hugging Face repository), `revision` (a 40-character commit hash), `path` (a file name, without `/`), `bytes` and `sha256`, all required |
| `runtime` | `null` to use the catalog's runtime, or another image pinned by `@sha256:` digest |
| `coord_format` | `grid_xyxy` (`[x1, y1, x2, y2]`) or `grid_yxyx` (`[ymin, xmin, ymax, xmax]`): the order in which the model writes the coordinates of a box, on a 0 to 1000 grid |
| `parameters` | Settings of llama-server for this model: `ctx` (context size, 8192 when missing), `reasoning` (`false` turns reasoning off), `image_min_tokens` (minimum tokens per image). The packaged entries also have `temperature: 0`, which the node does not read: the request uses 0 unless **Parameters** says otherwise |
| `vram_mb` | The VRAM the model needs, measured, in MB: a positive integer |
| `ram_mb_cpu` | `null` in the packaged entries; the node does not read it |
| `license` | `id` (shown in the editor) and `commercial` |
| `deprecated` | Optional: `true` hides the model from the list of nodes that do not use it yet |

The entry of the default model, as the package has it:

```json
{
  "id": "qwen3.5-9b-q4",
  "name": "Qwen3.5 9B",
  "files": {
    "model": {
      "repo": "unsloth/Qwen3.5-9B-GGUF",
      "revision": "3885219b6810b007914f3a7950a8d1b469d598a5",
      "path": "Qwen3.5-9B-Q4_K_M.gguf",
      "bytes": 5680522464,
      "sha256": "03b74727a860a56338e042c4420bb3f04b2fec5734175f4cb9fa853daf52b7e8"
    },
    "mmproj": {
      "repo": "unsloth/Qwen3.5-9B-GGUF",
      "revision": "3885219b6810b007914f3a7950a8d1b469d598a5",
      "path": "mmproj-F16.gguf",
      "bytes": 918166080,
      "sha256": "f70dc3509053962b0d0d3ee8a7eacebf5d60aa560cad78254ae8698516ae029f"
    }
  },
  "runtime": null,
  "coord_format": "grid_xyxy",
  "parameters": { "temperature": 0, "reasoning": false, "image_min_tokens": 1024, "ctx": 8192 },
  "vram_mb": 6970,
  "ram_mb_cpu": null,
  "license": { "id": "apache-2.0", "commercial": true }
}
```

A device file has the form `{ "models": [ ... ] }`, with `"runtime"` only to replace the default image.

## Example

Check the closure of a bottle when a sensor fires, with a closed vocabulary, and free the GPU when the line stops:

```
[sensor trigger] → [camera] → [VLM local: Find by Tags, Unload after 300 s without use] → [function] → [reject output]
```

Tags: `[{"name": "cap", "description": "the aluminium capsule on the neck"}, {"name": "missing cap", "description": "an open neck with no capsule"}]`

```javascript
// function: reject when no cap is seen with enough confidence
const caps = msg.inference.filter(d => d.tag === "cap" && d.confidence >= 0.6);
msg.payload = caps.length === 0;   // true = reject
return msg;
```

## Related

- [Overview](./index.md) - Installation, requirements and the models of the catalog
- `vlm-online-inferencer`, in the [Vision Platform documentation](https://docs.rosepetal.ai/flows/vision-platform) - The same detections from a hosted model, through the Rosepetal platform
