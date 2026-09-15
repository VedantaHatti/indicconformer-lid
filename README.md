# Language Identification Service

Spoken **language identification** using a shared Conformer encoder, shared CTC
output, language vocabulary masks, and `normalized_ctc_score`.

Not a separately trained LID classifier. Does not transcribe.

## Quick start

```bash
cd indicconformer-lid
python3 -m venv .venv && source .venv/bin/activate

make install   # install all pinned deps; ask cpu/gpu → writes .env
make setup     # download model (asks HF token; skips if model/ exists)
make run       # asks which port, then starts API + UI
```

Open **http://127.0.0.1:PORT/** (default `8007`).

## What each make target does

| Command | Does |
|---------|------|
| `make install` | `pip install -r requirements.txt` (+ Torch CPU index); ask cpu/gpu; write `.env` |
| `make setup` | Hugging Face download of LID-only artifacts (token prompt); skip if present |
| `make run` | Ask port → start uvicorn + browser UI |
| `make model` | Force re-download model |
| `make docker-build` | Build CUDA+cuDNN image |
| `make docker-run` | Ask host port; run with `--gpus all` and `./model` mounted |

## Will Docker help for GPU?

**Yes.** Bare-metal GPU needs NVIDIA driver + **CUDA 12** + **cuDNN 9** (`libcudnn.so.9`).
If that library is missing, ONNX Runtime cannot create `CUDAExecutionProvider`
(even when `onnxruntime-gpu` is installed).

The Docker image is based on `nvidia/cuda:12.4.1-cudnn-runtime`, so cuDNN 9 is
inside the container. Recommended GPU path when the host lacks cuDNN:

```bash
make install          # optional on host if you only use Docker for serving
make setup            # download model onto the host once
make docker-build
make docker-run       # asks port; LID_DEVICE=cuda; mounts ./model
```

Requires [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html).

## Bare-metal GPU

1. `make install` → choose **gpu** (`LID_DEVICE=cuda` in `.env`)
2. Ensure host has driver + CUDA 12 + **cuDNN 9**
3. `make setup && make run`

If startup says CUDA EP fell back to CPU / missing `libcudnn.so.9`, use Docker
or set `LID_DEVICE=cpu`.

## Architecture

```text
Audio → TorchScript preprocessor (CPU always)
     → ONNX Conformer encoder (once; CPU or CUDA)
     → ONNX shared CTC head (once; CPU or CUDA)
     → language masks → normalized_ctc_score → ranked languages
```

Torch CUDA is **never** required. GPU = ONNX Runtime CUDA only.

## Tested / pinned versions

| Package | Version |
|---------|---------|
| Python | 3.10 – 3.12 |
| fastapi | 0.115.6 |
| uvicorn | 0.34.0 |
| numpy | 1.26.4 |
| torch | 2.4.1 (CPU wheel) |
| onnxruntime-gpu | 1.19.2 |
| soundfile | 0.12.1 |
| huggingface_hub | 0.27.1 |
| python-dotenv | 1.0.1 |
| httpx / pytest | 0.28.1 / 8.3.4 |

All pins are in [`requirements.txt`](requirements.txt) (single file).

### System deps for bare-metal GPU (not in pip)

- NVIDIA driver
- CUDA 12 runtime
- cuDNN 9 (`libcudnn.so.9`)

## Model artifacts (LID only)

Kept: `preprocessor.ts`, `encoder.onnx` + shards, `ctc_decoder.onnx`, `language_masks.json`  

Excluded: `rnnt*`, `joint*`, `vocab.json`

After `make setup`, inference is offline. The server never downloads.

## Languages (22)

`as bn brx doi gu hi kn kok ks mai ml mni mr ne or pa sa sat sd ta te ur`  

Default candidates: `hi,kn,mr,ta,te`

## API

- `GET /` — UI  
- `GET /health` — device, providers, languages  
- `POST /identify` — multipart `audio`  

```bash
curl -s http://127.0.0.1:8007/health | jq
curl -s -X POST http://127.0.0.1:8007/identify -F "audio=@utt.wav" | jq
```

## Python

```python
from lid import LanguageIdentifier

lid = LanguageIdentifier(model_dir="./model", device="cpu")
print(lid.identify(open("utt.wav", "rb").read())["language"])
```

## Limitations

- Compatibility scoring, not a dedicated supervised LID model  
- Softmax confidence is heuristic  
- Batch size 1  
- Bare-metal GPU needs cuDNN 9; otherwise use Docker  
