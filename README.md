# Language Identification Service

Spoken **language identification** for Indian languages using a shared Conformer
encoder, shared CTC output, language vocabulary masks, and
`normalized_ctc_score` ranking.

This is **not** a separately trained LID classifier and does **not** transcribe.

## Quick start (3 commands)

```bash
cd indicconformer-lid
python3 -m venv .venv && source .venv/bin/activate
make setup    # install pins → ask cpu/gpu → download model (skip if present)
make run      # API + browser UI
```

Open **http://127.0.0.1:8007/**

Hold the mic button, speak, release.

## Architecture

```text
Audio (mono, 16 kHz)
  → TorchScript preprocessor   (CPU always)
  → shared Conformer encoder   (ONNX — once)
  → shared CTC head            (ONNX — once)
  → language masks + normalized_ctc_score
  → ranked languages
```

PyTorch is only used for the small preprocessor. **Torch CUDA is never required.**
GPU acceleration (optional) is ONNX Runtime CUDA for encoder/CTC only.

## Tested versions (pinned)

| Component | Version |
|-----------|---------|
| Python | 3.10 – 3.12 |
| fastapi | 0.115.6 |
| uvicorn | 0.34.0 |
| numpy | 1.26.4 |
| torch | 2.4.1 (**CPU** wheel) |
| onnxruntime | 1.19.2 (CPU default) |
| onnxruntime-gpu | 1.19.2 (optional overlay) |
| soundfile | 0.12.1 |
| huggingface_hub | 0.27.1 |
| python-dotenv | 1.0.1 |
| httpx / pytest | 0.28.1 / 8.3.4 |

Exact pins live in [`requirements.txt`](requirements.txt) and
[`requirements-gpu.txt`](requirements-gpu.txt).

## Setup details

`make setup` runs [`scripts/setup.py`](scripts/setup.py):

1. Checks Python 3.10–3.12  
2. Installs **CPU** pins (`torch` from the PyTorch CPU index + `onnxruntime`)  
3. Asks **cpu** (default) or **gpu**  
4. If gpu: uninstalls CPU `onnxruntime`, installs `onnxruntime-gpu`, checks `CUDAExecutionProvider`  
5. Downloads LID-only HF artifacts (asks for `HF_TOKEN` if needed) — **skips** if `model/` is complete  
6. Writes `.env`

Force re-download: `make model`

### Supported languages (22)

`as bn brx doi gu hi kn kok ks mai ml mni mr ne or pa sa sat sd ta te ur`

Default candidates: `hi,kn,mr,ta,te`

### Model artifacts (LID only)

Kept: `preprocessor.ts`, `encoder.onnx` + shards, `ctc_decoder.onnx`, `language_masks.json`  

Excluded: `rnnt*`, `joint*`, `vocab.json`

After setup, inference is **offline**. The server never downloads.

## Commands

| Command | Meaning |
|---------|---------|
| `make setup` | Full interactive setup |
| `make install` | CPU requirements only |
| `make run` | Start server (default **cpu**) |
| `make run-cpu` / `make run-gpu` | Force device |
| `make check` | Print config / model status |
| `make model` | Force model re-download |

## Python API

```python
from lid import LanguageIdentifier

lid = LanguageIdentifier(model_dir="./model", device="cpu")
result = lid.identify(open("utt.wav", "rb").read())
print(result["language"], result["margin"], result["providers"])
```

## API

- `GET /` — browser UI  
- `GET /health` — model, device, providers, languages  
- `POST /identify` — multipart `audio` (+ optional `candidate_languages`)  

```bash
curl -s http://127.0.0.1:8007/health | jq
curl -s -X POST http://127.0.0.1:8007/identify -F "audio=@utt.wav" | jq
```

## Docker

CPU image (default):

```bash
docker build -t indicconformer-lid .
docker run -p 8007:8007 \
  -e MODEL_DIR=/app/model -e LID_DEVICE=cpu \
  -v /absolute/path/to/model:/app/model \
  indicconformer-lid
```

GPU image (NVIDIA Container Toolkit):

```bash
docker build --build-arg INSTALL_GPU=1 -t indicconformer-lid:gpu .
docker run --gpus all -p 8007:8007 \
  -e MODEL_DIR=/app/model -e LID_DEVICE=cuda \
  -v /absolute/path/to/model:/app/model \
  indicconformer-lid:gpu
```

## GPU notes

- Needs a working NVIDIA driver and `onnxruntime-gpu==1.19.2`
- Do **not** install `onnxruntime` and `onnxruntime-gpu` together
- This app does **not** need a CUDA-enabled PyTorch build
- If CUDA EP is missing, use `LID_DEVICE=cpu` / `make run-cpu`

## Limitations

- Compatibility probe, not a dedicated supervised LID model  
- Softmax `confidence` is heuristic  
- Batch size 1  
- Encoder external weights must sit beside `encoder.onnx` (handled by setup)  
