# Language Identification Service

A multilingual speech language-identification system using a shared Conformer
encoder, shared CTC output, language-specific vocabulary masks, and normalized
CTC score comparison.

This is inference-only language identification. It is not a separately trained
LID classifier and does not transcribe speech.

## Architecture

```text
Audio
  → preprocessing (TorchScript mel filterbank)
  → shared Conformer encoder        (once per request)
  → shared CTC output head          (once per request)
  → language vocabulary masks
  → normalized_ctc_score
  → language ranking
```

### How language scores are obtained

For each candidate language, a boolean mask selects that language's slice of the
shared 5633-token CTC vocabulary (256 language tokens plus one shared blank).
The service applies `normalized_ctc_score` to the shared CTC log-probabilities:
at each frame it takes the best masked token, collapses consecutive repeats,
ignores blank tokens, and averages the surviving log-probabilities. If no
non-blank tokens remain, it falls back to a frame average.

### Why the encoder and CTC run only once

All languages share one encoder and one CTC head. A single forward pass produces
logits shaped `[batch, time, 5633]`. Language identification then scores every
candidate in parallel by indexing those shared logits with precomputed mask
indices. The encoder and CTC are never re-run per language.

## Supported languages

Default candidates: `hi`, `kn`, `mr`, `ta`, `te`.

Language masks for all 22 model languages are loaded from the Hugging Face
repository and can be enabled by passing `candidate_languages` to `/identify`:

`as`, `bn`, `brx`, `doi`, `gu`, `hi`, `kn`, `kok`, `ks`, `mai`, `ml`, `mni`,
`mr`, `ne`, `or`, `pa`, `sa`, `sat`, `sd`, `ta`, `te`, `ur`

## System dependencies

These are not installed by pip:

| Dependency | When needed |
|------------|-------------|
| Python 3.10, 3.11, or 3.12 | Always |
| `libsndfile1` | Audio decode via `soundfile` |
| NVIDIA driver | GPU mode (`DEVICE = "cuda"`) |
| CUDA 12 + cuDNN 9 (`libcudnn.so.9`) | GPU mode with ONNX Runtime CUDA |

On Ubuntu/Debian:

```bash
sudo apt-get update
sudo apt-get install -y libsndfile1
```

Verify GPU setup:

```bash
nvidia-smi
```

## Quick start

```bash
git clone <your-repo-url> indicconformer-lid
cd indicconformer-lid

python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu

# Install the Hugging Face CLI (skip if `hf` is already available)
curl -LsSf https://hf.co/cli/install.sh | bash

# Download model artifacts into ./assets/ (see below)
hf download vedantahatti/indic-lid --local-dir .

python server.py
```

The extra PyTorch index installs the CPU wheel used only for the TorchScript
preprocessor. Encoder and CTC inference use ONNX Runtime.

## Model artifacts (required)

The ONNX model files (~2.4 GB) are **not** stored in this git repository. Download
them from Hugging Face before starting the server:

**https://huggingface.co/vedantahatti/indic-lid**

This bundle contains **LID-only** artifacts derived from
[ai4bharat/indic-conformer-600m-multilingual](https://huggingface.co/ai4bharat/indic-conformer-600m-multilingual).
It does not include ASR/RNNT files (`joint_*`, `rnnt_*`, `vocab.json`).

### Where files must live

All model files must sit in the **`assets/` folder at the project root** — the
same directory as `server.py`:

```text
indicconformer-lid/
├── server.py
├── lid/
├── assets/                          ← model files go here
│   ├── encoder.onnx
│   ├── ctc_decoder.onnx
│   ├── preprocessor.ts
│   ├── language_masks.json
│   └── <encoder weight shards>      ← ~370 additional files
└── ...
```

The server loads models from `assets/` automatically (`ASSETS_DIR` in
`server.py`). Do not rename this folder or nest files under `assets/assets/`.

### Step-by-step download

**1. Clone this repository and enter the project directory**

```bash
git clone <your-repo-url> indicconformer-lid
cd indicconformer-lid
```

**2. Install the Hugging Face CLI** (one-time, if you do not already have `hf`)

```bash
curl -LsSf https://hf.co/cli/install.sh | bash
```

Re-open your terminal, or run `export PATH="$HOME/.local/bin:$PATH"`.

**3. Download artifacts into the project root**

Run this **from inside `indicconformer-lid/`** (the folder that contains
`server.py`):

```bash
hf download vedantahatti/indic-lid --local-dir .
```

The trailing `.` is important: it tells the CLI to write files relative to your
current directory. The repo stores files under an `assets/` prefix, so this
command creates `./assets/` with all required files.

**4. Verify the download**

```bash
ls assets/encoder.onnx assets/ctc_decoder.onnx assets/preprocessor.ts assets/language_masks.json
du -sh assets
```

You should see all four core files and a total size of roughly **2.4 GB**.

**5. Start the server**

```bash
python server.py
```

No Hugging Face token is required for this public model repo. Optional: set
`HF_TOKEN` for faster download rate limits.

### Troubleshooting downloads

| Problem | What to do |
|---------|------------|
| `No space left on device` | Free at least **5 GB** on disk, then re-run the download command |
| Missing files after a failed download | Delete partial files (`rm -rf assets/*`) and download again |
| `hf: command not found` | Install the CLI (step 2) or use `pip install huggingface_hub` and retry |

Confirmed ONNX interfaces:

| Stage | Inputs | Outputs |
|-------|--------|---------|
| Preprocessor | `input_signal`, `length` | mel features, feature lengths |
| Encoder | `audio_signal`, `length` | `outputs`, `encoded_lengths` |
| CTC head | `encoder_output` | `logprobs` `[B, T, 5633]` |

## CPU / GPU selection

In `server.py`:

```python
DEVICE = "cpu"
# DEVICE = "cuda"
```

CPU mode works out of the box. Switch to `"cuda"` only if you have a working
NVIDIA driver, CUDA 12, and cuDNN 9. Restart the server after changing the line.

## Run the server

```bash
python server.py
```

Health check:

```bash
curl -s http://127.0.0.1:8007/health
```

Example response:

```json
{
  "status": "ok",
  "device": "cpu",
  "model_loaded": true,
  "available_languages": ["as", "bn", "..."],
  "providers": ["CPUExecutionProvider"]
}
```

## API

### `GET /health`

Returns service status, active device, whether the model is loaded, all
available languages, and the ONNX Runtime execution providers in use.

### `POST /identify`

Multipart form upload with field `audio`.

Optional form field `candidate_languages` — comma-separated list, for example
`hi,kn,mr,ta,te`.

```bash
curl -s -X POST http://127.0.0.1:8007/identify \
  -F "audio=@tests/fixtures/speech-16k.wav"
```

Example response:

```json
{
  "language": "kn",
  "scores": {
    "hi": -0.12,
    "kn": -0.05,
    "mr": -0.18,
    "ta": -0.21,
    "te": -0.09
  },
  "margin": 0.04,
  "top_candidates": [
    {"language": "kn", "score": -0.05},
    {"language": "te", "score": -0.09}
  ]
}
```

No transcript or ASR output is returned.

## Limitations

- Scores come from shared CTC compatibility, not a dedicated supervised LID model.
- Softmax-derived confidence is a heuristic, not a calibrated probability.
- Batch size 1 only.
- Model artifacts must be downloaded separately (~2.4 GB) from
  [vedantahatti/indic-lid](https://huggingface.co/vedantahatti/indic-lid) into
  `./assets/` before the first run.
- GPU mode requires CUDA 12 and cuDNN 9 on the host.
