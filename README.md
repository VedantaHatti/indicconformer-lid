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

## Installation

```bash
cd indicconformer-lid
python3 -m venv .venv
source .venv/bin/activate

pip install -U pip
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
```

The extra PyTorch index installs the CPU wheel used only for the TorchScript
preprocessor. Encoder and CTC inference use ONNX Runtime.

## Hugging Face authentication

The model repository is gated:

`https://huggingface.co/ai4bharat/indic-conformer-600m-multilingual`

1. Create or sign in to a Hugging Face account.
2. Open the model page and accept the access conditions.
3. Create an access token at https://huggingface.co/settings/tokens
4. Export it before the first run:

```bash
export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxx
```

`HUGGING_FACE_HUB_TOKEN` is also accepted.

## Model download

On the first `python server.py` startup, the service downloads **only** the LID
artifacts from the Hugging Face repository:

- `preprocessor.ts`
- `encoder.onnx` and its external weight shards
- `ctc_decoder.onnx`
- `language_masks.json`

RNNT decoder, joint networks, language-specific RNNT heads, and `vocab.json`
are excluded by download patterns. After the first successful download, inference
works offline from the local Hugging Face cache.

Confirmed ONNX interfaces:

| Stage | Inputs | Outputs |
|-------|--------|---------|
| Preprocessor | `input_signal`, `length` | mel features, feature lengths |
| Encoder | `audio_signal`, `length` | `outputs`, `encoded_lengths` |
| CTC head | `encoder_output` | `logprobs` `[B, T, 5633]` |

## CPU / GPU selection

In `server.py`:

```python
DEVICE = "cuda"
# DEVICE = "cpu"
```

Change that one line and restart the server.

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
  "device": "cuda",
  "model_loaded": true,
  "available_languages": ["as", "bn", "..."],
  "providers": ["CUDAExecutionProvider", "CPUExecutionProvider"]
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
- First run requires Hugging Face access for the gated model.
- GPU mode requires CUDA 12 and cuDNN 9 on the host.
