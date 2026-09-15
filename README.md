# Language Identification Service

Spoken **language identification** for Indian languages using:

1. A shared Conformer audio encoder  
2. A shared CTC output projection  
3. Per-language vocabulary masks  
4. Validated `normalized_ctc_score` ranking  

This is **not** a separately trained supervised LID classifier. Language
evidence is derived from how well each language’s vocabulary mask fits the
shared CTC distribution for a single utterance.

The service does **not** produce transcripts and does **not** run RNNT / ASR
decoding.

## Architecture

```text
Audio (mono, 16 kHz)
  → preprocessor
  → shared Conformer encoder   (ONCE per request)
  → shared CTC head            (ONCE per request)
  → logits over shared vocab
  → apply each language mask
  → normalized_ctc_score
  → ranked languages + margin
```

Scoring many candidate languages does **not** re-run the encoder or CTC head.
Only mask application and scoring iterate over languages.

## Supported languages

Masks shipped with the model (22):

`as`, `bn`, `brx`, `doi`, `gu`, `hi`, `kn`, `kok`, `ks`, `mai`, `ml`, `mni`,
`mr`, `ne`, `or`, `pa`, `sa`, `sat`, `sd`, `ta`, `te`, `ur`

Default API candidates: `hi,kn,mr,ta,te` (override via env or request).

## Model setup (download once)

`scripts/setup_model.py` downloads **only** LID artifacts from the Hugging Face
checkpoint `ai4bharat/indic-conformer-600m-multilingual`:

| Kept | Purpose |
|------|---------|
| `assets/preprocessor.ts` | Feature frontend |
| `assets/encoder.onnx` + external weight shards | Shared encoder |
| `assets/ctc_decoder.onnx` | Shared CTC head |
| `assets/language_masks.json` | Language vocabulary masks |

| Excluded | Reason |
|----------|--------|
| `rnnt*` | RNNT ASR |
| `joint*` | RNNT joint / language heads |
| `vocab.json` | ASR detokenization |

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Prompts for HF_TOKEN if not already in the environment
python scripts/setup_model.py --dest model

# Or copy from an existing local assets directory (offline):
python scripts/setup_model.py --from-local-assets /path/to/assets --dest model
```

Token handling: `HF_TOKEN` / `HUGGING_FACE_HUB_TOKEN`, or interactive prompt.
The token is never printed.

After setup, inference is fully **offline**. `server.py` never downloads.

## Python usage

```python
from lid import LanguageIdentifier

lid = LanguageIdentifier(model_dir="./model", device="cuda")
result = lid.identify(open("utt.wav", "rb").read())

print(result["language"], result["margin"], result["languages"][:3])
print(result["providers"], result["latency_ms"])
```

Optional candidates:

```python
result = lid.identify(audio_bytes, candidate_languages=["hi", "kn", "te"])
```

## API

```bash
export MODEL_DIR=model
export LID_DEVICE=cpu   # or cuda when CUDAExecutionProvider is available
uvicorn server:app --host 0.0.0.0 --port 8007
```

### Browser test UI

Open the hold-to-talk page (same origin as the API):

```text
http://127.0.0.1:8007/
```

Hold the button, speak, release — the page posts WAV audio to `/identify` and shows the language code, margin, and top ranks. Use `localhost` or HTTPS so the browser allows the microphone.

### Health

```bash
curl -s http://127.0.0.1:8007/health | jq
```

Reports `model_loaded`, `device`, `providers`, and languages.

### Identify

```bash
curl -s -X POST http://127.0.0.1:8007/identify \
  -F "audio=@utt.wav" \
  -F "candidate_languages=hi,kn,mr,ta,te" | jq
```

Response includes `language`, `decision` (`accepted` / `uncertain`), scores,
ranked `languages`, `margin`, `latency_ms`, and actual ORT `providers`.
There is **no** transcription endpoint.

### Configuration

| Variable | Default | Meaning |
|----------|---------|---------|
| `MODEL_DIR` | `model` | Local artifact directory |
| `LID_DEVICE` | `cuda` | `cuda` or `cpu` |
| `LID_MARGIN_THRESHOLD` | `0.050965` | Below → `decision=uncertain` |
| `LID_CANDIDATE_LANGUAGES` | `hi,kn,mr,ta,te` | Default candidates |

`confidence` is a softmax over compatibility scores (heuristic), not a
calibrated probability.

## Docker (GPU)

Build (no weight download):

```bash
docker build -t indicconformer-lid .
```

Run with a host-mounted model directory prepared by `setup_model.py`:

```bash
docker run --gpus all -p 8007:8007 \
  -e MODEL_DIR=/app/model \
  -e LID_DEVICE=cuda \
  -e LID_MARGIN_THRESHOLD=0.050965 \
  -e LID_CANDIDATE_LANGUAGES=hi,kn,mr,ta,te \
  -v /absolute/path/to/model:/app/model \
  indicconformer-lid
```

## GPU / CPU

- `LID_DEVICE=cuda` requires ONNX Runtime `CUDAExecutionProvider` and fails
  clearly if it is missing (no silent CPU fallback).
- `LID_DEVICE=cpu` uses `CPUExecutionProvider`.
- `/health` reports the providers actually used by the loaded sessions.

## Offline inference

After `setup_model.py` finishes, no Hugging Face or network access is required
for identify/health. Point `MODEL_DIR` at the prepared directory.

## Tests

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=. pytest -q
```

## Limitations

- Derived from multilingual CTC vocabulary compatibility, not a dedicated LID
  training objective; confusable language pairs can remain uncertain.
- Confidence is heuristic.
- Batch size 1.
- Encoder external weights must sit beside `encoder.onnx` (handled by setup).
- CUDA Docker needs NVIDIA Container Toolkit and a compatible driver.
