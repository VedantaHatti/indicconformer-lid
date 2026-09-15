# Language Identification Service

Spoken **language identification** using a shared Conformer encoder, shared CTC
output, language vocabulary masks, and `normalized_ctc_score`.

This is not a separately trained LID classifier and does not transcribe.

The service is configured to run on **GPU by default**. To use CPU, change one
line in `server.py` (see below).

---

## Step-by-step setup

### 0. System dependencies (not installed by pip)

**Required for GPU (default):**

| Dependency | Why |
|------------|-----|
| NVIDIA driver | GPU access (`nvidia-smi` should work) |
| CUDA 12 runtime | Used by `onnxruntime-gpu` |
| **cuDNN 9** (`libcudnn.so.9`) | Required by ONNX Runtime CUDA; missing this is a common failure |
| `libsndfile1` | Used by the `soundfile` package for audio decode |

On Ubuntu/Debian:

```bash
sudo apt-get update
sudo apt-get install -y libsndfile1
```

Install NVIDIA driver / CUDA 12 / cuDNN 9 from NVIDIA’s docs if they are not
already on the machine. Verify:

```bash
nvidia-smi
# libcudnn.so.9 should be findable on the library path
```

**Python:** 3.10, 3.11, or 3.12.

---

### 1. Create a virtual environment and install Python packages

```bash
cd indicconformer-lid
python3 -m venv .venv
source .venv/bin/activate

pip install -U pip
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
```

The extra index is required so `torch==2.4.1` resolves to the **CPU** wheel.
Encoder/CTC still use **GPU** via `onnxruntime-gpu`.

All Python packages needed to run and to download the model are listed in
`requirements.txt` (pinned versions).

---

### 2. Download the model from Hugging Face

This downloads **only** the LID artifacts (encoder, CTC head, preprocessor,
language masks) from:

`ai4bharat/indic-conformer-600m-multilingual`

```bash
# Optional: export HF_TOKEN=hf_...   # if the Hub requires auth
python scripts/setup_model.py --dest model
```

You will be prompted for a Hugging Face token if `HF_TOKEN` /
`HUGGING_FACE_HUB_TOKEN` is not already set. The token is never printed.

After this finishes you should have files under `model/`, including:

- `preprocessor.ts`
- `encoder.onnx` (+ external weight shards next to it)
- `ctc_decoder.onnx`
- `language_masks.json`

RNNT / joint / `vocab.json` ASR pieces are **not** downloaded.

Re-download later with:

```bash
python scripts/setup_model.py --dest model --force
```

---

### 3. Run the server

```bash
python server.py
```

Then open:

```text
http://127.0.0.1:8007/
```

Hold the mic button, speak, release — the page calls `/identify`.

Health check:

```bash
curl -s http://127.0.0.1:8007/health
```

---

## Switch GPU / CPU

In [`server.py`](server.py), near the top:

```python
DEVICE = "cuda"   # default: GPU
# DEVICE = "cpu"  # use this if you have no working GPU / cuDNN
```

Change only that line, then run `python server.py` again.

If `DEVICE = "cuda"` but cuDNN/CUDA is missing, startup fails with a clear
error (it will not silently pretend to use the GPU).

---

## What runs where

```text
Audio
  → TorchScript preprocessor   (CPU — always)
  → ONNX Conformer encoder     (GPU when DEVICE=cuda)
  → ONNX shared CTC head       (GPU when DEVICE=cuda)
  → language masks + scoring
  → predicted language
```

---

## Pinned Python versions (`requirements.txt`)

| Package | Version |
|---------|---------|
| fastapi | 0.115.6 |
| uvicorn | 0.34.0 |
| numpy | 1.26.4 |
| torch | 2.4.1 (CPU wheel) |
| onnxruntime-gpu | 1.19.2 |
| soundfile | 0.12.1 |
| huggingface_hub | 0.27.1 |
| pydantic | 2.10.4 |
| python-dotenv | 1.0.1 |
| httpx / pytest | 0.28.1 / 8.3.4 |

---

## API

- `GET /` — browser UI  
- `GET /health` — device, providers, languages  
- `POST /identify` — multipart field `audio`  

```bash
curl -s -X POST http://127.0.0.1:8007/identify -F "audio=@utt.wav"
```

## Python usage

```python
from lid import LanguageIdentifier

lid = LanguageIdentifier(model_dir="./model", device="cuda")  # or "cpu"
result = lid.identify(open("utt.wav", "rb").read())
print(result["language"], result["providers"])
```

## Supported languages (22)

`as bn brx doi gu hi kn kok ks mai ml mni mr ne or pa sa sat sd ta te ur`

Default candidates in `server.py`: `hi, kn, mr, ta, te`.

## Limitations

- Derived CTC / mask scoring, not a dedicated supervised LID model  
- Softmax confidence is heuristic  
- Batch size 1  
- GPU needs CUDA 12 + cuDNN 9 on the host  
