# GPU Language ID image (CUDA 12 + cuDNN 9 runtime).
# Model weights are NOT downloaded at build or startup.
# Prepare ./model on the host with: make setup
# Run: make docker-run   (or see README)

FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MODEL_DIR=/app/model \
    LID_DEVICE=cuda \
    LID_MARGIN_THRESHOLD=0.050965 \
    LID_CANDIDATE_LANGUAGES=hi,kn,mr,ta,te \
    LID_HOST=0.0.0.0 \
    LID_PORT=8007 \
    PIP_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cpu

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 \
        python3-pip \
        python3-venv \
        libsndfile1 \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3 /usr/bin/python

WORKDIR /app

COPY requirements.txt .
RUN pip3 install --no-cache-dir --upgrade pip \
    && pip3 uninstall -y onnxruntime || true \
    && pip3 install --no-cache-dir -r requirements.txt

COPY lid ./lid
COPY server.py .
COPY scripts ./scripts
COPY static ./static

RUN mkdir -p /app/model

EXPOSE 8007

CMD ["python3", "-m", "uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8007"]
