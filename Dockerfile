# GPU language-ID service. Model weights are NOT downloaded at build or startup.
# Mount prepared artifacts at /app/model (run scripts/setup_model.py on the host).

FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MODEL_DIR=/app/model \
    LID_DEVICE=cuda \
    LID_MARGIN_THRESHOLD=0.050965 \
    LID_CANDIDATE_LANGUAGES=hi,kn,mr,ta,te

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 \
        python3-pip \
        libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

COPY lid ./lid
COPY server.py .
COPY scripts ./scripts
COPY static ./static

RUN mkdir -p /app/model

EXPOSE 8007

CMD ["python3", "-m", "uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8007"]
