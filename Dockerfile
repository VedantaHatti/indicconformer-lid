# Language ID service. Weights are NOT downloaded at build or startup.
# Prepare model/ on the host with: make setup
# Mount: -v /path/to/model:/app/model

FROM python:3.12-slim-bookworm

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MODEL_DIR=/app/model \
    LID_DEVICE=cpu \
    LID_MARGIN_THRESHOLD=0.050965 \
    LID_CANDIDATE_LANGUAGES=hi,kn,mr,ta,te \
    LID_HOST=0.0.0.0 \
    LID_PORT=8007 \
    PIP_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cpu

RUN apt-get update && apt-get install -y --no-install-recommends \
        libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Optional GPU image build: docker build --build-arg INSTALL_GPU=1 …
ARG INSTALL_GPU=0
COPY requirements-gpu.txt .
RUN if [ "$INSTALL_GPU" = "1" ]; then \
      pip uninstall -y onnxruntime && \
      pip install --no-cache-dir -r requirements-gpu.txt && \
      echo "GPU ORT installed"; \
    fi

COPY lid ./lid
COPY server.py .
COPY scripts ./scripts
COPY static ./static

RUN mkdir -p /app/model

EXPOSE 8007

CMD ["python", "-m", "uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8007"]
