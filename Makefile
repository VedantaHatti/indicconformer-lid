# Language ID
#
#   make install       Install all Python deps + choose cpu/gpu → .env
#   make setup         Download model (HF token); skip if present
#   make run           Ask port → start API + UI
#   make docker-build  Build CUDA+cuDNN image
#   make docker-run    Ask port → run container with ./model mounted

ROOT := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))
PYTHON ?= python3
PORT ?= 8007
HOST ?= 0.0.0.0
TORCH_CPU_INDEX ?= https://download.pytorch.org/whl/cpu
IMAGE ?= indicconformer-lid

ifneq (,$(wildcard $(ROOT).env))
  include $(ROOT).env
  export
endif

.PHONY: help install setup model run check docker-build docker-run

help:
	@echo "Language ID"
	@echo ""
	@echo "  make install        Install everything from requirements.txt; ask cpu/gpu"
	@echo "  make setup          Download LID model (asks HF token; skips if present)"
	@echo "  make run            Ask port and start server + UI"
	@echo "  make model          Force re-download model"
	@echo "  make check          Show .env / model status"
	@echo "  make docker-build   Build GPU image (CUDA + cuDNN)"
	@echo "  make docker-run     Ask port; run with --gpus all and ./model mounted"
	@echo ""
	@echo "Flow:  make install && make setup && make run"
	@echo "GPU without host cuDNN:  make setup && make docker-build && make docker-run"

install:
	$(PYTHON) scripts/install_env.py

setup:
	$(PYTHON) scripts/setup_model.py --dest model

model:
	$(PYTHON) scripts/setup_model.py --dest model --force

run:
	$(PYTHON) scripts/run_server.py

check:
	@echo "MODEL_DIR=$${MODEL_DIR:-model}"
	@echo "LID_DEVICE=$${LID_DEVICE:-cpu}"
	@echo "LID_PORT=$${LID_PORT:-$(PORT)}"
	@echo "LID_HOST=$${LID_HOST:-$(HOST)}"
	@if [ -f .env ]; then echo "(.env present)"; else echo "(.env missing — run make install)"; fi
	@if [ -f model/encoder.onnx ] && [ -f model/ctc_decoder.onnx ] && [ -f model/preprocessor.ts ] && [ -f model/language_masks.json ]; then \
	  echo "(model artifacts: present)"; \
	else \
	  echo "(model artifacts: missing — run make setup)"; \
	fi

docker-build:
	docker build -t $(IMAGE) .

docker-run:
	@P=$${LID_PORT:-$(PORT)}; \
	read -r -p "Host port [$$P]: " ENTERED; \
	P=$${ENTERED:-$$P}; \
	echo "Browser UI: http://127.0.0.1:$$P/"; \
	echo "Starting Docker with LID_DEVICE=cuda, mounting ./model → /app/model"; \
	docker run --rm --gpus all \
	  -p $$P:8007 \
	  -e MODEL_DIR=/app/model \
	  -e LID_DEVICE=cuda \
	  -e LID_PORT=8007 \
	  -v "$(ROOT)model:/app/model" \
	  $(IMAGE)
