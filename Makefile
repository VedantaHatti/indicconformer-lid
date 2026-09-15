# Language ID — minimal commands
#
#   make setup   Install pins → ask cpu/gpu → download model (skip if present) → .env
#   make run     Start API + UI
#   make help

ROOT := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))
PYTHON ?= python3
PORT ?= 8007
HOST ?= 0.0.0.0
TORCH_CPU_INDEX ?= https://download.pytorch.org/whl/cpu

ifneq (,$(wildcard $(ROOT).env))
  include $(ROOT).env
  export
endif

.PHONY: help setup install model run run-cpu run-gpu check ui-hint

help:
	@echo "Language ID"
	@echo ""
	@echo "  make setup     Full setup (deps + model + cpu/gpu → .env)"
	@echo "  make install   Install pinned CPU requirements only"
	@echo "  make model     Force re-download model artifacts"
	@echo "  make run       Start server (reads .env; default device=cpu)"
	@echo "  make run-cpu   Force CPU"
	@echo "  make run-gpu   Force GPU (needs requirements-gpu.txt)"
	@echo "  make check     Show config / model status"
	@echo ""
	@echo "UI: http://127.0.0.1:$${LID_PORT:-$(PORT)}/"

setup:
	$(PYTHON) scripts/setup.py

install:
	$(PYTHON) -m pip install --upgrade pip setuptools wheel
	-$(PYTHON) -m pip uninstall -y onnxruntime-gpu onnxruntime
	$(PYTHON) -m pip install -r requirements.txt --extra-index-url $(TORCH_CPU_INDEX)

model:
	$(PYTHON) scripts/setup_model.py --dest model --force

check:
	@echo "MODEL_DIR=$${MODEL_DIR:-model}"
	@echo "LID_DEVICE=$${LID_DEVICE:-cpu}"
	@echo "LID_PORT=$${LID_PORT:-$(PORT)}"
	@echo "LID_HOST=$${LID_HOST:-$(HOST)}"
	@if [ -f .env ]; then echo "(.env present)"; else echo "(.env missing — defaults apply, device=cpu)"; fi
	@if [ -f model/encoder.onnx ] && [ -f model/ctc_decoder.onnx ] && [ -f model/preprocessor.ts ] && [ -f model/language_masks.json ]; then \
	  echo "(model artifacts: present)"; \
	else \
	  echo "(model artifacts: missing — run make setup)"; \
	fi

run: ui-hint
	@DEVICE=$${LID_DEVICE:-cpu}; \
	P=$${LID_PORT:-$(PORT)}; \
	H=$${LID_HOST:-$(HOST)}; \
	echo "Starting with LID_DEVICE=$$DEVICE on $$H:$$P"; \
	MODEL_DIR=$${MODEL_DIR:-model} LID_DEVICE=$$DEVICE LID_PORT=$$P LID_HOST=$$H \
	  $(PYTHON) -m uvicorn server:app --host $$H --port $$P

run-cpu: ui-hint
	@P=$${LID_PORT:-$(PORT)}; H=$${LID_HOST:-$(HOST)}; \
	echo "Starting with LID_DEVICE=cpu on $$H:$$P"; \
	MODEL_DIR=$${MODEL_DIR:-model} LID_DEVICE=cpu \
	  $(PYTHON) -m uvicorn server:app --host $$H --port $$P

run-gpu: ui-hint
	@P=$${LID_PORT:-$(PORT)}; H=$${LID_HOST:-$(HOST)}; \
	echo "Starting with LID_DEVICE=cuda on $$H:$$P"; \
	MODEL_DIR=$${MODEL_DIR:-model} LID_DEVICE=cuda \
	  $(PYTHON) -m uvicorn server:app --host $$H --port $$P

ui-hint:
	@P=$${LID_PORT:-$(PORT)}; \
	echo "Browser UI: http://127.0.0.1:$$P/"
