#!/usr/bin/env python3
"""One-shot interactive setup for Language ID.

1. Check Python version (3.10–3.12)
2. Install pinned CPU requirements (Torch CPU wheels)
3. Ask cpu / gpu — if gpu, swap in onnxruntime-gpu and verify CUDA EP
4. Download model (HF token) or skip if already present
5. Write .env
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SETUP_MODEL = Path(__file__).resolve().parent / "setup_model.py"
REQUIREMENTS = ROOT / "requirements.txt"
REQUIREMENTS_GPU = ROOT / "requirements-gpu.txt"
ENV_PATH = ROOT / ".env"
MODEL_DIR = ROOT / "model"
TORCH_CPU_INDEX = "https://download.pytorch.org/whl/cpu"

REQUIRED_NAMES = (
    "encoder.onnx",
    "ctc_decoder.onnx",
    "preprocessor.ts",
    "language_masks.json",
)


def check_python() -> None:
    if sys.version_info < (3, 10) or sys.version_info >= (3, 13):
        raise SystemExit(
            f"Python 3.10–3.12 required; found {sys.version_info.major}.{sys.version_info.minor}.\n"
            "Create a venv with a supported interpreter, then re-run make setup."
        )


def model_ready(model_dir: Path = MODEL_DIR) -> bool:
    if not model_dir.is_dir():
        return False
    return all((model_dir / name).exists() for name in REQUIRED_NAMES)


def run(cmd: list[str]) -> int:
    print("+", " ".join(cmd))
    return subprocess.run(cmd, cwd=str(ROOT)).returncode


def install_cpu_requirements() -> int:
    print("Step 1/4 — Install pinned CPU dependencies")
    print(f"  Torch CPU index: {TORCH_CPU_INDEX}")
    print()
    code = run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--upgrade",
            "pip",
            "setuptools",
            "wheel",
        ]
    )
    if code != 0:
        return code
    # Uninstall conflicting ORT packages so a clean CPU install wins.
    run([sys.executable, "-m", "pip", "uninstall", "-y", "onnxruntime-gpu", "onnxruntime"])
    return run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-r",
            str(REQUIREMENTS),
            "--extra-index-url",
            TORCH_CPU_INDEX,
        ]
    )


def install_gpu_overlay() -> int:
    print()
    print("Installing GPU overlay (onnxruntime-gpu)…")
    code = run([sys.executable, "-m", "pip", "uninstall", "-y", "onnxruntime"])
    if code not in (0, 1):  # 1 = not installed
        # pip uninstall returns 0 even when missing on some versions; continue anyway
        pass
    return run([sys.executable, "-m", "pip", "install", "-r", str(REQUIREMENTS_GPU)])


def verify_cuda_ep() -> bool:
    try:
        import onnxruntime as ort

        providers = ort.get_available_providers()
        ok = "CUDAExecutionProvider" in providers
        print(f"  ONNX Runtime providers: {providers}")
        return ok
    except Exception as exc:  # noqa: BLE001
        print(f"  Could not import onnxruntime: {exc}")
        return False


def ask_device() -> str:
    print()
    print("Step 2/4 — Device")
    print("  1) cpu   (default — always works with requirements.txt)")
    print("  2) gpu   (needs NVIDIA driver + requirements-gpu.txt)")
    print("  Note: PyTorch stays on CPU either way; only ONNX encoder/CTC use GPU.")
    while True:
        raw = input("Choose [1/cpu, 2/gpu] (default: cpu): ").strip().lower()
        if raw in ("", "1", "c", "cpu"):
            return "cpu"
        if raw in ("2", "g", "gpu", "cuda"):
            return "cuda"
        print("Please enter 1/cpu or 2/gpu.")


def ask_port(default: str = "8007") -> str:
    print()
    print("Step 3/4 — Server port")
    raw = input(f"Server port [{default}]: ").strip()
    return raw or default


def download_model_if_needed() -> int:
    print()
    print("Step 4/4 — Model artifacts")
    if model_ready():
        print(f"  Model already present in {MODEL_DIR} — skipping download.")
        for name in REQUIRED_NAMES:
            print(f"    OK  {name}")
        return 0

    print(f"  Model missing/incomplete in {MODEL_DIR}")
    print("  Downloading LID-only artifacts from Hugging Face.")
    print("  You may be asked for HF_TOKEN (never printed).")
    print()
    return run([sys.executable, str(SETUP_MODEL), "--dest", "model"])


def write_env(device: str, port: str) -> None:
    lines = [
        "MODEL_DIR=model",
        f"LID_DEVICE={device}",
        "LID_MARGIN_THRESHOLD=0.050965",
        "LID_CANDIDATE_LANGUAGES=hi,kn,mr,ta,te",
        "LID_HOST=0.0.0.0",
        f"LID_PORT={port}",
        "",
    ]
    ENV_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {ENV_PATH} (LID_DEVICE={device}, LID_PORT={port})")


def main() -> int:
    os.chdir(ROOT)
    print("=== Language ID setup ===")
    print(f"Project: {ROOT}")
    print()

    try:
        check_python()
    except SystemExit as exc:
        print(exc, file=sys.stderr)
        return 1

    code = install_cpu_requirements()
    if code != 0:
        print("Dependency install failed.", file=sys.stderr)
        return code

    device = ask_device()
    if device == "cuda":
        code = install_gpu_overlay()
        if code != 0:
            print("GPU dependency install failed.", file=sys.stderr)
            return code
        if not verify_cuda_ep():
            print(
                "CUDAExecutionProvider not available after installing onnxruntime-gpu.\n"
                "Check NVIDIA drivers, or re-run setup and choose cpu.",
                file=sys.stderr,
            )
            return 1

    port = ask_port()

    code = download_model_if_needed()
    if code != 0:
        print("Model setup failed.", file=sys.stderr)
        return code

    write_env(device, port)
    print()
    print("Setup complete.")
    print("Start with:  make run")
    print(f"UI:          http://127.0.0.1:{port}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
