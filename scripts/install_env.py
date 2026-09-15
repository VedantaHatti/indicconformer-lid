#!/usr/bin/env python3
"""Install Python deps and write .env device preference (cpu/gpu)."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REQUIREMENTS = ROOT / "requirements.txt"
ENV_PATH = ROOT / ".env"
TORCH_CPU_INDEX = "https://download.pytorch.org/whl/cpu"


def check_python() -> None:
    if sys.version_info < (3, 10) or sys.version_info >= (3, 13):
        raise SystemExit(
            f"Python 3.10–3.12 required; found {sys.version_info.major}.{sys.version_info.minor}."
        )


def run(cmd: list[str]) -> int:
    print("+", " ".join(cmd))
    return subprocess.run(cmd, cwd=str(ROOT)).returncode


def ask_device() -> str:
    print()
    print("Device preference for inference:")
    print("  1) cpu   (default — works without CUDA/cuDNN)")
    print("  2) gpu   (needs NVIDIA driver + CUDA 12 + cuDNN 9, or Docker GPU)")
    print("  Note: PyTorch stays on CPU; only ONNX encoder/CTC use the GPU.")
    while True:
        raw = input("Choose [1/cpu, 2/gpu] (default: cpu): ").strip().lower()
        if raw in ("", "1", "c", "cpu"):
            return "cpu"
        if raw in ("2", "g", "gpu", "cuda"):
            return "cuda"
        print("Please enter 1/cpu or 2/gpu.")


def read_env() -> dict[str, str]:
    values: dict[str, str] = {}
    if not ENV_PATH.is_file():
        return values
    for line in ENV_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        values[key.strip()] = val.strip()
    return values


def write_env(device: str, *, port: str = "8007") -> None:
    existing = read_env()
    port = existing.get("LID_PORT", port)
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


def hint_gpu() -> None:
    print()
    print("GPU selected. Bare-metal CUDA needs:")
    print("  - NVIDIA driver")
    print("  - CUDA 12 runtime")
    print("  - cuDNN 9 (libcudnn.so.9)  ← missing this caused prior startup failures")
    print("If the host lacks cuDNN 9, use Docker instead:")
    print("  make docker-build && make docker-run")


def main() -> int:
    os.chdir(ROOT)
    print("=== Language ID — install ===")
    print(f"Project: {ROOT}")
    print()

    try:
        check_python()
    except SystemExit as exc:
        print(exc, file=sys.stderr)
        return 1

    print("Installing pinned dependencies from requirements.txt …")
    print(f"Torch CPU index: {TORCH_CPU_INDEX}")
    print()

    code = run(
        [sys.executable, "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"]
    )
    if code != 0:
        return code

    # Avoid conflicting CPU-only onnxruntime wheel beside onnxruntime-gpu.
    run([sys.executable, "-m", "pip", "uninstall", "-y", "onnxruntime"])

    code = run(
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
    if code != 0:
        print("pip install failed.", file=sys.stderr)
        return code

    device = ask_device()
    write_env(device)
    if device == "cuda":
        hint_gpu()

    print()
    print("Install complete. Next:")
    print("  make setup   # download model (HF token if needed)")
    print("  make run     # ask port and start server")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
