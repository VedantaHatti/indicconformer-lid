#!/usr/bin/env python3
"""Ask for a port and start the Language ID server."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    os.chdir(ROOT)
    load_dotenv(ROOT / ".env")

    default_port = os.getenv("LID_PORT", "8007").strip() or "8007"
    default_host = os.getenv("LID_HOST", "0.0.0.0").strip() or "0.0.0.0"
    device = os.getenv("LID_DEVICE", "cpu").strip().lower() or "cpu"
    model_dir = os.getenv("MODEL_DIR", "model").strip() or "model"

    print("=== Language ID — run ===")
    print(f"Device (from .env): {device}")
    print(f"Model dir:          {model_dir}")
    raw = input(f"Port [{default_port}]: ").strip()
    port = raw or default_port
    try:
        port_i = int(port)
        if not (1 <= port_i <= 65535):
            raise ValueError
    except ValueError:
        print(f"Invalid port: {port}", file=sys.stderr)
        return 1

    # Persist chosen port for next time.
    env_path = ROOT / ".env"
    if env_path.is_file():
        lines = env_path.read_text(encoding="utf-8").splitlines()
        out: list[str] = []
        seen = False
        for line in lines:
            if line.startswith("LID_PORT="):
                out.append(f"LID_PORT={port}")
                seen = True
            else:
                out.append(line)
        if not seen:
            out.append(f"LID_PORT={port}")
        env_path.write_text("\n".join(out) + "\n", encoding="utf-8")

    print(f"Browser UI: http://127.0.0.1:{port}/")
    print(f"Starting with LID_DEVICE={device} on {default_host}:{port}")
    env = os.environ.copy()
    env["MODEL_DIR"] = model_dir
    env["LID_DEVICE"] = device
    env["LID_PORT"] = port
    env["LID_HOST"] = default_host
    return subprocess.call(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "server:app",
            "--host",
            default_host,
            "--port",
            port,
        ],
        cwd=str(ROOT),
        env=env,
    )


if __name__ == "__main__":
    raise SystemExit(main())
