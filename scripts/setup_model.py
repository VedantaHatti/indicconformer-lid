#!/usr/bin/env python3
"""Download ONLY the artifacts required for language identification.

Fetches the shared Conformer encoder, shared CTC head, preprocessor, and
language vocabulary masks. Does NOT download RNNT / joint / vocab ASR pieces.

Auth: uses HF_TOKEN or HUGGING_FACE_HUB_TOKEN if set; otherwise prompts
interactively (token is never printed).
"""

from __future__ import annotations

import argparse
import getpass
import os
import shutil
import sys
from pathlib import Path

HF_REPO_ID = "ai4bharat/indic-conformer-600m-multilingual"

REQUIRED_NAMES = (
    "encoder.onnx",
    "ctc_decoder.onnx",
    "preprocessor.ts",
    "language_masks.json",
)

ASR_IGNORE_PREFIXES = ("rnnt", "joint")
ASR_IGNORE_NAMES = frozenset({"vocab.json"})

ALLOW_PATTERNS = ["assets/*"]
IGNORE_PATTERNS = [
    "assets/rnnt*",
    "assets/joint*",
    "assets/vocab.json",
]


def resolve_token(cli_token: str | None) -> str | None:
    if cli_token:
        return cli_token.strip() or None
    for key in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        value = os.environ.get(key)
        if value and value.strip():
            print(f"Using token from environment variable {key}.")
            return value.strip()
    print("Hugging Face authentication may be required to download model artifacts.")
    print("Create a token at https://huggingface.co/settings/tokens if needed.")
    entered = getpass.getpass("HF_TOKEN (leave blank if public access works): ")
    return entered.strip() or None


def is_lid_artifact(name: str) -> bool:
    if name in ASR_IGNORE_NAMES:
        return False
    if name.startswith(ASR_IGNORE_PREFIXES):
        return False
    return True


def find_assets_dir(root: Path) -> Path:
    if (root / "encoder.onnx").exists():
        return root
    if (root / "assets" / "encoder.onnx").exists():
        return root / "assets"
    raise FileNotFoundError(
        f"Could not find encoder.onnx under {root} or {root / 'assets'}"
    )


def materialize_flat(assets_dir: Path, dest: Path, *, force: bool) -> list[str]:
    dest.mkdir(parents=True, exist_ok=True)
    copied: list[str] = []
    for path in sorted(assets_dir.iterdir()):
        if not path.is_file() and not path.is_symlink():
            continue
        if not is_lid_artifact(path.name):
            continue
        target = dest / path.name
        if target.exists() or target.is_symlink():
            if not force:
                copied.append(path.name)
                continue
            target.unlink()
        shutil.copy2(path.resolve(), target)
        copied.append(path.name)

    missing = [name for name in REQUIRED_NAMES if not (dest / name).exists()]
    if missing:
        raise RuntimeError(f"LID setup incomplete; missing in {dest}: {missing}")
    return copied


def inspect_destination(dest: Path) -> None:
    names = sorted(p.name for p in dest.iterdir() if p.is_file() or p.is_symlink())
    print(f"\nModel directory: {dest}")
    print(f"Artifact count: {len(names)}")
    for required in REQUIRED_NAMES:
        print(f"  OK  {required}")
    leaked = [n for n in names if not is_lid_artifact(n)]
    if leaked:
        raise RuntimeError(f"ASR artifacts unexpectedly present: {leaked}")
    print("Excluded ASR artifacts: rnnt*, joint*, vocab.json")


def copy_from_local(src: Path, dest: Path, *, force: bool) -> list[str]:
    assets = find_assets_dir(src)
    print(f"Copying LID artifacts from: {assets}")
    return materialize_flat(assets, dest, force=force)


def download_from_hub(
    repo_id: str,
    dest: Path,
    *,
    token: str | None,
    force: bool,
    local_only: bool,
) -> list[str]:
    from huggingface_hub import snapshot_download

    print(f"Downloading LID-only artifacts from {repo_id} ...")
    print("Keeping: preprocessor.ts, encoder.onnx (+ external shards),")
    print("         ctc_decoder.onnx, language_masks.json")
    print("Ignoring: rnnt*, joint*, vocab.json")

    snapshot_dir = Path(
        snapshot_download(
            repo_id=repo_id,
            token=token,
            allow_patterns=ALLOW_PATTERNS,
            ignore_patterns=IGNORE_PATTERNS,
            local_files_only=local_only,
        )
    )
    print(f"Snapshot cached at: {snapshot_dir}")
    assets = find_assets_dir(snapshot_dir)
    return materialize_flat(assets, dest, force=force)


def model_ready(dest: Path) -> bool:
    return all((dest / name).exists() for name in REQUIRED_NAMES)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dest", type=Path, default=Path("model"))
    parser.add_argument("--repo-id", default=HF_REPO_ID)
    parser.add_argument("--token", default=None, help="Prefer HF_TOKEN env or prompt")
    parser.add_argument("--force", action="store_true", help="Re-download even if model exists")
    parser.add_argument(
        "--local-only",
        action="store_true",
        help="Use existing HF cache only (no network)",
    )
    parser.add_argument(
        "--from-local-assets",
        type=Path,
        default=None,
        help="Copy from a local assets directory instead of downloading",
    )
    args = parser.parse_args(argv)
    dest = args.dest.expanduser().resolve()

    try:
        if model_ready(dest) and not args.force and args.from_local_assets is None:
            print(f"Model already present in {dest} — skipping download.")
            for name in REQUIRED_NAMES:
                print(f"  OK  {name}")
            print("\nNext: make run")
            return 0

        if args.from_local_assets is not None:
            copied = copy_from_local(
                args.from_local_assets.expanduser().resolve(), dest, force=args.force
            )
        else:
            try:
                import huggingface_hub  # noqa: F401
            except ImportError:
                print(
                    "ERROR: install dependencies first:\n  make install",
                    file=sys.stderr,
                )
                return 1
            token = None if args.local_only else resolve_token(args.token)
            copied = download_from_hub(
                args.repo_id,
                dest,
                token=token,
                force=args.force,
                local_only=args.local_only,
            )

        print(f"Materialized {len(copied)} LID artifacts into {dest}")
        inspect_destination(dest)
        print("\nSetup complete. Inference runs offline from local files.")
        print("Next: make run")
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
