"""Download LID-only artifacts from Hugging Face into ``assets/``."""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any, Sequence

HF_REPO_ID = "vedantahatti/indic-lid"

REPO_ROOT = Path(__file__).resolve().parent.parent
# Flat ONNX + weight shards live directly under assets/ (gitignored except .gitkeep).
DEFAULT_ARTIFACTS_DIR = REPO_ROOT / "assets"

LID_HF_CORE_PATHS = (
    "assets/preprocessor.ts",
    "assets/encoder.onnx",
    "assets/ctc_decoder.onnx",
    "assets/language_masks.json",
)

REQUIRED_FILES = (
    "encoder.onnx",
    "ctc_decoder.onnx",
    "preprocessor.ts",
    "language_masks.json",
)

class DownloadError(ValueError):
    """Raised when LID artifacts cannot be downloaded or validated."""


def resolve_hf_token() -> str | None:
    for key in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        value = os.environ.get(key)
        if value and value.strip():
            return value.strip()
    return None


def find_assets_dir(root: Path) -> Path:
    """Resolve a directory that already contains ``encoder.onnx``."""

    if (root / "encoder.onnx").exists():
        return root
    nested = root / "assets"
    if (nested / "encoder.onnx").exists():
        return nested
    raise DownloadError(
        f"Could not find LID artifacts under {root} or {nested}. "
        f"Download them with: hf download {HF_REPO_ID} --local-dir ."
    )


def onnx_external_locations(onnx_path: Path) -> frozenset[str]:
    """Return external-data filenames referenced by an ONNX model."""

    try:
        import onnx
        from onnx import TensorProto
        from onnx.external_data_helper import _get_all_tensors
    except ImportError as exc:
        raise DownloadError(
            "The onnx package is required to resolve encoder weight shards. "
            "Install project dependencies with: pip install -r requirements.txt"
        ) from exc

    model = onnx.load(str(onnx_path), load_external_data=False)
    refs: set[str] = set()
    for tensor in _get_all_tensors(model):
        if tensor.data_location == TensorProto.EXTERNAL:
            for entry in tensor.external_data:
                if entry.key == "location":
                    refs.add(entry.value)
    return frozenset(refs)


def build_lid_hf_paths(encoder_external: frozenset[str]) -> tuple[str, ...]:
    """Hub paths for LID-only artifacts (core files + encoder external shards)."""

    paths = list(LID_HF_CORE_PATHS)
    paths.extend(f"assets/{name}" for name in sorted(encoder_external))
    return tuple(paths)


def _list_unexpected_files(assets_dir: Path, allowed_names: frozenset[str]) -> list[str]:
    unexpected: list[str] = []
    for path in assets_dir.iterdir():
        if path.is_file() and path.name not in allowed_names and path.name != ".gitkeep":
            unexpected.append(path.name)
    return sorted(unexpected)


def _artifacts_ready(assets_dir: Path, encoder_external: frozenset[str]) -> bool:
    allowed = frozenset(REQUIRED_FILES) | encoder_external
    missing = [name for name in REQUIRED_FILES if not (assets_dir / name).exists()]
    missing.extend(name for name in encoder_external if not (assets_dir / name).exists())
    if missing:
        return False
    return not _list_unexpected_files(assets_dir, allowed)


def _download_hf_files(
    repo_id: str,
    filenames: Sequence[str],
    local_dir: Path,
    *,
    token: str | None,
) -> None:
    from huggingface_hub import hf_hub_download

    local_dir.mkdir(parents=True, exist_ok=True)
    kwargs: dict[str, Any] = {
        "repo_id": repo_id,
        "local_dir": str(local_dir),
    }
    if token:
        kwargs["token"] = token
    for filename in filenames:
        hf_hub_download(filename=filename, **kwargs)


def _clear_assets_dir(assets_dir: Path) -> None:
    """Remove downloaded files but keep ``.gitkeep``."""

    assets_dir.mkdir(parents=True, exist_ok=True)
    for path in assets_dir.iterdir():
        if path.name == ".gitkeep":
            continue
        if path.is_file() or path.is_symlink():
            path.unlink()
        elif path.is_dir():
            shutil.rmtree(path)


def ensure_lid_artifacts(
    repo_id: str = HF_REPO_ID,
    *,
    token: str | None = None,
    artifacts_dir: str | Path | None = None,
) -> Path:
    """Download LID-only artifacts into an isolated local ``assets/`` directory.

    Two-phase download: fetch ``encoder.onnx``, parse ONNX external-data
    references, then fetch only those shards plus the other LID core files.
    """

    token = token if token is not None else resolve_hf_token()
    assets_dir = Path(artifacts_dir or DEFAULT_ARTIFACTS_DIR).expanduser().resolve()
    # Hub paths are ``assets/<file>``. Download with local_dir = parent so files
    # land flat under ``assets/`` without nesting ``assets/assets/``.
    download_root = assets_dir.parent if assets_dir.name == "assets" else assets_dir

    encoder_external: frozenset[str] | None = None
    if (assets_dir / "encoder.onnx").is_file():
        encoder_external = onnx_external_locations(assets_dir / "encoder.onnx")

    if encoder_external is not None and _artifacts_ready(assets_dir, encoder_external):
        return assets_dir

    _clear_assets_dir(assets_dir)

    try:
        _download_hf_files(
            repo_id,
            ["assets/encoder.onnx"],
            download_root,
            token=token,
        )
    except Exception as exc:  # noqa: BLE001
        raise DownloadError(
            f"Failed to download encoder.onnx from {repo_id}. "
            f"Underlying error: {exc}"
        ) from exc

    resolved = find_assets_dir(download_root)
    encoder_external = onnx_external_locations(resolved / "encoder.onnx")

    remaining = [
        path
        for path in build_lid_hf_paths(encoder_external)
        if path != "assets/encoder.onnx"
    ]
    try:
        _download_hf_files(
            repo_id,
            remaining,
            download_root,
            token=token,
        )
    except Exception as exc:  # noqa: BLE001
        raise DownloadError(
            f"Failed to download LID artifacts from {repo_id}. "
            f"Underlying error: {exc}"
        ) from exc

    assets_dir = find_assets_dir(download_root)
    if not _artifacts_ready(assets_dir, encoder_external):
        missing = [
            name
            for name in (*REQUIRED_FILES, *encoder_external)
            if not (assets_dir / name).exists()
        ]
        unexpected = _list_unexpected_files(
            assets_dir, frozenset(REQUIRED_FILES) | encoder_external
        )
        detail = []
        if missing:
            detail.append(f"missing={sorted(set(missing))}")
        if unexpected:
            detail.append(f"unexpected files={unexpected}")
        raise DownloadError(
            f"LID artifact setup failed under {assets_dir}"
            + (f" ({'; '.join(detail)})" if detail else "")
        )

    return assets_dir
