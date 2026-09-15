"""Model artifact validation tests (offline stubs)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from lid.model import REQUIRED_FILES, LanguageIdentifier, LanguageIdentifierError
from lid.scoring import LOCAL_BLANK_ID, SHARED_VOCAB_SIZE


def _write_masks(path: Path, languages: list[str]) -> None:
    masks = {}
    for lang in languages:
        mask = [False] * SHARED_VOCAB_SIZE
        for i in range(LOCAL_BLANK_ID):
            mask[i] = True
        mask[SHARED_VOCAB_SIZE - 1] = True
        masks[lang] = mask
    path.write_text(json.dumps(masks), encoding="utf-8")


def test_missing_model_directory(tmp_path: Path):
    with pytest.raises(LanguageIdentifierError, match="does not exist"):
        LanguageIdentifier(tmp_path / "missing", device="cpu")


def test_missing_required_files(tmp_path: Path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "encoder.onnx").write_bytes(b"x")
    with pytest.raises(LanguageIdentifierError, match="Missing required"):
        LanguageIdentifier(model_dir, device="cpu")


def test_cuda_unavailable(monkeypatch, tmp_path: Path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    for name in REQUIRED_FILES:
        if name == "language_masks.json":
            _write_masks(model_dir / name, ["hi", "kn"])
        else:
            (model_dir / name).write_bytes(b"x")

    import onnxruntime as ort

    monkeypatch.setattr(ort, "get_available_providers", lambda: ["CPUExecutionProvider"])
    with pytest.raises(LanguageIdentifierError, match="CUDAExecutionProvider"):
        LanguageIdentifier(model_dir, device="cuda")


def test_invalid_candidate_language(monkeypatch, tmp_path: Path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    for name in REQUIRED_FILES:
        if name == "language_masks.json":
            _write_masks(model_dir / name, ["hi", "kn"])
        else:
            (model_dir / name).write_bytes(b"x")

    class FakeSession:
        def __init__(self, *a, **k):
            self._providers = k.get("providers") or ["CPUExecutionProvider"]

        def get_providers(self):
            return list(self._providers)

    class FakeModule:
        def eval(self):
            return self

    import onnxruntime as ort
    import torch

    monkeypatch.setattr(ort, "get_available_providers", lambda: ["CPUExecutionProvider"])
    monkeypatch.setattr(ort, "InferenceSession", FakeSession)
    monkeypatch.setattr(torch.jit, "load", lambda *a, **k: FakeModule())

    with pytest.raises(ValueError, match="Unknown candidate"):
        LanguageIdentifier(model_dir, device="cpu", candidate_languages=["hi", "xx"])
