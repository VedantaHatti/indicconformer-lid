"""Single-pass encoder/CTC guarantee."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from lid.model import LanguageIdentifier
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


def _stub_model(monkeypatch, tmp_path: Path, languages: list[str]) -> LanguageIdentifier:
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    for name in ("encoder.onnx", "ctc_decoder.onnx", "preprocessor.ts"):
        (model_dir / name).write_bytes(b"x")
    _write_masks(model_dir / "language_masks.json", languages)

    class FakePreprocessor:
        def eval(self):
            return self

        def __call__(self, input_signal, length):
            t = max(1, int(input_signal.shape[-1]) // 160)
            return torch.zeros(input_signal.shape[0], 80, t), length

    class FakeEncoder:
        def __init__(self):
            self.calls = 0
            self._providers = ["CPUExecutionProvider"]

        def get_providers(self):
            return list(self._providers)

        def run(self, output_names, feeds):
            self.calls += 1
            audio = feeds["audio_signal"]
            b, _, t = audio.shape
            return np.zeros((b, 1024, t), dtype=np.float32), np.array([t], dtype=np.int64)

    class FakeCtc:
        def __init__(self):
            self.calls = 0
            self._providers = ["CPUExecutionProvider"]

        def get_providers(self):
            return list(self._providers)

        def run(self, output_names, feeds):
            self.calls += 1
            enc = feeds["encoder_output"]
            b, _, t = enc.shape
            return [np.random.randn(b, t, SHARED_VOCAB_SIZE).astype(np.float32)]

    enc, ctc = FakeEncoder(), FakeCtc()

    def fake_session(path, *a, **k):
        return enc if str(path).endswith("encoder.onnx") else ctc

    import onnxruntime as ort

    monkeypatch.setattr(ort, "get_available_providers", lambda: ["CPUExecutionProvider"])
    monkeypatch.setattr(ort, "InferenceSession", fake_session)
    monkeypatch.setattr(torch.jit, "load", lambda *a, **k: FakePreprocessor())

    model = LanguageIdentifier(
        model_dir,
        device="cpu",
        candidate_languages=languages,
        margin_threshold=0.0,
        min_probe_duration_ms=0.0,
        min_rms_energy=None,
    )
    model._test_encoder = enc  # type: ignore[attr-defined]
    model._test_ctc = ctc  # type: ignore[attr-defined]
    return model


def test_single_pass_many_languages(monkeypatch, tmp_path: Path):
    languages = ["hi", "kn", "mr", "ta", "te", "bn", "gu"]
    model = _stub_model(monkeypatch, tmp_path, languages)
    wav = torch.randn(1, 16_000) * 0.1
    model.reset_counters()
    result = model.identify(wav, candidate_languages=languages)
    assert result["encoder_calls"] == 1
    assert result["ctc_calls"] == 1
    assert model._test_encoder.calls == 1  # type: ignore[attr-defined]
    assert model._test_ctc.calls == 1  # type: ignore[attr-defined]
    assert len(result["languages"]) == len(languages)


def test_invalid_language_on_identify(monkeypatch, tmp_path: Path):
    model = _stub_model(monkeypatch, tmp_path, ["hi", "kn"])
    with pytest.raises(ValueError, match="Unknown candidate"):
        model.identify(torch.randn(1, 16_000) * 0.1, candidate_languages=["hi", "zz"])
