"""Unit and API tests for the LID pilot."""

from __future__ import annotations

import io
import json
import wave
from pathlib import Path

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient

from lid.model import (
    ALLOW_PATTERNS,
    IGNORE_PATTERNS,
    LOCAL_BLANK_ID,
    SHARED_VOCAB_SIZE,
    LanguageIdentifier,
    build_scoring_result,
    language_index_matrix,
    score_all_languages,
)
from server import create_app


def _wav_bytes(seconds: float = 1.0, sr: int = 16_000) -> bytes:
    n = int(seconds * sr)
    t = np.linspace(0, seconds, n, endpoint=False)
    samples = (np.sin(2 * np.pi * 220 * t) * 0.2 * 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(samples.tobytes())
    return buf.getvalue()


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
    assets = tmp_path / "assets"
    assets.mkdir()
    for name in ("encoder.onnx", "ctc_decoder.onnx", "preprocessor.ts"):
        (assets / name).write_bytes(b"x")
    _write_masks(assets / "language_masks.json", languages)

    class FakePreprocessor:
        def eval(self):
            return self

        def __call__(self, input_signal, length):
            t = max(8, int(input_signal.shape[-1]) // 160)
            return torch.zeros(input_signal.shape[0], 80, t), length

    class FakeSession:
        def __init__(self, path, *args, **kwargs):
            self.path = str(path)
            self._providers = kwargs.get("providers") or ["CPUExecutionProvider"]

        def get_providers(self):
            return list(self._providers)

        def run(self, output_names, feeds):
            if self.path.endswith("encoder.onnx"):
                audio = feeds["audio_signal"]
                b, _, t = audio.shape
                return np.zeros((b, 1024, t), dtype=np.float32), np.array([t], dtype=np.int64)
            enc = feeds["encoder_output"]
            b, _, t = enc.shape
            logits = np.zeros((b, t, SHARED_VOCAB_SIZE), dtype=np.float32)
            logits[:, :, :10] = 1.0
            return [logits]

    import onnxruntime as ort

    monkeypatch.setattr(ort, "get_available_providers", lambda: ["CPUExecutionProvider"])
    monkeypatch.setattr(ort, "InferenceSession", FakeSession)
    monkeypatch.setattr(torch.jit, "load", lambda *args, **kwargs: FakePreprocessor())

    return LanguageIdentifier(
        device="cpu",
        candidate_languages=languages,
        assets_dir=assets,
        margin_threshold=0.050965,
        min_probe_duration_ms=100.0,
        min_rms_energy=None,
    )


def test_no_rnnt_artifacts_in_download_patterns():
    assert any("joint" in pattern for pattern in IGNORE_PATTERNS)
    assert any("rnnt" in pattern for pattern in IGNORE_PATTERNS)
    assert "assets/vocab.json" in IGNORE_PATTERNS
    assert ALLOW_PATTERNS == ["assets/*"]


def test_normalized_ctc_score_shapes():
    languages = ["a", "b"]
    vocab, selected, blank = 16, 5, 4
    masks = {}
    for i, lang in enumerate(languages):
        mask = [False] * vocab
        for j in range(selected - 1):
            mask[i * 2 + j] = True
        mask[vocab - 1] = True
        masks[lang] = mask
    indices = language_index_matrix(masks, languages, vocab, torch.device("cpu"), blank_id=blank)
    logits = torch.randn(1, 8, vocab)
    scores = score_all_languages(
        logits, torch.tensor([8]), indices, "normalized_ctc_score", blank_id=blank
    )
    assert scores.shape == (1, 2)
    assert torch.isfinite(scores).all()


def test_ranking_and_margin():
    languages = ["hi", "kn", "te"]
    scores = torch.tensor([-0.05, -0.20, -0.12])
    result = build_scoring_result(
        scores, languages, "normalized_ctc_score", margin_threshold=0.050965
    )
    assert result.language == "hi"
    assert result.decision == "accepted"
    assert result.margin == pytest.approx(result.top_score - result.second_score)


def test_single_pass_many_languages(monkeypatch, tmp_path: Path):
    languages = ["hi", "kn", "mr", "ta", "te", "bn", "gu"]
    model = _stub_model(monkeypatch, tmp_path, languages)
    wav = torch.randn(1, 16_000) * 0.1
    model.reset_counters()
    result = model.identify(wav, candidate_languages=languages)
    assert result["encoder_calls"] == 1
    assert result["ctc_calls"] == 1
    assert len(result["top_candidates"]) == len(languages)


@pytest.fixture()
def client(monkeypatch, tmp_path: Path):
    languages = ["hi", "kn", "mr", "ta", "te"]
    model = _stub_model(monkeypatch, tmp_path, languages)
    app = create_app(model=model)
    with TestClient(app) as test_client:
        yield test_client


def test_health(client: TestClient):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert "CPUExecutionProvider" in body["providers"]
    assert "hi" in body["available_languages"]


def test_identify_response_shape(client: TestClient):
    response = client.post(
        "/identify",
        files={"audio": ("utt.wav", _wav_bytes(), "audio/wav")},
        data={"candidate_languages": "hi,kn,mr,ta,te"},
    )
    assert response.status_code == 200
    body = response.json()
    assert set(body.keys()) == {"language", "scores", "margin", "top_candidates"}
    assert "transcript" not in body
    assert set(body["scores"].keys()) == {"hi", "kn", "mr", "ta", "te"}
    assert len(body["top_candidates"]) == 5


def test_identify_empty(client: TestClient):
    response = client.post("/identify", files={"audio": ("e.wav", b"", "audio/wav")})
    assert response.status_code == 400


def test_identify_invalid_lang(client: TestClient):
    response = client.post(
        "/identify",
        files={"audio": ("utt.wav", _wav_bytes(), "audio/wav")},
        data={"candidate_languages": "hi,xx"},
    )
    assert response.status_code == 400
