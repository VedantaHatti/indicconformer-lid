"""API contract tests with stubbed model."""

from __future__ import annotations

import io
import json
import wave
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from lid.model import LanguageIdentifier
from lid.scoring import LOCAL_BLANK_ID, SHARED_VOCAB_SIZE
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


def _stub_model(monkeypatch, tmp_path: Path) -> LanguageIdentifier:
    languages = ["hi", "kn", "mr", "ta", "te"]
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    for name in ("encoder.onnx", "ctc_decoder.onnx", "preprocessor.ts"):
        (model_dir / name).write_bytes(b"x")
    _write_masks(model_dir / "language_masks.json", languages)

    class FakePreprocessor:
        def eval(self):
            return self

        def __call__(self, input_signal, length):
            import torch

            t = max(8, int(input_signal.shape[-1]) // 160)
            return torch.zeros(input_signal.shape[0], 80, t), length

    class FakeSession:
        def __init__(self, path, *a, **k):
            self.path = str(path)
            self._providers = ["CPUExecutionProvider"]

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
    import torch

    monkeypatch.setattr(ort, "get_available_providers", lambda: ["CPUExecutionProvider"])
    monkeypatch.setattr(ort, "InferenceSession", FakeSession)
    monkeypatch.setattr(torch.jit, "load", lambda *a, **k: FakePreprocessor())

    return LanguageIdentifier(
        model_dir,
        device="cpu",
        candidate_languages=languages,
        margin_threshold=0.050965,
        min_probe_duration_ms=100.0,
        min_rms_energy=None,
    )


@pytest.fixture()
def client(monkeypatch, tmp_path: Path):
    model = _stub_model(monkeypatch, tmp_path)
    app = create_app(model=model)
    with TestClient(app) as test_client:
        yield test_client


def test_health(client: TestClient):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert "CPUExecutionProvider" in body["providers"]


def test_identify(client: TestClient):
    r = client.post(
        "/identify",
        files={"audio": ("utt.wav", _wav_bytes(), "audio/wav")},
        data={"candidate_languages": "hi,kn,mr,ta,te"},
    )
    assert r.status_code == 200
    body = r.json()
    assert "language" in body
    assert "transcript" not in body
    assert body["encoder_calls"] == 1
    assert body["ctc_calls"] == 1


def test_identify_empty(client: TestClient):
    r = client.post("/identify", files={"audio": ("e.wav", b"", "audio/wav")})
    assert r.status_code == 400


def test_identify_invalid_lang(client: TestClient):
    r = client.post(
        "/identify",
        files={"audio": ("utt.wav", _wav_bytes(), "audio/wav")},
        data={"candidate_languages": "hi,xx"},
    )
    assert r.status_code == 400
