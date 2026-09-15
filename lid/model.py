"""Offline language identification: shared Conformer encoder + shared CTC + masks."""

from __future__ import annotations

import io
import json
import wave
from dataclasses import asdict
from pathlib import Path
from time import perf_counter
from typing import Any, Sequence

import numpy as np
import onnxruntime as ort
import torch

from lid.scoring import (
    GLOBAL_BLANK_ID,
    LOCAL_BLANK_ID,
    SHARED_VOCAB_SIZE,
    ScoringError,
    as_lengths,
    build_scoring_result,
    language_index_matrix,
    score_all_languages,
    validate_candidates,
)

REQUIRED_FILES = (
    "encoder.onnx",
    "ctc_decoder.onnx",
    "preprocessor.ts",
    "language_masks.json",
)

DEFAULT_CANDIDATES = ("hi", "kn", "mr", "ta", "te")


class LanguageIdentifierError(ValueError):
    """Raised for model loading or inference failures."""


class AudioError(LanguageIdentifierError):
    """Raised when audio cannot be decoded."""


def _decode_audio_bytes(data: bytes) -> tuple[np.ndarray, int]:
    if not data:
        raise AudioError("Audio payload is empty")
    if data[:4] == b"RIFF" and data[8:12] == b"WAVE":
        try:
            with wave.open(io.BytesIO(data), "rb") as reader:
                channels = reader.getnchannels()
                sample_width = reader.getsampwidth()
                sample_rate = reader.getframerate()
                frames = reader.readframes(reader.getnframes())
        except wave.Error as exc:
            raise AudioError(f"Invalid WAV audio: {exc}") from exc
        if sample_width == 2 and channels >= 1:
            samples = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
            if channels > 1:
                samples = samples.reshape(-1, channels).mean(axis=1)
            return samples.copy(), int(sample_rate)
    try:
        import soundfile as sf
    except ImportError as exc:
        raise AudioError("Install soundfile to decode non-PCM16 audio") from exc
    try:
        samples, sample_rate = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
    except Exception as exc:  # noqa: BLE001
        raise AudioError(f"Failed to decode audio: {exc}") from exc
    return samples.mean(axis=1), int(sample_rate)


def _resample_linear(samples: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    if source_rate == target_rate:
        return samples.astype(np.float32, copy=False)
    if source_rate <= 0 or target_rate <= 0:
        raise AudioError("Sample rates must be positive")
    if samples.size == 0:
        return samples.astype(np.float32, copy=False)
    duration = samples.shape[0] / float(source_rate)
    target_length = max(1, int(round(duration * target_rate)))
    source_x = np.linspace(0.0, 1.0, num=samples.shape[0], endpoint=False)
    target_x = np.linspace(0.0, 1.0, num=target_length, endpoint=False)
    return np.interp(target_x, source_x, samples.astype(np.float64)).astype(np.float32)


def _to_waveform(audio: bytes | torch.Tensor | np.ndarray, sample_rate: int) -> tuple[torch.Tensor, float]:
    if isinstance(audio, (bytes, bytearray)):
        samples, sr = _decode_audio_bytes(bytes(audio))
        samples = _resample_linear(samples, sr, sample_rate)
        wav = torch.from_numpy(np.ascontiguousarray(samples)).unsqueeze(0)
    elif isinstance(audio, np.ndarray):
        tensor = torch.from_numpy(np.asarray(audio, dtype=np.float32))
        wav = tensor.unsqueeze(0) if tensor.ndim == 1 else tensor
    elif isinstance(audio, torch.Tensor):
        wav = audio.unsqueeze(0) if audio.ndim == 1 else audio
    else:
        raise LanguageIdentifierError("audio must be bytes, numpy array, or torch.Tensor")
    if wav.ndim != 2 or wav.shape[-1] == 0:
        raise AudioError("Audio waveform is empty or has invalid shape")
    if not torch.isfinite(wav).all():
        raise AudioError("Audio contains NaN or infinite values")
    duration_ms = float(wav.shape[-1]) * 1000.0 / float(sample_rate)
    return wav, duration_ms


def _rms_energy(wav: torch.Tensor) -> float:
    return float(wav.float().square().mean().sqrt().item())


class LanguageIdentifier:
    """Identify spoken language using shared encoder + CTC + vocabulary masks.

    Per ``identify`` call: exactly one encoder ONNX run and one CTC ONNX run,
    then vectorized mask scoring on the shared logits.
    """

    def __init__(
        self,
        model_dir: str | Path = "./model",
        device: str = "cuda",
        candidate_languages: Sequence[str] | None = None,
        *,
        margin_threshold: float = 0.050965,
        confidence_threshold: float | None = None,
        min_probe_duration_ms: float = 500.0,
        min_rms_energy: float | None = 1e-4,
        sample_rate: int = 16_000,
        blank_id: int = LOCAL_BLANK_ID,
    ) -> None:
        self.model_dir = Path(model_dir).expanduser().resolve()
        self.device_request = device.lower().strip()
        if self.device_request not in {"cuda", "cpu"}:
            raise LanguageIdentifierError("device must be 'cuda' or 'cpu'")
        self.margin_threshold = margin_threshold
        self.confidence_threshold = confidence_threshold
        self.min_probe_duration_ms = min_probe_duration_ms
        self.min_rms_energy = min_rms_energy
        self.sample_rate = sample_rate
        self.blank_id = blank_id
        self.scoring_method = "normalized_ctc_score"

        self.encoder_calls = 0
        self.ctc_calls = 0

        self._validate_artifacts()
        self.providers = self._resolve_providers(self.device_request)
        self.torch_device = torch.device(
            "cuda" if self.providers[0] == "CUDAExecutionProvider" else "cpu"
        )

        self.preprocessor = torch.jit.load(
            str(self.model_dir / "preprocessor.ts"),
            map_location=self.torch_device,
        )
        self.preprocessor.eval()

        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.encoder = ort.InferenceSession(
            str(self.model_dir / "encoder.onnx"),
            sess_options=session_options,
            providers=self.providers,
        )
        self.ctc_decoder = ort.InferenceSession(
            str(self.model_dir / "ctc_decoder.onnx"),
            sess_options=session_options,
            providers=self.providers,
        )
        self._assert_active_providers()

        with open(self.model_dir / "language_masks.json", encoding="utf-8") as reader:
            raw_masks = json.load(reader)
        if not isinstance(raw_masks, dict) or not raw_masks:
            raise LanguageIdentifierError("language_masks.json must be a non-empty object")
        self.language_masks: dict[str, list[bool]] = {
            str(lang): list(mask) for lang, mask in raw_masks.items()
        }
        self._validate_masks()
        self.available_languages = tuple(sorted(self.language_masks.keys()))
        if candidate_languages is None:
            # Prefer a sensible default subset when available; else all masks.
            defaults = [c for c in DEFAULT_CANDIDATES if c in self.language_masks]
            self.candidate_languages = tuple(defaults) if defaults else self.available_languages
        else:
            self.candidate_languages = validate_candidates(
                candidate_languages, self.available_languages
            )
        self._index_cache: dict[tuple[str, ...], torch.Tensor] = {}

    def _validate_artifacts(self) -> None:
        if not self.model_dir.is_dir():
            raise LanguageIdentifierError(f"Model directory does not exist: {self.model_dir}")
        missing = [name for name in REQUIRED_FILES if not (self.model_dir / name).exists()]
        if missing:
            raise LanguageIdentifierError(
                f"Missing required model artifacts in {self.model_dir}: {', '.join(missing)}. "
                "Run: python scripts/setup_model.py"
            )

    def _validate_masks(self) -> None:
        lengths = {len(mask) for mask in self.language_masks.values()}
        if lengths != {SHARED_VOCAB_SIZE}:
            raise LanguageIdentifierError(
                f"Expected language masks of length {SHARED_VOCAB_SIZE}, got {sorted(lengths)}"
            )
        for language, mask in self.language_masks.items():
            selected = sum(1 for value in mask if value)
            if selected != LOCAL_BLANK_ID + 1:
                raise LanguageIdentifierError(
                    f"Mask for {language!r} selects {selected} tokens; "
                    f"expected {LOCAL_BLANK_ID + 1}"
                )
            if not mask[GLOBAL_BLANK_ID]:
                raise LanguageIdentifierError(
                    f"Mask for {language!r} does not include global blank id {GLOBAL_BLANK_ID}"
                )

    @staticmethod
    def _resolve_providers(device: str) -> list[str]:
        available = ort.get_available_providers()
        if device == "cuda":
            if "CUDAExecutionProvider" not in available:
                raise LanguageIdentifierError(
                    "device='cuda' requested but CUDAExecutionProvider is unavailable "
                    f"(available={available}). Install onnxruntime-gpu or use device='cpu'."
                )
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]
        return ["CPUExecutionProvider"]

    def _assert_active_providers(self) -> None:
        if self.device_request != "cuda":
            return
        for name, session in (("encoder", self.encoder), ("ctc", self.ctc_decoder)):
            providers = session.get_providers()
            if not providers or providers[0] != "CUDAExecutionProvider":
                raise LanguageIdentifierError(
                    f"CUDA was requested but the {name} session is not using "
                    f"CUDAExecutionProvider (active={providers})"
                )

    @property
    def active_providers(self) -> list[str]:
        return list(self.encoder.get_providers())

    @property
    def runtime_device(self) -> str:
        providers = self.active_providers
        if providers and providers[0] == "CUDAExecutionProvider":
            return "cuda"
        return "cpu"

    def reset_counters(self) -> None:
        self.encoder_calls = 0
        self.ctc_calls = 0

    def _language_indices(self, languages: Sequence[str], device: torch.device) -> torch.Tensor:
        key = tuple(languages)
        cached = self._index_cache.get(key)
        if cached is not None and cached.device == device:
            return cached
        matrix = language_index_matrix(
            self.language_masks,
            languages,
            SHARED_VOCAB_SIZE,
            device,
            blank_id=self.blank_id,
        )
        self._index_cache[key] = matrix
        return matrix

    def encode(self, wav: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
        if wav.ndim == 1:
            wav = wav.unsqueeze(0)
        if wav.ndim != 2 or wav.shape[0] != 1:
            raise LanguageIdentifierError("Only batch size 1 waveform [1, samples] is supported")
        with torch.inference_mode():
            audio_signal, length = self.preprocessor(
                input_signal=wav.to(self.torch_device),
                length=torch.tensor([wav.shape[-1]], device=self.torch_device),
            )
        self.encoder_calls += 1
        outputs, encoded_lengths = self.encoder.run(
            ["outputs", "encoded_lengths"],
            {
                "audio_signal": audio_signal.detach().cpu().numpy(),
                "length": length.detach().cpu().numpy(),
            },
        )
        return outputs, encoded_lengths

    def project_ctc(self, encoder_outputs: np.ndarray) -> torch.Tensor:
        encoded = np.asarray(encoder_outputs)
        if encoded.ndim != 3:
            raise LanguageIdentifierError(
                f"Expected encoder output shaped [B, H, T], got {tuple(encoded.shape)}"
            )
        self.ctc_calls += 1
        try:
            output = self.ctc_decoder.run(["logprobs"], {"encoder_output": encoded})[0]
        except Exception as exc:  # noqa: BLE001
            raise LanguageIdentifierError(f"Shared CTC projection failed: {exc}") from exc
        result = torch.as_tensor(output)
        if result.ndim != 3 or result.shape[-1] != SHARED_VOCAB_SIZE:
            raise LanguageIdentifierError(
                f"Expected CTC logits [B, T, {SHARED_VOCAB_SIZE}], got {tuple(result.shape)}"
            )
        return result

    def identify(
        self,
        audio: bytes | torch.Tensor | np.ndarray,
        *,
        candidate_languages: Sequence[str] | None = None,
    ) -> dict[str, Any]:
        total_started = perf_counter()
        audio_ms = 0.0
        encoder_ms = 0.0
        ctc_ms = 0.0
        scoring_ms = 0.0

        prep_started = perf_counter()
        wav, duration_ms = _to_waveform(audio, self.sample_rate)
        audio_ms = (perf_counter() - prep_started) * 1000.0

        if candidate_languages is None:
            languages = self.candidate_languages
        else:
            languages = validate_candidates(candidate_languages, self.available_languages)

        if duration_ms < self.min_probe_duration_ms:
            return self._guard_result(
                "insufficient_audio",
                duration_ms,
                self._latency(audio_ms, 0, 0, 0, (perf_counter() - total_started) * 1000),
            )

        energy = _rms_energy(wav)
        if self.min_rms_energy is not None and energy < self.min_rms_energy:
            payload = self._guard_result(
                "insufficient_speech_energy",
                duration_ms,
                self._latency(audio_ms, 0, 0, 0, (perf_counter() - total_started) * 1000),
            )
            payload["rms_energy"] = energy
            return payload

        before_enc, before_ctc = self.encoder_calls, self.ctc_calls
        enc_started = perf_counter()
        encoder_outputs, encoded_lengths = self.encode(wav)
        encoder_ms = (perf_counter() - enc_started) * 1000.0

        ctc_started = perf_counter()
        ctc_logits = self.project_ctc(encoder_outputs)
        ctc_ms = (perf_counter() - ctc_started) * 1000.0

        if self.encoder_calls - before_enc != 1 or self.ctc_calls - before_ctc != 1:
            raise LanguageIdentifierError(
                "Single-pass guarantee violated: "
                f"encoder_delta={self.encoder_calls - before_enc}, "
                f"ctc_delta={self.ctc_calls - before_ctc}"
            )

        score_started = perf_counter()
        try:
            lengths = as_lengths(
                encoded_lengths, ctc_logits.shape[0], ctc_logits.shape[1], ctc_logits.device
            )
            indices = self._language_indices(languages, ctc_logits.device)
            scores = score_all_languages(
                ctc_logits, lengths, indices, self.scoring_method, blank_id=self.blank_id
            )[0]
            result = build_scoring_result(
                scores,
                languages,
                self.scoring_method,
                margin_threshold=self.margin_threshold,
                confidence_threshold=self.confidence_threshold,
            )
        except ScoringError as exc:
            raise LanguageIdentifierError(str(exc)) from exc
        scoring_ms = (perf_counter() - score_started) * 1000.0
        total_ms = (perf_counter() - total_started) * 1000.0

        return {
            "language": result.language,
            "top_language": result.top_language,
            "decision": result.decision,
            "confidence": result.confidence,
            "margin": result.margin,
            "top_score": result.top_score,
            "second_score": result.second_score,
            "languages": [asdict(item) for item in result.languages],
            "scoring_method": result.scoring_method,
            "reason": result.reason,
            "duration_ms": duration_ms,
            "encoder_calls": 1,
            "ctc_calls": 1,
            "latency_ms": self._latency(audio_ms, encoder_ms, ctc_ms, scoring_ms, total_ms),
            "providers": self.active_providers,
            "device": self.runtime_device,
        }

    @staticmethod
    def _latency(
        audio_ms: float, encoder_ms: float, ctc_ms: float, scoring_ms: float, total_ms: float
    ) -> dict[str, float]:
        return {
            "audio_preprocessing": round(audio_ms, 3),
            "encoder": round(encoder_ms, 3),
            "ctc": round(ctc_ms, 3),
            "scoring": round(scoring_ms, 3),
            "total": round(total_ms, 3),
        }

    def _guard_result(
        self, reason: str, duration_ms: float, latency_ms: dict[str, float]
    ) -> dict[str, Any]:
        return {
            "language": None,
            "top_language": None,
            "decision": reason,
            "confidence": None,
            "margin": None,
            "top_score": None,
            "second_score": None,
            "languages": [],
            "scoring_method": self.scoring_method,
            "reason": reason,
            "duration_ms": duration_ms,
            "encoder_calls": 0,
            "ctc_calls": 0,
            "latency_ms": latency_ms,
            "providers": self.active_providers,
            "device": self.runtime_device,
        }

    def health(self) -> dict[str, Any]:
        return {
            "status": "ok",
            "model_loaded": True,
            "device": self.runtime_device,
            "device_requested": self.device_request,
            "providers": self.active_providers,
            "languages_loaded": list(self.candidate_languages),
            "languages_available": list(self.available_languages),
            "model_dir": str(self.model_dir),
            "scoring_method": self.scoring_method,
            "margin_threshold": self.margin_threshold,
        }
