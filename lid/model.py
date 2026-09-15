"""Language identification: preprocessing, shared encoder, shared CTC, masks."""

from __future__ import annotations

import io
import json
import os
import wave
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Literal, Sequence

import numpy as np
import onnxruntime as ort
import torch
import torch.nn.functional as F

HF_REPO_ID = "ai4bharat/indic-conformer-600m-multilingual"

ALLOW_PATTERNS = ["assets/*"]
IGNORE_PATTERNS = ["assets/rnnt*", "assets/joint*", "assets/vocab.json"]

REQUIRED_FILES = (
    "encoder.onnx",
    "ctc_decoder.onnx",
    "preprocessor.ts",
    "language_masks.json",
)

ASR_IGNORE_PREFIXES = ("rnnt", "joint")
ASR_IGNORE_NAMES = frozenset({"vocab.json"})

DEFAULT_CANDIDATES = ("hi", "kn", "mr", "ta", "te")

ScoringMethod = Literal["normalized_ctc_score"]
SCORING_METHODS: tuple[str, ...] = ("normalized_ctc_score",)

LOCAL_BLANK_ID = 256
SHARED_VOCAB_SIZE = 5633
GLOBAL_BLANK_ID = 5632


class LanguageIdentifierError(ValueError):
    """Raised for model loading or inference failures."""


class AudioError(LanguageIdentifierError):
    """Raised when audio cannot be decoded."""


class ScoringError(ValueError):
    """Raised when CTC logits / masks are incompatible with scoring."""


@dataclass(frozen=True)
class RankedLanguage:
    language: str
    score: float
    rank: int


@dataclass(frozen=True)
class ScoringResult:
    language: str | None
    top_language: str | None
    decision: str
    confidence: float | None
    margin: float | None
    top_score: float | None
    second_score: float | None
    languages: list[RankedLanguage]
    scoring_method: str
    reason: str | None = None


def resolve_hf_token() -> str | None:
    for key in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        value = os.environ.get(key)
        if value and value.strip():
            return value.strip()
    return None


def is_lid_artifact(name: str) -> bool:
    if name in ASR_IGNORE_NAMES:
        return False
    if name.startswith(ASR_IGNORE_PREFIXES):
        return False
    return True


def find_assets_dir(root: Path) -> Path:
    if (root / "encoder.onnx").exists():
        return root
    assets = root / "assets"
    if (assets / "encoder.onnx").exists():
        return assets
    raise LanguageIdentifierError(
        f"Could not find LID artifacts under {root} or {assets}. "
        "Set HF_TOKEN if the Hugging Face model is gated."
    )


def ensure_lid_artifacts(
    repo_id: str = HF_REPO_ID,
    *,
    token: str | None = None,
) -> Path:
    """Download or resolve LID-only artifacts from the Hugging Face Hub cache."""

    from huggingface_hub import snapshot_download

    token = token if token is not None else resolve_hf_token()
    kwargs: dict[str, Any] = {
        "repo_id": repo_id,
        "allow_patterns": ALLOW_PATTERNS,
        "ignore_patterns": IGNORE_PATTERNS,
    }
    if token:
        kwargs["token"] = token

    try:
        snapshot_dir = Path(snapshot_download(**kwargs, local_files_only=True))
    except Exception:
        snapshot_dir = Path(snapshot_download(**kwargs))

    assets_dir = find_assets_dir(snapshot_dir)
    missing = [name for name in REQUIRED_FILES if not (assets_dir / name).exists()]
    if missing:
        raise LanguageIdentifierError(
            f"LID setup incomplete; missing in {assets_dir}: {missing}"
        )

    leaked = [
        path.name
        for path in assets_dir.iterdir()
        if path.is_file() and not is_lid_artifact(path.name)
    ]
    if leaked:
        raise LanguageIdentifierError(f"ASR artifacts unexpectedly present: {sorted(leaked)}")

    return assets_dir


def frame_mask(lengths: torch.Tensor, frames: int) -> torch.Tensor:
    return torch.arange(frames, device=lengths.device).unsqueeze(0) < lengths.unsqueeze(1)


def as_lengths(
    encoded_lengths: Any,
    batch_size: int,
    max_frames: int,
    device: torch.device,
) -> torch.Tensor:
    lengths = torch.as_tensor(encoded_lengths, device=device).reshape(-1).long()
    if lengths.numel() != batch_size:
        raise ScoringError(
            f"encoded_lengths has {lengths.numel()} entries for a batch of {batch_size}"
        )
    if (lengths <= 0).any() or (lengths > max_frames).any():
        raise ScoringError(
            f"encoded_lengths must be in [1, {max_frames}], got {lengths.detach().cpu().tolist()}"
        )
    return lengths


def language_index_matrix(
    language_masks: dict[str, Sequence[bool] | torch.Tensor],
    languages: Sequence[str],
    vocabulary_size: int,
    device: torch.device,
    blank_id: int = LOCAL_BLANK_ID,
) -> torch.Tensor:
    indices: list[torch.Tensor] = []
    for language in languages:
        if language not in language_masks:
            raise ScoringError(f"Missing language mask for {language!r}")
        mask = torch.as_tensor(language_masks[language], dtype=torch.bool, device=device)
        if mask.numel() != vocabulary_size:
            raise ScoringError(
                f"Mask for {language!r} has {mask.numel()} entries "
                f"but CTC vocabulary has {vocabulary_size}"
            )
        if not bool(mask.any()):
            raise ScoringError(f"Mask for {language!r} selects no CTC tokens")
        indices.append(torch.nonzero(mask, as_tuple=False).squeeze(1))
    sizes = {index.numel() for index in indices}
    if len(sizes) != 1:
        raise ScoringError(
            "Equal-size language masks are required for batched scoring; "
            f"found selected sizes {sorted(sizes)}"
        )
    result = torch.stack(indices)
    if not 0 <= blank_id < result.shape[1]:
        raise ScoringError(
            f"BLANK_ID={blank_id} is invalid for {result.shape[1]}-token masked CTC vocabularies"
        )
    return result


def score_all_languages(
    ctc_logits: torch.Tensor,
    lengths: torch.Tensor,
    language_indices: torch.Tensor,
    method: ScoringMethod = "normalized_ctc_score",
    blank_id: int = LOCAL_BLANK_ID,
) -> torch.Tensor:
    if method not in SCORING_METHODS:
        raise ScoringError(f"Unknown scoring method: {method!r}")
    if ctc_logits.ndim != 3:
        raise ScoringError(f"Expected CTC logits [B, T, V], got {tuple(ctc_logits.shape)}")

    language_logits = ctc_logits[:, :, language_indices]
    log_probs = F.log_softmax(language_logits.float(), dim=-1)
    if not torch.isfinite(log_probs).all():
        raise ScoringError("CTC output contains NaN or infinite values")

    valid = frame_mask(lengths, log_probs.shape[1]).unsqueeze(-1)
    valid_count = lengths.to(log_probs.dtype).unsqueeze(1)
    best_logprob, best_token = log_probs.max(dim=-1)

    changed = torch.ones_like(best_token, dtype=torch.bool)
    if best_token.shape[1] > 1:
        changed[:, 1:, :] = best_token[:, 1:, :] != best_token[:, :-1, :]
    kept = valid & changed & (best_token != blank_id)
    kept_count = kept.sum(dim=1)
    token_score = (best_logprob * kept).sum(dim=1) / kept_count.clamp_min(1)
    frame_score = (best_logprob * valid).sum(dim=1) / valid_count
    return torch.where(kept_count > 0, token_score, frame_score)


def build_scoring_result(
    scores: torch.Tensor,
    languages: Sequence[str],
    method: str,
    *,
    margin_threshold: float | None,
    confidence_threshold: float | None = None,
) -> ScoringResult:
    if scores.ndim != 1:
        raise ScoringError(f"Expected score vector [L], got {tuple(scores.shape)}")
    if scores.numel() != len(languages):
        raise ScoringError("Score vector length must match candidate languages")

    confidence_values = torch.softmax(scores, dim=0)
    ranked_scores, ranked_indices = scores.sort(dim=0, descending=True)
    ranking = [
        RankedLanguage(
            language=languages[index],
            score=float(ranked_scores[rank]),
            rank=rank + 1,
        )
        for rank, index in enumerate(ranked_indices.tolist())
    ]
    top_index = int(ranked_indices[0])
    top_language = languages[top_index]
    top_score = float(ranked_scores[0])
    second_score = float(ranked_scores[1]) if len(languages) > 1 else None
    margin = top_score - second_score if second_score is not None else None
    confidence = float(confidence_values[top_index])

    accepted = (
        (confidence_threshold is None or confidence >= confidence_threshold)
        and (margin_threshold is None or margin is None or margin >= margin_threshold)
    )
    return ScoringResult(
        language=top_language,
        top_language=top_language,
        decision="accepted" if accepted else "uncertain",
        confidence=confidence,
        margin=margin,
        top_score=top_score,
        second_score=second_score,
        languages=ranking,
        scoring_method=method,
        reason=None if accepted else "language_uncertain",
    )


def validate_candidates(
    requested: Sequence[str] | None,
    available: Sequence[str],
) -> tuple[str, ...]:
    available_set = set(available)
    if requested is None:
        return tuple(available)
    languages = tuple(requested)
    if not languages:
        raise ValueError("At least one candidate language is required")
    if len(set(languages)) != len(languages):
        raise ValueError("Candidate languages must not contain duplicates")
    unsupported = sorted(set(languages) - available_set)
    if unsupported:
        raise ValueError(
            f"Unknown candidate languages {unsupported}; "
            f"supported: {', '.join(available)}"
        )
    return languages


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
    """Identify spoken language using shared encoder + CTC + vocabulary masks."""

    def __init__(
        self,
        device: str = "cpu",
        candidate_languages: Sequence[str] | None = None,
        *,
        assets_dir: str | Path | None = None,
        hf_repo_id: str = HF_REPO_ID,
        hf_token: str | None = None,
        margin_threshold: float = 0.050965,
        confidence_threshold: float | None = None,
        min_probe_duration_ms: float = 500.0,
        min_rms_energy: float | None = 1e-4,
        sample_rate: int = 16_000,
        blank_id: int = LOCAL_BLANK_ID,
    ) -> None:
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
        self.hf_repo_id = hf_repo_id

        self.encoder_calls = 0
        self.ctc_calls = 0

        if assets_dir is None:
            self.assets_dir = ensure_lid_artifacts(hf_repo_id, token=hf_token)
        else:
            self.assets_dir = find_assets_dir(Path(assets_dir).expanduser().resolve())

        self.providers = self._resolve_providers(self.device_request)
        self.torch_device = torch.device("cpu")

        self.preprocessor = torch.jit.load(
            str(self.assets_dir / "preprocessor.ts"),
            map_location="cpu",
        )
        self.preprocessor.eval()

        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        try:
            self.encoder = ort.InferenceSession(
                str(self.assets_dir / "encoder.onnx"),
                sess_options=session_options,
                providers=self.providers,
            )
            self.ctc_decoder = ort.InferenceSession(
                str(self.assets_dir / "ctc_decoder.onnx"),
                sess_options=session_options,
                providers=self.providers,
            )
        except Exception as exc:  # noqa: BLE001
            if self.device_request == "cuda":
                raise LanguageIdentifierError(
                    "Failed to create ONNX sessions with CUDA. "
                    "Need CUDA 12 + cuDNN 9, or set DEVICE='cpu' in server.py. "
                    f"Underlying error: {exc}"
                ) from exc
            raise LanguageIdentifierError(f"Failed to create ONNX sessions: {exc}") from exc
        self._assert_active_providers()

        with open(self.assets_dir / "language_masks.json", encoding="utf-8") as reader:
            raw_masks = json.load(reader)
        if not isinstance(raw_masks, dict) or not raw_masks:
            raise LanguageIdentifierError("language_masks.json must be a non-empty object")
        self.language_masks: dict[str, list[bool]] = {
            str(lang): list(mask) for lang, mask in raw_masks.items()
        }
        self._validate_masks()
        self.available_languages = tuple(sorted(self.language_masks.keys()))
        if candidate_languages is None:
            defaults = [c for c in DEFAULT_CANDIDATES if c in self.language_masks]
            self.candidate_languages = tuple(defaults) if defaults else self.available_languages
        else:
            self.candidate_languages = validate_candidates(
                candidate_languages, self.available_languages
            )
        self._index_cache: dict[tuple[str, ...], torch.Tensor] = {}

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
                    f"(available={available}). Install CUDA 12 + cuDNN 9, or set DEVICE='cpu'."
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
                    f"CUDAExecutionProvider (active={providers}). "
                    "Usually missing libcudnn.so.9 (cuDNN 9). Set DEVICE='cpu' to fall back."
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
        if candidate_languages is None:
            languages = self.candidate_languages
        else:
            languages = validate_candidates(candidate_languages, self.available_languages)

        wav, duration_ms = _to_waveform(audio, self.sample_rate)

        if duration_ms < self.min_probe_duration_ms:
            return self._empty_result(languages)

        energy = _rms_energy(wav)
        if self.min_rms_energy is not None and energy < self.min_rms_energy:
            return self._empty_result(languages)

        before_enc, before_ctc = self.encoder_calls, self.ctc_calls
        encoder_outputs, encoded_lengths = self.encode(wav)
        ctc_logits = self.project_ctc(encoder_outputs)

        if self.encoder_calls - before_enc != 1 or self.ctc_calls - before_ctc != 1:
            raise LanguageIdentifierError(
                "Single-pass guarantee violated: "
                f"encoder_delta={self.encoder_calls - before_enc}, "
                f"ctc_delta={self.ctc_calls - before_ctc}"
            )

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

        return {
            "language": result.language,
            "margin": result.margin,
            "scores": {item.language: item.score for item in result.languages},
            "top_candidates": [
                {"language": item.language, "score": item.score} for item in result.languages
            ],
            "encoder_calls": 1,
            "ctc_calls": 1,
        }

    def _empty_result(self, languages: Sequence[str]) -> dict[str, Any]:
        return {
            "language": None,
            "margin": None,
            "scores": {lang: None for lang in languages},
            "top_candidates": [],
            "encoder_calls": 0,
            "ctc_calls": 0,
        }

    def health(self) -> dict[str, Any]:
        return {
            "status": "ok",
            "device": self.runtime_device,
            "model_loaded": True,
            "available_languages": list(self.available_languages),
            "providers": self.active_providers,
        }
