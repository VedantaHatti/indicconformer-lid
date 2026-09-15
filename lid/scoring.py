"""Validated CTC language scoring (normalized_ctc_score and helpers).

Scores come from masked shared-CTC vocabulary distributions. Softmax
``confidence`` is a heuristic diagnostic, not a calibrated probability.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Sequence

import torch
import torch.nn.functional as F

ScoringMethod = Literal["normalized_ctc_score"]
SCORING_METHODS: tuple[str, ...] = ("normalized_ctc_score",)

LOCAL_BLANK_ID = 256
SHARED_VOCAB_SIZE = 5633
GLOBAL_BLANK_ID = 5632


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
    """Return token indices shaped ``[L, K]`` for equal-size language masks."""

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
    """Vectorized compatibility scores for all masks, returned as ``[B, L]``."""

    if method not in SCORING_METHODS:
        raise ScoringError(f"Unknown scoring method: {method!r}")
    if ctc_logits.ndim != 3:
        raise ScoringError(f"Expected CTC logits [B, T, V], got {tuple(ctc_logits.shape)}")

    # [B, T, V] indexed with [L, K] → [B, T, L, K]
    language_logits = ctc_logits[:, :, language_indices]
    log_probs = F.log_softmax(language_logits.float(), dim=-1)
    if not torch.isfinite(log_probs).all():
        raise ScoringError("CTC output contains NaN or infinite values")

    valid = frame_mask(lengths, log_probs.shape[1]).unsqueeze(-1)
    valid_count = lengths.to(log_probs.dtype).unsqueeze(1)
    best_logprob, best_token = log_probs.max(dim=-1)

    # normalized_ctc_score: collapse-aware non-blank token avg, else frame avg
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
