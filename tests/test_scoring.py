"""Scoring and ranking unit tests."""

from __future__ import annotations

import pytest
import torch

from lid.scoring import (
    build_scoring_result,
    language_index_matrix,
    score_all_languages,
)


def test_ranking_and_margin():
    languages = ["hi", "kn", "te"]
    scores = torch.tensor([-0.05, -0.20, -0.12])
    result = build_scoring_result(
        scores, languages, "normalized_ctc_score", margin_threshold=0.050965
    )
    assert result.language == "hi"
    assert result.decision == "accepted"
    assert result.languages[0].language == "hi"
    assert result.languages[0].rank == 1
    assert result.languages[1].language == "te"
    assert result.margin == pytest.approx(result.top_score - result.second_score)


def test_uncertain_when_margin_low():
    result = build_scoring_result(
        torch.tensor([-0.10, -0.12]),
        ["hi", "kn"],
        "normalized_ctc_score",
        margin_threshold=0.050965,
    )
    assert result.decision == "uncertain"
    assert result.language == "hi"
    assert result.reason == "language_uncertain"


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
