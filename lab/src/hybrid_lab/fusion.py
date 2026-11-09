"""Rank fusion: RRF and linear score fusion."""
from __future__ import annotations


def rrf_fuse(
    rank_lists: list[list[tuple[str, float]]],
    k: int = 60,
) -> list[tuple[str, float]]:
    scores: dict[str, float] = {}
    for ranking in rank_lists:
        for rank, (doc_id, _score) in enumerate(ranking, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda x: (-x[1], x[0]))


def _minmax(ranking: list[tuple[str, float]]) -> dict[str, float]:
    if not ranking:
        return {}
    vals = [s for _, s in ranking]
    lo, hi = min(vals), max(vals)
    if hi - lo < 1e-12:
        return {d: 1.0 for d, _ in ranking}
    return {d: (s - lo) / (hi - lo) for d, s in ranking}


def linear_fuse(
    rank_lists: list[list[tuple[str, float]]],
    weights: list[float] | None = None,
) -> list[tuple[str, float]]:
    if not rank_lists:
        return []
    if weights is None:
        weights = [1.0 / len(rank_lists)] * len(rank_lists)
    if len(weights) != len(rank_lists):
        raise ValueError("weights length must match rank_lists")
    scores: dict[str, float] = {}
    for ranking, w in zip(rank_lists, weights):
        normed = _minmax(ranking)
        for doc_id, s in normed.items():
            scores[doc_id] = scores.get(doc_id, 0.0) + w * s
    return sorted(scores.items(), key=lambda x: (-x[1], x[0]))
