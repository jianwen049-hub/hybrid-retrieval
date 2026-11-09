"""Toy bag-of-char-ngram dense embeddings + cosine search."""
from __future__ import annotations

import math
from dataclasses import dataclass, field


def _char_ngrams(text: str, n: int = 3) -> list[str]:
    t = text.lower().strip()
    if len(t) < n:
        return [t] if t else []
    return [t[i : i + n] for i in range(len(t) - n + 1)]


def embed(text: str, dim: int = 128) -> tuple[float, ...]:
    vec = [0.0] * dim
    grams = _char_ngrams(text, 3)
    if not grams:
        return tuple(vec)
    for g in grams:
        # stable within process; fine for offline lab
        bucket = hash(g) % dim
        vec[bucket] += 1.0
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return tuple(v / norm for v in vec)


def cosine(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    return float(sum(x * y for x, y in zip(a, b)))


@dataclass
class DenseIndex:
    docs: dict[str, str]
    vectors: dict[str, tuple[float, ...]] = field(default_factory=dict)
    dim: int = 128

    @classmethod
    def from_docs(cls, docs: dict[str, str], dim: int = 128) -> "DenseIndex":
        idx = cls(docs=dict(docs), dim=dim)
        for doc_id, text in docs.items():
            idx.vectors[doc_id] = embed(text, dim=dim)
        return idx

    def search(self, query: str, k: int = 5) -> list[tuple[str, float]]:
        qv = embed(query, dim=self.dim)
        scored = [(doc_id, cosine(qv, vec)) for doc_id, vec in self.vectors.items()]
        scored.sort(key=lambda x: (-x[1], x[0]))
        return scored[:k]
