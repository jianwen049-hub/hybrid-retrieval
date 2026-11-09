"""Hybrid search pipeline."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from hybrid_lab.dense import DenseIndex
from hybrid_lab.fusion import linear_fuse, rrf_fuse
from hybrid_lab.sparse import SparseIndex


@dataclass
class HybridSearcher:
    sparse: SparseIndex
    dense: DenseIndex
    method: str = "rrf"  # rrf | linear
    candidate_k: int = 5

    @classmethod
    def from_docs(cls, docs: dict[str, str], method: str = "rrf") -> "HybridSearcher":
        return cls(
            sparse=SparseIndex.from_docs(docs),
            dense=DenseIndex.from_docs(docs),
            method=method,
        )

    def search(self, query: str, k: int = 5, explain: bool = False) -> dict[str, Any] | list[tuple[str, float]]:
        sparse_hits = self.sparse.search(query, k=self.candidate_k)
        dense_hits = self.dense.search(query, k=self.candidate_k)
        if self.method == "linear":
            fused = linear_fuse([sparse_hits, dense_hits], weights=[0.55, 0.45])
        else:
            fused = rrf_fuse([sparse_hits, dense_hits])
        fused = fused[:k]
        if explain:
            return {
                "query": query,
                "method": self.method,
                "sparse": sparse_hits,
                "dense": dense_hits,
                "fused": fused,
            }
        return fused
