"""BM25-like sparse index (stdlib only)."""
from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field

from hybrid_lab.tokenize import tokenize


@dataclass
class SparseIndex:
    docs: dict[str, str]
    df: Counter[str] = field(default_factory=Counter)
    doc_len: dict[str, int] = field(default_factory=dict)
    tf: dict[str, Counter[str]] = field(default_factory=dict)
    avgdl: float = 0.0
    n_docs: int = 0
    k1: float = 1.5
    b: float = 0.75

    @classmethod
    def from_docs(cls, docs: dict[str, str], k1: float = 1.5, b: float = 0.75) -> "SparseIndex":
        idx = cls(docs=dict(docs), k1=k1, b=b)
        lengths: list[int] = []
        for doc_id, text in docs.items():
            toks = tokenize(text)
            tf = Counter(toks)
            idx.tf[doc_id] = tf
            idx.doc_len[doc_id] = len(toks)
            lengths.append(len(toks))
            for t in tf:
                idx.df[t] += 1
        idx.n_docs = len(docs)
        idx.avgdl = (sum(lengths) / len(lengths)) if lengths else 0.0
        return idx

    def idf(self, term: str) -> float:
        df = self.df.get(term, 0)
        # BM25 IDF with smoothing
        return math.log(1.0 + (self.n_docs - df + 0.5) / (df + 0.5))

    def score(self, query: str, doc_id: str) -> float:
        q_terms = tokenize(query)
        if not q_terms or doc_id not in self.tf:
            return 0.0
        dl = self.doc_len[doc_id]
        avgdl = self.avgdl or 1.0
        tf = self.tf[doc_id]
        s = 0.0
        for t in q_terms:
            f = tf.get(t, 0)
            if f <= 0:
                continue
            idf = self.idf(t)
            denom = f + self.k1 * (1.0 - self.b + self.b * dl / avgdl)
            s += idf * (f * (self.k1 + 1.0)) / denom
        return s

    def search(self, query: str, k: int = 5) -> list[tuple[str, float]]:
        scored = [(doc_id, self.score(query, doc_id)) for doc_id in self.docs]
        scored.sort(key=lambda x: (-x[1], x[0]))
        return [(d, s) for d, s in scored[:k] if s > 0]
