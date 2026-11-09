#!/usr/bin/env python3
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hybrid_lab.cli import cmd_search
from hybrid_lab.corpus import demo_corpus, load_corpus
from hybrid_lab.pipeline import HybridSearcher
from hybrid_lab.sparse import SparseIndex


def main() -> int:
    docs = demo_corpus()
    corpus_path = ROOT / "data" / "corpus.jsonl"
    if corpus_path.exists():
        docs = load_corpus(corpus_path)

    sparse = SparseIndex.from_docs(docs)
    hits = sparse.search("ORD-2024-881", k=3)
    assert any(d == "d2" for d, _ in hits), hits

    searcher = HybridSearcher.from_docs(docs, method="rrf")
    explained = searcher.search("退款多久到账", k=3, explain=True)
    assert explained["fused"], explained
    assert cmd_search("发票邮箱", "rrf", str(corpus_path) if corpus_path.exists() else None) == 0

    loader = unittest.TestLoader()
    suite = loader.discover(str(ROOT / "tests"))
    result = unittest.TextTestRunner(verbosity=1).run(suite)
    if not result.wasSuccessful():
        return 1
    print("smoke_check OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
