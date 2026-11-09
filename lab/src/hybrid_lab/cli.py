"""CLI demos for hybrid retrieval."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from hybrid_lab.corpus import demo_corpus, load_corpus
from hybrid_lab.pipeline import HybridSearcher


def cmd_search(query: str, method: str, data: str | None) -> int:
    if data:
        docs = load_corpus(data)
    else:
        docs = demo_corpus()
        default = Path(__file__).resolve().parents[2] / "data" / "corpus.jsonl"
        if default.exists():
            docs = load_corpus(default)
    searcher = HybridSearcher.from_docs(docs, method=method)
    result = searcher.search(query, k=5, explain=True)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="hybrid_lab")
    sub = p.add_subparsers(dest="cmd", required=True)
    sp = sub.add_parser("search")
    sp.add_argument("query")
    sp.add_argument("--method", choices=["rrf", "linear"], default="rrf")
    sp.add_argument("--data", default=None, help="path to corpus.jsonl")
    args = p.parse_args(argv)
    if args.cmd == "search":
        return cmd_search(args.query, args.method, args.data)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
