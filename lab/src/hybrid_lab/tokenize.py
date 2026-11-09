"""Lightweight tokenizer for EN words + CJK char n-grams."""
from __future__ import annotations

import re

_WORD = re.compile(r"[A-Za-z0-9_\-]+")
_CJK_RUN = re.compile(r"[\u4e00-\u9fff]+")


def tokenize(text: str) -> list[str]:
    if not text:
        return []
    text = text.lower()
    tokens: list[str] = []
    for m in _WORD.finditer(text):
        tokens.append(m.group(0))
    for m in _CJK_RUN.finditer(text):
        run = m.group(0)
        tokens.extend(list(run))
        tokens.extend(run[i : i + 2] for i in range(len(run) - 1))
    return tokens
