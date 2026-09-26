"""Deterministic, contiguous evidence spans. IDs bind offsets to exact chunk bytes."""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

from .schemas import ChunkRecord


@dataclass(frozen=True)
class EvidenceSpan:
    span_id: str
    start: int
    end: int
    text: str


def evidence_spans(chunk: ChunkRecord) -> list[EvidenceSpan]:
    # Paragraph boundaries preserve code blocks, case, punctuation, and negation.
    spans = []
    for match in re.finditer(r"\S(?:.*?\S)?(?=\n\s*\n|\Z)", chunk.text, re.DOTALL):
        start, end = match.span()
        text = chunk.text[start:end]
        digest = hashlib.sha256(f"{chunk.chunk_id}:{start}:{end}:{text}".encode()).hexdigest()[:12]
        spans.append(EvidenceSpan(f"s-{digest}", start, end, text))
    return spans
