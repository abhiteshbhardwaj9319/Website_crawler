"""Structural source linkage, not semantic entailment. V2 selects exact local passages;
legacy quotes require a case-sensitive contiguous match with whitespace normalization.
"""

from __future__ import annotations

import re
from .evidence import evidence_spans
from .schemas import AnswerDraft, AnswerSelection, ChunkRecord, RejectedClaim, ValidatedCitation, ValidatedClaim

MIN_QUOTE_CHARS = 8


def normalize(text: str) -> str:
    """Whitespace only: never fold code case, punctuation, Unicode, or negation."""
    return re.sub(r"\s+", " ", text).strip()


def quote_in_text(quote: str, text: str) -> bool:
    needle = normalize(quote)
    return len(needle) >= MIN_QUOTE_CHARS and needle in normalize(text)


def validate_claims(
    draft: AnswerDraft | AnswerSelection, context: dict[str, ChunkRecord]
) -> tuple[list[ValidatedClaim], list[ValidatedCitation], list[RejectedClaim]]:
    claims: list[ValidatedClaim] = []
    citations: list[ValidatedCitation] = []
    rejected: list[RejectedClaim] = []
    marker_for: dict[tuple[str, str], int] = {}
    span_maps = {cid: {s.span_id: s for s in evidence_spans(c)} for cid, c in context.items()}

    for claim in draft.claims:
        reasons: list[str] = []
        pending: list[tuple[ChunkRecord, str]] = []
        if not claim.evidence:
            reasons.append("claim has no evidence")
        for ev in claim.evidence:
            chunk = context.get(ev.chunk_id)
            if chunk is None:
                reasons.append(f"chunk id '{ev.chunk_id[:40]}' was not in the supplied evidence")
            elif isinstance(draft, AnswerSelection):
                span = span_maps[ev.chunk_id].get(ev.span_id)
                if span is None:
                    reasons.append(f"passage id '{ev.span_id[:40]}' was not supplied for chunk {ev.chunk_id}")
                else:
                    pending.append((chunk, span.text))
            elif not quote_in_text(ev.quote, chunk.text):
                reasons.append(f"quote not found in chunk {ev.chunk_id}")
            else:
                pending.append((chunk, ev.quote))
        if reasons:
            rejected.append(RejectedClaim(text=claim.text, reasons=reasons))
            continue
        markers = []
        for chunk, quote in pending:
            key = (chunk.chunk_id, normalize(quote))
            if key not in marker_for:
                marker_for[key] = len(citations) + 1
                citations.append(ValidatedCitation(
                    marker=marker_for[key], chunk_id=chunk.chunk_id, quote=quote.strip(),
                    url=chunk.citation_url, title=chunk.title, section=chunk.section_label,
                ))
            if marker_for[key] not in markers:
                markers.append(marker_for[key])
        claims.append(ValidatedClaim(text=claim.text, citations=markers))
    return claims, citations, rejected
