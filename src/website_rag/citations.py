"""Deterministic validation of model-produced claims against the supplied context.

Checks (structural, not semantic):
- every cited chunk_id must be one of the chunks actually sent to the model for this run;
- every quote must occur in that chunk's text after normalization (Unicode NFKC,
  typographic quotes/dashes folded, markdown code fences/backticks removed, whitespace
  collapsed, case-folded). A quote may use "..." to join excerpts that occur in order;
- quotes shorter than MIN_QUOTE_CHARS after normalization are rejected as non-evidence.
A claim with any invalid evidence is rejected as a whole. Source URLs, titles and sections
are rendered from indexed metadata, never from model output.

A valid ID and quote prove the passage was supplied and quoted, not that it entails the
claim; semantic support is judged in evaluation.
"""

from __future__ import annotations

import re
import unicodedata

from .schemas import AnswerDraft, ChunkRecord, RejectedClaim, ValidatedCitation, ValidatedClaim

MIN_QUOTE_CHARS = 8
_FOLD = str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"', "–": "-",
                       "—": "-", " ": " ", "…": "..."})


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).translate(_FOLD)
    text = text.replace("```", " ").replace("`", "")
    text = re.sub(r"\s+", " ", text)
    return text.strip().casefold()


def quote_in_text(quote: str, text: str) -> bool:
    haystack = normalize(text)
    parts = [p.strip(" .,;:") for p in normalize(quote).split("...")]
    parts = [p for p in parts if p]
    if not parts or sum(len(p) for p in parts) < MIN_QUOTE_CHARS:
        return False
    pos = 0
    for part in parts:
        found = haystack.find(part, pos)
        if found < 0:
            return False
        pos = found + len(part)
    return True


def validate_claims(
    draft: AnswerDraft, context: dict[str, ChunkRecord]
) -> tuple[list[ValidatedClaim], list[ValidatedCitation], list[RejectedClaim]]:
    claims: list[ValidatedClaim] = []
    citations: list[ValidatedCitation] = []
    rejected: list[RejectedClaim] = []
    marker_for: dict[tuple[str, str], int] = {}

    for claim in draft.claims:
        reasons: list[str] = []
        pending: list[tuple[ChunkRecord, str]] = []
        if not claim.evidence:
            reasons.append("claim has no evidence")
        for ev in claim.evidence:
            chunk = context.get(ev.chunk_id)
            if chunk is None:
                reasons.append(f"chunk id '{ev.chunk_id[:40]}' was not in the supplied evidence")
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
