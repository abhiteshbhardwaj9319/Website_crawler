"""Section-aware chunking with stable identities.

Chunks never cross section boundaries, so every chunk has exactly one heading path and at
most one real anchor. Oversized sections are split on block boundaries (paragraph, list
item, code block) with a small block-level overlap; a single block larger than the target
is split by lines. The page title and heading path are prefixed to the text used for
embedding and BM25 (not to the evidence text that answers quote).
"""

from __future__ import annotations

import hashlib
import os
from functools import lru_cache

from .schemas import ChunkRecord, PageRecord

CHUNKER_VERSION = "chunk-v1"


@lru_cache(maxsize=1)
def _encoding():
    import tiktoken

    os.environ.setdefault("TIKTOKEN_CACHE_DIR", os.path.join("data", "cache", "tiktoken"))
    return tiktoken.get_encoding("o200k_base")


def count_tokens(text: str) -> int:
    return len(_encoding().encode(text, disallowed_special=()))


def chunk_id_for(site_id: str, source_url: str, ordinal: int, text: str) -> str:
    raw = f"{CHUNKER_VERSION}|{site_id}|{source_url}|{ordinal}|{hashlib.sha256(text.encode()).hexdigest()}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


def embedding_text(chunk: ChunkRecord) -> str:
    return f"{chunk.title} > {chunk.section_label}\n\n{chunk.text}" if chunk.heading_path else f"{chunk.title}\n\n{chunk.text}"


def _split_large_block(block: str, target: int) -> list[str]:
    pieces, current, current_tokens = [], [], 0
    for line in block.splitlines():
        t = count_tokens(line) + 1
        if current and current_tokens + t > target:
            pieces.append("\n".join(current))
            current, current_tokens = [], 0
        current.append(line)
        current_tokens += t
    if current:
        pieces.append("\n".join(current))
    # keep code fences balanced in each piece
    fixed = []
    for piece in pieces:
        if piece.count("```") % 2 == 1:
            piece = piece + "\n```" if piece.lstrip().startswith("```") else "```\n" + piece
        fixed.append(piece)
    return fixed


def split_section(text: str, target: int, overlap: int) -> list[str]:
    blocks: list[str] = []
    for block in text.split("\n\n"):
        block = block.strip()
        if not block:
            continue
        if count_tokens(block) > target:
            blocks.extend(_split_large_block(block, target))
        else:
            blocks.append(block)

    chunks: list[str] = []
    current: list[tuple[str, int]] = []
    tokens = 0
    for block in blocks:
        bt = count_tokens(block)
        if current and tokens + bt > target:
            chunks.append("\n\n".join(b for b, _ in current))
            carried: list[tuple[str, int]] = []
            carried_tokens = 0
            for b, t in reversed(current):
                if carried_tokens + t > overlap:
                    break
                carried.insert(0, (b, t))
                carried_tokens += t
            current, tokens = carried, carried_tokens
        current.append((block, bt))
        tokens += bt
    if current:
        text_out = "\n\n".join(b for b, _ in current)
        if not chunks or text_out not in chunks[-1]:
            chunks.append(text_out)
    return chunks


def chunk_pages(pages: list[PageRecord], target: int, overlap: int) -> list[ChunkRecord]:
    records: list[ChunkRecord] = []
    for page in pages:
        ordinal = 0
        for section in page.sections:
            for piece in split_section(section.text, target, overlap):
                cid = chunk_id_for(page.site_id, page.final_url, ordinal, piece)
                records.append(
                    ChunkRecord(
                        chunk_id=cid,
                        site_id=page.site_id,
                        corpus_id=page.corpus_id,
                        page_id=page.page_id,
                        source_url=page.final_url,
                        title=page.title,
                        heading_path=section.heading_path,
                        anchor=section.anchor,
                        ordinal=ordinal,
                        text=piece,
                        token_count=count_tokens(piece),
                        content_hash=hashlib.sha256(piece.encode("utf-8")).hexdigest(),
                        fetched_at=page.fetched_at,
                    )
                )
                ordinal += 1
    return records
