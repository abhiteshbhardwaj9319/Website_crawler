"""Retrieval modes and evidence-context selection for exactly one corpus.

Modes (compared in docs/evaluation.md):
- dense:          cosine similarity over local bge-small embeddings in Qdrant.
- bm25:           BM25Okapi over the same chunks (title + heading path + text).
- hybrid:         reciprocal rank fusion (RRF) of dense and BM25 rankings.
- hybrid_rerank:  hybrid candidates re-scored by a local cross-encoder.

Scores are ranking signals only. RRF and cosine values are not calibrated probabilities
and are never used as answerability confidence.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import time

from .chunk import embedding_text
from .config import RetrievalMode, Settings
from .embeddings import Embedder, get_embedder, get_reranker
from .errors import ErrorCode, RagError
from .index import CorpusStore
from .schemas import ChunkRecord, RetrievedChunk, SiteRecord


def _check_isolation(chunk: ChunkRecord, site_id: str, corpus_id: str) -> None:
    if chunk.site_id != site_id or chunk.corpus_id != corpus_id:
        raise RagError(
            ErrorCode.INDEX_INCOMPATIBLE,
            "Retrieved evidence does not belong to the selected website corpus; refusing to answer.",
            stage="retrieve",
            details={"expected": [site_id, corpus_id], "got": [chunk.site_id, chunk.corpus_id]},
        )


def rrf_fuse(rankings: dict[str, list[ChunkRecord]], k: int) -> list[tuple[ChunkRecord, float, dict[str, int]]]:
    scores: dict[str, float] = {}
    ranks: dict[str, dict[str, int]] = {}
    by_id: dict[str, ChunkRecord] = {}
    for name, ranked in rankings.items():
        for rank, chunk in enumerate(ranked, start=1):
            by_id[chunk.chunk_id] = chunk
            scores[chunk.chunk_id] = scores.get(chunk.chunk_id, 0.0) + 1.0 / (k + rank)
            ranks.setdefault(chunk.chunk_id, {})[name] = rank
    ordered = sorted(scores, key=lambda cid: (-scores[cid], cid))
    return [(by_id[cid], scores[cid], ranks[cid]) for cid in ordered]


@dataclass
class Retriever:
    settings: Settings
    site: SiteRecord
    store: CorpusStore
    embedder: Embedder | None = None
    reranker: object | None = None
    timings: dict[str, float] = field(default_factory=dict)

    def _embedder(self) -> Embedder:
        if self.embedder is None:
            self.embedder = get_embedder(
                self.settings.embedding_model, str(self.settings.model_cache_dir), self.settings.embedding_batch_size
            )
        return self.embedder

    def _reranker(self):  # noqa: ANN202
        if self.reranker is None:
            self.reranker = get_reranker(self.settings.reranker_model, str(self.settings.model_cache_dir))
        return self.reranker

    def search(self, question: str, mode: RetrievalMode, k: int | None = None) -> list[RetrievedChunk]:
        k = k or self.settings.candidate_k
        site_id, corpus_id = self.store.site_id, self.store.corpus_id
        if site_id != self.site.site_id:
            raise RagError(ErrorCode.INDEX_INCOMPATIBLE, "Store/site mismatch.", stage="retrieve")

        dense: list[tuple[ChunkRecord, float]] = []
        lexical: list[tuple[ChunkRecord, float]] = []
        self.timings = {}
        if mode in ("dense", "hybrid", "hybrid_rerank"):
            t = time.perf_counter()
            embedder = self._embedder()
            self.timings['embedding_model_load_ms'] = (time.perf_counter() - t) * 1000
            t = time.perf_counter()
            vector = embedder.embed_query(question)
            self.timings['query_embedding_ms'] = (time.perf_counter() - t) * 1000
            t = time.perf_counter()
            dense = self.store.dense_search(vector, k)
            self.timings['dense_ms'] = (time.perf_counter() - t) * 1000
        if mode in ("bm25", "hybrid", "hybrid_rerank"):
            t = time.perf_counter()
            lexical = self.store.bm25_search(question, k)
            self.timings['lexical_ms'] = (time.perf_counter() - t) * 1000
        for chunk, _ in (*dense, *lexical):
            _check_isolation(chunk, site_id, corpus_id)

        if mode == "dense":
            return [RetrievedChunk(chunk=c, retriever="dense", rank=i, score=s, component_ranks={"dense": i})
                    for i, (c, s) in enumerate(dense, start=1)]
        if mode == "bm25":
            return [RetrievedChunk(chunk=c, retriever="bm25", rank=i, score=s, component_ranks={"bm25": i})
                    for i, (c, s) in enumerate(lexical, start=1)]

        fused = rrf_fuse({"dense": [c for c, _ in dense], "bm25": [c for c, _ in lexical]}, self.settings.rrf_k)[:k]
        if mode == "hybrid":
            return [RetrievedChunk(chunk=c, retriever="hybrid", rank=i, score=s, component_ranks=r)
                    for i, (c, s, r) in enumerate(fused, start=1)]

        depth = min(len(fused), self.settings.rerank_k)
        if not depth:
            return []
        t = time.perf_counter()
        reranker = self._reranker()
        self.timings['reranker_model_load_ms'] = (time.perf_counter() - t) * 1000
        t = time.perf_counter()
        scores = reranker.score(question, [embedding_text(c) for c, _, _ in fused[:depth]])
        self.timings['rerank_ms'] = (time.perf_counter() - t) * 1000
        order = sorted(range(depth), key=lambda i: -scores[i]) + list(range(depth, len(fused)))
        out = []
        for new_rank, i in enumerate(order, start=1):
            c, _, r = fused[i]
            out.append(RetrievedChunk(chunk=c, retriever="hybrid_rerank", rank=new_rank, score=scores[i] if i < depth else 0,
                                      component_ranks={**r, "hybrid": i + 1}))
        return out


def select_context(
    retrieved: list[RetrievedChunk], max_chunks: int, token_budget: int, site_id: str, corpus_id: str,
    policy: str = 'ranked', corpus_chunks: list[ChunkRecord] | None = None,
) -> list[RetrievedChunk]:
    """Take ranked chunks in order within the chunk and token budgets (no truncation mid-chunk)."""
    for item in retrieved:
        _check_isolation(item.chunk, site_id, corpus_id)
    if policy == 'diverse':
        # Soft section diversity: allow two chunks per section first, then fill remaining space.
        first, deferred, sections = [], [], {}
        for item in retrieved:
            key = (item.chunk.source_url, item.chunk.anchor, tuple(item.chunk.heading_path))
            sections[key] = sections.get(key, 0) + 1
            (first if sections[key] <= 2 else deferred).append(item)
        retrieved = first + deferred
    elif policy == 'neighbors':
        by_position = {(c.page_id, c.ordinal): c for c in corpus_chunks or []}
        expanded = []
        for item in retrieved:
            expanded.append(item)
            if item.rank <= 3:
                for offset in (-1, 1):
                    c = by_position.get((item.chunk.page_id, item.chunk.ordinal + offset))
                    if c and c.anchor == item.chunk.anchor and c.heading_path == item.chunk.heading_path:
                        _check_isolation(c, site_id, corpus_id)
                        expanded.append(RetrievedChunk(chunk=c, retriever='neighbor', rank=item.rank,
                                                       score=item.score, component_ranks={'neighbor_of_rank': item.rank}))
        retrieved = expanded
    elif policy != 'ranked':
        raise RagError(ErrorCode.CONFIG_INVALID, 'Unknown context policy.', stage='context')
    selected: list[RetrievedChunk] = []
    used = 0
    seen: set[str] = set()
    selected_text: list[str] = []
    for item in retrieved:
        _check_isolation(item.chunk, site_id, corpus_id)
        if item.chunk.chunk_id in seen:
            continue
        normalized = ' '.join(item.chunk.text.split())
        if policy != 'ranked' and any(normalized in text for text in selected_text):
            continue
        if used + item.chunk.token_count > token_budget:
            continue  # a later, smaller chunk may still fit
        selected.append(item)
        seen.add(item.chunk.chunk_id)
        selected_text.append(normalized)
        used += item.chunk.token_count
        if len(selected) >= max_chunks:
            break
    return selected
