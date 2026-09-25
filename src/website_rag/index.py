"""Per-corpus storage: Qdrant dense collection, chunk manifest, BM25 lexical index.

Isolation model: every corpus (one ingestion of one site) has its own Qdrant collection,
its own chunks.jsonl (the lexical index source), and its own manifest. A query opens
exactly one corpus. Dense search additionally filters on corpus_id and every returned
payload is re-validated against the selected site/corpus, so a mixed or corrupted
collection fails loudly instead of leaking evidence.
"""

from __future__ import annotations

import hashlib
import json
import re
import uuid
from functools import lru_cache
from pathlib import Path

import numpy as np
from qdrant_client import QdrantClient, models

from .config import Settings
from .errors import ErrorCode, RagError
from .schemas import ChunkRecord, IndexManifest, IndexStatus, SiteRecord

_STOPWORDS = set(
    "a an and are as at be by can do does for from how i if in is it its of on or that the this to was "
    "what when where which who why will with you your me my we our not no".split()
)
_TOKEN = re.compile(r"[a-z0-9_]+")


def lexical_tokens(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in _STOPWORDS]


def point_id(chunk_id: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"chunk:{chunk_id}"))


def collection_name(corpus_id: str) -> str:
    return "corpus_" + re.sub(r"[^A-Za-z0-9_-]", "_", corpus_id)


def config_fingerprint(settings: Settings, embedding_model: str, extra: dict | None = None) -> str:
    from .chunk import CHUNKER_VERSION
    from .extract import EXTRACTOR_VERSION

    payload = {
        "embedding_model": embedding_model,
        "chunker": CHUNKER_VERSION,
        "extractor": EXTRACTOR_VERSION,
        "target": settings.chunk_target_tokens,
        "overlap": settings.chunk_overlap_tokens,
        **(extra or {}),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


_CLIENTS: dict[str, QdrantClient] = {}


def get_qdrant(path: Path) -> QdrantClient:
    """One local-mode client per storage path per process (local mode locks the directory)."""
    key = str(path.resolve())
    if key not in _CLIENTS:
        path.mkdir(parents=True, exist_ok=True)
        try:
            _CLIENTS[key] = QdrantClient(path=key)
        except RuntimeError as exc:
            raise RagError(
                ErrorCode.INDEX_INCOMPLETE,
                "The local vector store is locked by another process.",
                hint="Close other `rag` processes (e.g. an open `rag chat`) and retry.",
                stage="index",
            ) from exc
    return _CLIENTS[key]


def close_qdrant() -> None:
    for client in _CLIENTS.values():
        try:
            client.close()
        except Exception:  # noqa: BLE001 - closing is best effort
            pass
    _CLIENTS.clear()


class CorpusStore:
    def __init__(self, settings: Settings, site_id: str, corpus_id: str) -> None:
        self.settings = settings
        self.site_id = site_id
        self.corpus_id = corpus_id
        self.dir = settings.sites_dir / site_id / "corpora" / corpus_id
        self.collection = collection_name(corpus_id)

    # ------------------------------------------------------------------ files
    @property
    def manifest_path(self) -> Path:
        return self.dir / "manifest.json"

    def save_manifest(self, manifest: IndexManifest) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        tmp = self.manifest_path.with_suffix(".tmp")
        tmp.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
        tmp.replace(self.manifest_path)

    def load_manifest(self) -> IndexManifest:
        if not self.manifest_path.exists():
            raise RagError(
                ErrorCode.INDEX_INCOMPLETE,
                f"Corpus {self.corpus_id} has no manifest.",
                hint="Re-run ingestion for this site.",
                stage="index",
            )
        return IndexManifest.model_validate_json(self.manifest_path.read_text(encoding="utf-8"))

    def save_chunks(self, chunks: list[ChunkRecord], vectors: np.ndarray) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        with (self.dir / "chunks.jsonl").open("w", encoding="utf-8") as fh:
            for c in chunks:
                fh.write(c.model_dump_json() + "\n")
        np.save(self.dir / "vectors.npy", vectors)

    def load_chunks(self) -> list[ChunkRecord]:
        return _load_chunks_cached(str(self.dir / "chunks.jsonl"))

    def load_vectors(self) -> np.ndarray | None:
        path = self.dir / "vectors.npy"
        return np.load(path) if path.exists() else None

    # ------------------------------------------------------------------ dense index
    def build_collection(self, chunks: list[ChunkRecord], vectors: np.ndarray, batch: int = 128) -> None:
        client = get_qdrant(self.settings.qdrant_dir)
        if client.collection_exists(self.collection):
            client.delete_collection(self.collection)
        client.create_collection(
            self.collection,
            vectors_config=models.VectorParams(size=int(vectors.shape[1]), distance=models.Distance.COSINE),
        )
        for start in range(0, len(chunks), batch):
            part = chunks[start : start + batch]
            client.upsert(
                self.collection,
                points=[
                    models.PointStruct(
                        id=point_id(c.chunk_id),
                        vector=vectors[start + i].tolist(),
                        payload=json.loads(c.model_dump_json()),
                    )
                    for i, c in enumerate(part)
                ],
            )

    def point_count(self) -> int:
        client = get_qdrant(self.settings.qdrant_dir)
        if not client.collection_exists(self.collection):
            return 0
        return client.count(self.collection, exact=True).count

    def drop_collection(self) -> None:
        client = get_qdrant(self.settings.qdrant_dir)
        if client.collection_exists(self.collection):
            client.delete_collection(self.collection)

    # ------------------------------------------------------------------ readiness
    def verify_ready(self, site: SiteRecord, embedding_model: str) -> IndexManifest:
        manifest = self.load_manifest()
        if manifest.site_id != site.site_id or manifest.corpus_id != self.corpus_id:
            raise RagError(
                ErrorCode.INDEX_INCOMPATIBLE,
                "Corpus manifest does not belong to the selected site.",
                stage="index",
                details={"manifest_site": manifest.site_id, "selected_site": site.site_id},
            )
        if manifest.status != IndexStatus.COMPLETE:
            raise RagError(
                ErrorCode.INDEX_INCOMPLETE,
                f"Corpus {self.corpus_id} is {manifest.status.value}, not complete.",
                hint=f"Re-run `rag ingest --site {site.number}`.",
                stage="index",
            )
        if manifest.embedding_model != embedding_model:
            raise RagError(
                ErrorCode.INDEX_INCOMPATIBLE,
                f"Corpus was embedded with {manifest.embedding_model}, but the configured "
                f"embedding model is {embedding_model}. Refusing to mix embedding spaces.",
                hint=f"Set RAG_EMBEDDING_MODEL={manifest.embedding_model} or re-ingest the site.",
                stage="index",
            )
        count = self.point_count()
        if count != manifest.chunk_count:
            raise RagError(
                ErrorCode.INDEX_INCOMPLETE,
                f"Vector collection has {count} points but the manifest expects {manifest.chunk_count}.",
                hint=f"Re-run `rag ingest --site {site.number}`.",
                stage="index",
            )
        return manifest

    # ------------------------------------------------------------------ search
    def dense_search(self, vector: np.ndarray, limit: int) -> list[tuple[ChunkRecord, float]]:
        client = get_qdrant(self.settings.qdrant_dir)
        response = client.query_points(
            self.collection,
            query=vector.tolist(),
            limit=limit,
            with_payload=True,
            query_filter=models.Filter(
                must=[
                    models.FieldCondition(key="corpus_id", match=models.MatchValue(value=self.corpus_id)),
                    models.FieldCondition(key="site_id", match=models.MatchValue(value=self.site_id)),
                ]
            ),
        )
        return [(ChunkRecord.model_validate(p.payload), float(p.score)) for p in response.points]

    def bm25_search(self, query: str, limit: int) -> list[tuple[ChunkRecord, float]]:
        chunks = self.load_chunks()
        bm25 = _bm25_cached(str(self.dir / "chunks.jsonl"))
        scores = bm25.get_scores(lexical_tokens(query))
        order = np.argsort(-scores)[:limit]
        return [(chunks[i], float(scores[i])) for i in order if scores[i] > 0]


@lru_cache(maxsize=8)
def _load_chunks_cached(path: str) -> list[ChunkRecord]:
    p = Path(path)
    if not p.exists():
        raise RagError(ErrorCode.INDEX_INCOMPLETE, "Chunk manifest is missing for this corpus.", stage="index")
    return [ChunkRecord.model_validate_json(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


@lru_cache(maxsize=8)
def _bm25_cached(path: str):  # noqa: ANN202
    from rank_bm25 import BM25Okapi

    from .chunk import embedding_text

    chunks = _load_chunks_cached(path)
    return BM25Okapi([lexical_tokens(embedding_text(c)) for c in chunks])


def clear_caches() -> None:
    _load_chunks_cached.cache_clear()
    _bm25_cached.cache_clear()
