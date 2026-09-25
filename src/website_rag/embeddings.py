"""Local embedding and reranking models (ONNX via fastembed; no API calls).

The embedding model is pinned per corpus in its manifest. Queries are always embedded with
the manifest's model; a mismatch is a hard error rather than a silent substitution.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Protocol

import numpy as np


class Embedder(Protocol):
    model_name: str
    dim: int

    def embed_documents(self, texts: list[str]) -> np.ndarray: ...

    def embed_query(self, text: str) -> np.ndarray: ...


class FastEmbedder:
    def __init__(self, model_name: str, cache_dir: Path, batch_size: int = 32) -> None:
        from fastembed import TextEmbedding

        cache_dir.mkdir(parents=True, exist_ok=True)
        self.model_name = model_name
        self.batch_size = batch_size
        self._model = TextEmbedding(model_name=model_name, cache_dir=str(cache_dir))
        self.dim = int(self.embed_query("dimension probe").shape[0])

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        vectors = list(self._model.passage_embed(texts, batch_size=self.batch_size))
        return np.asarray(vectors, dtype=np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        return np.asarray(next(iter(self._model.query_embed(text))), dtype=np.float32)


class FastReranker:
    def __init__(self, model_name: str, cache_dir: Path) -> None:
        from fastembed.rerank.cross_encoder import TextCrossEncoder

        cache_dir.mkdir(parents=True, exist_ok=True)
        self.model_name = model_name
        self._model = TextCrossEncoder(model_name=model_name, cache_dir=str(cache_dir))

    def score(self, query: str, documents: list[str]) -> list[float]:
        return [float(s) for s in self._model.rerank(query, documents)]


@lru_cache(maxsize=4)
def get_embedder(model_name: str, cache_dir: str, batch_size: int = 32) -> FastEmbedder:
    return FastEmbedder(model_name, Path(cache_dir), batch_size)


@lru_cache(maxsize=2)
def get_reranker(model_name: str, cache_dir: str) -> FastReranker:
    return FastReranker(model_name, Path(cache_dir))
