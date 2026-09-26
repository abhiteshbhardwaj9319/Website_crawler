"""Explicit runtime configuration.

Values come from environment variables or an ignored `.env` file. Provider keys are
held as SecretStr so they never appear in reprs, logs, or JSON output.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import AliasChoices, Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "config"

ProviderName = Literal["openai", "groq"]
ProviderMode = Literal["openai", "groq", "auto"]
RetrievalMode = Literal["dense", "bm25", "hybrid", "hybrid_rerank"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", env_prefix="RAG_", extra="ignore"
    )

    data_dir: Path = Path("data")

    # Provider credentials use the providers' conventional names.
    openai_api_key: SecretStr | None = Field(
        default=None, validation_alias=AliasChoices("OPENAI_API_KEY", "RAG_OPENAI_API_KEY")
    )
    groq_api_key: SecretStr | None = Field(
        default=None, validation_alias=AliasChoices("GROQ_API_KEY", "RAG_GROQ_API_KEY")
    )

    # Generation (checked against official model pages on 2026-09-26).
    provider: ProviderMode = "auto"
    openai_model: str = "gpt-4.1-mini-2025-04-14"
    groq_model: str = "openai/gpt-oss-120b"
    groq_reasoning_effort: Literal["low", "medium", "high"] = "low"
    temperature: float = Field(default=0.0, ge=0.0, le=1.0)
    max_output_tokens: int = Field(default=1200, ge=128, le=8000)

    # Provider transport policy: the application owns retries (SDK retries disabled).
    request_timeout_s: float = Field(default=45.0, gt=0, le=300)
    max_attempts_per_provider: int = Field(default=3, ge=1, le=5)
    retry_budget_s: float = Field(default=60.0, gt=0, le=600)
    max_retry_after_s: float = Field(default=20.0, ge=0, le=120)

    # Embeddings / reranking (local ONNX; pinned per corpus in the index manifest).
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    reranker_model: str = "Xenova/ms-marco-MiniLM-L-6-v2"
    embedding_batch_size: int = Field(default=32, ge=1, le=256)

    # Retrieval and context budget.
    retrieval_mode: RetrievalMode = "hybrid"  # selected on the dev split; see docs/evaluation.md
    candidate_k: int = Field(default=20, ge=1, le=100)
    context_max_chunks: int = Field(default=10, ge=1, le=20)
    evidence_token_budget: int = Field(default=3000, ge=200, le=20000)
    rrf_k: int = Field(default=60, ge=1, le=1000)

    # Chunking (starting hypotheses, not tuned optima).
    chunk_target_tokens: int = Field(default=320, ge=50, le=1000)
    chunk_overlap_tokens: int = Field(default=48, ge=0, le=300)
    max_index_chunks: int = Field(default=30000, ge=100, le=100000)

    # Spending guards for evaluation / batch runs.
    eval_max_requests: int = Field(default=60, ge=1)
    eval_max_cost_usd: float = Field(default=0.50, ge=0)

    # Debug: include sanitized stack traces in local traces.
    debug: bool = False

    @field_validator("chunk_overlap_tokens")
    @classmethod
    def _overlap_smaller_than_target(cls, v: int, info):  # noqa: ANN001
        target = info.data.get("chunk_target_tokens", 320)
        if v >= target:
            raise ValueError("chunk_overlap_tokens must be smaller than chunk_target_tokens")
        return v

    # ---- derived paths -------------------------------------------------
    @property
    def registry_path(self) -> Path:
        return self.data_dir / "registry.json"

    @property
    def sites_dir(self) -> Path:
        return self.data_dir / "sites"

    @property
    def qdrant_dir(self) -> Path:
        return self.data_dir / "qdrant"

    @property
    def runs_dir(self) -> Path:
        return self.data_dir / "runs"

    @property
    def traces_dir(self) -> Path:
        return self.data_dir / "traces"

    @property
    def ledger_path(self) -> Path:
        return self.data_dir / "usage" / "ledger.jsonl"

    @property
    def model_cache_dir(self) -> Path:
        return self.data_dir / "cache" / "models"

    def key_for(self, provider: ProviderName) -> SecretStr | None:
        return self.openai_api_key if provider == "openai" else self.groq_api_key

    def model_for(self, provider: ProviderName) -> str:
        return self.openai_model if provider == "openai" else self.groq_model


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
