"""Typed data contracts shared by ingestion, retrieval, generation, and evaluation."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field


def utcnow() -> datetime:
    return datetime.now(UTC)


# --------------------------------------------------------------------------- sites


class SiteStatus(StrEnum):
    REGISTERED = "registered"  # known, never ingested
    INGESTING = "ingesting"  # an ingestion run is in progress (or was interrupted)
    READY = "ready"  # has a complete active corpus
    FAILED = "failed"  # last ingestion failed and no usable corpus exists


class CrawlLimits(BaseModel):
    max_pages: int = Field(default=40, ge=1, le=500)
    max_depth: int = Field(default=4, ge=0, le=20)
    max_bytes_per_page: int = Field(default=3_000_000, ge=10_000)
    timeout_s: float = Field(default=20.0, gt=0, le=120)
    max_total_s: float = Field(default=300.0, gt=0, le=3600)
    delay_s: float = Field(default=0.5, ge=0, le=30)
    concurrency: int = Field(default=2, ge=1, le=8)
    max_retries: int = Field(default=2, ge=0, le=5)
    min_page_words: int = Field(default=80, ge=0)
    max_sitemaps: int = Field(default=8, ge=0, le=50)
    max_sitemap_depth: int = Field(default=2, ge=0, le=5)
    max_sitemap_urls: int = Field(default=3000, ge=1, le=10000)


class SiteRecord(BaseModel):
    site_id: str
    number: int = Field(ge=1)
    display_name: str
    seed_url: str
    allowed_host: str
    allowed_path_prefix: str
    additional_path_prefixes: list[str] = Field(default_factory=list)
    seed_urls: list[str] = Field(default_factory=list)
    sitemap_urls: list[str] = Field(default_factory=list)
    scope_version: int = 1
    crawl_complete: bool = False
    crawl_stop_reason: str | None = None
    pending_urls: int = 0
    description: str = ""
    crawl: CrawlLimits = Field(default_factory=CrawlLimits)
    exclude_patterns: list[str] = Field(default_factory=list)
    status: SiteStatus = SiteStatus.REGISTERED
    active_corpus_id: str | None = None
    pending_corpus_id: str | None = None
    accepted_pages: int = 0
    chunk_count: int = 0
    embedding_model: str | None = None
    last_error: str | None = None
    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)

    @property
    def is_queryable(self) -> bool:
        return self.active_corpus_id is not None


# --------------------------------------------------------------------------- crawl


class FetchOutcome(StrEnum):
    ACCEPTED = "accepted"
    SKIPPED = "skipped"
    FAILED = "failed"
    DEFERRED = "deferred"


class Section(BaseModel):
    heading_path: list[str]
    anchor: str | None = None  # only set when the id exists in the page HTML
    text: str


class PageRecord(BaseModel):
    page_id: str
    site_id: str
    corpus_id: str
    requested_url: str
    final_url: str
    title: str
    fetched_at: datetime
    content_hash: str
    depth: int
    word_count: int
    sections: list[Section]


class CrawlLogEntry(BaseModel):
    url: str
    final_url: str | None = None
    outcome: FetchOutcome
    reason: str
    status_code: int | None = None
    depth: int = 0
    bytes: int = 0
    elapsed_ms: int = 0
    at: datetime = Field(default_factory=utcnow)


# --------------------------------------------------------------------------- index


class ChunkRecord(BaseModel):
    chunk_id: str
    site_id: str
    corpus_id: str
    page_id: str
    source_url: str
    title: str
    heading_path: list[str]
    anchor: str | None
    ordinal: int  # position of the chunk within its page
    text: str  # evidence text shown to the model and quoted
    token_count: int
    content_hash: str
    fetched_at: datetime

    @property
    def section_label(self) -> str:
        return " > ".join(self.heading_path) if self.heading_path else self.title

    @property
    def citation_url(self) -> str:
        return f"{self.source_url}#{self.anchor}" if self.anchor else self.source_url


class IndexStatus(StrEnum):
    BUILDING = "building"
    COMPLETE = "complete"
    FAILED = "failed"


class IndexManifest(BaseModel):
    corpus_id: str
    site_id: str
    status: IndexStatus
    seed_url: str
    allowed_host: str
    allowed_path_prefix: str
    collection_name: str
    accepted_pages: int = 0
    chunk_count: int = 0
    embedding_model: str
    embedding_dim: int
    chunker_version: str
    extractor_version: str
    chunk_target_tokens: int
    chunk_overlap_tokens: int
    config_fingerprint: str
    scope_fingerprint: str = "legacy"
    scope: dict = Field(default_factory=dict)
    started_at: datetime = Field(default_factory=utcnow)
    completed_at: datetime | None = None
    error: str | None = None


# --------------------------------------------------------------------------- retrieval


class RetrievedChunk(BaseModel):
    chunk: ChunkRecord
    retriever: str  # dense | bm25 | hybrid | hybrid_rerank
    rank: int
    score: float  # raw retriever score; not a calibrated probability
    component_ranks: dict[str, int] = Field(default_factory=dict)


# --------------------------------------------------------------------------- generation


class EvidenceQuote(BaseModel):
    chunk_id: str = Field(description="ID of one supplied evidence chunk, copied exactly.")
    quote: str = Field(
        description="Exact contiguous excerpt copied from that chunk that supports the claim."
    )


class DraftClaim(BaseModel):
    text: str = Field(description="One factual statement made in the answer.")
    evidence: list[EvidenceQuote] = Field(
        description="One or more supporting chunk IDs with exact quotations."
    )


class AnswerDraft(BaseModel):
    """Schema the language model must return (strict structured output)."""

    status: Literal["answered", "partially_answered", "insufficient_evidence"] = Field(
        description="answered: fully supported; partially_answered: only part is supported; "
        "insufficient_evidence: the evidence does not answer the question."
    )
    answer: str = Field(description="Concise answer using only the evidence. Empty if insufficient.")
    claims: list[DraftClaim] = Field(description="Every factual claim in the answer with evidence.")
    missing_information: str = Field(
        description="What the question asked that the evidence does not establish; empty if none."
    )
    premise_issue: str = Field(
        description="If the question contains a premise the evidence contradicts or does not "
        "support, explain it here; empty otherwise."
    )


class EvidenceSelection(BaseModel):
    chunk_id: str = Field(description="Exact supplied chunk ID.")
    span_id: str = Field(description="Exact supplied passage ID from that chunk.")


class SelectedClaim(BaseModel):
    text: str = Field(description="One concise factual statement supported by the selected passages.")
    evidence: list[EvidenceSelection]


class AnswerSelection(BaseModel):
    """V2 provider contract. No separate unchecked answer or premise prose."""

    status: Literal["answered", "partially_answered", "insufficient_evidence"]
    claims: list[SelectedClaim]


class ValidatedCitation(BaseModel):
    marker: int
    chunk_id: str
    quote: str
    url: str
    title: str
    section: str


class ValidatedClaim(BaseModel):
    text: str
    citations: list[int]  # markers into AnswerResult.citations


class RejectedClaim(BaseModel):
    text: str
    reasons: list[str]


AnswerStatus = Literal["answered", "partially_answered", "insufficient_evidence", "error"]


class ProviderAttempt(BaseModel):
    provider: str
    model: str
    attempt: int
    outcome: Literal["success", "error"]
    error_code: str | None = None
    message: str | None = None
    latency_ms: int = 0
    request_id: str | None = None
    retry_after_s: float | None = None


class AnswerResult(BaseModel):
    run_id: str
    question: str
    site_id: str | None
    site_number: int | None
    site_name: str | None
    corpus_id: str | None
    status: AnswerStatus
    answer: str = ""
    claims: list[ValidatedClaim] = Field(default_factory=list)
    citations: list[ValidatedCitation] = Field(default_factory=list)
    rejected_claims: list[RejectedClaim] = Field(default_factory=list)
    missing_information: str = ""
    premise_issue: str = ""
    retrieval_mode: str | None = None
    retrieved: list[RetrievedChunk] = Field(default_factory=list)
    context_chunk_ids: list[str] = Field(default_factory=list)
    provider: str | None = None
    model: str | None = None
    fallback_reason: str | None = None
    provider_attempts: list[ProviderAttempt] = Field(default_factory=list)
    error: dict | None = None
    prompt_version: str | None = None
    usage: dict = Field(default_factory=dict)
    latency_ms: int = 0
    created_at: datetime = Field(default_factory=utcnow)


# --------------------------------------------------------------------------- accounting


Phase = Literal["ingestion", "query", "evaluation"]
Measurement = Literal["provider_reported", "local_count", "unknown"]


class UsageEvent(BaseModel):
    event_id: str
    run_id: str
    operation_id: str  # one logical operation (e.g. generation for a query)
    attempt: int  # 1-based attempt within the operation; retries are separate events
    phase: Phase
    site_id: str | None
    corpus_id: str | None
    provider: str  # openai | groq | local
    model: str
    operation: Literal["generation", "embedding", "rerank"]
    input_tokens: int | None = None
    cached_input_tokens: int | None = None
    output_tokens: int | None = None
    reasoning_tokens: int | None = None
    measurement: Measurement
    cost_usd: float | None = None  # None means unknown, never assumed zero
    pricing_version: str | None = None
    outcome: Literal["success", "error"]
    error_code: str | None = None
    latency_ms: int = 0
    request_id: str | None = None
    at: datetime = Field(default_factory=utcnow)
