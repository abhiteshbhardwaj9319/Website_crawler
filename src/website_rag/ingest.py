"""Ingestion pipeline: crawl -> extract -> chunk -> embed -> index -> activate.

A new corpus version is built beside the currently active one. The registry switches
`active_corpus_id` only after the new manifest is COMPLETE and the collection point count
matches, so a failed or interrupted refresh never removes the last usable index.
"""

from __future__ import annotations

import time
import json
import shutil
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Callable

import numpy as np

from .chunk import CHUNKER_VERSION, chunk_pages, count_tokens, embedding_text
from .checkpoint import coverage, scope_fingerprint, scope_identity
from .config import Settings
from .crawl import CrawlResult, crawl_site, save_crawl
from .embeddings import Embedder, get_embedder
from .errors import ErrorCode, RagError
from .extract import EXTRACTOR_VERSION
from .index import CorpusStore, clear_caches, config_fingerprint
from .registry import SiteRegistry
from .schemas import FetchOutcome, IndexManifest, IndexStatus, SiteRecord, SiteStatus, UsageEvent, utcnow
from .tracing import Tracer, new_id
from .usage import Pricing, UsageLedger

ProgressFn = Callable[[str, dict], None]


@dataclass
class IngestReport:
    site: SiteRecord
    corpus_id: str
    accepted_pages: int
    skipped: int
    failed: int
    chunk_count: int
    embedded: int
    reused_embeddings: int
    embedding_tokens: int
    stopped_reason: str
    elapsed_s: float
    skip_reasons: dict[str, int] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    run_id: str = ""
    pending: int = 0
    crawl_complete: bool = False


def _reason_key(reason: str) -> str:
    return "duplicate_content" if reason.startswith("duplicate_of") else reason


def ingest_site(
    site: SiteRecord,
    settings: Settings,
    registry: SiteRegistry,
    tracer: Tracer,
    ledger: UsageLedger,
    progress: ProgressFn | None = None,
    embedder: Embedder | None = None,
    crawl_kwargs: dict | None = None,
    min_pages: int = 1,
    resume: bool = False,
) -> IngestReport:
    progress = progress or (lambda kind, data: None)
    started = time.monotonic()
    corpus_id = f"{site.site_id}--{datetime.now(UTC).strftime('%Y%m%dT%H%M%S')}-{new_id()[:6]}"
    if corpus_id in (site.active_corpus_id, site.pending_corpus_id):
        raise RagError(ErrorCode.INTERNAL, "Corpus id collision; retry ingestion.", stage="ingest")
    store = CorpusStore(settings, site.site_id, corpus_id)
    previous_corpus = site.active_corpus_id
    work_root = settings.sites_dir / site.site_id / 'crawl-work'
    pointer = work_root / 'current.json'
    previous_work = json.loads(pointer.read_text(encoding='utf-8'))['directory'] if resume and pointer.exists() else None
    work_dir = work_root / (previous_work or corpus_id)
    work_dir.mkdir(parents=True, exist_ok=True)
    from .registry import _atomic_write
    _atomic_write(pointer, json.dumps({'directory': work_dir.name}))
    tracer.bind(site_id=site.site_id, corpus_id=corpus_id)

    embedder = embedder or get_embedder(
        settings.embedding_model, str(settings.model_cache_dir), settings.embedding_batch_size
    )
    manifest = IndexManifest(
        corpus_id=corpus_id, site_id=site.site_id, status=IndexStatus.BUILDING,
        seed_url=site.seed_url, allowed_host=site.allowed_host,
        allowed_path_prefix=site.allowed_path_prefix, collection_name=store.collection,
        embedding_model=embedder.model_name, embedding_dim=embedder.dim,
        chunker_version=CHUNKER_VERSION, extractor_version=EXTRACTOR_VERSION,
        chunk_target_tokens=settings.chunk_target_tokens,
        chunk_overlap_tokens=settings.chunk_overlap_tokens,
        config_fingerprint=config_fingerprint(settings, embedder.model_name),
        scope_fingerprint=scope_fingerprint(site), scope=scope_identity(site),
    )
    store.save_manifest(manifest)
    site.pending_corpus_id = corpus_id
    registry.mark_status(site, SiteStatus.INGESTING)

    try:
        with tracer.span("crawl", seed_url=site.seed_url, max_pages=site.crawl.max_pages) as span:
            kwargs = dict(crawl_kwargs or {})
            kwargs.setdefault('checkpoint_dir', work_dir)
            if resume and not previous_work and previous_corpus:
                kwargs.setdefault('legacy_dir', CorpusStore(settings, site.site_id, previous_corpus).dir)
            result: CrawlResult = crawl_site(site, corpus_id, progress=progress, **kwargs)
            save_crawl(result, store.dir)
            if (work_dir / 'frontier.json').exists():
                shutil.copyfile(work_dir / 'frontier.json', store.dir / 'frontier.json')
            skip_reasons: dict[str, int] = {}
            for e in result.log:
                if e.outcome != FetchOutcome.ACCEPTED:
                    skip_reasons[_reason_key(e.reason)] = skip_reasons.get(_reason_key(e.reason), 0) + 1
            span.set(accepted=len(result.pages), log_entries=len(result.log),
                     stopped_reason=result.stopped_reason, skip_reasons=skip_reasons)
            if len(result.pages) < min_pages:
                js = skip_reasons.get("likely_javascript_rendered", 0)
                raise RagError(
                    ErrorCode.LOW_CONTENT,
                    f"No usable content pages were extracted from {site.seed_url} "
                    f"({len(result.log)} URLs examined).",
                    hint=(
                        "The site appears to render content with JavaScript, which this static-HTML "
                        "crawler does not execute." if js else
                        "Check that the URL points to a public HTML section with text content; "
                        "see `rag trace <run-id>` for per-URL reasons."
                    ),
                    stage="crawl",
                    details={"skip_reasons": skip_reasons},
                )
            # A partial transport failure must not turn old accepted pages into deletions.
            if previous_corpus and any(e.outcome == FetchOutcome.FAILED for e in result.log):
                raise RagError(ErrorCode.FETCH_FAILED, 'Crawl has failed URLs; previous index remains active.',
                               hint='Inspect coverage, then ingest --resume to retry failed URLs.', stage='crawl')
        progress("stage", {"name": "chunk"})
        with tracer.span("chunk", target=settings.chunk_target_tokens, overlap=settings.chunk_overlap_tokens) as span:
            chunks = []
            for page in result.pages:
                batch = chunk_pages([page], settings.chunk_target_tokens, settings.chunk_overlap_tokens)
                if len(chunks) + len(batch) > settings.max_index_chunks:
                    raise RagError(ErrorCode.CONFIG_INVALID, 'Index chunk limit exceeded; previous index remains active.',
                                   hint='Narrow the crawl scope or explicitly raise RAG_MAX_INDEX_CHUNKS.', stage='chunk')
                chunks.extend(batch)
            span.set(chunks=len(chunks))

        with tracer.span("embed", model=embedder.model_name, chunks=len(chunks)) as span:
            t0 = time.monotonic()
            vectors, reused, embed_tokens = _embed_with_reuse(
                chunks, embedder, settings, site, previous_corpus, progress
            )
            op_id = new_id("op-")
            ledger.record(UsageEvent(
                event_id=new_id("u-"), run_id=tracer.run_id, operation_id=op_id, attempt=1,
                phase="ingestion", site_id=site.site_id, corpus_id=corpus_id, provider="local",
                model=embedder.model_name, operation="embedding", input_tokens=embed_tokens,
                output_tokens=0, measurement="local_count", cost_usd=0.0,
                pricing_version=Pricing().version, outcome="success",
                latency_ms=int((time.monotonic() - t0) * 1000),
            ))
            span.set(embedded=len(chunks) - reused, reused=reused, input_tokens_o200k=embed_tokens)

        progress("stage", {"name": "index"})
        with tracer.span("index", collection=store.collection) as span:
            store.save_chunks(chunks, vectors)
            store.build_collection(chunks, vectors)
            count = store.point_count()
            span.set(points=count)
            if count != len(chunks):
                raise RagError(ErrorCode.INDEX_INCOMPLETE, f"Indexed {count} of {len(chunks)} chunks.", stage="index")

        manifest.status = IndexStatus.COMPLETE
        manifest.accepted_pages = len(result.pages)
        manifest.chunk_count = len(chunks)
        manifest.completed_at = utcnow()
        store.save_manifest(manifest)
        clear_caches()

        site.active_corpus_id = corpus_id
        site.pending_corpus_id = None
        site.accepted_pages = len(result.pages)
        site.chunk_count = len(chunks)
        site.embedding_model = embedder.model_name
        cov = coverage(work_dir)
        site.crawl_complete = cov['crawl_complete']
        site.crawl_stop_reason = result.stopped_reason
        site.pending_urls = len(result.pending)
        registry.mark_status(site, SiteStatus.READY)

        warnings = []
        if len(result.pages) < 20:
            warnings.append(f"Only {len(result.pages)} content pages were accepted; coverage is limited.")
        failed = sum(1 for e in result.log if e.outcome == FetchOutcome.FAILED)
        skipped = sum(1 for e in result.log if e.outcome == FetchOutcome.SKIPPED)
        return IngestReport(
            site=site, corpus_id=corpus_id, accepted_pages=len(result.pages), skipped=skipped,
            failed=failed, chunk_count=len(chunks), embedded=len(chunks) - reused,
            reused_embeddings=reused, embedding_tokens=embed_tokens,
            stopped_reason=result.stopped_reason, elapsed_s=time.monotonic() - started,
            skip_reasons=skip_reasons, warnings=warnings, run_id=tracer.run_id,
            pending=len(result.pending), crawl_complete=site.crawl_complete,
        )
    except BaseException as exc:  # includes KeyboardInterrupt: record, keep last usable index
        reason = "interrupted by user" if isinstance(exc, KeyboardInterrupt) else str(getattr(exc, "message", exc))
        manifest.status = IndexStatus.FAILED
        manifest.error = reason
        try:
            store.save_manifest(manifest)
            if corpus_id != previous_corpus:  # never drop the collection that is still serving queries
                store.drop_collection()
        except Exception:  # noqa: BLE001 - never mask the original failure
            pass
        site.pending_corpus_id = None
        site.active_corpus_id = previous_corpus
        registry.mark_status(site, SiteStatus.READY if previous_corpus else SiteStatus.FAILED, reason)
        raise


def _embed_with_reuse(chunks, embedder, settings, site, previous_corpus, progress):  # noqa: ANN001, ANN202
    """Embed chunks, reusing vectors from the previous corpus when the embedded text is identical."""
    cache: dict[str, np.ndarray] = {}
    if previous_corpus:
        prev = CorpusStore(settings, site.site_id, previous_corpus)
        try:
            prev_manifest = prev.load_manifest()
            prev_vectors = prev.load_vectors()
            if prev_manifest.embedding_model == embedder.model_name and prev_vectors is not None:
                for c, v in zip(prev.load_chunks(), prev_vectors):
                    cache[embedding_text(c)] = v
        except RagError:
            cache = {}
    texts = [embedding_text(c) for c in chunks]
    vectors = np.zeros((len(chunks), embedder.dim), dtype=np.float32)
    todo = [i for i, t in enumerate(texts) if t not in cache]
    for i, t in enumerate(texts):
        if t in cache:
            vectors[i] = cache[t]
    tokens = sum(count_tokens(texts[i]) for i in todo)
    batch = max(1, settings.embedding_batch_size) * 4
    for start in range(0, len(todo), batch):
        idx = todo[start : start + batch]
        vectors[idx] = embedder.embed_documents([texts[i] for i in idx])
        progress("embed", {"done": min(start + batch, len(todo)), "total": len(todo)})
    return vectors, len(chunks) - len(todo), tokens
