# Walkthrough script (10-15 minutes)

Recording and submission are done by the author. **Video link: pending (not yet recorded).**

Before recording: `uv run rag doctor` shows both keys configured and sites 1 and 2 ready. Use a terminal at least 120 columns wide. Close other `rag` processes (local Qdrant allows one process). Pre-ingest sites 1 and 2; the live "add a URL" step uses a small site so it finishes in about a minute.

## 0:00-1:30 Problem and scope

- Goal: answer questions about a website using only that website's pages, with source URLs, and say clearly when the pages do not contain the answer.
- What I built: a Python CLI. Numbered websites, each with its own isolated index; hybrid retrieval; a small LangGraph workflow; verified citations; typed provider errors with a visible OpenAI to Groq fallback; local traces and cost accounting; a frozen evaluation set.
- Out of scope on purpose: web UI, deployment, agent loops.

## 1:30-4:00 Architecture

Show [docs/diagrams/architecture-overview.svg](diagrams/architecture-overview.svg).

- Ingestion: bounded, robots-aware crawler with scope and redirect checks, extractor that keeps headings, real anchors, and code; section-aware chunks; local bge-small embeddings (no API dependency for indexing or retrieval).
- Isolation: one Qdrant collection + BM25 index + manifest per site version; a query opens exactly one; payloads are re-validated; citations must reference chunks sent in this run. Not enforced by prompt.
- Query graph: validate, retrieve, context, generate, check citations, finalize; abstain without a model call when nothing is retrieved; every failure is a typed error with a next step.
- Why LangGraph: explicit, testable routing and traceable state, not accuracy by itself.

## 4:00-7:30 Live demo

```bash
uv run rag sites list
uv run rag ask "Which command creates a new Scrapy project?" --site 1
```
Point out: status, answer, verified claims with markers, sources with URL#anchor, section and exact quote, provider/model, tokens, cost, run ID.

```bash
uv run rag ask "If I enable AutoThrottle, can it ever use a delay shorter than DOWNLOAD_DELAY, and what DOWNLOAD_DELAY does a project created with startproject use by default?" --site 1 --show-retrieval
```
Multi-page: sources from `autothrottle.html` and `settings.html`; show ranks from dense and BM25 in the retrieval table.

```bash
uv run rag ask "Why does Scrapy set DOWNLOAD_DELAY to 0 by default, even in new projects created with startproject?" --site 1
```
False premise: the 2.19 docs say default 1 (fallback 0). Note that a model answering from memory would likely say 0.

```bash
uv run rag ask "How much does a Zyte Scrapy Cloud subscription cost per month?" --site 1
```
Unanswerable: "The indexed pages for Scrapy 2.19 documentation do not contain enough information..." The wording is about the indexed pages, not the whole internet.

Switch sites in chat:
```bash
uv run rag chat
ask> How do I enable an item pipeline component?      # answered on site 1
ask> 2                                                  # switch to Python tutorial; evidence cleared
ask> How do I enable an item pipeline component?      # abstains; no Scrapy sources
ask> How do I create a virtual environment?           # answered from venv.html
```

Add a site (about a minute):
```bash
uv run rag sites add https://packaging.python.org/en/latest/tutorials/ --max-pages 8 --max-depth 1
uv run rag ask "How do I create a pyproject.toml for a package?" --site 3
uv run rag sites add https://example.com/        # low-content site fails clearly, with a trace ID
```

## 7:30-9:00 Errors and traces

```bash
uv run rag trace <run-id-from-multi-page-answer>
uv run rag ask "Which command creates a new Scrapy project?" --provider openai   # with an invalid/exhausted key, if safe to show
```
Explain the classification table ([architecture](architecture.md#providers-and-failures)): exhausted credits vs spend limit vs ambiguous quota vs rate limit; no retries on billing errors; bounded retries on transient ones; one visible fallback in `auto`; `rag trace` names the failing stage. (If showing a failure live is not practical, show `tests/test_pipeline.py::test_fallback_to_groq_is_visible_and_counted_once_per_attempt` and its trace output.)

## 9:00-11:30 Evaluation

Show [docs/evaluation.md](evaluation.md).

- 15 frozen test questions across five categories plus a separate dev set and isolation set; labels from source review, with evidence groups and verbatim quotes checked against the index.
- Retrieval comparison on identical inputs: dense, BM25, hybrid, hybrid + rerank. Hybrid chosen on dev; on test the reranker was best and hybrid did not beat dense on context recall. Say why the default was not changed after seeing test results.
- Answer-level results: status accuracy per category, abstention, citation validity, and the manual support review (fill in from the live run).
- Isolation: zero leaks.
- Limits: tiny sample, strict quote-level scoring, snapshot drift.

## 11:30-13:00 Costs

Show [docs/cost-analysis.md](cost-analysis.md) and `uv run rag costs`.

- Ingestion: $0 API (local embeddings), measured token counts and time.
- One query: ~3K input tokens, measured composition; output from the live ledger.
- 100 / 1,000 / 10,000 queries on gpt-4.1-mini and Groq; Groq free-tier daily token cap means it is not capacity for 1,000+ queries.

## 13:00-14:30 Improvements

- Larger dev set to decide between hybrid and hybrid + rerank (the reranker is already implemented as `--mode hybrid_rerank`).
- Fix the T11-type retrieval gap (false-premise questions that name a nonexistent feature): query expansion or a second retrieval pass, measured on dev.
- Claim-level support checking with an entailment model, evaluated before trusting it.
- JavaScript-rendered sites via a headless browser, sitemap-seeded crawling, incremental refresh.
- Hosted tracing (Langfuse) behind explicit configuration; a small API or UI if needed.
