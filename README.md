# Website-Grounded RAG Agent

**Status (2026-09-26): implementation in progress.** Crawling, extraction, per-site indexing, the LangGraph query workflow, citation validation, provider error handling, tracing, and the CLI are implemented and covered by an offline test suite. Live provider answers, evaluation results, and cost projections are not yet recorded; see the sections marked *pending*.

A command-line system that crawls public documentation websites, builds an isolated searchable index per website, and answers questions **only** from the selected website's indexed pages, with source URLs, section headings, and verified quotes.

## Setup

Requires Python 3.11-3.13 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync                      # installs the pinned, locked dependency set
cp .env.example .env         # then put your OPENAI_API_KEY and/or GROQ_API_KEY in .env
uv run rag doctor            # environment, key presence (never values), site readiness
```

Embeddings run locally (BAAI/bge-small-en-v1.5 via ONNX, ~67 MB downloaded on first ingestion), so crawling, indexing, and retrieval need no API key. A key is needed only to generate answers.

## Commands (verified)

```bash
uv run rag sites list                         # numbered websites (1 = default)
uv run rag ingest --site 1                    # crawl + index Scrapy 2.19 docs (45 pages)
uv run rag ingest --site 2                    # crawl + index the Python 3.13 tutorial (17 pages)
uv run rag ask "How do I enable an item pipeline?" --site 1 --show-retrieval
uv run rag ask "..." --site 2 --provider groq --mode hybrid --json
uv run rag sites add https://example.org/docs/ --max-pages 20   # new public site, then ingest
uv run rag chat                               # interactive; type a number to switch sites
uv run rag sources <run-id>                   # evidence used by a run
uv run rag trace <run-id>                     # stage-by-stage trace, failures, fallback
uv run rag costs                              # usage ledger by phase/provider/model
uv run pytest -q                              # offline test suite (no keys needed)
```

All commands accept `--json` for machine-readable output; `rag --no-color ...` disables color.

## Design summary

- **Websites and isolation.** A persistent registry gives each website a stable number and its own corpus: one Qdrant collection, one BM25 index, one manifest per ingestion. Queries open exactly one corpus; retrieved payloads are re-validated against the selected site, and citations may reference only chunks sent to the model for that run.
- **Ingestion.** Bounded breadth-first crawl (pages, depth, bytes, time, rate, concurrency, retries) with robots.txt, manual redirect scope checks, and private-network blocking. Sphinx/HTML extraction keeps headings, real anchors, code, lists, and tables; duplicates and thin pages are skipped with reasons. A refresh builds a new corpus beside the active one and switches only on success.
- **Query workflow.** A bounded LangGraph graph: validate → retrieve → context → generate → check citations → finalize, with explicit abstain and error paths. Generation uses LangChain `ChatOpenAI` / `ChatGroq` with strict JSON-schema output: claims with chunk IDs and verbatim quotes, which code validates before display. URLs come from index metadata, never the model.
- **Providers.** `--provider openai|groq|auto`. Auto tries OpenAI (`gpt-4.1-mini-2025-04-14`) and falls back once, visibly, to Groq (`openai/gpt-oss-120b`) on quota/availability/transient failures. Credit exhaustion, spend limits, ambiguous quota errors, rate limits, auth, and timeouts are distinguished; SDK retries are disabled so the application's bounded retry policy is the only one.
- **Observability.** Every run writes a local JSONL trace (spans with site/corpus IDs, retrieval ranks, provider attempts, errors) and usage-ledger events (tokens, cost or explicit *unknown*). Secrets are redacted.

Detailed documentation: [architecture](docs/architecture.md), [implementation scope](docs/implementation-scope.md).

## Pending

Evaluation results, retrieval comparison, cost analysis, walkthrough script, and live provider verification.
