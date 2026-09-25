# Website-Grounded RAG Agent

**Status: implementation authorized; handoff prepared. No application code, runnable commands, evaluation results, or measured costs exist yet.**

This assessment will build a small system that crawls public websites, indexes at least 20 useful pages for the primary demo site, and answers questions using only the selected website's retrieved content with supporting source URLs. It will use LangChain and/or LangGraph and report ingestion/query token usage and estimated costs.

The proposed design is documented in [the architecture](docs/architecture.md), with a standalone [overview diagram](docs/diagrams/architecture-overview.svg). These describe intended behavior, not implemented features.

The accepted direction is a polished Python CLI with stable numbered website selection, default site 1, and the ability to ingest a new public URL. The proposed technical defaults are a bounded LangGraph query workflow, persistent local Qdrant, and local embeddings with OpenAI/Groq generation. Dense and hybrid retrieval will be compared; a reranker will be evaluated if practical. Structured local traces will explain sources, costs, and errors. See the [latest implementation scope](docs/implementation-scope.md) for the requirements and rationale. Exact models, dependency versions, and site viability still require implementation checks.

## Planned submission contents

- Source code, pinned dependencies, safe `.env.example`, and setup/run instructions.
- Architecture and technical decisions, with limitations.
- At least 10 evaluation questions covering straightforward, paraphrased, multi-page, misleading, and unanswerable cases, plus actual results.
- Measured example usage and estimated costs for ingestion and 100, 1,000, and 10,000 queries.
- A 10-15 minute recorded walkthrough link.

## Reproducibility and local files

Internal planning and personal reference material are intentionally not published. Application setup must not depend on those files. Secrets, crawl snapshots, vector stores, and raw traces will remain local; reviewed evaluation fixtures and aggregate results will be tracked separately.

Setup instructions and command examples will be added after the first implementation milestone verifies them. No API key is needed to read this repository.
