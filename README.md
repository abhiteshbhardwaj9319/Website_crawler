# Website-Grounded RAG Agent

A Python CLI that crawls scoped documentation, keeps a separate index for each numbered website, and generates source-linked answers from retrieved passages. Citation checks enforce supplied chunk/passage IDs; semantic correctness remains a measured limitation.

**Verified on 2026-09-26:** 106 offline tests passed; 35 OpenAI and 3 independent Groq requests completed. Scrapy grew from 45 to 144 pages; Python from 17 to 90. The `input()` question now answers from its indexed definition. [Evaluation](docs/evaluation.md) records improvements, regressions and unsupported claims without treating citation validity as truth.

![Implemented architecture](docs/diagrams/implemented-architecture.visual-check.1440x900.light.png)

[Interactive Archify architecture and evidence workflow](docs/diagrams/README.md) · [Demo walkthrough](docs/walkthrough.md) · [Recorded CLI preview](docs/diagrams/cli-input-replay.svg)

## Setup

Requires Python 3.11–3.13 and [uv](https://docs.astral.sh/uv/). The main suite ran on Windows/Python 3.13; a separate locked installation and CLI bootstrap were checked on Python 3.11.7.

```powershell
uv sync --locked
# Only if .env is absent, create it from .env.example and enter your keys locally.
# Preserve any existing .env and credentials.
uv run rag doctor
uv run rag sites list
```

Existing indexed websites are ready to query. For a **new checkout** (indexes are intentionally not committed):

```powershell
uv run rag ingest --site 1
uv run rag ingest --site 2
```

Ingestion/retrieval use local embeddings and require no provider key. Initial model downloads and crawling take time; new website snapshots may differ from the recorded evaluation. Keep one `rag` process open at a time because local Qdrant owns an exclusive store lock.

## Ask and inspect

```powershell
uv run rag demo project --provider openai
uv run rag demo input --provider openai
uv run rag demo abstain --provider openai
uv run rag demo isolation --provider openai
uv run rag ask "what is python" --site 2 --mode hybrid_rerank --provider openai
uv run rag chat
uv run rag sources <run-id>
uv run rag trace <run-id>
uv run rag costs
```

Normal output shows one accepted-claim answer, deduplicated sources, actual provider, tokens, cost and run ID. `--explain` shows exact evidence and explicitly marked rejected candidates; `--show-retrieval` shows ranks. A partial citation failure says what was withheld. `rag demo --replay <run-id>` labels the saved timestamp, corpus, model and run; it makes no live call. Local run IDs are absent in public clones.

Data/report commands support `--json`; interactive chat and help remain terminal interfaces. `rag --no-color ...` or `NO_COLOR=1` disables colors; `rag --ascii ...` uses ASCII table borders while preserving source/code text. Expected application errors exit 2, including structured JSON errors. CLI usage errors remain parser messages.

Providers: `--provider openai|groq|auto`; auto allows one visible bounded fallback. Explicit providers do not fall back. Retrieval: `--mode dense|bm25|hybrid|hybrid_rerank`. The code default is `hybrid_rerank`, selected on development before the new holdout; existing `.env` overrides are respected. `--mode hybrid` is the measured faster alternative.

## Coverage and resume

```powershell
uv run rag sites coverage 1
uv run rag sites coverage 2
uv run rag sites configure 2 --max-pages 120 --max-total-s 600
uv run rag ingest --site 2 --resume
# For an intentionally new snapshot:
uv run rag ingest --site 2 --refresh
```

Scrapy's discovered eligible `/en/2.19/` frontier is exhausted at 144 pages/1,895 chunks. Python's explicit tutorial/library/builtins scope has 90 pages/4,611 chunks and 186 pending URLs at its cap. “Index ready” differs from “crawl complete”; neither promises every host URL. Interrupted crawls resume durable work, and failed replacement builds preserve the active index. [Exact scopes and limits](docs/corpus.md).

## Measured results

| Measure | Result |
| --- | --- |
| Development context recall | Rerank 19/19; hybrid 16/19 |
| New holdout context recall | Rerank 7/10; hybrid **8/10** |
| Holdout mean retrieval | Rerank 2,038 ms; hybrid 133 ms |
| Live site/corpus leaks | 0/38 cases |
| Structurally accepted claims | 50/51 generated claims |
| Explicit semantic review | 45/50 displayed claims fully supported; 18/26 answerable cases both complete and grounded |
| Final provider list-price cost | $0.0725245 for 38 requests |

Reranking did not win on the holdout, and valid citations did not prevent all factual mistakes. The old observed test also fell from 17/21 to 16/21 context recall after corpus/scoring changes; this is not an isolated ranker comparison. OpenAI's observed mean cost was $0.002005/query (~$2.01 per 1,000 at the same workload). [Full evaluation](docs/evaluation.md) · [Measured costs and projections](docs/cost-analysis.md).

```powershell
uv run pytest
uv run rag evaluate --split holdout_v1 --modes dense,bm25,hybrid,hybrid_rerank --retrieval-only --out artifacts/recheck
```

## Documentation and limits

- [Architecture](docs/architecture.md), [decisions](docs/decisions.md), [evolution evidence](docs/evolution.md), [implementation scope](docs/implementation-scope.md).
- [10–15 minute walkthrough](docs/walkthrough.md): commands rehearsed; user recording/submission remains pending.
- Static HTML only, finite crawl/index bounds, single-process local storage; no conditional HTTP refresh or hosted telemetry.
- Small evaluation sets and model variability limit generalization. Known failures include multi-part retrieval, false-premise correction and claim entailment. No runtime semantic verifier is claimed.

`.env`, crawl snapshots, vector stores, raw traces, local runs and internal planning remain excluded from Git. Only reviewed public evidence is tracked. Existing indexes and stable site numbers are preserved; the repository can be understood without internal planning files.
