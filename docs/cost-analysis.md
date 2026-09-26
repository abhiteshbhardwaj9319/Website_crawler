# Measured costs

The final acceptance run made **35 OpenAI requests and 3 independent Groq requests**, costing **$0.0725245 at published list rates**. All token counts were provider-reported; no retries, fallback, cache discounts or unknown-cost events occurred. These are calculated usage costs, not a billing-invoice reconciliation. Groq's actual account tier is not inferred from a successful API call.

## Rates checked on 2026-09-26

| Model | Input per million | Cached input per million | Output per million |
| --- | --- | --- | --- |
| OpenAI `gpt-4.1-mini-2025-04-14` | $0.40 | $0.10 | $1.60 |
| Groq `openai/gpt-oss-120b` | $0.15 | Not assumed | $0.60 |

Sources: [OpenAI model pricing](https://developers.openai.com/api/docs/models/gpt-4.1-mini), [Groq model pricing](https://console.groq.com/docs/models). The arithmetic uses [config/pricing.json](../config/pricing.json); recheck rates before budgeting future use. Cached tokens, if reported, are a subset of input and are charged once. Groq reasoning tokens are part of reported output, not an extra charge added again.

## Actual final workload

[Measured cost data](../eval/results/evolution-live/measured-costs.json) and [per-answer records](../eval/results/evolution-live/answers.jsonl) support these numbers.

| Provider | Requests | Input tokens | Output tokens | Reasoning subset | Total USD | Mean USD |
| --- | --- | --- | --- | --- | --- | --- |
| OpenAI | 35 | 151,749 | 5,929 | 0 | 0.070186 | 0.0020053 |
| Groq | 3 | 13,782 | 452 | 127 | 0.0023385 | 0.0007795 |

Mean OpenAI input/output: 4,335.7/169.4 tokens. Mean Groq: 4,594/150.7. These populations differ: Groq only ran two simple questions and an abstention, so their means are not a controlled provider comparison. The run included abstentions and incorrect/incomplete answers; costs are not costs per successful answer.

Example `input()` answer, OpenAI run `r-e82762d31b534721`: 4,609 input × $0.40/M + 124 output × $1.60/M = **$0.002042**. It cited the expanded Python definition. The old insufficient-evidence run cost $0.001418; improved coverage and a useful answer were more expensive. The new passage-ID prompt increases input overhead while reducing copied output; no blanket cost reduction is claimed.

## Projections from observed means

Assumes the same workload/token distribution, one call per query, unchanged prices, no retries or fallback, and no cache discount. These are projections, not measured workloads.

| Requests | OpenAI observed 35-case mean | Groq observed 3-case mean at list rates |
| --- | --- | --- |
| 100 | $0.201 | $0.078 |
| 1,000 | $2.005 | $0.780 |
| 10,000 | $20.053 | $7.795 |

Groq's documented free limits for this model are 30 RPM, 1,000 RPD, 8,000 TPM and 200,000 TPD. At the observed ~4,745 total tokens/query, the daily token bound allows about 42 such queries/day before other limits; 100/1,000/10,000 requests would take at least about 3/24/238 days at that average, not burst capacity. Actual tier and enforcement can differ. See [official rate limits](https://console.groq.com/docs/rate-limits). A free-tier allowance does not establish unlimited capacity or zero compute cost.

## Ingestion and build costs

Local BGE embeddings and cross-encoder reranking have **$0 provider API charge**, but consume CPU, memory, disk, download bandwidth and electricity. Those resource costs were not monetized.

| Ingestion | New local embedding tokens (o200k estimate) | Reused / new vectors | Recorded elapsed |
| --- | --- | --- | --- |
| Original Scrapy 45 pages | 158,018 | 0 / 945 | 200 s, including model download |
| Original Python 17 pages | 63,819 | 0 / 251 | 62 s |
| Expanded Scrapy 144 pages | 225,797 | 945 / 950 | 249.9 s |
| Expanded Python 90 pages | 556,050 | 251 / 4,360 | 755.5 s |

These counts describe successive builds, not four independent active indexes to sum into per-query usage. Reusing vectors avoids embedding work, not fetching/extraction/indexing work. Resume reuses persisted pages; refresh fetches bodies. The Python 600-second setting bounds crawl time, not the later embedding/index phase. [Ingestion evidence](../eval/results/evolution-scope/) records the manifests and events.

Evolution provider evidence comprises $0.0052812 for three baseline calls, $0.0019364 for the E3 `input()` coverage check, and $0.0725245 for final acceptance: **$0.0797421**. This is the documented evolution subset, not a claim about all historical account spend. Earlier user runs and any other ledger events remain separate. The current ledger is available through `uv run rag costs --json`.

Evaluation stopping caps count every provider attempt. A call already in flight can cross the known-cost threshold; unknown usage is counted explicitly, never assumed free. Final acceptance used at most 40 calls and a $0.15 known-cost stopping threshold. Ordinary evaluation uses configurable `RAG_EVAL_MAX_REQUESTS` and `RAG_EVAL_MAX_COST_USD`. Projections exclude local compute, taxes, hosting, retries, fallback and future evaluations.
