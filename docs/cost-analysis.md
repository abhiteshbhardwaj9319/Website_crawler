# Cost analysis

**Status (2026-09-26):** ingestion usage and per-query **input** tokens are measured. Per-query **output** tokens and provider-reported usage are **estimates** until the first live run (no provider keys were available in this environment). Every number is labeled *measured* or *estimated*. Prices are standard-tier list prices checked on 2026-09-26 ([config/pricing.json](../config/pricing.json)); they change, so recheck before relying on the projections.

## Prices used

| Model | Input / 1M | Cached input / 1M | Output / 1M | Source |
| --- | --- | --- | --- | --- |
| OpenAI `gpt-4.1-mini` (snapshot 2025-04-14) | $0.40 | $0.10 | $1.60 | [OpenAI pricing](https://developers.openai.com/api/docs/pricing) |
| Groq `openai/gpt-oss-120b` | $0.15 | n/a | $0.60 | [Groq models](https://console.groq.com/docs/models) |
| Local `BAAI/bge-small-en-v1.5` embeddings, local cross-encoder | $0 API | - | - | runs on your CPU |

Groq free tier for `openai/gpt-oss-120b` (checked 2026-09-26): 30 requests/min, 1,000 requests/day, 8,000 tokens/min, 200,000 tokens/day ([rate limits](https://console.groq.com/docs/rate-limits)).

## Ingestion (measured)

| Site | Pages | Chunks | Embedded tokens (o200k count) | API cost | Local time |
| --- | --- | --- | --- | --- | --- |
| Scrapy 2.19 docs | 45 | 945 | 158,018 | $0 | 200 s including the one-time ~67 MB model download |
| Python 3.13 tutorial | 17 | 251 | 63,819 | $0 | 62 s |

Embeddings are local, so ingestion has no provider charge; it costs CPU time and a one-time model download. For comparison only, the same 221,837 tokens through OpenAI `text-embedding-3-small` ($0.02/1M on the same pricing page) would cost about $0.0044. The local choice exists for availability (an exhausted OpenAI balance cannot disable indexing or retrieval), not for the negligible savings. Re-ingesting unchanged content reuses stored vectors.

## One example query

Question T08 (multi-page): *"If I enable AutoThrottle, can it ever use a delay shorter than DOWNLOAD_DELAY, and what DOWNLOAD_DELAY does a project created with startproject use by default?"* Hybrid retrieval, 10 chunks from `autothrottle.html`, `optimize.html`, `settings.html`.

| Component | Tokens | Status |
| --- | --- | --- |
| Query embedding (local) | 32 | measured, $0 |
| System prompt ([answer_v1](../src/website_rag/prompts/answer_v1.md)) | 406 | measured (o200k) |
| User message: site line + 10 evidence chunks (1,599) + question | 2,103 | measured (o200k) |
| Structured-output JSON schema | ~471 | estimated (schema text count; the provider's internal formatting may differ) |
| **Input total** | **~2,980** | measured prompt + estimated schema overhead |
| Output (answer, claims with quotes) | ~350 (range 150-600) | **estimated** |

Cost of that query, gpt-4.1-mini: 2,980 x $0.40/1M + 350 x $1.60/1M = $0.00119 + $0.00056 = **~$0.0018** (estimate).
Same query on Groq gpt-oss-120b (adding ~200 reasoning tokens at `reasoning_effort=low`, also estimated): 2,980 x $0.15/1M + 550 x $0.60/1M = **~$0.0008**.

The OpenAI cached-input discount is not assumed: the static part of the prompt (system prompt + schema, ~880 tokens) is expected to fall below OpenAI's 1,024-token minimum cacheable prefix, and the evidence differs per question. The ledger records any cached tokens the provider actually reports.

## Workload mean (input measured, output estimated)

Across the 25 dev + test questions with the default configuration (hybrid, 10 chunks), system + user prompt tokens were **mean 2,363, median 2,281, min 1,672, max 3,527** (measured with o200k). Category means: direct 2,173; paraphrased 2,576; multi-page 2,230; misleading 2,570; unanswerable 2,266. Adding ~471 schema tokens gives ~2,834 input tokens per query.

Workload assumption for projections: every question is answered with one generation call (retrieval always returns chunks for these corpora, so abstentions also call the model), no retries, no fallback. Output 350 tokens (range 150-600).

## Projections

Per-query estimate: gpt-4.1-mini $0.00169 (range $0.00137-$0.00209); gpt-oss-120b on Groq's paid tier $0.00076 (range $0.00064-$0.00091).

| Queries | gpt-4.1-mini (range) | Groq gpt-oss-120b at published rates (range) | Groq free tier |
| --- | --- | --- | --- |
| 100 | $0.17 ($0.14-$0.21) | $0.08 ($0.06-$0.09) | $0 cash; ~2 days at ~3.3K tokens/query under the 200K tokens/day cap |
| 1,000 | $1.69 ($1.37-$2.09) | $0.76 ($0.64-$0.91) | not feasible within free daily limits (~17 days) |
| 10,000 | $16.94 ($13.74-$20.94) | $7.55 ($6.35-$9.05) | not feasible; requires a paid tier |

Ingestion is a one-time $0 API cost per corpus version (local embeddings) and is not included in the per-query rows. With OpenAI embeddings instead, first ingestion of both sites would add ~$0.004, and query embeddings ~$0.0000006 each.

Excluded: local compute and electricity, hosting, taxes, re-crawls, retries and fallback attempts (each is billed as a separate call if it reaches the model), evaluation/build runs, price changes, and Groq's observed free-tier charge being $0 (a free-tier cash charge of $0 does not imply capacity at scale).

Sensitivity: input dominates (about two-thirds of the gpt-4.1-mini cost). Dropping the context from 10 to 6 chunks would save roughly 600-700 input tokens per query (~$0.00026 on gpt-4.1-mini) at the cost of the evidence recall measured in [evaluation](evaluation.md).

## Build and evaluation costs

API spend during implementation so far: **$0** (no live provider calls were made; the usage ledger contains only local embedding events). The pending answer-level evaluation (about 30 generation calls on gpt-4.1-mini) is estimated at ~$0.05 and is capped by `RAG_EVAL_MAX_COST_USD=0.50`.

## How to see real numbers

After any live run, `uv run rag costs` aggregates the ledger by phase (ingestion / query / evaluation), provider, and model, including attempts, errors, and events with unknown cost. Each answer prints its tokens and cost, and `rag trace <run-id>` shows per-attempt usage.
