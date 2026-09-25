# Technical decisions

Each entry: the choice, why, the alternative considered, and what would change it. "Measured" means supported by results in this repository; otherwise it is a design judgment.

## 1. Primary website: Scrapy 2.19 documentation

Chosen for structured technical content with exact facts, paraphrasable concepts, and genuine cross-page topics. Pinned to `/en/2.19/` because `robots.txt` disallows `/en/stable/` and `/en/latest/` drifts. Version-specific defaults (e.g. `DOWNLOAD_DELAY: Default: 1 (fallback: 0)`) differ from older Scrapy releases, which tests whether answers follow the evidence rather than model memory. Second site: the Python 3.13 tutorial, a different and smaller corpus that overlaps on a few topics. Details: [corpus](corpus.md).

## 2. Isolation by construction: one corpus per site

Each ingestion writes its own Qdrant collection, BM25 source, and manifest; a query opens exactly one. Dense search also filters on site/corpus, and every retrieved payload is re-validated. Alternative: one shared collection with a metadata filter. That is fine at scale, but a missing or wrong filter silently leaks; separate collections make leakage structurally hard and easy to test. Prompt instructions are not relied on for isolation. Measured: 0 leaks on the isolation split in all four retrieval modes; offline tests cover contradictory cross-site content.

## 3. Safe refresh and stable numbering

The registry assigns numbers from a monotonic counter and never reuses them. A refresh builds a new corpus version beside the active one and switches only when the manifest is complete and the point count matches. A failure (or Ctrl+C) keeps the previous corpus serving queries. This is tested, including a bug the test found: second-resolution corpus IDs could collide and the failure cleanup deleted the active collection; IDs now carry a random suffix and cleanup never touches the active corpus.

## 4. Local embeddings: BAAI/bge-small-en-v1.5 via fastembed (ONNX)

MIT-licensed, 384-dimensional, ~67 MB, CPU-only ONNX runtime (no PyTorch), installs cleanly on Windows. Keeps ingestion and retrieval working when an OpenAI balance or quota is exhausted, and keeps generation-provider changes from touching the embedding space. The model name is pinned in each corpus manifest; querying with a different configured model fails with `index_incompatible` rather than silently mixing spaces. Trade-off: a larger or API embedding model might retrieve better; not measured here. Changing it requires re-ingestion.

## 5. Section-aware chunks (~320 tokens, block overlap)

Chunks never cross a heading boundary, so every chunk has one heading path and a real anchor for citations. The token target is a starting hypothesis (kept below the 512-token limit of the embedding model), not a tuned optimum. Title and heading path are prepended for embedding/BM25 only, so quotes match the page text.

## 6. Retrieval: hybrid (dense + BM25, reciprocal rank fusion) by default; reranker optional

Four modes implemented and compared on identical inputs ([evaluation](evaluation.md)). The default was chosen on the dev split: hybrid tied BM25 and the reranker on dev context recall (10/10) with better top-6 recall than BM25 and ~40 ms latency; the cross-encoder (`Xenova/ms-marco-MiniLM-L-6-v2`, local ONNX) improved ranking (MRR 0.90) but added ~1.3 s per query without more evidence reaching the model on dev. On the frozen test split the reranker was best (17/21 context recall vs 14/21 hybrid and 15/21 dense). The default was not changed after seeing test results; a larger dev set is the recommended way to decide. RRF uses k=60 and ranks only; fused scores are not treated as confidence. Hybrid fusion is done in the application over the corpus's own BM25 index, not Qdrant's native sparse vectors (simpler, and one fewer model to manage).

## 7. Context: up to 10 chunks within 3,000 evidence tokens

Chosen on dev: moving from 6 to 10 chunks recovered one evidence group at rank 10 for ~700 more input tokens (~$0.0003 per query on gpt-4.1-mini). Chunks are taken in rank order without mid-chunk truncation.

## 8. LangGraph for the query workflow, LangChain for model access

A fixed `StateGraph`: validate → retrieve → context → generate → check_citations → finalize, with explicit abstain and error edges. The value is explicit, typed, inspectable routing that tests exercise and traces display; it does not by itself improve accuracy. No agent loop, query rewriting, tool choice, or judge call: one query embedding and one generation call per normal question keeps cost and latency predictable. LangChain's `ChatOpenAI`/`ChatGroq` provide the provider integrations and strict structured output. Alternative: a plain LangChain runnable sequence would work; the graph makes the failure paths first-class.

## 9. Structured claims + deterministic citation checks

The model returns claims, each with chunk IDs and verbatim quotes, under a strict JSON schema. Code verifies that IDs were in the supplied context and quotes occur in those chunks, and builds URLs from index metadata. Invalid claims are withheld (and the free-text answer with them). This catches invented sources and fabricated quotes cheaply and deterministically. It does not prove the passage entails the claim; semantic support is judged in evaluation. An LLM-judge verification call was not added: it doubles cost and latency and would itself need evaluation.

## 10. Generation models

- OpenAI `gpt-4.1-mini-2025-04-14`: pinned snapshot, non-reasoning (predictable latency and tokens), supports strict Structured Outputs, $0.40/$1.60 per 1M tokens (checked 2026-09-26).
- Groq `openai/gpt-oss-120b`: production model with strict `json_schema` support on Groq, $0.15/$0.60, `reasoning_effort=low`. Groq's Llama models are listed as enterprise/contact-sales as of the check date, and strict schema support is limited to the GPT-OSS/Qwen models.

Both are configurable (`RAG_OPENAI_MODEL`, `RAG_GROQ_MODEL`); the pricing file must be updated with them.

## 11. Provider failures and fallback

Errors are classified from HTTP status plus the provider's error `code`/`type`, because the same 429 can mean a temporary rate limit, confirmed exhausted credits, a spend limit, or an ambiguous quota error, each needing a different action. Permanent billing/auth errors are never retried; transient ones get at most 3 attempts within 60 s, honoring `Retry-After` up to 20 s. SDK retries are disabled so attempts cannot multiply and every attempt is counted once. `auto` falls back from OpenAI to Groq once, visibly, for quota/availability/transient failures only; invalid keys do not trigger fallback because they need fixing, not masking. Groq's free tier is rate-limited (8K tokens/min, 200K/day), so it is a fallback, not unlimited capacity.

## 12. Local observability first

JSONL traces and a usage ledger under `data/` work without accounts, Docker, or network services, and are what `rag trace`, `rag sources`, and `rag costs` read. Langfuse was not added: it would need an account or a self-hosted stack and would upload traces, which the local design avoids by default. It remains a straightforward extension (LangChain callback) if hosted tracing is wanted.

## 13. CLI rather than a UI

The assessment does not require a UI. A Typer/Rich CLI keeps the demo reproducible and scriptable (`--json` on every command), and renders untrusted page/model text as plain text with terminal control sequences stripped.
