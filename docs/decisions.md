# Technical decisions

Each entry: the choice, why, the alternative considered, and what would change it. "Measured" means supported by results in this repository; otherwise it is a design judgment.

## 1. Primary website: Scrapy 2.19 documentation

Chosen for structured technical documentation with exact facts and cross-page topics. Pinned to `/en/2.19/` after the recorded robots/access check. Version-specific defaults (e.g. `DOWNLOAD_DELAY: Default: 1 (fallback: 0)`) test whether answers follow evidence over memory; final evaluation also records a failure on that distinction. The second corpus began as the Python 3.13 tutorial and now deliberately includes library/builtins reference pages. Details: [corpus](corpus.md).

## 2. Isolation by construction: one corpus per site

Each ingestion writes its own Qdrant collection, BM25 source, and manifest; a query opens exactly one. Dense search also filters on site/corpus, and every retrieved payload is re-validated. Alternative: one shared collection with a metadata filter. That is fine at scale, but a missing or wrong filter silently leaks; separate collections make leakage structurally hard and easy to test. Prompt instructions are not relied on for isolation. Measured: 0 leaks on the isolation split in all four retrieval modes; offline tests cover contradictory cross-site content.

## 3. Safe refresh and stable numbering

The registry assigns numbers from a monotonic counter and never reuses them. A refresh builds a new corpus version beside the active one and switches only when the manifest is complete and the point count matches. A failure (or Ctrl+C) keeps the previous corpus serving queries. This is tested, including a bug the test found: second-resolution corpus IDs could collide and the failure cleanup deleted the active collection; IDs now carry a random suffix and cleanup never touches the active corpus.

## 4. Local embeddings: BAAI/bge-small-en-v1.5 via fastembed (ONNX)

MIT-licensed, 384-dimensional, ~67 MB, CPU-only ONNX runtime (no PyTorch), installs cleanly on Windows. Keeps ingestion and retrieval working when an OpenAI balance or quota is exhausted, and keeps generation-provider changes from touching the embedding space. The model name is pinned in each corpus manifest; querying with a different configured model fails with `index_incompatible` rather than silently mixing spaces. Trade-off: a larger or API embedding model might retrieve better; not measured here. Changing it requires re-ingestion.

## 5. Section-aware chunks (~320 tokens, block overlap)

Chunks never cross a heading boundary, so every chunk has one heading path and a real anchor for citations. The token target is a starting hypothesis (kept below the 512-token limit of the embedding model), not a tuned optimum. Title and heading path are prepended for embedding/BM25 only, so quotes match the page text.

## 6. Retrieval: development-selected reranking, with a faster hybrid option

The original implementation selected hybrid on its 10-question development set. Evolution froze 20 development and 12 new holdout questions before tuning. Rerank20 recovered 19/19 development groups versus hybrid20 16/19; reranking 40 candidates was slower and recovered 18/19. Twenty-candidate reranking became the code default before opening the new holdout. The holdout favored hybrid (8/10) over rerank (7/10), at 133 versus 2,038 ms mean retrieval. Retain that negative finding and offer `--mode hybrid`; do not quietly retune on holdout. Existing environment overrides remain respected. [Full experiment](evaluation.md).

RRF uses ranks with k=60; its scores are not confidence. Separate candidate and reranker depths bound cross-encoder work. Local models/stores are reused within one process and corpus activation keys invalidate store caches. No additional store process or hosted infrastructure was added.

## 7. Context: up to 10 chunks within 3,000 evidence tokens

Ranked whole chunks remain the default. Development experiments added optional section diversity with containment deduplication and same-section neighbor expansion; neither beat ranked reranking. Policies all enforce selected site/corpus and token limits. No query rewrite was justified by residual development misses. The holdout exposes multi-part candidate misses and a context-budget miss; those are future hypotheses requiring a new holdout. The o200k evidence count is a generation budget estimate, not BGE's tokenizer; local models may truncate long inputs.

## 8. LangGraph for the query workflow, LangChain for model access

A fixed `StateGraph`: validate → retrieve → context → generate → check_citations → finalize, with explicit abstain and error edges. The value is explicit, typed, inspectable routing that tests exercise and traces display; it does not by itself improve accuracy. No agent loop, query rewriting, tool choice, or judge call: one query embedding and one generation call per normal question keeps cost and latency predictable. LangChain's `ChatOpenAI`/`ChatGroq` provide the provider integrations and strict structured output. Alternative: a plain LangChain runnable sequence would work; the graph makes the failure paths first-class.

## 9. Source-linked claims with deterministic passage IDs

`answer_v2` returns only status and claims referencing supplied chunk/passage IDs. Quotes and URLs are reconstructed locally from deterministic paragraph spans. The answer is built only from accepted claims; no independent model answer/premise prose can bypass checks. Legacy records remain readable with strict contiguous whitespace-normalized, case-preserving quote checks. Rejected candidates appear only in explicit diagnostics. No index migration or repair model call is needed.

This verifies evidence identity, not entailment. Explicit final review found 45/50 displayed claims fully supported and 18/26 answerable cases both complete and grounded. We report those failures instead of calling all claims verified. An additional semantic verifier was not introduced without its own measured development evidence.

## 10. Generation models

- OpenAI `gpt-4.1-mini-2025-04-14`: pinned snapshot, non-reasoning (predictable latency and tokens), supports strict Structured Outputs, $0.40/$1.60 per 1M tokens (checked 2026-09-26).
- Groq `openai/gpt-oss-120b`: production model with strict `json_schema` support on Groq, $0.15/$0.60, `reasoning_effort=low`. Groq's Llama models are listed as enterprise/contact-sales as of the check date, and strict schema support is limited to the GPT-OSS/Qwen models.

Both are configurable (`RAG_OPENAI_MODEL`, `RAG_GROQ_MODEL`); the pricing file must be updated with them.

## 11. Provider failures and fallback

Errors are classified from HTTP status plus the provider's error `code`/`type`, because the same 429 can mean a temporary rate limit, confirmed exhausted credits, a spend limit, or an ambiguous quota error, each needing a different action. Permanent billing/auth errors are never retried; transient ones get at most 3 attempts within 60 s, honoring `Retry-After` up to 20 s. SDK retries are disabled so attempts cannot multiply and every attempt is counted once. `auto` falls back from OpenAI to Groq once, visibly, for quota/availability/transient failures only; invalid keys do not trigger fallback because they need fixing, not masking. Groq's free tier is rate-limited (8K tokens/min, 200K/day), so it is a fallback, not unlimited capacity.

## 12. Local observability first

JSONL traces and a usage ledger under `data/` work without accounts, Docker, or network services, and are what `rag trace`, `rag sources`, and `rag costs` read. Langfuse was not added: it would need an account or a self-hosted stack and would upload traces, which the local design avoids by default. It remains a straightforward extension (LangChain callback) if hosted tracing is wanted.

## 13. CLI rather than a UI

The assessment does not require a UI. A Typer/Rich CLI keeps the demo reproducible and scriptable (structured `--json` on data/report commands), and renders untrusted page/model text as plain text with terminal control sequences stripped.

## 14. Resume and coverage instead of an unlimited crawl

Persist page bodies and frontier/outcomes incrementally; reserve accepted-page capacity before scheduling a batch. Keep active and working corpora separate and activate only after a complete index write. Failed refresh retains the previous index. Scope/limit additions can resume; narrowed or incompatible policies require refresh. No conditional GET was added because cached-body consistency would need additional state and tests. Bounded XML sitemap discovery complements HTML links; these two documented sites required no sitemap.

Scrapy reached 144 pages with its eligible frontier exhausted. Python reached 90 pages with 186 pending at the cap, explicitly including the redirected `/3.13/builtins/` scope. Neither number is advertised as total host coverage. Page storage is incremental but the chunk/vector matrix remains bounded by a configurable 30,000-chunk cap.

## 15. Presentation and evidence

Keep the existing CLI. Compact accepted claims, deduplicated URL markers, optional diagnostics, ASCII borders and explicit replay metadata serve the demo without another application. Two standalone Archify HTML diagrams explain actual modules; nine showcase checks, desktop containment and screenshot inspection are recorded. Local traces stay local; only reviewed public answer evidence is committed. User video recording and Windows Terminal GUI inspection are not claimed as completed.
