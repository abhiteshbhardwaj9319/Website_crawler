# Historical evaluation (pre-evolution)

**Historical status at initial implementation (superseded by [current evaluation](evaluation.md)):** retrieval comparison on the dev, test, and isolation splits is complete and recorded below. **Answer-level evaluation with real OpenAI/Groq models has not been run yet** because no provider keys were available in this environment. That section says exactly what will be measured and how to run it.

## Question sets (frozen)

| Split | File | Cases | Purpose | SHA-256 (first 12) |
| --- | --- | --- | --- | --- |
| test | [eval/questions_test.jsonl](../eval/questions_test.jsonl) | 15: 3 direct, 3 paraphrased, 3 multi-page, 3 misleading, 3 unanswerable | final reported numbers | `e9b6ad94c975` |
| dev | [eval/questions_dev.jsonl](../eval/questions_dev.jsonl) | 10 (2 per category), disjoint topics | choosing retrieval mode and context size | `5dbfde17798d` |
| isolation | [eval/questions_isolation.jsonl](../eval/questions_isolation.jsonl) | 5: four cross-site probes + one positive control | site leakage | `da1826b26b5f` |

How labels were made: each question was written from a source passage found in the indexed snapshot (not from generated answers). Each case lists expected behavior (`answer`, `correct_premise`, or `abstain`), expected claims, and **evidence groups**: every group is one fact the answer needs, satisfied by any listed alternative (URL + verbatim quote). A script checked that every labeled quote occurs in an indexed chunk at that URL (0 missing). Labels were finalized before the first test-split run; one pre-run revision added legitimate alternative sources found while reviewing dev misses (for example, the FAQ entry that also documents `CloseSpider`). Every result file records the question-file hash. Labels were authored by Claude Code from source review and should be spot-checked by a human reviewer.

Category design notes: misleading questions contain a false premise that the docs contradict (T10 asks why `DOWNLOAD_DELAY` defaults to 0 in new projects; 2.19 documents `Default: 1 (fallback: 0)`, and the AutoThrottle page still says "default download delay of zero", a deliberate distractor). Unanswerable questions are plausible in-domain facts absent from the snapshot (Zyte Cloud price, current maintainer, 2.19 release date; release notes are excluded from the corpus).

## Metrics

- **Evidence Recall@k**: evidence groups with a matching chunk (URL + normalized quote) in the top k / all groups of answerable cases.
- **Context recall**: the same, restricted to the chunks actually sent to the model (default 10 chunks, 3,000-token budget). This is the retrieval number that matters for answering.
- **Multi-page complete**: multi-page cases where every evidence group is in context, from at least two distinct pages.
- **MRR@20**: reciprocal rank of the first matching chunk.
- **Isolation leaks**: retrieved, context, or cited URLs from a forbidden host.

Quote-level matching is strict: a chunk from the right section that does not contain the labeled sentence counts as a miss, so these numbers understate what the model can see (see T12 below).

## Retrieval comparison

Identical corpus snapshot, questions, chunking, embedding model, and context budget for every mode. Latency is local CPU time per query (retrieval only), warm models.

### Dev split: used to choose the default (8 answerable cases, 10 evidence groups)

| Mode | R@3 | R@6 | R@10 | R@20 | Context recall | Multi-page complete | MRR@20 | Mean ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dense | 6/10 | 8/10 | 8/10 | 9/10 | 8/10 | 2/2 | 0.694 | 67 |
| bm25 | 6/10 | 8/10 | 10/10 | 10/10 | 10/10 | 2/2 | 0.622 | 10 |
| hybrid (RRF) | 7/10 | 9/10 | 10/10 | 10/10 | 10/10 | 2/2 | 0.606 | 39 |
| hybrid + rerank | 7/10 | 9/10 | 10/10 | 10/10 | 10/10 | 2/2 | 0.900 | 1,346 |

Decision on dev: **hybrid** as default, context size **10** chunks. With 6 context chunks, hybrid and rerank reached 9/10 and dense and BM25 8/10; raising the context to 10 recovered one hybrid evidence group at rank 10 for about 700 more input tokens per query. Hybrid tied BM25 and rerank on context recall and beat BM25 at R@6; rerank ordered evidence better (MRR 0.90) but added about 1.3 s per query without putting more evidence in front of the model. These are differences of one or two evidence groups; they are weak evidence.

### Test split: frozen labels (12 answerable cases, 21 evidence groups)

| Mode | R@3 | R@6 | R@10 | R@20 | Context recall | Multi-page complete | MRR@20 | Mean ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dense | 8/21 | 13/21 | 15/21 | 18/21 | 15/21 | 2/3 | 0.427 | 57 |
| bm25 | 10/21 | 10/21 | 12/21 | 15/21 | 12/21 | 1/3 | 0.500 | 10 |
| **hybrid (default)** | 10/21 | 13/21 | 14/21 | 18/21 | 14/21 | 2/3 | 0.656 | 41 |
| hybrid + rerank | 12/21 | 15/21 | 17/21 | 18/21 | 17/21 | 2/3 | 0.711 | 1,431 |

Honest reading:

- The dev-selected default (hybrid) did **not** beat dense on test context recall (14/21 vs 15/21); it ranks gold evidence higher (MRR 0.66 vs 0.43).
- Hybrid + rerank was best on test (17/21, MRR 0.71) at roughly 1.4 s extra CPU latency. It was not selected because dev showed no context-recall gain; changing the default after seeing test results would tune on the test set. The recommended next step is a larger dev set to decide between hybrid and hybrid + rerank; the reranker is implemented and available with `--mode hybrid_rerank`.
- BM25 alone ranked paraphrased evidence lower (T04 at rank 9, T05 at rank 7) and completed only 1 of 3 multi-page cases; it was fastest and did well on exact identifiers (T01-T03 at ranks 1-3).
- With 21 evidence groups, a one-group difference is about 5 percentage points; none of these differences is statistically meaningful.

### Test retrieval failures (evidence not in context)

| Case | Mode(s) | What happened |
| --- | --- | --- |
| T05 (paraphrased AutoThrottle) | dense, hybrid | The question never says "AutoThrottle"; dense ranked Optimization pages first. Rerank put the AutoThrottle design-goals chunk at rank 1. The second group (the `AUTOTHROTTLE_ENABLED` settings list) was missed in every mode. |
| T07 (downloader vs spider middleware) | hybrid | "Activating" chunks for both at ranks 1-2; the spider-middleware definition chunk ranked 14. 3 of 4 groups in context. |
| T08 (AutoThrottle floor + default delay) | dense | `DOWNLOAD_DELAY` settings chunk at rank 12. Hybrid got both pages. |
| T09 (jobs + security) | rerank | Rerank promoted jobs-directory chunks and pushed the `JOBDIR` command chunk to rank 11. |
| T10 (false premise about `DOWNLOAD_DELAY`) | hybrid | The settings chunk with `Default: 1 (fallback: 0)` ranked 12; the "documented as default" chunk ranked 4. |
| T11 (false premise: built-in JS rendering) | all | Dynamic-content page chunks were retrieved, but not the two labeled chunks (intro recommendation, headless-browser section). Lexical and dense signals both key on "JavaScript" wording elsewhere on the page. The largest remaining retrieval gap. |
| T12 (guarantee against bans) | dense, hybrid | The first chunk of the "Avoiding getting banned" section was retrieved (rank 2 in hybrid, and it does advise identifying yourself via `USER_AGENT`), but the labeled quote lives in the section's second chunk (rank 19). Quote-level scoring counts this as a miss although related evidence was in context. |

### Isolation (retrieval level, real corpora)

All four modes, 5 cases: **0 leaked URLs** in retrieved or context chunks. The positive control (I05, "How do I create a virtual environment?" on site 2) retrieved `tutorial/venv.html` at rank 1. The overlapping-topic probe I01 on site 1 retrieved only Scrapy's installation page (which discusses virtual environments) and never the Python tutorial's `python -m venv tutorial-env` passage. Offline integration tests additionally check dense, BM25 and hybrid isolation with contradictory values, rejection of a citation to the other site's chunk, and a tampered lexical index (fails closed).

## Answer-level evaluation (pending: requires provider keys)

Command (bounded by `RAG_EVAL_MAX_REQUESTS=60` and `RAG_EVAL_MAX_COST_USD=0.50`):

```bash
uv run rag evaluate --split test --modes dense,hybrid --provider openai
uv run rag evaluate --split test --modes hybrid --provider groq      # provider-stratified
uv run rag evaluate --split isolation --modes hybrid
```

Recorded per case in `eval/results/`: status, answer, premise correction, missing information, verified claims, citations (URL, section, quote), rejected claims, provider/model actually used, fallback reason, latency, tokens, cost, and the retrieval metrics above. Automatic scores: behavior correctness by category, correct and inappropriate abstention, citation validity (verified claims / generated claims), gold evidence cited, infrastructure errors (kept separate from quality errors), leakage. **Semantic support and completeness are reviewed manually** per claim against the cited passage and recorded in a review file; a valid quote shows the passage was supplied, not that it supports the claim.

Estimated cost of that run with `gpt-4.1-mini`: 30 queries x ~$0.0018 = ~$0.05 (see [cost analysis](cost-analysis.md)).

## Reproduce

```bash
uv run rag evaluate --split dev  --modes dense,bm25,hybrid,hybrid_rerank --retrieval-only --out eval/results/dev
uv run rag evaluate --split test --modes dense,bm25,hybrid,hybrid_rerank --retrieval-only --out eval/results/test
uv run rag evaluate --split isolation --modes dense,bm25,hybrid,hybrid_rerank --retrieval-only --out eval/results/isolation
```

Results depend on the crawl snapshot; a fresh crawl may change chunk ranks. The per-case JSONL files in `eval/results/` are the recorded evidence for the tables above.
