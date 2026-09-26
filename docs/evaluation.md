# Evaluation

The evolution compares retrieval on expanded corpora, with a development set and a new holdout frozen before tuning. Results are mixed: the selected reranker wins on development, but hybrid wins on holdout. The [historical report](evaluation-baseline.md) and original result files are preserved.

## Retrieval experiment

[Label manifest](../eval/labels-evolution-v1.json) records the frozen files. `evolution_dev` contains 20 questions on both sites and 19 evidence groups; `holdout_v1` contains 12 questions and 10 groups. Questions cover direct facts, identifiers, paraphrases, multiple pages, false premises, absent information and isolation. Labels were authored by Codex against indexed source text, not generated answers; human spot-checking remains useful. The old 15-case test is already observed and is not a new holdout.

All variants use the same expanded corpus snapshots, BGE-small embeddings, 10 context chunks and a 3,000-token evidence budget. A hit requires the labeled URL and contiguous quote, normalizing whitespace only. Case and punctuation remain significant. [Reports](../eval/results/evolution-retrieval/) contain question/model/prompt/corpus fingerprints, per-case ranks, context IDs, failure stages and timings.

| Variant | Dev context recall | Mean retrieval ms | Holdout context recall | Mean retrieval ms |
| --- | --- | --- | --- | --- |
| Dense, 20 candidates | 16/19 | 136 | 6/10 | 161 |
| BM25, 20 | 17/19 | 30 | 7/10 | 49 |
| Hybrid, 20 | 16/19 | 124 | **8/10** | 133 |
| Hybrid + rerank, 20/20 | **19/19** | 1,742 | 7/10 | 2,038 |
| Hybrid, 40 | 18/19 | 127 | Not selected | — |
| Rerank, 40/40 | 18/19 | 3,682 | Not selected | — |
| Rerank, 40/20 | 19/19 | 1,881 | Not selected | — |
| Hybrid + section diversity/dedup | 16/19 | 117 | Not selected | — |
| Hybrid + same-section neighbors | 16/19 | 124 | Not selected | — |

The [selection record](../eval/results/evolution-retrieval/selection.json) predates the holdout run. The code default is `hybrid_rerank`, 20 candidates and 20 reranked, with ordinary ranked context. Existing environment overrides remain respected. Use `--mode hybrid` for the measured faster alternative. No claim of a universal quality or latency improvement is justified. No reformulation call was added: the selected development configuration had no residual evidence misses. The holdout is now observed; further selection needs a fresh split.

Holdout misses in the selected configuration: H03 cookies evidence was absent from candidates; H09 arithmetic progression evidence was absent from candidates; H04's labeled selector introduction was retrieved but omitted by context limits. Other CSS passages were present, so a label miss does not alone explain its later generation abstention. All 12 holdout cases and every tested development variant had zero foreign site/corpus chunks. These small denominators do not establish production accuracy.

## Latency and budgets

The ablation means include the first query and varying process/cache state. [Two dedicated observations](../eval/results/evolution-retrieval/cold-warm.json) separately measure a fresh process with already cached model files and a repeated query in the same process. Store verification was 1,045/1.5 ms; embedding model access 552/0.05 ms; reranker model access 475/0.15 ms; total retrieval 3,617/2,608 ms; context selection 0.33/0.34 ms. OS caches were not cleared; the offline test/browser jobs overlapped this pair, so it is evidence of reuse, not an isolated latency benchmark. A comparable dedicated cold baseline was not captured; no before/after cold-speed claim is made.

Chunk headings, titles, identifiers and definition anchors are preserved. The evidence budget uses `o200k_base`, suitable for the configured OpenAI generation model; it is an estimate for Groq and is not BGE's tokenizer. Embedding/reranker inputs can be truncated at their model limits. No chunking change was selected on these results. The 30,000-chunk index cap bounds the in-memory chunk/vector matrix; page HTML is persisted incrementally.

## Live answer acceptance

The bounded final run and semantic review are recorded in the final acceptance checkpoint. Citation identity, factual support and answer completeness are separate metrics. An answered status or a matching source passage alone is not proof of a correct answer.

## Reproduction

```powershell
uv run rag evaluate --split holdout_v1 --modes dense,bm25,hybrid,hybrid_rerank --retrieval-only --out artifacts/recheck
uv run rag evaluate --split holdout_v1 --modes hybrid_rerank --provider openai --out artifacts/live-recheck
uv run pytest
```

Do not run two processes against the same local Qdrant store. Paid evaluation uses configured request/spend stopping caps; a call already in flight can take known spend past the threshold. Use a new output directory and retain original evidence. `scripts/compare_evolution.py` reproduces the original ablation layout; archive its output separately before rerunning because it writes deterministic filenames.
