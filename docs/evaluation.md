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

The [38-case final run](../eval/results/evolution-live/answers.jsonl), [summaries](../eval/results/evolution-live/summary.json), [fingerprints](../eval/results/evolution-live/fingerprints.json) and [claim review](../eval/results/evolution-live/semantic-review.json) preserve the evidence. It used 35 explicit OpenAI requests and three independent Groq requests, one attempt each, no fallback. A 40-request/$0.15 known-spend stopping cap bounded the run; recorded cost was $0.0725245. There were **0/38 operational errors**, **0 unknown usage events** and **0/38 site/corpus leaks** in retrieved chunks; cited URLs were also checked against their selected hosts.

| Suite | Cases | Structural behavior proxy | Correct abstentions | Accepted/generated claims | Gold groups cited |
| --- | --- | --- | --- | --- | --- |
| New holdout, OpenAI | 12 | 11/12 | 4/4 | 12/12 | 7/10 |
| Observed original test, OpenAI | 15 | 13/15 | 3/3 | 27/27 | 16/21 |
| Isolation, OpenAI | 5 | 5/5 | 4/4 | 2/3 | 1/1 |
| Three user questions, OpenAI | 3 | 3/3 | n/a | 7/7 | No gold labels |
| Independent Groq smoke cases | 3 | 3/3 | 1/1 | 2/2 | No gold labels |

The behavior proxy accepts answered/partial statuses and checks gold citation for false-premise cases; it does **not** measure semantic correctness or completeness. All 12 labeled abstentions succeeded. Two additional false-premise cases wrongly abstained. One generated claim was rejected for a passage ID not supplied for its chunk (isolation I05), leaving a partial answer with an accurate citation-failure notice. All 50 accepted claims had structurally valid source links; that does not establish meaning.

**Explicit semantic review:** Codex compared every displayed claim with its selected quotations, then checked every requested answer part. This is a documented reviewer process, not an independent human review or a runtime entailment guarantee. **45/50 accepted claims were fully supported; 20/26 answerable cases were complete; 18/26 were both complete and fully source-supported.** The remaining 12 cases were expected abstentions. A strict rubric counts a multi-assertion claim as unsupported if any substantive part exceeds its selected excerpts.

Five support failures remain visible in the review: H03 adds command-line precedence without citing that rule; H09 substitutes `getrandbits()` (one integer) for an integer sequence; T10 accepts the wrong new-project delay premise; T12's universal no-guarantee assertion exceeds its mitigation excerpts; Groq U03 cites `input()` signatures but omits the passage establishing user input. The source needed for Groq U03 was retrieved, so that is evidence selection rather than coverage. Incomplete cases include H03's missing cookies half, H09's missing arithmetic sequence, H04/T11 false-premise abstentions, T10's incorrect correction, and I05's withheld command plus dangling “this command” reference. These are measured limitations, not hidden successful outcomes. No prompt retuning on this observed holdout was performed.

Original test labels are retained against the expanded corpus: its context recall is 16/21 versus historical rerank 17/21. Coverage and strict quote normalization both changed, so that is not an isolated ranker regression estimate. The original release-notes exclusion remains unchanged; no label was revised to manufacture improvement.

## User-query before/after

| Question | Original corpus / answer_v1 | Expanded corpus / answer_v2 |
| --- | --- | --- |
| New Scrapy project command | Answered; 2,592 in / 251 out; $0.0014384; 8,422 ms | Answered; 4,451 / 185; $0.0020764; 3,554 ms |
| What is Python | Partial; 3 accepted + 1 rejected; 3,534 / 632; $0.0024248; 6,204 ms | Answered; 5 accepted; 4,703 / 427; $0.0025644; 4,930 ms |
| Python command to get input | Insufficient evidence; 3,385 / 40; $0.001418; 1,645 ms | Answered from `builtins/functions.html#input`; 4,609 / 124; $0.002042; 3,034 ms |

The project question is a closely related paraphrase, not byte-identical. Each row compares one observation; corpus, retrieval, prompt and provider latency changed. Input costs increased; fewer output tokens do not by themselves mean a cheaper answer. Scope expansion fixed the input coverage gap. No guaranteed speedup or aggregate pre/post semantic improvement is claimed.

Mean live OpenAI stage times: retrieve 1,793 ms, context 0.06 ms, generation 1,950 ms (provider attempt 1,946 ms), citation checks 0.71 ms, render 1.85 ms; total 3,771 ms. Groq's three smoke cases averaged 3,939 ms total, including 2,321 ms provider attempts. Trace durations are integer milliseconds, so sub-ms stages can report zero. Rendering was measured separately and is not included in saved query latency.

## Verification and rehearsal

Final offline suite: **106 passed, 3 opt-in live tests skipped**. The explicit 38-request acceptance script supplies live evidence separately. Tests cover the free-text bypass, altered/foreign evidence, partial/all-invalid output, interruption recovery, scope expansion, sitemap constraints, failed refresh preservation, bounded batches, context policies and UI/JSON behavior. [24 CLI rehearsal checks](../eval/results/evolution-live/cli-rehearsal.json) exercised actual saved answers at 80/100/140 columns, PowerShell UTF-8 pipes, NO_COLOR/--no-color, ASCII borders, sources, traces, coverage, replay and error JSON. Windows Terminal GUI appearance was not opened or claimed. [Clean locked installation](../eval/results/evolution-live/clean-setup.json) passed four CLI/bootstrap checks on Python 3.11.7 with credentials removed and a separate registry; it did not repeat the live crawl or model evaluation. Both Archify diagrams have separately recorded [browser validation and visual review](diagrams/README.md).

## Reproduction

```powershell
uv run rag evaluate --split holdout_v1 --modes dense,bm25,hybrid,hybrid_rerank --retrieval-only --out artifacts/recheck
uv run rag evaluate --split holdout_v1 --modes hybrid_rerank --provider openai --out artifacts/live-recheck
uv run pytest
```

Do not run two processes against the same local Qdrant store. Paid evaluation uses configured request/spend stopping caps; a call already in flight can take known spend past the threshold. Use a new output directory and retain original evidence. `scripts/compare_evolution.py --out artifacts/new-experiment` reproduces the original development ablation layout; `--holdout` evaluates the frozen selection. Capture scripts refuse to overwrite completed frozen experiments.
