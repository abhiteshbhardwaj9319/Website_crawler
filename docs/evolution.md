# Evolution evidence

## E1 baseline (2026-09-26)

Baseline application commit: `17bd0f88bdd3f2b2574668a569cbeebf06bc5c1a`.
The offline suite passed all 78 offline tests; 3 live tests were skipped. Windows sandbox file access initially prevented pytest temporary directories; the same suite passed with filesystem access. Qdrant correctly rejected a concurrent session; evaluation ran after the user closed it.

Frozen [fingerprints and bounded live runs](../eval/results/evolution-baseline/fingerprints.json) record corpus/configuration, model, prompt, and chunk-file hashes. Original benchmark files are preserved. The repeated test context recall matches the original: dense 15/21, BM25 12/21, hybrid 14/21, rerank 17/21. Reranker mean retrieval time was 1,750 ms versus hybrid 66 ms on this test run; these are mixed process-cache timings, not a dedicated cold-load benchmark.

| User query | Before status | Input/output tokens | Cost USD | Total ms |
| --- | --- | --- | --- | --- |
| Scrapy project command | answered | 2592/251 | 0.0014384 | 8422 |
| what is python | partial; 3 accepted, 1 rejected claim | 3534/632 | 0.0024248 | 6204 |
| what is python command to get input | insufficient evidence | 3385/40 | 0.0014180 | 1645 |

These were three explicit OpenAI requests, total $0.0052812, under a 3-request/$0.03 stopping cap. The Python corpus has no `input` anchor and excludes `/3.13/library/`; this is a coverage gap before claiming a ranking failure. Safe CLI snapshots at 80/100/140 columns are stored locally under ignored `artifacts/evolution/`.

Confirmed defects: unchecked `draft.answer`/premise prose can bypass claim checks; normal output prints rejected candidates; quote normalization folds code case and accepts ellipsis that can omit negation; crawl batch processing can silently discard fetched results at the accepted-page cap. Coverage expansion, new holdout evaluation, and presentation acceptance remain separate milestones.

## E2 grounding boundary

88 offline tests pass, 3 live tests skipped. Regression cases cover valid claims plus unsupported answer/premise prose, partial rejection, invented IDs/passages, altered quotes, cross-site citations, and terminal widths 80/100/140. Default output contains accepted claims once and deduplicates sources; `--explain` exposes clearly labeled rejected candidates.

`answer_v2` uses `AnswerSelection` (status, claims, chunk/passsage IDs); exact passages are generated locally. No index migration or paid repair call is needed. Existing `answer_v1` records and their original prompt remain available; new output does not display their unchecked prose. Legacy quote validation now preserves case and treats ellipses literally. This establishes structural linkage, not semantic truth. Live v2 quality/cost measurement follows after coverage work.
