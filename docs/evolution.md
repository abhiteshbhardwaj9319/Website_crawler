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

## E4 retrieval selection

The expanded development set was frozen before tuning, followed by a separately invoked new holdout. Rerank20 recovered 19/19 development evidence groups versus hybrid20 16/19, at 1,742 versus 124 ms mean retrieval. The preselected reranker recovered 7/10 on holdout; hybrid recovered 8/10 at 133 versus 2,038 ms. Candidate-depth, diversity/dedup and neighbor ablations did not beat the selected development tradeoff. These are measured tradeoffs, not a universal retrieval improvement. Original observed-test context recall on the expanded corpus is 16/21 with reranking, versus historical 17/21 on the smaller corpus; corpus composition and stricter quote scoring also changed. See [evaluation](evaluation.md) for stage failures and cold/warm measurement limits.

## E3 scope, checkpoint and ingestion results

95 offline tests pass, 3 live tests skipped. Targeted resume/interruption/sitemap/scope tests also pass after checkpoint recovery hardening. Scrapy: 45 ? 144 accepted pages, 945 ? 1,895 chunks, 0 pending, 0 failures; 945 vectors reused, 950 new vectors, 225,797 local embedding tokens, 249.9 seconds. Python: 17 ? 90 pages, 251 ? 4,611 chunks, 186 pending at its declared cap, 0 failures; 251 vectors reused, 4,360 new, 556,050 local tokens, 755.5 seconds including embedding/indexing (the 600-second bound is crawl time only). See [coverage details](corpus.md).

The Python source moved from `library/functions.html` to `builtins/functions.html` within 3.13. The initially rejected redirect is recorded; adding that explicit path fixed scope rather than bypassing checks. Reference definition IDs are preserved by extract-v2; imported old pages keep their original extraction, timestamps, and chunks. The JSON output/manifest/trace confirm completed activation. PowerShell's combined command reported nonzero for Python ingestion despite successful recorded stages; the active corpus is separately checked by a fresh query. Never infer success from that shell status alone.


## E5 presentation and E6 acceptance

Final offline suite: 106 passed, 3 opt-in live tests skipped. Twenty-four actual CLI replay/report checks passed at 80/100/140 columns and through UTF-8 PowerShell pipes; ASCII, NO_COLOR, JSON errors, sources, traces and coverage were checked. Windows Terminal GUI inspection was not performed. Two source-backed Archify artifacts passed all nine showcase checks, zero errors/warnings, four-size browser containment and explicit light/dark screenshot review. [Diagrams and receipts](diagrams/README.md).

Final acceptance: 35 OpenAI + 3 independent Groq calls, no operational errors/fallback/unknown usage, $0.0725245 list-price cost. All 12 expected abstentions succeeded; two false-premise cases also abstained incorrectly. One of 51 generated claims was rejected for a wrong passage ID. Explicit review found 45/50 displayed claims fully supported; 20/26 answerable cases complete, 18/26 both complete and grounded. Failures and per-case judgments are retained in [evaluation](evaluation.md), not relabeled away. [Costs](cost-analysis.md) use observed output tokens; [walkthrough](walkthrough.md) distinguishes rehearsed commands from the user's pending video.

Final audit also repaired a nested-sitemap depth-limit path that could falsely report complete discovery. Depth/count deferrals now remain incomplete; failed sitemap HTTP/redirects raise typed errors; DTD rejection also checks parsed XML declarations, including UTF-16. The recorded corpora did not use sitemap discovery, so these changes do not alter their retrieval/live evidence. Capture scripts now refuse to overwrite frozen results. Original corpus chunk hashes still match the E1 fingerprint; the separate clean Python 3.11.7 installation passed four CLI/bootstrap checks with no credentials or old data copied.

Verified remote checkpoints before final acceptance: E1 `d4eefea`, E2 `fdaa886`, E3 `50960c3`, E4 `06f799b`, E5 `1655783`. Each push was compared with `git ls-remote origin refs/heads/main`. Final acceptance is committed separately; inspect `git log` for its exact hash. Credentials, stores, internal planning and unrelated edits are excluded from these checkpoints.
