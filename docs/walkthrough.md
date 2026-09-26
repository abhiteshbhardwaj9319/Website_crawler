# Walkthrough: 10–15 minutes

The CLI sequence was rehearsed against actual saved runs at 80/100/140 columns, with live requests exercised in the acceptance run. A separate clean locked install passed bootstrap checks. This is a technical rehearsal, not a recorded or timed narration. **User video recording and submission remain pending.** Windows PowerShell/piped output was checked; a Windows Terminal GUI session was not opened.

Before presenting, close other `rag` processes, run `uv run rag doctor`, and use about 100–140 terminal columns. Existing corpora can be reused. Public clones must ingest first. Do not expose `.env` or manufacture a billing failure by editing credentials.

## 0:00–3:00 Architecture and evidence boundary

Open [implemented architecture](diagrams/implemented-architecture.html) locally in a browser; use the [PNG fallback](diagrams/implemented-architecture.visual-check.2048x1320.light.png) on GitHub. Explain numbered sites, separate Qdrant/BM25 corpora, bounded crawl checkpoints and atomic activation. Show [question workflow](diagrams/evidence-path.html): context selection, provider call, deterministic passage checks and explicit abstention/error branches. These are static explanations, not live telemetry. LangGraph makes routing inspectable; it is not an accuracy guarantee.

## 3:00–7:00 Live questions and sources

```powershell
uv run rag sites list
uv run rag demo project --provider openai
uv run rag demo input --provider openai
uv run rag sources <run-id-from-input-answer>
uv run rag trace <run-id-from-input-answer>
```

Explain the single answer, inline markers and deduplicated source URLs, then inspect exact passages behind the compact view. Point out actual provider, tokens, cost and run ID. Python's source is now `https://docs.python.org/3.13/builtins/functions.html#input` after an official redirect; no answer is hard-coded.

If live access is unavailable, explicitly use a recorded replay from this workspace:

```powershell
uv run rag demo --replay r-002102cc0ef246bd
uv run rag demo --replay r-e82762d31b534721
```

The replay prints its timestamp, corpus, model and run ID. These IDs are local artifacts, not files included in a public clone. Portable reviewed evidence is [here](../eval/results/evolution-live/answers.jsonl).

## 7:00–9:00 Isolation, abstention and coverage

```powershell
uv run rag demo isolation --provider openai
uv run rag demo abstain --provider openai
uv run rag sites coverage 1
uv run rag sites coverage 2
uv run rag chat
```

In chat, enter `2`, then `/help`, then `/quit`. Site changes clear prior evidence. The isolation preset asks a Scrapy pipeline question while site 2 is selected; its recorded acceptance case abstained without Scrapy citations. The annual-profit preset also abstained. Live model responses may vary.

Coverage: Scrapy 45 → 144 pages with zero pending; Python 17 → 90 with 186 pending at the cap. Show the scope and stop reason. Explain that raising the cap and `--resume` continues durable work, while `--refresh` fetches a new snapshot. Avoid a long crawl during the timed demo. The interrupted-resume and failed-refresh tests demonstrate recovery without deliberately breaking the working index.

## 9:00–11:30 Results and an honest failure

Show [evaluation](evaluation.md). Reranking recovered 19/19 development groups but only 7/10 holdout groups; hybrid got 8/10 much faster. The holdout did not confirm a universal gain. Cite 106 offline tests, 38 live calls and zero observed cross-site leakage.

Show the review of H09 (`getrandbits` is one integer, not a sequence) or T10 (false delay premise). A valid passage ID does not prove entailment. Of 50 displayed claims, 45 were fully supported by their selected excerpts; 18/26 answerable cases were complete and fully grounded. Explain the difference between citation identity, support and completeness. Do not demonstrate T10 as a reliably corrected premise; the recorded answer failed it.

For partial withholding, the recorded I05 run `r-23e7d64f9f564af6` rejected one command claim for an invalid passage ID. Normal replay shows the useful remainder plus the citation-failure notice; `rag sources` exposes explicitly marked diagnostics. Its remaining dangling reference is documented as a limitation.

## 11:30–13:30 Costs and close

```powershell
uv run rag costs
```

Show [cost analysis](cost-analysis.md): local ingestion has no API charge; final acceptance cost $0.0725245 at list prices. OpenAI's observed mean projects to ~$0.20/$2.01/$20.05 for 100/1,000/10,000 similar requests. Groq was independently exercised, but only on three smoke cases; its free-tier token limits constrain throughput and the account's cash tier was not inferred.

End with the remaining measured problems: multi-part candidate misses, evidence lost to context limits, false-premise correction and semantic entailment. Future changes need a fresh holdout. No extra frontend, hosted trace service or autonomous loop is required to explain this implementation.
