# Current corpus coverage (E3, 2026-09-26)

| Site | Before | Active index | Remaining frontier | Crawl state |
| --- | --- | --- | --- | --- |
| 1 Scrapy 2.19 | 45 pages / 945 chunks | 144 pages / 1,895 chunks | 0 | Discovered eligible frontier exhausted; 5 low-content skips; 0 failures |
| 2 Python 3.13 | 17 pages / 251 chunks | 90 pages / 4,611 chunks | 186 | Index ready, crawl limited at 90 accepted pages; 8 skips; 0 failures |

[Machine-readable coverage, fingerprints and ingestion accounting](../eval/results/evolution-scope/) preserve the measured scope and limits. Scrapy includes linked source-code pages under `/en/2.19/_modules/`; the original `news.html` exclusion remains unchanged. This is finite configured scope, not a claim about every host URL. Python allows only `/3.13/tutorial/`, `/3.13/library/`, and `/3.13/builtins/` on `docs.python.org`. Its original stable site ID/number is preserved.

The supplied `/3.13/library/functions.html` now returns HTTP 301 to [the official builtins page](https://docs.python.org/3.13/builtins/functions.html#input), independently checked with HTTP and browsing. Scope was explicitly versioned to include that pinned path. The requested and final URLs remain in page provenance; the indexed `input` definition has its actual anchor. Original indexes and historical evaluation files remain intact.

```
uv run rag sites coverage 1
uv run rag sites show 2 --json
uv run rag sites configure 2 --max-pages 120 --max-total-s 600
uv run rag ingest --site 2 --resume
uv run rag ingest --site 2 --refresh
```

Resume loads persisted accepted pages and retries failed/deferred URLs; refresh starts a new snapshot. No conditional GET is implemented: a refresh fetches full bodies, and HTTP 304 without a body is a fetch failure. Disk checkpoints and compressed raw bodies are saved per page; raw HTML is not retained for an entire site. Chunking loads one page at a time, embeddings/index writes use batches, and a 30,000-chunk configurable guard bounds the in-memory chunk/vector matrix. This remains a bounded local assessment tool, not a streaming unlimited-corpus service.

Checkpoint scope identity rejects narrowed paths/changed exclusions or extraction thresholds; safe additive scope/limit changes preserve completed pages. An interrupted checkpoint recovers durable page records. Any failed URL during refresh/resume of an existing corpus prevents activation, leaving the prior index queryable. Accepted-page batch capacity is reserved before fetch, so fetched results are not silently discarded. Robots, public-host and redirect checks apply before page requests; XML discovery has document/count/depth/byte/URL limits and rejects DTD/entities. Sitemap limits make coverage incomplete. No sitemap source is necessary for these linked documentation corpora.

`processed_of_discovered` counts current outcomes over discovered in-scope URLs; it is not an estimate of the website's unknown total. Deferred URLs are not skipped or failed. A crawl cap is displayed as a resource bound, never as a known-total progress bar.

## Historical corpus report (pre-evolution)

# Corpus coverage

Snapshot used for the evaluation: crawled 2026-09-25 22:43-22:47 UTC. Websites change; re-running `rag ingest` produces a new snapshot and may change results.

| # | Website | Scope (host + path prefix) | Accepted pages | Chunks | Embedded tokens (o200k count) | Ingestion time |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | Scrapy 2.19 documentation | `docs.scrapy.org/en/2.19/` | 45 | 945 (mean 151 tokens, max 363) | 158,018 | 200 s (includes first model download) |
| 2 | Python 3.13 tutorial | `docs.python.org/3.13/tutorial/` | 17 | 251 (mean 227, max 348) | 63,819 | 62 s |

Every chunk in both corpora has a page-defined section anchor. Embedding is local, so ingestion had no API cost.

## Site 1: why Scrapy 2.19

- Structured technical documentation with direct facts (setting defaults, commands), cross-page topics (middlewares, throttling, jobs/security), and version-specific details that differ from older releases, which tests grounding over model memory (e.g. `DOWNLOAD_DELAY` is documented as `Default: 1 (fallback: 0)` in 2.19).
- Access checked on 2026-09-26: `robots.txt` disallows `/en/stable/` and several old versions and allows `/en/latest/` and `/en/2.19/`. The current release on PyPI was 2.19.0. The corpus is pinned to `/en/2.19/`; `latest` would drift and `stable` is disallowed.
- Pages declare `<link rel="canonical">` pointing to `/en/latest/...`. The crawler ignores canonical tags for scope: a canonical cannot move the corpus to another version.
- Coverage is partial by design: breadth-first discovery found 128 in-scope pages; the 45-page cap (`config/sites.default.json`) kept intro, topics, FAQ, and practices pages reachable within depth 1 and left 83 URLs unfetched (logged as `not_fetched_limit_reached`). `news.html` (release notes, a very long changelog) is excluded so it does not dominate retrieval; questions about release dates are therefore unanswerable in this corpus.
- Crawl: 45 accepted, 0 failed, 0 duplicate/low-content skips.

## Site 2: Python 3.13 tutorial

A genuinely different, smaller corpus (the complete tutorial, 17 pages) used to demonstrate site switching and isolation. It overlaps with site 1 on some topics (virtual environments, Python basics), which the isolation tests exploit. It does not meet the 20-page threshold on its own; the assessment requirement is met by site 1.

## Crawl and extraction policy

- Discovery: breadth-first from the seed; links from the whole page (including navigation) feed discovery; only in-scope, robots-allowed URLs are queued.
- Normalization: lowercase scheme/host, drop default ports, fragments, and query strings; collapse `//`, resolve `.`/`..`; `index.html` equals its directory.
- Excluded paths: Sphinx `_sources`, `_static`, `_images`, `_downloads`, `genindex`, `search`, `py-modindex`, login/logout/register, `/tag/`, `/page/N`, and non-HTML extensions.
- Limits for site 1: 45 pages, depth 3, 3 MB per page, 20 s timeout, 300 s total, 0.5 s between requests, concurrency 2, 2 retries.
- Skips with reasons: out-of-scope redirects, robots-disallowed URLs, non-HTML responses, oversized pages, `noindex`, fewer than 80 words (or likely JavaScript-rendered), and content-hash duplicates. Duplicate or empty pages never count toward the page total.
- Raw HTML is kept gzipped under `data/` (ignored) so extraction can be audited.

## Extraction review

Three representative pages were inspected after extraction:

- `topics/item-pipeline.html`: heading hierarchy preserved (`Item Pipeline > Item pipeline example > Write items to MongoDB`), code blocks fenced intact, list items kept, permalink glyphs and sidebar removed.
- `topics/settings.html` (7,661 words): 129 sections, each with its real anchor (e.g. `#download-delay`); each setting's `Default:` line stays with its description.
- `tutorial/venv.html`: 3 sections with anchors; the `python -m venv tutorial-env` command block preserved.

## Adding a site during a demo

`rag sites add <url> --max-pages 20` registers the URL's directory as the scope and ingests it immediately (or `/add <url>` in `rag chat`). Static HTML documentation-style sites work best. Sites that block crawlers, require JavaScript to render content, or have little text fail with `low_content` or `robots_disallowed` and a next step; the previous index of any existing site is never affected.
