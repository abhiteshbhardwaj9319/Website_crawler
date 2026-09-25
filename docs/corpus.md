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
