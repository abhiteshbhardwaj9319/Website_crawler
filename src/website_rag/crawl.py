"""Bounded, scoped, robots-aware website crawler.

Breadth-first internal-link discovery from the seed URL. Bounds: accepted pages, total
fetch attempts, depth, bytes per page, per-request timeout, total wall time, request rate,
concurrency, and retries. Redirects are followed manually so every hop is scope-checked.
Every URL considered ends up in the crawl log as accepted, skipped, or failed with a reason.
"""

from __future__ import annotations

import asyncio
import gzip
import hashlib
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable
from urllib.robotparser import RobotFileParser

import httpx

from .errors import ErrorCode, RagError
from .extract import ExtractedPage, extract_page
from .schemas import CrawlLogEntry, FetchOutcome, PageRecord, SiteRecord, utcnow
from .urls import Scope, assert_public_host, normalize_url

USER_AGENT = "website-rag-assessment/0.1 (+https://github.com/abhiteshbhardwaj9319/Website_crawler)"
MAX_REDIRECTS = 5
RETRY_STATUS = {429, 500, 502, 503, 504}


def page_id_for(url: str) -> str:
    return hashlib.sha256(url.encode("utf-8")).hexdigest()[:16]


def content_hash(sections) -> str:  # noqa: ANN001
    joined = "\n".join(s.text for s in sections)
    return hashlib.sha256(" ".join(joined.split()).encode("utf-8")).hexdigest()


@dataclass
class FetchResult:
    url: str
    final_url: str | None
    status: int | None
    html: str | None
    outcome: FetchOutcome
    reason: str
    bytes: int = 0
    elapsed_ms: int = 0


@dataclass
class CrawlResult:
    pages: list[PageRecord]
    log: list[CrawlLogEntry]
    raw_html: dict[str, str] = field(default_factory=dict)  # page_id -> html
    stopped_reason: str = "frontier_exhausted"
    robots_url: str | None = None


ProgressFn = Callable[[str, dict], None]


class _RateLimiter:
    def __init__(self, min_interval: float) -> None:
        self.min_interval = min_interval
        self._lock = asyncio.Lock()
        self._last = 0.0

    async def wait(self) -> None:
        async with self._lock:
            now = time.monotonic()
            delay = self._last + self.min_interval - now
            if delay > 0:
                await asyncio.sleep(delay)
            self._last = time.monotonic()


class Crawler:
    def __init__(
        self,
        site: SiteRecord,
        corpus_id: str,
        transport: httpx.AsyncBaseTransport | None = None,
        check_network: bool = True,
        progress: ProgressFn | None = None,
    ) -> None:
        self.site = site
        self.corpus_id = corpus_id
        self.limits = site.crawl
        self.scope = Scope(site.allowed_host, site.allowed_path_prefix, tuple(site.exclude_patterns))
        self.transport = transport
        self.check_network = check_network
        self.progress = progress or (lambda kind, data: None)
        self.robots: RobotFileParser | None = None

    # ------------------------------------------------------------------ robots
    async def _load_robots(self, client: httpx.AsyncClient) -> str:
        robots_url = f"https://{self.site.allowed_host}/robots.txt"
        if self.site.seed_url.startswith("http://"):
            robots_url = f"http://{self.site.allowed_host}/robots.txt"
        parser = RobotFileParser()
        try:
            resp = await client.get(robots_url, timeout=self.limits.timeout_s)
        except httpx.HTTPError as exc:
            raise RagError(
                ErrorCode.FETCH_FAILED,
                f"Could not fetch robots.txt from {self.site.allowed_host} ({type(exc).__name__}).",
                hint="Check connectivity; crawling is not attempted when robots.txt is unreachable.",
                stage="crawl",
            ) from exc
        if 400 <= resp.status_code < 500:
            parser.parse([])  # RFC 9309: unavailable robots.txt -> no restrictions
        elif resp.status_code >= 500:
            raise RagError(
                ErrorCode.FETCH_FAILED,
                f"robots.txt returned HTTP {resp.status_code}; treating the site as disallowed.",
                hint="Try again later.",
                stage="crawl",
            )
        else:
            parser.parse(resp.text.splitlines())
        self.robots = parser
        return robots_url

    def _robots_allows(self, url: str) -> bool:
        return self.robots is None or self.robots.can_fetch(USER_AGENT, url)

    # ------------------------------------------------------------------ fetch
    async def _fetch(self, client: httpx.AsyncClient, url: str, limiter: _RateLimiter) -> FetchResult:
        started = time.monotonic()
        current = url
        attempt = 0
        hops = 0
        while True:
            await limiter.wait()
            try:
                async with client.stream("GET", current, timeout=self.limits.timeout_s) as resp:
                    status = resp.status_code
                    if status in (301, 302, 303, 307, 308):
                        location = resp.headers.get("location")
                        hops += 1
                        if not location or hops > MAX_REDIRECTS:
                            return self._result(url, current, status, "too_many_redirects", started, FetchOutcome.FAILED)
                        try:
                            target = normalize_url(location, base=current)
                        except RagError:
                            return self._result(url, current, status, "redirect_invalid_url", started, FetchOutcome.SKIPPED)
                        in_scope, why = self.scope.contains(target)
                        if not in_scope:
                            return self._result(url, target, status, f"redirect_{why}", started, FetchOutcome.SKIPPED)
                        if not self._robots_allows(target):
                            return self._result(url, target, status, "redirect_robots_disallowed", started, FetchOutcome.SKIPPED)
                        current = target
                        continue
                    if status in RETRY_STATUS and attempt < self.limits.max_retries:
                        attempt += 1
                        retry_after = resp.headers.get("retry-after", "")
                        wait = float(retry_after) if retry_after.isdigit() else 1.5 * 2 ** (attempt - 1)
                        if wait > 30:
                            return self._result(url, current, status, f"http_{status}_retry_after_too_long", started, FetchOutcome.FAILED)
                        await asyncio.sleep(wait)
                        continue
                    if status != 200:
                        return self._result(url, current, status, f"http_{status}", started, FetchOutcome.FAILED)
                    ctype = resp.headers.get("content-type", "").lower()
                    if "html" not in ctype:
                        return self._result(url, current, status, "not_html", started, FetchOutcome.SKIPPED)
                    declared = resp.headers.get("content-length")
                    if declared and declared.isdigit() and int(declared) > self.limits.max_bytes_per_page:
                        return self._result(url, current, status, "too_large", started, FetchOutcome.SKIPPED)
                    chunks: list[bytes] = []
                    size = 0
                    async for part in resp.aiter_bytes():
                        size += len(part)
                        if size > self.limits.max_bytes_per_page:
                            return self._result(url, current, status, "too_large", started, FetchOutcome.SKIPPED, size)
                        chunks.append(part)
                    body = b"".join(chunks)
                    encoding = resp.encoding or "utf-8"
                    html = body.decode(encoding, errors="replace")
                    result = self._result(url, current, status, "fetched", started, FetchOutcome.ACCEPTED, size)
                    result.html = html
                    return result
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                if attempt < self.limits.max_retries:
                    attempt += 1
                    await asyncio.sleep(1.5 * 2 ** (attempt - 1))
                    continue
                reason = "timeout" if isinstance(exc, httpx.TimeoutException) else "connection_error"
                return self._result(url, current, None, reason, started, FetchOutcome.FAILED)

    @staticmethod
    def _result(url, final, status, reason, started, outcome, size=0) -> FetchResult:  # noqa: ANN001
        return FetchResult(url, final, status, None, outcome, reason, size, int((time.monotonic() - started) * 1000))

    # ------------------------------------------------------------------ crawl
    async def run(self) -> CrawlResult:
        seed = normalize_url(self.site.seed_url)
        in_scope, why = self.scope.contains(seed)
        if not in_scope:
            raise RagError(ErrorCode.URL_REJECTED, f"Seed URL is outside its own scope ({why}).", stage="crawl")
        if self.check_network:
            assert_public_host(self.site.allowed_host)

        limits = self.limits
        client_kwargs = dict(
            headers={"User-Agent": USER_AGENT, "Accept": "text/html,application/xhtml+xml"},
            follow_redirects=False,
            limits=httpx.Limits(max_connections=limits.concurrency),
        )
        if self.transport is not None:
            client_kwargs["transport"] = self.transport

        result = CrawlResult(pages=[], log=[])
        async with httpx.AsyncClient(**client_kwargs) as client:
            result.robots_url = await self._load_robots(client)
            if not self._robots_allows(seed):
                raise RagError(
                    ErrorCode.ROBOTS_DISALLOWED,
                    f"robots.txt disallows crawling the seed URL {seed}.",
                    hint="Choose a different section or site.",
                    stage="crawl",
                )
            delay = limits.delay_s
            if self.robots is not None:
                crawl_delay = self.robots.crawl_delay(USER_AGENT)
                if crawl_delay:
                    delay = max(delay, float(crawl_delay))
            limiter = _RateLimiter(delay)

            frontier: deque[tuple[str, int]] = deque([(seed, 0)])
            seen = {seed}
            hashes: dict[str, str] = {}
            deadline = time.monotonic() + limits.max_total_s
            max_fetches = limits.max_pages * 3
            fetches = 0

            while frontier:
                if len(result.pages) >= limits.max_pages:
                    result.stopped_reason = "max_pages_reached"
                    break
                if time.monotonic() > deadline:
                    result.stopped_reason = "time_budget_exhausted"
                    break
                if fetches >= max_fetches:
                    result.stopped_reason = "fetch_budget_exhausted"
                    break
                batch = []
                while frontier and len(batch) < limits.concurrency:
                    batch.append(frontier.popleft())
                fetches += len(batch)
                fetched = await asyncio.gather(*(self._fetch(client, u, limiter) for u, _ in batch))
                for (url, depth), fr in zip(batch, fetched):
                    self._process(fr, depth, result, frontier, seen, hashes)
                    if len(result.pages) >= limits.max_pages:
                        break
            for url, depth in frontier:
                result.log.append(CrawlLogEntry(url=url, outcome=FetchOutcome.SKIPPED, reason="not_fetched_limit_reached", depth=depth))
        return result

    def _process(self, fr: FetchResult, depth: int, result: CrawlResult, frontier, seen, hashes) -> None:  # noqa: ANN001
        entry = CrawlLogEntry(
            url=fr.url, final_url=fr.final_url, outcome=fr.outcome, reason=fr.reason,
            status_code=fr.status, depth=depth, bytes=fr.bytes, elapsed_ms=fr.elapsed_ms,
        )
        if fr.outcome != FetchOutcome.ACCEPTED or fr.html is None:
            result.log.append(entry)
            self.progress("page", {"url": fr.url, "outcome": entry.outcome.value, "reason": entry.reason})
            return

        final_url = fr.final_url or fr.url
        page: ExtractedPage = extract_page(fr.html)

        if not page.nofollow and depth < self.limits.max_depth:
            for href in page.links:
                try:
                    link = normalize_url(href, base=final_url)
                except RagError:
                    continue
                if link in seen:
                    continue
                seen.add(link)
                ok, why = self.scope.contains(link)
                if not ok:
                    continue  # out-of-scope links are not logged individually (too noisy)
                if not self._robots_allows(link):
                    result.log.append(CrawlLogEntry(url=link, outcome=FetchOutcome.SKIPPED, reason="robots_disallowed", depth=depth + 1))
                    continue
                frontier.append((link, depth + 1))

        reason = None
        if page.noindex:
            reason = "meta_robots_noindex"
        elif page.word_count < self.limits.min_page_words:
            reason = "likely_javascript_rendered" if page.js_required_hint else "low_content"
        chash = content_hash(page.sections)
        if reason is None and chash in hashes:
            reason = f"duplicate_of:{hashes[chash]}"
        if final_url != fr.url and final_url in {p.final_url for p in result.pages}:
            reason = "duplicate_final_url"

        if reason:
            entry.outcome = FetchOutcome.SKIPPED
            entry.reason = reason
            result.log.append(entry)
            self.progress("page", {"url": fr.url, "outcome": "skipped", "reason": reason})
            return

        pid = page_id_for(final_url)
        hashes[chash] = final_url
        record = PageRecord(
            page_id=pid, site_id=self.site.site_id, corpus_id=self.corpus_id,
            requested_url=fr.url, final_url=final_url, title=page.title, fetched_at=utcnow(),
            content_hash=chash, depth=depth, word_count=page.word_count, sections=page.sections,
        )
        result.pages.append(record)
        result.raw_html[pid] = fr.html
        entry.reason = "accepted"
        result.log.append(entry)
        self.progress("page", {"url": final_url, "outcome": "accepted", "title": page.title, "words": page.word_count, "accepted": len(result.pages)})


def crawl_site(site: SiteRecord, corpus_id: str, **kwargs) -> CrawlResult:  # noqa: ANN003
    return asyncio.run(Crawler(site, corpus_id, **kwargs).run())


def save_crawl(result: CrawlResult, corpus_dir: Path) -> None:
    corpus_dir.mkdir(parents=True, exist_ok=True)
    with (corpus_dir / "pages.jsonl").open("w", encoding="utf-8") as fh:
        for page in result.pages:
            fh.write(page.model_dump_json() + "\n")
    with (corpus_dir / "crawl_log.jsonl").open("w", encoding="utf-8") as fh:
        for entry in result.log:
            fh.write(entry.model_dump_json() + "\n")
    raw_dir = corpus_dir / "raw"
    raw_dir.mkdir(exist_ok=True)
    for pid, html in result.raw_html.items():
        (raw_dir / f"{pid}.html.gz").write_bytes(gzip.compress(html.encode("utf-8")))


def load_pages(corpus_dir: Path) -> list[PageRecord]:
    path = corpus_dir / "pages.jsonl"
    return [PageRecord.model_validate_json(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
