"""M2: extraction and crawling against an offline mock website."""

from __future__ import annotations

import httpx
import pytest

from website_rag.crawl import crawl_site
from website_rag.errors import ErrorCode, RagError
from website_rag.extract import extract_page
from website_rag.schemas import CrawlLimits, SiteRecord

LOREM = " ".join(f"word{i}" for i in range(120))


def page(title: str, body: str, links: str = "") -> str:
    return f"""<html><head><title>{title} - Docs</title>
    <link rel="canonical" href="https://other.example/latest/x.html"></head>
    <body><nav><a href="/docs/nav-only.html">Nav</a>{links}</nav>
    <div role="main"><section id="{title.lower()}"><h1>{title}<a class="headerlink" href="#{title.lower()}">¶</a></h1>
    {body}</section></div><footer>Copyright footer text</footer></body></html>"""


SPHINX_LIKE = """<html><head><title>Item Pipeline</title></head><body>
<div class="sphinxsidebar">Sidebar navigation should vanish</div>
<div itemprop="articleBody">
<section id="item-pipeline"><h1>Item Pipeline<a class="headerlink" href="#item-pipeline">¶</a></h1>
<p>Pipelines process   items.</p>
<ul><li><p>cleansing HTML</p></li><li>validating data</li></ul>
<section id="activating"><h2>Activating a component<a class="headerlink" href="#activating">¶</a></h2>
<div class="highlight"><pre>ITEM_PIPELINES = {
    "myproject.pipelines.PricePipeline": 300,
}</pre></div>
<table><tr><th>Setting</th><th>Default</th></tr><tr><td>LOG_LEVEL</td><td>DEBUG</td></tr></table>
<h3>No anchor heading</h3><p>Text under a heading that has no id.</p>
</section></section></div></body></html>"""


def test_extract_sections_anchors_code_tables_and_navigation_removal():
    page_ = extract_page(SPHINX_LIKE)
    assert page_.title == "Item Pipeline"
    paths = [s.heading_path for s in page_.sections]
    assert paths[0] == ["Item Pipeline"] and page_.sections[0].anchor == "item-pipeline"
    assert paths[1] == ["Item Pipeline", "Activating a component"] and page_.sections[1].anchor == "activating"
    # heading without its own id keeps the nearest real ancestor anchor, never an invented one
    assert paths[2][-1] == "No anchor heading" and page_.sections[2].anchor == "activating"
    text = "\n".join(s.text for s in page_.sections)
    assert "Sidebar navigation" not in text and "¶" not in text
    assert '```\nITEM_PIPELINES = {\n    "myproject.pipelines.PricePipeline": 300,\n}\n```' in text
    assert "- cleansing HTML" in text and "LOG_LEVEL | DEBUG" in text
    assert "Pipelines process items." in text


def test_extract_meta_robots_and_js_hint():
    html = '<html><head><meta name="robots" content="noindex,nofollow"></head><body><div id="root"></div><noscript>Please enable JavaScript</noscript></body></html>'
    p = extract_page(html)
    assert p.noindex and p.nofollow and p.js_required_hint and p.word_count == 0


# ------------------------------------------------------------------ crawler


def make_site(**limits) -> SiteRecord:
    return SiteRecord(
        site_id="mock", number=9, display_name="Mock", seed_url="https://docs.example.com/docs/",
        allowed_host="docs.example.com", allowed_path_prefix="/docs/",
        crawl=CrawlLimits(delay_s=0, concurrency=2, max_retries=1, min_page_words=50, **limits),
    )


def mock_site(robots: str = "User-agent: *\nDisallow: /docs/private/\n"):
    calls: dict[str, int] = {}
    pages = {
        "/docs/": page("Home", f"<p>{LOREM}</p>", '<a href="a.html#frag">A</a><a href="b.html?utm=1">B</a>'
                       '<a href="https://external.example/docs/x.html">ext</a><a href="/other/x.html">other path</a>'
                       '<a href="private/secret.html">private</a><a href="dup.html">dup</a><a href="thin.html">thin</a>'
                       '<a href="old.html">old</a><a href="escape.html">escape</a><a href="img.png">img</a>'
                       '<a href="flaky.html">flaky</a><a href="missing.html">missing</a><a href="pdf.html">pdf</a>'),
        "/docs/a.html": page("Alpha", f"<p>alpha {LOREM}</p>", '<a href="../docs/./b.html#x">B again</a>'),
        "/docs/b.html": page("Beta", f"<p>beta {LOREM}</p>"),
        "/docs/dup.html": page("Alpha", f"<p>alpha {LOREM}</p>"),
        "/docs/thin.html": page("Thin", "<p>too short</p>"),
        "/docs/new.html": page("New", f"<p>new {LOREM}</p>"),
    }

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        calls[path] = calls.get(path, 0) + 1
        if path == "/robots.txt":
            return httpx.Response(200, text=robots)
        if path == "/docs/old.html":
            return httpx.Response(301, headers={"location": "/docs/new.html"})
        if path == "/docs/escape.html":
            return httpx.Response(302, headers={"location": "https://evil.example/steal"})
        if path == "/docs/flaky.html":
            if calls[path] == 1:
                return httpx.Response(503)
            return httpx.Response(200, html=page("Flaky", f"<p>flaky {LOREM}</p>"))
        if path == "/docs/pdf.html":
            return httpx.Response(200, content=b"%PDF", headers={"content-type": "application/pdf"})
        if path in pages:
            return httpx.Response(200, html=pages[path])
        return httpx.Response(404)

    return httpx.MockTransport(handler), calls


def test_crawl_scope_redirects_robots_dedupe_and_retries():
    transport, calls = mock_site()
    result = crawl_site(make_site(max_pages=20), "c1", transport=transport, check_network=False)
    accepted = {p.final_url for p in result.pages}
    assert accepted == {
        "https://docs.example.com/docs/", "https://docs.example.com/docs/a.html",
        "https://docs.example.com/docs/b.html", "https://docs.example.com/docs/new.html",
        "https://docs.example.com/docs/flaky.html",
    }
    reasons = {e.url.rsplit("/", 1)[-1]: e.reason for e in result.log}
    assert reasons["secret.html"] == "robots_disallowed"
    assert reasons["dup.html"].startswith("duplicate_of:")
    assert reasons["thin.html"] == "low_content"
    assert reasons["escape.html"] == "redirect_other_host"
    assert reasons["missing.html"] == "http_404"
    assert reasons["pdf.html"] == "not_html"
    assert calls["/docs/flaky.html"] == 2  # one bounded retry
    assert "/other/x.html" not in calls and "/docs/img.png" not in calls
    assert calls["/docs/b.html"] == 1  # fragment/query/relative variants deduplicated
    new = next(p for p in result.pages if p.final_url.endswith("new.html"))
    assert new.requested_url.endswith("old.html")  # provenance of the redirect preserved
    assert all(p.site_id == "mock" and p.corpus_id == "c1" for p in result.pages)


def test_crawl_page_limit_is_enforced():
    transport, _ = mock_site()
    result = crawl_site(make_site(max_pages=2), "c1", transport=transport, check_network=False)
    assert len(result.pages) == 2 and result.stopped_reason == "max_pages_reached"


def test_crawl_robots_disallowing_seed_fails_clearly():
    transport, _ = mock_site(robots="User-agent: *\nDisallow: /\n")
    with pytest.raises(RagError) as err:
        crawl_site(make_site(), "c1", transport=transport, check_network=False)
    assert err.value.code == ErrorCode.ROBOTS_DISALLOWED
