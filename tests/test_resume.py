"""Checkpoint, coverage and bounded discovery regressions using a mock website."""
import json

import httpx
import pytest

from website_rag.checkpoint import coverage
from website_rag.crawl import crawl_site
from website_rag.errors import RagError
from website_rag.extract import extract_page
from .test_crawl_extract import make_site, mock_site, page, LOREM
from .test_pipeline import env, EMB
from website_rag.ingest import ingest_site
from website_rag.tracing import Tracer
from .helpers import mock_transport, SITE_A_PAGES


def test_resume_no_refetch_and_no_lost_batch(tmp_path):
    transport, calls = mock_site()
    site = make_site(max_pages=2)
    first = crawl_site(site, 'one', transport=transport, check_network=False, checkpoint_dir=tmp_path)
    assert len(first.pages) == 2 and first.pending
    before = dict(calls)
    site.crawl.max_pages = 20
    second = crawl_site(site, 'two', transport=transport, check_network=False, checkpoint_dir=tmp_path)
    assert len(second.pages) == 5
    for p in first.pages:
        assert calls[httpx.URL(p.requested_url).path] == before[httpx.URL(p.requested_url).path]
    outcomes = {e.url for e in second.log}
    assert all('https://docs.example.com' + p in outcomes for p in calls if p not in ('/robots.txt', '/docs/new.html'))
    assert not second.raw_html  # raw bodies are persisted, not retained
    assert coverage(tmp_path)['failed'] == 2  # missing.html and navigation-only fixture URL


def test_interrupt_keeps_frontier_and_saved_pages(tmp_path):
    transport, calls = mock_site()
    def interrupt(kind, data):
        if data.get('accepted') == 2:
            raise KeyboardInterrupt()
    site = make_site(max_pages=10)
    with pytest.raises(KeyboardInterrupt):
        crawl_site(site, 'one', transport=transport, check_network=False, checkpoint_dir=tmp_path, progress=interrupt)
    assert coverage(tmp_path)['stop_reason'] == 'interrupted'
    r = crawl_site(site, 'two', transport=transport, check_network=False, checkpoint_dir=tmp_path)
    assert len(r.pages) == 5 and calls['/docs/'] == 1 and calls['/docs/a.html'] == 1


def test_depth_deferred_then_resumed(tmp_path):
    transport, calls = mock_site()
    site = make_site(max_pages=20, max_depth=0)
    crawl_site(site, 'one', transport=transport, check_network=False, checkpoint_dir=tmp_path)
    assert coverage(tmp_path)['pending'] > 0 and not coverage(tmp_path)['crawl_complete']
    assert '/docs/a.html' not in calls
    site.crawl.max_depth = 3
    r = crawl_site(site, 'two', transport=transport, check_network=False, checkpoint_dir=tmp_path)
    assert len(r.pages) == 5 and calls['/docs/'] == 1


def test_sitemap_scope_redirect_and_entities(tmp_path):
    calls = []
    body = '<urlset><url><loc>https://docs.example.com/docs/only.html</loc></url><url><loc>https://evil.example/x</loc></url></urlset>'
    def handler(req):
        calls.append(str(req.url))
        if req.url.path == '/robots.txt':
            return httpx.Response(404)
        if req.url.path.endswith('.xml'):
            return httpx.Response(200, text=body)
        return httpx.Response(200, html=page(req.url.path, f'<p>{req.url.path} {LOREM}</p>'))
    site = make_site(max_pages=10)
    site.sitemap_urls = ['https://docs.example.com/docs/sitemap.xml']
    r = crawl_site(site, 'one', transport=httpx.MockTransport(handler), check_network=False, checkpoint_dir=tmp_path)
    assert any(p.final_url.endswith('only.html') for p in r.pages)
    assert not any('evil.example' in u for u in calls)
    body = '<!DOCTYPE x [<!ENTITY secret SYSTEM "file:///secret">]><urlset>&secret;</urlset>'
    with pytest.raises(RagError, match='DTD/entities'):
        crawl_site(site, 'two', transport=httpx.MockTransport(handler), check_network=False)


def test_reference_definition_keeps_real_anchor():
    extracted = extract_page('<main><h1>Functions</h1><dl><dt id="input">input(prompt)</dt><dd><p>Read a line from input.</p></dd>'
                             '<dt id="len">len(object)</dt><dd><p>Return the length.</p></dd></dl></main>')
    assert [(s.anchor, s.text) for s in extracted.sections] == [
        ('input', 'input(prompt)\n\nRead a line from input.'), ('len', 'len(object)\n\nReturn the length.')]


def test_scope_expansion_retries_blocked_redirect(tmp_path):
    def handler(req):
        if req.url.path == '/robots.txt':
            return httpx.Response(404)
        if req.url.path == '/docs/':
            return httpx.Response(301, headers={'location': '/reference/functions.html'})
        return httpx.Response(200, html=page('Functions', f'<p>{LOREM}</p>'))
    site = make_site(max_pages=10)
    first = crawl_site(site, 'one', transport=httpx.MockTransport(handler), check_network=False, checkpoint_dir=tmp_path)
    assert len(first.pages) == 0
    site.additional_path_prefixes = ['/reference/']
    second = crawl_site(site, 'two', transport=httpx.MockTransport(handler), check_network=False, checkpoint_dir=tmp_path)
    assert len(second.pages) == 1
    assert second.pages[0].requested_url.endswith('/docs/') and second.pages[0].final_url.endswith('/reference/functions.html')


def test_partial_transport_failure_cannot_delete_old_pages(env):
    settings, reg, ledger = env
    site = reg.get(1)
    old = site.active_corpus_id
    pages = dict(SITE_A_PAGES)
    del pages['/docs/settings.html']
    with pytest.raises(RagError, match='previous index remains active'):
        ingest_site(site, settings, reg, Tracer(None), ledger, embedder=EMB,
                    crawl_kwargs={'transport': mock_transport(pages), 'check_network': False})
    assert site.active_corpus_id == old
