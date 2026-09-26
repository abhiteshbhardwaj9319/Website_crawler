"""Bounded XML discovery. Sitemaps obey the same host/path, robots and redirect policy."""
from __future__ import annotations

from urllib.parse import urlsplit

from lxml import etree

from .errors import ErrorCode, RagError
from .urls import assert_public_host, normalize_url


async def discover(crawler, client, limiter) -> list[str]:
    limits = crawler.limits
    queue = [(u, 0) for u in crawler.site.sitemap_urls]
    seen, pages = set(), set()
    # Robots discovery is optional; XML outside selected paths is never fetched.
    queue.extend((u, 0) for u in (crawler.robots.site_maps() or []))
    while queue and len(seen) < limits.max_sitemaps and len(pages) < limits.max_sitemap_urls:
        raw, depth = queue.pop(0)
        url = normalize_url(raw)
        if url in seen or depth > limits.max_sitemap_depth:
            continue
        seen.add(url)
        for _ in range(6):
            parts = urlsplit(url)
            # Scope.contains excludes XML pages; use a temporary HTML extension for policy only.
            policy_url = url.rsplit('.', 1)[0] + '.html' if parts.path.endswith('.xml') else url
            if not crawler.scope.contains(policy_url)[0] or not crawler._robots_allows(url):
                break
            if crawler.check_network:
                assert_public_host(parts.hostname or '')
            await limiter.wait()
            async with client.stream('GET', url, timeout=limits.timeout_s) as response:
                if response.status_code in (301, 302, 303, 307, 308):
                    if not response.headers.get('location'):
                        break
                    url = normalize_url(response.headers['location'], base=url)
                    continue
                if response.status_code != 200:
                    break
                body = bytearray()
                async for part in response.aiter_bytes():
                    body.extend(part)
                    if len(body) > limits.max_bytes_per_page:
                        raise RagError(ErrorCode.FETCH_FAILED, 'Sitemap exceeds byte limit.', stage='crawl')
                if b'<!DOCTYPE' in body.upper() or b'<!ENTITY' in body.upper():
                    raise RagError(ErrorCode.URL_REJECTED, 'Sitemap DTD/entities are forbidden.', stage='crawl')
                try:
                    root = etree.fromstring(bytes(body), etree.XMLParser(resolve_entities=False, no_network=True))
                except etree.XMLSyntaxError as exc:
                    raise RagError(ErrorCode.FETCH_FAILED, 'Sitemap XML is malformed.', stage='crawl') from exc
                nested = etree.QName(root).localname == 'sitemapindex'
                for element in root.iter():
                    if not isinstance(element.tag, str) or etree.QName(element).localname != 'loc' or not element.text:
                        continue
                    target = normalize_url(element.text, base=url)
                    if nested:
                        if len(queue) < limits.max_sitemaps:
                            queue.append((target, depth + 1))
                        else:
                            crawler.discovery_limited = True
                    elif crawler.scope.contains(target)[0] and crawler._robots_allows(target):
                        pages.add(target)
                        if len(pages) >= limits.max_sitemap_urls:
                            break
                break
    if queue or len(pages) >= limits.max_sitemap_urls:
        crawler.discovery_limited = True
    return sorted(pages)
