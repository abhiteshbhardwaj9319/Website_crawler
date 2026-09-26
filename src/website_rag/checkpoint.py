"""Disk-backed crawl pages and atomic frontier checkpoints; never changes active indexes."""
from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator, Sequence
from pathlib import Path

from .errors import ErrorCode, RagError
from .registry import _atomic_write
from .schemas import CrawlLogEntry, FetchOutcome, PageRecord, SiteRecord


def scope_identity(site: SiteRecord) -> dict:
    return {"host": site.allowed_host, "paths": [site.allowed_path_prefix, *site.additional_path_prefixes],
            "seeds": [site.seed_url, *site.seed_urls], "exclusions": site.exclude_patterns,
            "sitemaps": site.sitemap_urls, "version": site.scope_version}


def scope_fingerprint(site: SiteRecord) -> str:
    return hashlib.sha256(json.dumps(scope_identity(site), sort_keys=True).encode()).hexdigest()[:16]


class DiskPages(Sequence):
    """Page bodies load one at a time; only page filenames stay in memory."""
    def __init__(self, directory: Path, corpus_id: str):
        self.directory, self.corpus_id = directory, corpus_id
        directory.mkdir(parents=True, exist_ok=True)
        self.paths = sorted(directory.glob('*.json'))

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[i] for i in range(*index.indices(len(self))) ]
        page = PageRecord.model_validate_json(self.paths[index].read_text(encoding='utf-8'))
        return page.model_copy(update={"corpus_id": self.corpus_id})

    def __iter__(self) -> Iterator[PageRecord]:
        for i in range(len(self)):
            yield self[i]

    def append(self, page: PageRecord) -> None:
        path = self.directory / f'{page.page_id}.json'
        _atomic_write(path, page.model_dump_json())
        if path not in self.paths:
            self.paths.append(path)


def load_checkpoint(directory: Path, site: SiteRecord) -> dict | None:
    path = directory / 'frontier.json'
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding='utf-8'))
    old, new = data['scope'], scope_identity(site)
    # Raising limits and explicitly adding scope are safe. Narrowing requires refresh.
    if old['host'] != new['host'] or not set(old['paths']).issubset(new['paths']) or old['exclusions'] != new['exclusions']:
        raise RagError(ErrorCode.CONFIG_INVALID, 'Resume scope is incompatible with the saved crawl.',
                       hint='Use ingest --refresh to start a new snapshot.', stage='crawl')
    old_limits = data.get('limits', {})
    if old_limits.get('min_page_words', site.crawl.min_page_words) != site.crawl.min_page_words:
        raise RagError(ErrorCode.CONFIG_INVALID, 'Extraction policy changed; start a refresh.', stage='crawl')
    return data


def save_checkpoint(directory: Path, site: SiteRecord, result, frontier, seen) -> None:
    # One current outcome per URL, retained failures can be retried on the next resume.
    latest = {e.url: e for e in result.log if e.outcome != FetchOutcome.DEFERRED}
    deferred = [(e.url, e.depth) for e in result.log if e.outcome == FetchOutcome.DEFERRED]
    pending = list(dict.fromkeys((u, d) for u, d in [*frontier, *deferred] if u not in latest))
    payload = {'schema_version': 1, 'scope': scope_identity(site), 'limits': site.crawl.model_dump(),
               'scope_fingerprint': scope_fingerprint(site), 'frontier': pending, 'seen': sorted(seen),
               'outcomes': [e.model_dump(mode='json') for e in latest.values()],
               'stopped_reason': result.stopped_reason}
    _atomic_write(directory/'frontier.json', json.dumps(payload, indent=2))


def coverage(directory: Path) -> dict:
    path = directory/'frontier.json'
    if not path.exists():
        return {'crawl_complete': False, 'stop_reason': 'legacy_snapshot_no_checkpoint', 'pending': None}
    data = json.loads(path.read_text(encoding='utf-8'))
    rows = data['outcomes']
    reasons: dict[str, int] = {}
    for e in rows:
        key = 'duplicate_content' if e['reason'].startswith('duplicate_of:') else e['reason']
        reasons[key] = reasons.get(key, 0) + 1
    counts = {k: sum(e['outcome'] == k for e in rows) for k in ('accepted','skipped','failed')}
    pending = len(data['frontier'])
    discovered = len({e['url'] for e in rows} | {u for u, _ in data['frontier']})
    return {'scope': data['scope'], 'scope_fingerprint': data['scope_fingerprint'], 'limits': data['limits'],
            'discovery_sources': ['HTML links', 'explicit seeds', 'bounded sitemaps'], **counts,
            'pending': pending, 'discovered_in_scope': discovered, 'reasons': reasons,
            'processed_of_discovered': f'{len(rows)}/{discovered}',
            'crawl_complete': not pending and not counts['failed'] and data['stopped_reason'] == 'frontier_exhausted',
            'stop_reason': data['stopped_reason']}
