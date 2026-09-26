"""Persistent site registry with stable display numbers.

Numbers are assigned once from a monotonic counter and never reassigned, regardless of
sorting, status changes, or failed ingestions. The registry is a single JSON file written
atomically; the tool is a single-process CLI, so no cross-process locking is attempted.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from pydantic import BaseModel, Field

from .config import CONFIG_DIR
from .errors import ErrorCode, RagError
from .schemas import CrawlLimits, SiteRecord, SiteStatus, utcnow
from .urls import assert_public_host, derive_scope, normalize_url, slugify

DEFAULT_SITES_FILE = CONFIG_DIR / "sites.default.json"


class RegistryFile(BaseModel):
    version: int = 1
    next_number: int = 1
    sites: list[SiteRecord] = Field(default_factory=list)


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


class SiteRegistry:
    def __init__(self, path: Path, defaults_file: Path | None = DEFAULT_SITES_FILE) -> None:
        self.path = path
        self.defaults_file = defaults_file
        self._data = self._load()

    # ------------------------------------------------------------------ io
    def _load(self) -> RegistryFile:
        if self.path.exists():
            try:
                return RegistryFile.model_validate_json(self.path.read_text(encoding="utf-8"))
            except ValueError as exc:
                raise RagError(
                    ErrorCode.CONFIG_INVALID,
                    f"Site registry at {self.path} is unreadable.",
                    hint="Restore it from backup or delete it to re-create the default sites.",
                    stage="registry",
                ) from exc
        data = RegistryFile()
        if self.defaults_file and self.defaults_file.exists():
            for entry in json.loads(self.defaults_file.read_text(encoding="utf-8")):
                host, prefix = derive_scope(entry["seed_url"])
                data.sites.append(
                    SiteRecord(
                        site_id=entry["site_id"],
                        number=data.next_number,
                        display_name=entry["display_name"],
                        seed_url=normalize_url(entry["seed_url"]),
                        allowed_host=host,
                        allowed_path_prefix=entry.get("allowed_path_prefix", prefix),
                        description=entry.get("description", ""),
                        crawl=CrawlLimits(**entry.get("crawl", {})),
                        exclude_patterns=entry.get("exclude_patterns", []),
                        additional_path_prefixes=entry.get('additional_path_prefixes', []),
                        seed_urls=entry.get('seed_urls', []),
                        sitemap_urls=entry.get('sitemap_urls', []),
                    )
                )
                data.next_number += 1
        self._data = data
        self.save()
        return data

    def save(self) -> None:
        _atomic_write(self.path, self._data.model_dump_json(indent=2))

    # ------------------------------------------------------------------ queries
    def list(self) -> list[SiteRecord]:
        return sorted(self._data.sites, key=lambda s: s.number)

    def get(self, ref: str | int | None) -> SiteRecord:
        """Resolve a site by number or site_id. None selects site 1 (the default)."""
        if ref is None or (isinstance(ref, str) and not ref.strip()):
            ref = 1
        if isinstance(ref, str) and ref.strip().isdigit():
            ref = int(ref.strip())
        for site in self._data.sites:
            if (isinstance(ref, int) and site.number == ref) or site.site_id == ref:
                return site
        numbers = ", ".join(str(s.number) for s in self.list()) or "none"
        raise RagError(
            ErrorCode.SITE_NOT_FOUND,
            f"Website '{ref}' is not registered.",
            hint=f"Registered website numbers: {numbers}. Run `rag sites list`.",
            stage="site_selection",
        )

    def require_queryable(self, ref: str | int | None) -> SiteRecord:
        site = self.get(ref)
        if not site.is_queryable:
            state = site.status.value
            raise RagError(
                ErrorCode.SITE_NOT_READY,
                f"Website {site.number} ({site.display_name}) has no completed index "
                f"(status: {state}).",
                hint=f"Run `rag ingest --site {site.number}` first.",
                stage="site_selection",
            )
        return site

    def find_equivalent(self, seed_url: str) -> SiteRecord | None:
        host, prefix = derive_scope(seed_url)
        for site in self._data.sites:
            if site.allowed_host == host and site.allowed_path_prefix == prefix:
                return site
        return None

    # ------------------------------------------------------------------ mutations
    def add(
        self,
        seed_url: str,
        display_name: str | None = None,
        crawl: CrawlLimits | None = None,
        check_network: bool = True,
    ) -> tuple[SiteRecord, bool]:
        """Register a website. Returns (site, created). Equivalent URLs return the existing site."""
        normalized = normalize_url(seed_url)
        existing = self.find_equivalent(normalized)
        if existing:
            return existing, False
        host, prefix = derive_scope(normalized)
        if check_network:
            assert_public_host(host)
        base_id = slugify(f"{host}{prefix}")
        site_id, n = base_id, 2
        taken = {s.site_id for s in self._data.sites}
        while site_id in taken:
            site_id, n = f"{base_id}-{n}", n + 1
        site = SiteRecord(
            site_id=site_id,
            number=self._data.next_number,
            display_name=display_name or f"{host}{prefix}".rstrip("/"),
            seed_url=normalized,
            allowed_host=host,
            allowed_path_prefix=prefix,
            crawl=crawl or CrawlLimits(),
        )
        self._data.next_number += 1
        self._data.sites.append(site)
        self.save()
        return site, True

    def update(self, site: SiteRecord) -> SiteRecord:
        site.updated_at = utcnow()
        for i, existing in enumerate(self._data.sites):
            if existing.site_id == site.site_id:
                if existing.number != site.number:
                    raise RagError(ErrorCode.INTERNAL, "Site numbers are immutable.")
                self._data.sites[i] = site
                self.save()
                return site
        raise RagError(ErrorCode.SITE_NOT_FOUND, f"Unknown site_id {site.site_id}.")

    def mark_status(self, site: SiteRecord, status: SiteStatus, error: str | None = None) -> None:
        site.status = status
        site.last_error = error
        self.update(site)
