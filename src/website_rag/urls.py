"""URL normalization, crawl scope, and network-safety checks.

Policy (documented in docs/corpus.md):
- Only http/https. Scheme and host are lowercased, default ports and fragments removed.
- Query strings are dropped: the supported targets are static pages whose content does not
  vary by query, and dropping them removes search/sort/session URL traps.
- A trailing `index.html` is equivalent to its directory URL.
- Scope is an exact host match plus a path prefix. Subdomains are out of scope.
- Hosts that are, or resolve to, non-global addresses (loopback, private, link-local,
  reserved) are rejected. Resolution is checked before every request, including each
  redirect hop.
"""

from __future__ import annotations

import ipaddress
import posixpath
import re
import socket
from dataclasses import dataclass
from urllib.parse import urlsplit, urlunsplit

from .errors import ErrorCode, RagError

NON_HTML_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".gif", ".svg", ".webp", ".ico", ".css", ".js", ".json",
    ".xml", ".pdf", ".zip", ".gz", ".tar", ".tgz", ".whl", ".txt", ".rst", ".md",
    ".mp4", ".mp3", ".woff", ".woff2", ".ttf", ".eot", ".epub", ".csv", ".py",
}

DEFAULT_EXCLUDE_PATTERNS = [
    r"/_sources/", r"/_static/", r"/_images/", r"/_downloads/",
    r"/genindex(\.html)?$", r"/search(\.html)?$", r"/py-modindex(\.html)?$",
    r"/(login|logout|signin|signup|register)(/|$)", r"/tag/", r"/page/\d+",
]


@dataclass(frozen=True)
class Scope:
    host: str
    path_prefix: str
    exclude_patterns: tuple[str, ...] = ()
    additional_prefixes: tuple[str, ...] = ()

    def contains(self, url: str) -> tuple[bool, str]:
        """Return (in_scope, reason)."""
        parts = urlsplit(url)
        if parts.scheme not in ("http", "https"):
            return False, "non_http_scheme"
        if (parts.hostname or "") != self.host:
            return False, "other_host"
        path = parts.path or "/"
        if not any(path.startswith(p) for p in (self.path_prefix, *self.additional_prefixes)):
            return False, "outside_path_prefix"
        ext = posixpath.splitext(path)[1].lower()
        if ext in NON_HTML_EXTENSIONS:
            return False, "non_html_extension"
        for pattern in (*DEFAULT_EXCLUDE_PATTERNS, *self.exclude_patterns):
            if re.search(pattern, path):
                return False, "excluded_pattern"
        return True, "in_scope"


def normalize_url(url: str, base: str | None = None) -> str:
    """Return the canonical form used for deduplication and fetching."""
    from urllib.parse import urljoin

    raw = urljoin(base, url.strip()) if base else url.strip()
    parts = urlsplit(raw)
    scheme = parts.scheme.lower()
    if scheme not in ("http", "https"):
        raise RagError(
            ErrorCode.URL_REJECTED,
            f"Only http(s) URLs are supported (got scheme '{scheme or 'none'}').",
            stage="url",
        )
    host = (parts.hostname or "").lower().rstrip(".")
    if not host:
        raise RagError(ErrorCode.URL_REJECTED, "URL has no host.", stage="url")
    port = parts.port
    netloc = host
    if port and not ((scheme == "http" and port == 80) or (scheme == "https" and port == 443)):
        netloc = f"{host}:{port}"
    path = parts.path or "/"
    path = re.sub(r"/{2,}", "/", path)
    trailing = path.endswith("/")
    path = posixpath.normpath(path)
    if path == ".":
        path = "/"
    if trailing and not path.endswith("/"):
        path += "/"
    if path.endswith("/index.html"):
        path = path[: -len("index.html")]
    return urlunsplit((scheme, netloc, path, "", ""))


def derive_scope(seed_url: str) -> tuple[str, str]:
    """Allowed host and path prefix for a seed URL: the seed's directory."""
    normalized = normalize_url(seed_url)
    parts = urlsplit(normalized)
    path = parts.path
    prefix = path if path.endswith("/") else posixpath.dirname(path) + "/"
    return parts.hostname or "", prefix


def assert_public_host(host: str) -> None:
    """Reject hosts that are or resolve to non-global addresses."""
    if host in ("localhost",) or host.endswith(".localhost") or host.endswith(".local"):
        raise RagError(
            ErrorCode.URL_REJECTED,
            f"Refusing to crawl local host '{host}'.",
            hint="Only public websites can be ingested.",
            stage="url",
        )
    try:
        literal = ipaddress.ip_address(host)
        addresses = [literal]
    except ValueError:
        try:
            infos = socket.getaddrinfo(host, None, proto=socket.IPPROTO_TCP)
        except socket.gaierror as exc:
            raise RagError(
                ErrorCode.FETCH_FAILED,
                f"Could not resolve host '{host}'.",
                hint="Check the URL spelling and your network connection.",
                stage="url",
            ) from exc
        addresses = [ipaddress.ip_address(info[4][0].split("%")[0]) for info in infos]
    for addr in addresses:
        if not addr.is_global:
            raise RagError(
                ErrorCode.URL_REJECTED,
                f"Refusing to crawl '{host}': it resolves to a non-public address.",
                hint="Private, loopback, and link-local networks are blocked.",
                stage="url",
            )


def slugify(text: str, max_len: int = 48) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug[:max_len].strip("-") or "site"
