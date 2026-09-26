"""Main-content extraction with heading paths and real anchors.

- Links for discovery are collected from the whole page (navigation included) before any
  cleanup; content extraction then works on the main container only.
- Navigation, sidebars, footers, scripts, forms, and Sphinx permalink glyphs are removed.
- Headings split the page into sections. A section anchor is set only when the page
  itself defines that id (heading id, permalink href, or enclosing section id).
- Code blocks are kept as fenced blocks, lists as "- " items, tables as " | " rows,
  and API definition lists as signature + description.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from bs4 import BeautifulSoup, NavigableString, Tag

from .schemas import Section

EXTRACTOR_VERSION = "extract-v2"

MAIN_SELECTORS = [
    '[itemprop="articleBody"]', '[role="main"]', "main", "article", "#content", ".document", "body",
]
REMOVE_SELECTORS = [
    "script", "style", "noscript", "nav", "footer", "aside", "form", "button", "svg", "iframe",
    "template", "a.headerlink", ".headerlink", ".toctree-wrapper", ".sphinxsidebar", ".related",
    ".breadcrumbs", ".wy-breadcrumbs", '[role="navigation"]', ".rst-footer-buttons",
    ".prev-next-area", ".cookie-banner", "#cookie-banner", ".skip-link", '[aria-hidden="true"]',
]
HEADINGS = {"h1": 1, "h2": 2, "h3": 3, "h4": 4, "h5": 5, "h6": 6}
CONTAINERS = {
    "div", "section", "article", "main", "blockquote", "figure", "details", "summary",
    "center", "header", "span", "dd", "li", "body", "html",
}


@dataclass
class ExtractedPage:
    title: str
    sections: list[Section]
    links: list[str]
    canonical: str | None
    noindex: bool
    nofollow: bool
    word_count: int
    js_required_hint: bool = False
    notes: list[str] = field(default_factory=list)


def _collapse(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _inline_text(node: Tag) -> str:
    return _collapse(node.get_text(""))


def _table_text(table: Tag) -> str:
    rows = []
    for tr in table.find_all("tr"):
        cells = [_inline_text(c) for c in tr.find_all(["th", "td"])]
        if any(cells):
            rows.append(" | ".join(cells))
    return "\n".join(rows)


def _pre_text(pre: Tag) -> str:
    code = pre.get_text("").rstrip("\n")
    code = "\n".join(line.rstrip() for line in code.splitlines())
    return f"```\n{code}\n```" if code.strip() else ""


class _SectionBuilder:
    def __init__(self, page_title: str, ids: set[str]) -> None:
        self.ids = ids
        self.stack: list[tuple[int, str, str | None]] = []  # (level, text, anchor)
        self.blocks: list[str] = []
        self.sections: list[Section] = []
        self.page_title = page_title

    def heading(self, level: int, text: str, anchor: str | None) -> None:
        self.flush()
        while self.stack and self.stack[-1][0] >= level:
            self.stack.pop()
        self.stack.append((level, text, anchor if anchor in self.ids else None))

    def block(self, text: str) -> None:
        if text and text.strip():
            self.blocks.append(text.strip())

    def flush(self) -> None:
        if not self.blocks:
            return
        path = [t for _, t, _ in self.stack] or [self.page_title]
        anchor = next((a for _, _, a in reversed(self.stack) if a), None)
        # Only the innermost heading's own anchor is precise; fall back to the nearest ancestor.
        self.sections.append(Section(heading_path=path, anchor=anchor, text="\n\n".join(self.blocks)))
        self.blocks = []


def _heading_anchor(h: Tag) -> str | None:
    if h.get("id"):
        return str(h["id"])
    link = h.find("a", class_="headerlink", href=True)
    if link and str(link["href"]).startswith("#") and len(str(link["href"])) > 1:
        return str(link["href"])[1:]
    parent = h.parent
    if isinstance(parent, Tag) and parent.name in ("section", "div") and parent.get("id"):
        first_heading = parent.find(list(HEADINGS))
        if first_heading is h:
            return str(parent["id"])
    return None


def _walk(node: Tag, builder: _SectionBuilder) -> None:
    for child in node.children:
        if isinstance(child, NavigableString):
            if type(child).__name__ in ("Comment", "Doctype", "ProcessingInstruction"):
                continue
            builder.block(_collapse(str(child)))
            continue
        if not isinstance(child, Tag):
            continue
        name = child.name
        if name in HEADINGS:
            builder.heading(HEADINGS[name], _inline_text(child), child.get("data-anchor"))
        elif name == "pre":
            builder.block(_pre_text(child))
        elif name == "table":
            builder.block(_table_text(child))
        elif name in ("ul", "ol"):
            for li in child.find_all("li", recursive=False):
                if li.find(["pre", "table", *HEADINGS]):
                    _walk(li, builder)
                else:
                    builder.block("- " + _inline_text(li))
        elif name == "dl":
            for item in child.children:
                if isinstance(item, Tag) and item.name == "dt":
                    if item.get('id') in builder.ids:
                        builder.heading(6, _inline_text(item), str(item['id']))
                    builder.block(_inline_text(item))
                elif isinstance(item, Tag) and item.name == "dd":
                    _walk(item, builder)
        elif name in ("p", "caption", "figcaption", "label"):
            builder.block(_inline_text(child))
        elif name in CONTAINERS or name not in ("img", "br", "hr", "input", "select", "option", "link", "meta"):
            _walk(child, builder)


def extract_page(html: str) -> ExtractedPage:
    soup = BeautifulSoup(html, "lxml")

    meta_robots = ""
    for meta in soup.find_all("meta", attrs={"name": re.compile("^robots$", re.I)}):
        meta_robots += " " + str(meta.get("content", "")).lower()
    canonical_tag = soup.find("link", rel=lambda v: v and "canonical" in v)
    canonical = str(canonical_tag["href"]) if canonical_tag and canonical_tag.get("href") else None

    links = [str(a["href"]) for a in soup.find_all("a", href=True)]
    ids = {str(t["id"]) for t in soup.find_all(id=True)}

    html_title = _collapse(soup.title.get_text()) if soup.title else ""
    main = None
    for selector in MAIN_SELECTORS:
        main = soup.select_one(selector)
        if main is not None:
            break
    if main is None:
        main = soup

    # Resolve heading anchors before removing permalink elements.
    for h in main.find_all(list(HEADINGS)):
        anchor = _heading_anchor(h)
        if anchor:
            h["data-anchor"] = anchor
    for selector in REMOVE_SELECTORS:
        for el in main.select(selector):
            el.decompose()

    h1 = main.find("h1")
    title = _inline_text(h1) if h1 else html_title
    title = title or "Untitled page"

    builder = _SectionBuilder(title, ids)
    _walk(main, builder)
    builder.flush()

    word_count = sum(len(s.text.split()) for s in builder.sections)
    page_text = soup.get_text(" ").lower()
    js_hint = word_count < 50 and (
        "enable javascript" in page_text or "requires javascript" in page_text
        or bool(soup.find(id=re.compile("^(root|app|__next)$")))
    )
    return ExtractedPage(
        title=title,
        sections=builder.sections,
        links=links,
        canonical=canonical,
        noindex="noindex" in meta_robots,
        nofollow="nofollow" in meta_robots,
        word_count=word_count,
        js_required_hint=js_hint,
    )
