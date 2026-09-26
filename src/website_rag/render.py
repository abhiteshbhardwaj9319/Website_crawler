"""Terminal rendering helpers.

All crawled text and model output is untrusted: it is wrapped in rich.Text (never parsed as
markup) after stripping terminal control sequences, so page content cannot recolor the
terminal, move the cursor, or inject fake UI. Color semantics are consistent: cyan = site/
identity, green = answered/ok, yellow = partial/warning/abstention, red = error, dim = meta.
"""

from __future__ import annotations

import re

from rich.console import Console, Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich import box

from .schemas import AnswerResult, SiteRecord

_ANSI = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]|\x1b\][^\x07]*(\x07|\x1b\\)|\x1b[@-Z\\-_]")
_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")
ASCII_UI = False


def ui_box():
    return box.ASCII if ASCII_UI else box.ROUNDED

STATUS_STYLE = {
    "answered": ("ANSWERED", "bold green"),
    "partially_answered": ("PARTIALLY ANSWERED", "bold yellow"),
    "insufficient_evidence": ("INSUFFICIENT EVIDENCE", "bold yellow"),
    "error": ("ERROR", "bold red"),
    "ready": ("ready", "green"),
    "registered": ("not ingested", "dim"),
    "ingesting": ("ingesting", "yellow"),
    "failed": ("failed", "red"),
}


def clean(text: str | None) -> str:
    if not text:
        return ""
    return _CONTROL.sub("", _ANSI.sub("", str(text)))


def safe(text: str | None, style: str = "") -> Text:
    return Text(clean(text), style=style)


def sites_table(sites: list[SiteRecord], selected: int | None = None) -> Table:
    table = Table(title="Websites", title_justify="left", header_style="bold", expand=False,
                  box=ui_box())
    table.add_column("#", justify="right", style="cyan")
    table.add_column("Website", overflow="fold")
    table.add_column("Scope", style="dim", overflow="fold")
    table.add_column("Status")
    table.add_column("Pages", justify="right")
    table.add_column("Chunks", justify="right")
    for site in sites:
        label, style = STATUS_STYLE.get(site.status.value, (site.status.value, ""))
        if site.status.value == "ingesting" and site.active_corpus_id:
            label = "ready (refresh in progress)"
        elif site.is_queryable:
            label = 'ready / crawl complete' if site.crawl_complete else 'ready / crawl limited'
        marker = " *" if selected == site.number else ""
        table.add_row(
            f"{site.number}{marker}", safe(site.display_name), safe(site.allowed_host + ' ' + ', '.join([site.allowed_path_prefix, *site.additional_path_prefixes])),
            Text(label, style=style), str(site.accepted_pages or "-"), str(site.chunk_count or "-"),
        )
    return table


def site_banner(site: SiteRecord) -> Text:
    t = Text()
    t.append("Website ", style="dim")
    t.append(f"{site.number}", style="bold cyan")
    t.append("  ")
    t.append(clean(site.display_name), style="cyan")
    t.append(f"  ({site.accepted_pages} pages, {site.chunk_count} chunks indexed from ", style="dim")
    paths = ', '.join([site.allowed_path_prefix, *site.additional_path_prefixes])
    t.append(clean(f"{site.allowed_host} {paths}"), style="dim")
    t.append(")", style="dim")
    return t


def claim_text(text: str) -> Text:
    """Render only code delimiters; never interpret arbitrary markup or links."""
    out = Text()
    for i, part in enumerate(re.split(r"(```[\s\S]*?```|`[^`\n]+`)", clean(text))):
        if part.startswith("```") and part.endswith("```"):
            out.append(part[3:-3].strip("\n"), style="cyan")
        elif part.startswith("`") and part.endswith("`"):
            out.append(part[1:-1], style="bold cyan")
        else:
            out.append(part)
    return out


def answer_panel(result: AnswerResult, show_retrieval: bool = False, explain: bool = False) -> Group:
    label, style = STATUS_STYLE[result.status]
    parts: list = [Text(label, style=style)]
    # Reconstruct even old saved runs from their accepted claims. Never replay unchecked prose.
    sources: dict[str, int] = {}
    marker_map: dict[int, int] = {}
    for c in result.citations:
        sources.setdefault(c.url, len(sources) + 1)
        marker_map[c.marker] = sources[c.url]
    if result.status in ("answered", "partially_answered"):
        for claim in result.claims:
            line = claim_text(claim.text)
            markers = sorted({marker_map[m] for m in claim.citations if m in marker_map})
            line.append(" " + "".join(f"[{m}]" for m in markers), style="cyan")
            parts.extend([line, Text("")])
    elif result.status == "insufficient_evidence":
        parts.append(safe(f"The indexed pages for {result.site_name} do not contain enough information to answer this question."))
    elif result.error:
        parts.append(safe(result.error.get("message", ""), "red"))
        parts.append(safe("Next step: " + result.error.get("hint", "")))
    if result.rejected_claims and result.status != "error":
        parts.append(Text(f"{len(result.rejected_claims)} statement(s) withheld: citation checks failed.", style="yellow"))
    elif result.status == "partially_answered":
        parts.append(Text("The retrieved evidence supports only part of the requested answer.", style="yellow"))
    for url, marker in sources.items():
        parts.append(safe(f"[{marker}] {url}", "dim"))
    if explain:
        for c in result.citations:
            parts.append(safe(f"\nEvidence [{marker_map[c.marker]}] / original marker {c.marker}: {c.section}\n"
                              f"{c.chunk_id}\n{c.quote}"))
        for r in result.rejected_claims:
            parts.append(safe(f"\nUNTRUSTED REJECTED CANDIDATE: {r.text}\nReasons: {'; '.join(r.reasons)}", "red"))
    if show_retrieval and result.retrieved:
        parts.append(retrieval_table(result))
    parts.append(meta_line(result))
    return Group(*parts)


def retrieval_table(result: AnswerResult, limit: int = 10) -> Table:
    in_context = set(result.context_chunk_ids)
    t = Table(title=f"Retrieved ({result.retrieval_mode})", title_justify="left", header_style="bold", box=None)
    for col in ("rank", "ctx", "score", "components", "page / section"):
        t.add_column(col)
    for r in result.retrieved[:limit]:
        comps = " ".join(f"{k}:{v}" for k, v in r.component_ranks.items())
        page = Text(clean(r.chunk.source_url.split("//", 1)[-1]))
        page.append(f"\n{clean(r.chunk.section_label)}", style="dim")
        t.add_row(str(r.rank), "yes" if r.chunk.chunk_id in in_context else "", f"{r.score:.4f}", comps, page)
    return t


def meta_line(result: AnswerResult) -> Text:
    u = result.usage or {}
    t = Text(style="dim")
    provider = clean(f"{result.provider}/{result.model}") if result.provider else "no generation call"
    t.append(f"{result.latency_ms / 1000:.1f}s | {provider} | mode {result.retrieval_mode} | ")
    t.append(f"tokens in {u.get('input_tokens', 0)} out {u.get('output_tokens', 0)} | ")
    cost = u.get("cost_usd", 0.0)
    unknown = u.get("unknown_cost_events", 0)
    t.append(f"cost ${cost:.6f}" + (f" (+{unknown} unknown)" if unknown else "") + " | ")
    t.append(f"run {result.run_id}")
    if result.fallback_reason:
        t.append("\nFallback: ", style="bold yellow")
        t.append(clean(result.fallback_reason), style="yellow")
    if u.get("accounting_write_failures"):
        t.append("\nWarning: usage ledger could not be written.", style="bold red")
    return t


def make_console(no_color: bool = False, stderr: bool = False) -> Console:
    return Console(no_color=no_color, highlight=False, stderr=stderr, emoji=False)
