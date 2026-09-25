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

from .schemas import AnswerResult, SiteRecord

_ANSI = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]|\x1b\][^\x07]*(\x07|\x1b\\)|\x1b[@-Z\\-_]")
_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")

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
    table = Table(title="Websites", title_justify="left", header_style="bold", expand=False)
    table.add_column("#", justify="right", style="cyan")
    table.add_column("Website")
    table.add_column("Scope", style="dim")
    table.add_column("Status")
    table.add_column("Pages", justify="right")
    table.add_column("Chunks", justify="right")
    for site in sites:
        label, style = STATUS_STYLE.get(site.status.value, (site.status.value, ""))
        if site.status.value == "ingesting" and site.active_corpus_id:
            label = "ready (refresh in progress)"
        marker = " *" if selected == site.number else ""
        table.add_row(
            f"{site.number}{marker}", safe(site.display_name), safe(f"{site.allowed_host}{site.allowed_path_prefix}"),
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
    t.append(clean(f"{site.allowed_host}{site.allowed_path_prefix}"), style="dim")
    t.append(")", style="dim")
    return t


def answer_panel(result: AnswerResult, show_retrieval: bool = False) -> Group:
    label, style = STATUS_STYLE[result.status]
    parts: list = []
    body = Text()
    if result.answer:
        body.append(clean(result.answer))
    if result.status == "error" and result.error:
        body.append(clean(result.error.get("message", "")), style="red")
        if result.error.get("hint"):
            body.append("\nNext step: ", style="bold")
            body.append(clean(result.error["hint"]))
    if result.premise_issue:
        body.append("\n\nPremise check: ", style="bold yellow")
        body.append(clean(result.premise_issue))
    if result.missing_information and result.status != "answered":
        body.append("\n\nNot covered by the indexed pages: ", style="bold")
        body.append(clean(result.missing_information))
    parts.append(Panel(body if body.plain else Text("(no answer text)", style="dim"),
                       title=Text(label, style=style), title_align="left", border_style=style.split()[-1]))

    if result.claims:
        claims = Table.grid(padding=(0, 1))
        claims.add_column(style="dim", no_wrap=True)
        claims.add_column()
        for claim in result.claims:
            markers = "".join(f"[{m}]" for m in claim.citations)
            claims.add_row(markers, safe(claim.text))
        parts.append(Panel(claims, title="Verified claims", title_align="left", border_style="dim"))

    if result.citations:
        src = Table(show_header=True, header_style="bold", expand=False, box=None, padding=(0, 1))
        src.add_column("#", style="cyan", justify="right")
        src.add_column("Source")
        for c in result.citations:
            cell = Text()
            cell.append(clean(c.title), style="bold")
            cell.append("  ")
            cell.append(clean(c.section), style="dim")
            cell.append("\n")
            cell.append(clean(c.url), style="underline blue")
            cell.append("\n“", style="dim")
            quote = clean(c.quote)
            cell.append(quote if len(quote) <= 220 else quote[:217] + "...", style="italic")
            cell.append("”", style="dim")
            src.add_row(str(c.marker), cell)
        parts.append(Panel(src, title="Sources", title_align="left", border_style="cyan"))

    if result.rejected_claims:
        rej = Text()
        for r in result.rejected_claims:
            rej.append("- ", style="red")
            rej.append(clean(r.text))
            rej.append(f"  ({'; '.join(r.reasons)})\n", style="dim")
        parts.append(Panel(rej, title="Withheld statements (failed citation checks)", title_align="left", border_style="red"))

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
    provider = f"{result.provider}/{result.model}" if result.provider else "no generation call"
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
