"""Command-line interface (Typer + Rich).

Every command has a non-interactive form; `--json` prints machine-readable output with no
color or decoration. Expected failures print a short explanation and next step (exit 2);
Ctrl+C exits with code 130 and leaves ingestion state consistent.
"""

from __future__ import annotations

import json
import os
import sys
from typing import Optional

import typer
from rich.progress import BarColumn, MofNCompleteColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table
from rich.text import Text

from . import render
from .config import Settings, get_settings
from .errors import ErrorCode, RagError
from .registry import SiteRegistry
from .schemas import CrawlLimits, SiteRecord
from .tracing import Tracer, load_trace
from .usage import UsageLedger

app = typer.Typer(add_completion=False, no_args_is_help=True, help="Website-grounded RAG: crawl, index, and answer with cited evidence.")
sites_app = typer.Typer(no_args_is_help=True, help="List, add, and inspect websites.")
app.add_typer(sites_app, name="sites")


class Ctx:
    no_color: bool = False
    debug: bool = False


CTX = Ctx()


@app.callback()
def _global(
    no_color: bool = typer.Option(False, "--no-color", help="Disable colors (also honors NO_COLOR)."),
    debug: bool = typer.Option(False, "--debug", help="Record sanitized stack traces in local traces."),
) -> None:
    CTX.no_color = no_color or bool(os.environ.get("NO_COLOR"))
    CTX.debug = debug


def _settings() -> Settings:
    s = get_settings()
    if CTX.debug:
        s.debug = True
    os.environ.setdefault("TIKTOKEN_CACHE_DIR", str(s.data_dir / "cache" / "tiktoken"))
    return s


def _console():
    return render.make_console(CTX.no_color)


def _registry(s: Settings) -> SiteRegistry:
    return SiteRegistry(s.registry_path)


def _emit_json(payload) -> None:  # noqa: ANN001
    sys.stdout.write(json.dumps(payload, indent=2, default=str, ensure_ascii=False) + "\n")


def _site_json(site: SiteRecord) -> dict:
    return json.loads(site.model_dump_json())


# ----------------------------------------------------------------------------- doctor


@app.command()
def doctor(
    check_providers: bool = typer.Option(False, "--check-providers", help="Verify keys with a free model-listing call (no tokens)."),
    as_json: bool = typer.Option(False, "--json"),
) -> None:
    """Check environment, configuration, provider keys (presence only), and site readiness."""
    import platform
    from importlib.metadata import version

    s = _settings()
    reg = _registry(s)
    checks: list[dict] = []

    def add(name: str, ok: bool | None, detail: str) -> None:
        checks.append({"check": name, "status": "ok" if ok else ("warn" if ok is None else "fail"), "detail": detail})

    add("python", sys.version_info >= (3, 11), platform.python_version())
    for pkg in ("langgraph", "langchain-openai", "langchain-groq", "qdrant-client", "fastembed"):
        add(pkg, True, version(pkg))
    add("data dir", True, str(s.data_dir.resolve()))
    for provider in ("openai", "groq"):
        configured = s.key_for(provider) is not None
        add(f"{provider} key", configured or None, ("configured" if configured else "not set") + f"; model {s.model_for(provider)}")
    add("provider mode", True, s.provider)
    add("embedding model", True, f"{s.embedding_model} (local ONNX)")
    add("retrieval mode", True, s.retrieval_mode)
    for site in reg.list():
        add(f"site {site.number}", True if site.is_queryable else None,
            f"{site.display_name}: {site.status.value}, {site.accepted_pages} pages, corpus {site.active_corpus_id or '-'}")
    if check_providers:
        for provider in ("openai", "groq"):
            if s.key_for(provider) is None:
                continue
            ok, detail = _check_provider(provider, s)
            add(f"{provider} live check", ok, detail)
    if as_json:
        _emit_json({"checks": checks})
        return
    console = _console()
    table = Table(title="rag doctor", title_justify="left", header_style="bold")
    table.add_column("Check")
    table.add_column("Status")
    table.add_column("Detail")
    style = {"ok": "green", "warn": "yellow", "fail": "red"}
    for c in checks:
        table.add_row(c["check"], Text(c["status"], style=style[c["status"]]), render.safe(c["detail"]))
    console.print(table)


def _check_provider(provider: str, s: Settings) -> tuple[bool, str]:
    from .providers import classify

    try:
        if provider == "openai":
            import openai

            client = openai.OpenAI(api_key=s.openai_api_key.get_secret_value(), max_retries=0, timeout=15)
            client.models.retrieve(s.openai_model)
        else:
            import groq

            client = groq.Groq(api_key=s.groq_api_key.get_secret_value(), max_retries=0, timeout=15)
            ids = {m.id for m in client.models.list().data}
            if s.groq_model not in ids:
                return False, f"model {s.groq_model} not listed for this key"
        return True, "key accepted and model available (no tokens used)"
    except Exception as exc:  # noqa: BLE001
        failure = classify(exc, provider)
        return False, f"{failure.code}: {failure.message}"


# ----------------------------------------------------------------------------- sites


@sites_app.command("list")
def sites_list(as_json: bool = typer.Option(False, "--json")) -> None:
    """List registered websites with their stable numbers."""
    reg = _registry(_settings())
    if as_json:
        _emit_json([_site_json(s) for s in reg.list()])
        return
    _console().print(render.sites_table(reg.list()))


@sites_app.command("show")
def sites_show(site: str = typer.Argument(..., help="Site number or id."), as_json: bool = typer.Option(False, "--json")) -> None:
    """Show one website's scope, corpus, and indexed pages."""
    s = _settings()
    record = _registry(s).get(site)
    pages = []
    if record.active_corpus_id:
        from .crawl import load_pages

        corpus_dir = s.sites_dir / record.site_id / "corpora" / record.active_corpus_id
        if (corpus_dir / "pages.jsonl").exists():
            pages = [{"url": p.final_url, "title": p.title, "words": p.word_count} for p in load_pages(corpus_dir)]
    if as_json:
        _emit_json({"site": _site_json(record), "pages": pages})
        return
    console = _console()
    console.print(render.site_banner(record))
    console.print(render.safe(f"Seed: {record.seed_url}   Corpus: {record.active_corpus_id or '-'}   "
                              f"Embedding: {record.embedding_model or '-'}", "dim"))
    if record.last_error:
        console.print(render.safe(f"Last error: {record.last_error}", "red"))
    if pages:
        t = Table(header_style="bold", box=None)
        t.add_column("Title")
        t.add_column("Words", justify="right")
        t.add_column("URL", style="dim")
        for p in pages:
            t.add_row(render.safe(p["title"]), str(p["words"]), render.safe(p["url"]))
        console.print(t)


@sites_app.command("add")
def sites_add(
    url: str = typer.Argument(..., help="Public http(s) URL; its directory becomes the crawl scope."),
    name: Optional[str] = typer.Option(None, "--name", help="Display name."),
    max_pages: int = typer.Option(25, "--max-pages", min=1, max=200),
    max_depth: int = typer.Option(3, "--max-depth", min=0, max=10),
    ingest: bool = typer.Option(True, "--ingest/--no-ingest", help="Ingest immediately after registering."),
    as_json: bool = typer.Option(False, "--json"),
) -> None:
    """Register a new website (equivalent URLs resolve to the existing site) and ingest it."""
    s = _settings()
    reg = _registry(s)
    site, created = reg.add(url, display_name=name, crawl=CrawlLimits(max_pages=max_pages, max_depth=max_depth))
    console = _console()
    if not as_json:
        if created:
            console.print(Text.assemble(("Registered website ", "green"), (str(site.number), "bold cyan"), ": ", render.safe(site.display_name)))
        else:
            console.print(Text.assemble(("This URL matches existing website ", "yellow"), (str(site.number), "bold cyan"), ": ",
                                        render.safe(site.display_name), (" (no new site created)", "dim")))
    report = None
    if ingest and (created or not site.is_queryable):
        report = _run_ingest(site, s, reg, as_json)
    if as_json:
        _emit_json({"site": _site_json(reg.get(site.site_id)), "created": created, "ingest": _report_json(report)})


# ----------------------------------------------------------------------------- ingest


def _report_json(report) -> dict | None:  # noqa: ANN001
    if report is None:
        return None
    return {k: v for k, v in report.__dict__.items() if k != "site"} | {"site_number": report.site.number}


def _run_ingest(site: SiteRecord, s: Settings, reg: SiteRegistry, as_json: bool):  # noqa: ANN202
    from .ingest import ingest_site

    console = render.make_console(CTX.no_color, stderr=as_json)
    tracer = Tracer(s.traces_dir, kind="ingestion", debug=s.debug, site_id=site.site_id)
    ledger = UsageLedger(s.ledger_path)
    progress = Progress(
        SpinnerColumn(), TextColumn("{task.description}"), BarColumn(), MofNCompleteColumn(), TimeElapsedColumn(),
        console=console, transient=False, disable=not console.is_terminal,
    )
    crawl_task = progress.add_task(f"Crawling {site.allowed_host}{site.allowed_path_prefix}", total=site.crawl.max_pages)
    embed_task = None
    counts = {"skipped": 0, "failed": 0}

    def on_progress(kind: str, data: dict) -> None:
        nonlocal embed_task
        if kind == "page":
            if data["outcome"] == "accepted":
                progress.update(crawl_task, completed=data["accepted"])
                if not console.is_terminal:
                    console.print(render.safe(f"  accepted {data['accepted']:>3}  {data['url']}", "dim"))
            else:
                counts["skipped" if data["outcome"] == "skipped" else "failed"] += 1
            progress.update(crawl_task, description=f"Crawling (skipped {counts['skipped']}, failed {counts['failed']})")
        elif kind == "stage" and data["name"] == "chunk":
            progress.update(crawl_task, total=progress.tasks[crawl_task].completed)
        elif kind == "embed":
            if embed_task is None:
                embed_task = progress.add_task("Embedding chunks (local)", total=data["total"])
            progress.update(embed_task, completed=data["done"])

    console.print(Text.assemble(("Ingesting website ", ""), (str(site.number), "bold cyan"), ": ", render.safe(site.display_name)))
    try:
        with progress:
            report = ingest_site(site, s, reg, tracer, ledger, progress=on_progress)
    except KeyboardInterrupt:
        console.print(Text("Ingestion interrupted. The previous index (if any) is still active.", style="yellow"))
        raise
    except RagError as err:
        err.details["run_id"] = tracer.run_id
        raise
    summary = Table.grid(padding=(0, 2))
    summary.add_row("Accepted pages", str(report.accepted_pages))
    summary.add_row("Skipped / failed URLs", f"{report.skipped} / {report.failed}")
    if report.skip_reasons:
        summary.add_row("Skip reasons", ", ".join(f"{k}: {v}" for k, v in sorted(report.skip_reasons.items())))
    summary.add_row("Chunks", f"{report.chunk_count} ({report.embedded} embedded, {report.reused_embeddings} reused)")
    summary.add_row("Embedding tokens (o200k count)", f"{report.embedding_tokens:,} - local model, $0 API cost")
    summary.add_row("Stopped because", report.stopped_reason)
    summary.add_row("Time", f"{report.elapsed_s:.1f}s")
    summary.add_row("Corpus / run", f"{report.corpus_id} / {report.run_id}")
    console.print(summary)
    for w in report.warnings:
        console.print(Text(f"Warning: {w}", style="yellow"))
    console.print(Text(f"Website {site.number} is ready. Ask with: rag ask \"...\" --site {site.number}", style="green"))
    return report


@app.command()
def ingest(
    site: Optional[str] = typer.Option(None, "--site", "-s", help="Site number or id (default 1)."),
    as_json: bool = typer.Option(False, "--json"),
) -> None:
    """Crawl and (re)index a website. The previous index stays active until the new one completes."""
    s = _settings()
    reg = _registry(s)
    record = reg.get(site)
    report = _run_ingest(record, s, reg, as_json)
    if as_json:
        _emit_json(_report_json(report))


# ----------------------------------------------------------------------------- ask


def _deps(s: Settings, reg: SiteRegistry, phase: str = "query"):  # noqa: ANN202
    from .graph import QueryDeps

    return QueryDeps(settings=s, registry=reg, ledger=UsageLedger(s.ledger_path),
                     tracer=Tracer(s.traces_dir, kind=phase, debug=s.debug), phase=phase)


@app.command()
def ask(
    question: str = typer.Argument(..., help="Natural-language question."),
    site: Optional[str] = typer.Option(None, "--site", "-s", help="Site number or id (default 1)."),
    provider: Optional[str] = typer.Option(None, "--provider", "-p", help="openai | groq | auto"),
    mode: Optional[str] = typer.Option(None, "--mode", "-m", help="dense | bm25 | hybrid | hybrid_rerank"),
    show_retrieval: bool = typer.Option(False, "--show-retrieval", help="Show ranked retrieved chunks."),
    explain: bool = typer.Option(False, "--explain", help="Show exact evidence and clearly marked rejected candidates."),
    as_json: bool = typer.Option(False, "--json"),
) -> None:
    """Answer one question from the selected website's indexed evidence."""
    from .graph import answer_question

    s = _settings()
    provider = _validate_choice(provider, ("openai", "groq", "auto"), "provider")
    mode = _validate_choice(mode, ("dense", "bm25", "hybrid", "hybrid_rerank"), "mode")
    reg = _registry(s)
    console = _console()
    record = reg.get(site)
    if not as_json:
        console.print(render.site_banner(record))
    result = answer_question(_deps(s, reg), question, record.number, mode=mode, provider_mode=provider)
    if as_json:
        _emit_json(json.loads(result.model_dump_json()))
    else:
        console.print(render.answer_panel(result, show_retrieval=show_retrieval, explain=explain))
    if result.status == "error":
        raise typer.Exit(2)


def _validate_choice(value: str | None, choices: tuple, name: str) -> str | None:
    if value is None:
        return None
    if value not in choices:
        raise RagError(ErrorCode.INVALID_INPUT, f"Unknown {name} '{value}'.", hint=f"Choose one of: {', '.join(choices)}.")
    return value


# ----------------------------------------------------------------------------- chat


CHAT_HELP = """Commands:
  <number>          select a website by number (same as /use <number>)
  /sites            list websites          /use <n>        select website n
  /add <url>        register + ingest a new public URL, then select it
  /provider <p>     openai | groq | auto   /mode <m>       dense | bm25 | hybrid | hybrid_rerank
  /sources          sources of the last answer   /trace   trace of the last answer
  /help             this help              /quit           exit
Each question is answered independently (no conversation memory)."""


@app.command()
def chat(
    site: Optional[str] = typer.Option(None, "--site", "-s", help="Starting site (default 1)."),
    provider: Optional[str] = typer.Option(None, "--provider", "-p"),
    mode: Optional[str] = typer.Option(None, "--mode", "-m"),
) -> None:
    """Interactive session with numbered website selection."""
    from .graph import answer_question

    s = _settings()
    reg = _registry(s)
    console = _console()
    provider = _validate_choice(provider, ("openai", "groq", "auto"), "provider") or s.provider
    mode = _validate_choice(mode, ("dense", "bm25", "hybrid", "hybrid_rerank"), "mode") or s.retrieval_mode
    current = reg.get(site)
    last_run: str | None = None
    console.print(Text("Website-grounded RAG", style="bold"))
    console.print(render.sites_table(reg.list(), selected=current.number))
    console.print(Text(CHAT_HELP, style="dim"))

    while True:
        console.print()
        console.print(Text.assemble(render.site_banner(current), (f"  provider {provider}, mode {mode}", "dim")))
        try:
            line = console.input("[bold cyan]ask>[/] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\nBye.")
            return
        if not line:
            continue
        try:
            cmd, _, arg = line.partition(" ")
            if line.isdigit() or cmd == "/use":
                target = line if line.isdigit() else arg.strip()
                new_site = reg.get(target)
                if not new_site.is_queryable:
                    console.print(Text(f"Website {new_site.number} is not ingested yet (status {new_site.status.value}). "
                                       f"Staying on website {current.number}.", style="yellow"))
                    continue
                current, last_run = new_site, None  # clear evidence state on switch
                console.print(Text(f"Switched to website {current.number}. Previous answers and evidence cleared.", style="green"))
            elif cmd == "/sites":
                console.print(render.sites_table(reg.list(), selected=current.number))
            elif cmd == "/add":
                if not arg.strip():
                    console.print(Text("Usage: /add <url>", style="yellow"))
                    continue
                new_site, created = reg.add(arg.strip())
                if not created:
                    console.print(Text(f"That URL matches existing website {new_site.number}.", style="yellow"))
                if not new_site.is_queryable:
                    _run_ingest(new_site, s, reg, as_json=False)
                current, last_run = reg.get(new_site.site_id), None
                console.print(Text(f"Selected website {current.number}.", style="green"))
            elif cmd == "/provider":
                provider = _validate_choice(arg.strip(), ("openai", "groq", "auto"), "provider") or provider
            elif cmd == "/mode":
                mode = _validate_choice(arg.strip(), ("dense", "bm25", "hybrid", "hybrid_rerank"), "mode") or mode
            elif cmd == "/sources":
                if last_run:
                    sources(last_run, as_json=False)
                else:
                    console.print(Text("No answer yet on this website.", style="dim"))
            elif cmd == "/trace":
                if last_run:
                    trace(last_run, as_json=False)
                else:
                    console.print(Text("No answer yet on this website.", style="dim"))
            elif cmd == "/help":
                console.print(Text(CHAT_HELP, style="dim"))
            elif cmd in ("/quit", "/exit", "/q"):
                return
            elif cmd.startswith("/"):
                console.print(Text(f"Unknown command {cmd}. Type /help.", style="yellow"))
            else:
                result = answer_question(_deps(s, reg), line, current.number, mode=mode, provider_mode=provider)
                last_run = result.run_id
                console.print(render.answer_panel(result))
        except RagError as err:
            _print_error(err, console)
        except KeyboardInterrupt:
            console.print(Text("\nCancelled.", style="yellow"))


# ----------------------------------------------------------------------------- inspection


@app.command()
def sources(run_id: str = typer.Argument(...), as_json: bool = typer.Option(False, "--json")) -> None:
    """Show the evidence used by a saved run: context chunks, ranks, and cited excerpts."""
    from .graph import load_run

    s = _settings()
    result = load_run(s, run_id)
    if as_json:
        _emit_json({"run_id": run_id, "site_id": result.site_id, "corpus_id": result.corpus_id,
                    "citations": [c.model_dump() for c in result.citations],
                    "context": [r.model_dump(mode="json") for r in result.retrieved if r.chunk.chunk_id in result.context_chunk_ids]})
        return
    console = _console()
    console.print(Text.assemble(("Run ", "dim"), (run_id, "bold"), (f"  website {result.site_number} ", "dim"),
                                render.safe(result.site_name, "cyan"), (f"  corpus {result.corpus_id}", "dim")))
    console.print(render.safe(f"Q: {result.question}", "bold"))
    console.print(render.retrieval_table(result, limit=len(result.retrieved)))
    for r in result.retrieved:
        if r.chunk.chunk_id in result.context_chunk_ids:
            console.print(Text.assemble((f"\n[rank {r.rank}] ", "cyan"), render.safe(r.chunk.citation_url, "underline"),
                                        ("  " + r.chunk.chunk_id, "dim")))
            console.print(render.safe(r.chunk.text[:700] + ("..." if len(r.chunk.text) > 700 else ""), "dim"))


@app.command()
def trace(run_id: str = typer.Argument(...), as_json: bool = typer.Option(False, "--json")) -> None:
    """Explain what happened in a run, stage by stage, including failures and provider fallback."""
    s = _settings()
    events = load_trace(s.traces_dir, run_id)
    if not events:
        raise RagError(ErrorCode.INVALID_INPUT, f"No trace found for run '{run_id}'.", hint="Run IDs are printed after each answer/ingestion.")
    if as_json:
        _emit_json(events)
        return
    console = _console()
    first = events[0]
    console.print(Text.assemble(("Trace ", "bold"), run_id, (f"  kind {first.get('kind')}  site {first.get('site_id')}  corpus {first.get('corpus_id')}", "dim")))
    t = Table(header_style="bold", box=None)
    for col in ("start", "stage", "status", "ms", "details"):
        t.add_column(col)
    events = sorted(events, key=lambda e: e.get("ts", ""))
    depth: dict[str, int] = {}
    for e in events:
        parent = e.get("parent_id")
        level = depth.get(parent, -1) + 1 if parent else 0
        depth[e.get("span_id")] = level
        status = e.get("status", "event")
        style = {"ok": "green", "error": "red"}.get(status, "cyan")
        detail = _trace_detail(e)
        t.add_row(e.get("ts", "")[11:23], "  " * level + str(e.get("stage")), Text(status, style=style),
                  str(e.get("duration_ms", "")), render.safe(detail))
    console.print(t)
    failed = [e for e in events if e.get("status") == "error"]
    if failed:
        root = failed[0]
        console.print(Text.assemble(("Failed at stage ", "bold red"), (str(root.get("stage")), "bold"), ": ",
                                    render.safe(f"{root.get('error_code')} - {root.get('error_message')}")))


def _trace_detail(e: dict) -> str:
    a = e.get("attrs", {})
    stage = e.get("stage")
    if e.get("status") == "error":
        return f"{e.get('error_code')}: {e.get('error_message')}"
    if stage == "retrieve":
        top = a.get("results", [])[:3]
        return f"{a.get('count')} results; top: " + ", ".join(f"{r['chunk_id'][:8]}({r['page'].rsplit('/', 1)[-1]})" for r in top)
    if stage == "context":
        return f"{len(a.get('chunk_ids', []))} chunks, {a.get('tokens')} tokens, {len(a.get('pages', []))} pages"
    if stage == "provider_attempt":
        base = f"{a.get('provider')}/{a.get('model')} attempt {a.get('attempt')}: {a.get('outcome')}"
        if a.get("error_code"):
            base += f" ({a.get('error_code')})"
        u = a.get("usage") or {}
        if u.get("input_tokens") is not None:
            base += f" in={u.get('input_tokens')} out={u.get('output_tokens')} ${a.get('cost_usd') or 0:.6f}"
        return base
    if stage == "provider_fallback":
        return a.get("reason", "")
    if stage == "generate":
        return f"{a.get('provider')}/{a.get('model')}, attempts {a.get('attempts')}, draft {a.get('status')}"
    if stage == "check_citations":
        return f"draft {a.get('draft_status')}: {a.get('valid')} valid / {a.get('claims')} claims, {a.get('rejected')} rejected"
    if stage == "finalize":
        return f"status {a.get('status')}, {a.get('citations')} citations, ${a.get('cost_usd', 0):.6f}"
    if stage == "crawl":
        return f"{a.get('accepted')} accepted, stop: {a.get('stopped_reason')}, skips: {a.get('skip_reasons')}"
    if stage in ("chunk", "embed", "index", "validate"):
        return ", ".join(f"{k}={v}" for k, v in a.items() if not isinstance(v, (list, dict)))
    return ", ".join(f"{k}={v}" for k, v in a.items() if not isinstance(v, (list, dict)))[:160]


@app.command()
def runs(limit: int = typer.Option(15, "--limit"), as_json: bool = typer.Option(False, "--json")) -> None:
    """List recent saved query runs."""
    s = _settings()
    files = sorted(s.runs_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)[:limit] if s.runs_dir.exists() else []
    rows = []
    for f in files:
        d = json.loads(f.read_text(encoding="utf-8"))
        rows.append({"run_id": d["run_id"], "site": d.get("site_number"), "status": d["status"],
                     "provider": d.get("provider"), "question": d["question"], "created_at": d.get("created_at")})
    if as_json:
        _emit_json(rows)
        return
    t = Table(header_style="bold", box=None)
    for col in ("run", "site", "status", "provider", "question"):
        t.add_column(col)
    for r in rows:
        t.add_row(r["run_id"], str(r["site"]), r["status"], str(r["provider"] or "-"), render.safe(r["question"][:70]))
    _console().print(t)


@app.command()
def costs(as_json: bool = typer.Option(False, "--json")) -> None:
    """Summarize the usage ledger by phase, provider, and model (ingestion vs query vs evaluation)."""
    from .usage import Pricing, group_summary

    s = _settings()
    events = UsageLedger(s.ledger_path).read_all()
    rows = group_summary(events)
    pricing = Pricing()
    if as_json:
        _emit_json({"pricing_version": pricing.version, "rows": rows})
        return
    console = _console()
    if not rows:
        console.print(Text("The usage ledger is empty. Ingest a site or ask a question first.", style="dim"))
        return
    t = Table(title=f"Usage ledger (prices as of {pricing.version}, {pricing.currency})", title_justify="left", header_style="bold")
    for col, j in (("phase", "left"), ("provider", "left"), ("model", "left"), ("op", "left"), ("runs", "right"),
                   ("attempts", "right"), ("errors", "right"), ("input tok", "right"), ("cached", "right"),
                   ("output tok", "right"), ("cost USD", "right"), ("unknown", "right")):
        t.add_column(col, justify=j)
    for r in rows:
        t.add_row(r["phase"], r["provider"], r["model"], r["operation"], str(r["runs"]), str(r["attempts"]),
                  str(r["errors"]), f"{r['input_tokens']:,}", f"{r['cached_input_tokens']:,}", f"{r['output_tokens']:,}",
                  f"{r['cost_usd']:.6f}", str(r["unknown_cost_events"]))
    console.print(t)
    console.print(Text("Local embedding/reranking has no API charge but uses local compute. 'unknown' counts attempts "
                       "whose billed usage was not reported (e.g. failed requests); they are not assumed to be free.", style="dim"))


@app.command()
def evaluate(
    site: Optional[str] = typer.Option("1", "--site", "-s"),
    split: str = typer.Option("test", "--split", help="test | dev | isolation"),
    modes: str = typer.Option("dense", "--modes", help="Comma-separated retrieval modes to compare."),
    provider: Optional[str] = typer.Option(None, "--provider", "-p"),
    retrieval_only: bool = typer.Option(False, "--retrieval-only", help="Score retrieval without generation (no API calls)."),
    out: Optional[str] = typer.Option(None, "--out", help="Directory for results (default eval/results)."),
    as_json: bool = typer.Option(False, "--json"),
) -> None:
    """Run the frozen evaluation set; writes per-case JSONL and a summary."""
    from .evaluate import run_evaluation

    s = _settings()
    provider = _validate_choice(provider, ("openai", "groq", "auto"), "provider")
    mode_list = [m.strip() for m in modes.split(",") if m.strip()]
    for m in mode_list:
        _validate_choice(m, ("dense", "bm25", "hybrid", "hybrid_rerank"), "mode")
    summary = run_evaluation(s, _registry(s), site, split, mode_list, provider, retrieval_only, out,
                             console=None if as_json else _console())
    if as_json:
        _emit_json(summary)


# ----------------------------------------------------------------------------- entry point


def _print_error(err: RagError, console=None) -> None:  # noqa: ANN001
    console = console or render.make_console(CTX.no_color, stderr=True)
    msg = Text.assemble(("Error", "bold red"), (f" [{err.code}]", "red"), ": ", render.safe(err.message))
    if err.hint:
        msg.append("\nNext step: ", style="bold")
        msg.append(render.clean(err.hint))
    if err.details.get("run_id"):
        msg.append(f"\nTrace: rag trace {err.details['run_id']}", style="dim")
    console.print(msg)


def main() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
        except (AttributeError, ValueError):
            pass
    try:
        rc = app(standalone_mode=False)
        if isinstance(rc, int) and rc:
            sys.exit(rc)
    except RagError as err:
        _print_error(err)
        sys.exit(2)
    except KeyboardInterrupt:
        render.make_console(CTX.no_color, stderr=True).print("Interrupted.")
        sys.exit(130)
    except (typer.Exit, SystemExit) as exc:
        code = getattr(exc, "exit_code", getattr(exc, "code", 0))
        sys.exit(code or 0)
    except Exception as exc:  # click usage errors and unexpected failures
        # Typer vendors its own Click, so match usage errors by class hierarchy name.
        names = {cls.__name__ for cls in type(exc).__mro__}
        if "ClickException" in names:
            exc.show()  # type: ignore[attr-defined]
            sys.exit(getattr(exc, "exit_code", 2))
        if "Abort" in names:
            sys.exit(130)
        from .tracing import redact

        render.make_console(CTX.no_color, stderr=True).print(
            Text(f"Unexpected error: {type(exc).__name__}: {redact(str(exc))[:500]}\nRe-run with --debug for a local stack trace.", style="red"))
        if CTX.debug:
            import traceback

            sys.stderr.write(redact("".join(traceback.format_exception(exc)))[-4000:])
        sys.exit(1)
    finally:
        from .index import close_qdrant

        close_qdrant()


if __name__ == "__main__":
    main()
