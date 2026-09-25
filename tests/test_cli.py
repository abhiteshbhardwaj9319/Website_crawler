"""CLI behavior through the real entry point: exit codes, default site, JSON, no-color, errors."""

from __future__ import annotations

import json
import sys

import pytest

from website_rag import cli
from website_rag.config import get_settings


@pytest.fixture
def run(tmp_path, monkeypatch):
    monkeypatch.setenv("RAG_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.chdir(tmp_path)  # no .env here
    get_settings.cache_clear()

    def _run(*args: str) -> tuple[int, str, str]:
        monkeypatch.setattr(sys, "argv", ["rag", *args])
        cli.CTX.no_color = False
        code = 0
        try:
            cli.main()
        except SystemExit as exc:
            code = exc.code or 0
        return code

    yield _run
    get_settings.cache_clear()


def test_sites_list_json_has_stable_numbers_and_default_site(run, capsys):
    assert run("sites", "list", "--json") == 0
    sites = json.loads(capsys.readouterr().out)
    assert [s["number"] for s in sites] == [1, 2]
    assert sites[0]["site_id"] == "scrapy-docs-2-19" and sites[0]["status"] == "registered"


def test_ask_defaults_to_site_1_and_reports_not_ready(run, capsys):
    code = run("ask", "What is a spider?")
    err = capsys.readouterr()
    assert code == 2
    assert "Website 1" in err.out and "has no completed index" in err.out
    assert "rag ingest --site 1" in err.out


def test_unknown_site_and_invalid_provider_are_readable_errors(run, capsys):
    assert run("ask", "q", "--site", "7") == 2
    out = capsys.readouterr()
    assert "site_not_found" in out.err and "Registered website numbers: 1, 2" in out.err
    assert "Traceback" not in out.err + out.out
    assert run("ask", "q", "--provider", "claude") == 2
    assert "Unknown provider 'claude'" in capsys.readouterr().err


def test_usage_error_is_not_reported_as_unexpected(run, capsys):
    code = run("trace")
    out = capsys.readouterr()
    assert code == 2 and "Unexpected error" not in out.err and "Missing argument" in out.err


def test_ask_json_error_payload(run, capsys):
    code = run("ask", "q", "--json")
    payload = json.loads(capsys.readouterr().out)
    assert code == 2 and payload["status"] == "error" and payload["error"]["code"] == "site_not_ready"


def test_no_color_and_markup_escaping(run, capsys):
    assert run("--no-color", "sites", "add", "https://1.1.1.1/[bold]x[/bold]/", "--no-ingest", "--name", "[red]Evil\x1b[31m") == 0
    out = capsys.readouterr().out
    assert "\x1b[" not in out  # no ANSI color codes and no injected escape sequence
    assert "[red]Evil" in out  # markup shown literally, not interpreted


def test_doctor_json_reports_key_presence_only(run, capsys, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk" + "-should-never-be-printed-12345")
    get_settings.cache_clear()
    assert run("doctor", "--json") == 0
    out = capsys.readouterr().out
    assert "should-never-be-printed" not in out
    checks = {c["check"]: c for c in json.loads(out)["checks"]}
    assert checks["openai key"]["status"] == "ok" and checks["groq key"]["status"] == "warn"


def test_chat_session_switching_and_sources_render_as_text(run, capsys, monkeypatch):
    import io

    monkeypatch.setattr(sys, "stdin", io.StringIO("9\n2\nWhat is a spider?\n/sources\n/bogus\n/quit\n"))
    assert run("--no-color", "chat") == 0
    out = capsys.readouterr().out
    assert "Website '9' is not registered" in out  # invalid number: explicit message
    assert "Website 2 is not ingested yet" in out and "Staying on website 1" in out  # never silently switch
    assert "has no completed index" in out  # question on unready site: explicit error
    assert "Run r-" in out and '"run_id"' not in out  # /sources renders text, not JSON
    assert "Unknown command /bogus" in out
