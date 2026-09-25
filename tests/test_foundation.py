"""M1 foundation: URL policy, registry, redaction, accounting, provider errors and retries."""

from __future__ import annotations

import json

import groq
import httpx
import httpx2
import openai
import pytest

from website_rag.config import Settings
from website_rag.errors import ErrorCode, RagError
from website_rag.providers import (
    ProviderFailure,
    RetryPolicy,
    call_with_retries,
    classify,
    provider_chain,
    run_with_fallback,
)
from website_rag.registry import SiteRegistry
from website_rag.schemas import SiteStatus
from website_rag.tracing import Tracer, redact
from website_rag.urls import Scope, assert_public_host, derive_scope, normalize_url
from website_rag.usage import Pricing, UsageLedger, summarize

# ---------------------------------------------------------------- URL policy


def test_normalize_url_drops_fragment_query_default_port_and_index():
    assert normalize_url("HTTPS://Docs.Scrapy.org:443/en/2.19/index.html?x=1#top") == (
        "https://docs.scrapy.org/en/2.19/"
    )
    assert normalize_url("../topics/a.html", base="https://h.org/en/2.19/intro/b.html") == (
        "https://h.org/en/2.19/topics/a.html"
    )
    assert normalize_url("https://h.org/a//b/./c/../d.html") == "https://h.org/a/b/d.html"


@pytest.mark.parametrize("url", ["ftp://h.org/x", "javascript:alert(1)", "mailto:a@b.c", "file:///etc/passwd"])
def test_normalize_url_rejects_non_http(url):
    with pytest.raises(RagError) as err:
        normalize_url(url)
    assert err.value.code == ErrorCode.URL_REJECTED


def test_scope_rules():
    scope = Scope("docs.scrapy.org", "/en/2.19/")
    assert scope.contains("https://docs.scrapy.org/en/2.19/topics/spiders.html") == (True, "in_scope")
    assert scope.contains("https://docs.scrapy.org/en/latest/topics/spiders.html")[1] == "outside_path_prefix"
    assert scope.contains("https://evil.docs.scrapy.org/en/2.19/")[1] == "other_host"
    assert scope.contains("https://docs.scrapy.org/en/2.19/_static/x.html")[1] == "excluded_pattern"
    assert scope.contains("https://docs.scrapy.org/en/2.19/_images/a.png")[1] == "non_html_extension"
    assert scope.contains("https://docs.scrapy.org/en/2.19/genindex.html")[1] == "excluded_pattern"


def test_derive_scope_uses_seed_directory():
    assert derive_scope("https://docs.python.org/3.13/tutorial/index.html") == ("docs.python.org", "/3.13/tutorial/")
    assert derive_scope("https://example.com") == ("example.com", "/")


@pytest.mark.parametrize("host", ["localhost", "127.0.0.1", "10.0.0.5", "192.168.1.1", "169.254.169.254", "::1"])
def test_private_hosts_rejected(host):
    with pytest.raises(RagError) as err:
        assert_public_host(host)
    assert err.value.code == ErrorCode.URL_REJECTED


# ---------------------------------------------------------------- registry


def test_registry_defaults_stable_numbers_and_equivalence(tmp_path):
    reg = SiteRegistry(tmp_path / "registry.json")
    sites = reg.list()
    assert [s.number for s in sites] == [1, 2]
    assert reg.get(None).number == 1  # default site
    assert reg.get("2").site_id == sites[1].site_id

    new, created = reg.add("https://example.org/docs/", check_network=False)
    assert created and new.number == 3
    same, created2 = reg.add("https://EXAMPLE.org/docs/index.html#x", check_network=False)
    assert not created2 and same.site_id == new.site_id

    # numbers survive reload and status changes; next number never reuses
    reg.mark_status(sites[0], SiteStatus.FAILED, "boom")
    reloaded = SiteRegistry(tmp_path / "registry.json")
    assert [s.number for s in reloaded.list()] == [1, 2, 3]
    another, _ = reloaded.add("https://example.net/", check_network=False)
    assert another.number == 4


def test_registry_unknown_and_unready_sites(tmp_path):
    reg = SiteRegistry(tmp_path / "registry.json")
    with pytest.raises(RagError) as err:
        reg.get(99)
    assert err.value.code == ErrorCode.SITE_NOT_FOUND and "1, 2" in err.value.hint
    with pytest.raises(RagError) as err:
        reg.require_queryable(1)
    assert err.value.code == ErrorCode.SITE_NOT_READY and "rag ingest --site 1" in err.value.hint


# ---------------------------------------------------------------- secrets / tracing


def test_redaction_and_settings_repr_hide_keys(settings, tmp_path):
    text = "key " + "sk" + "-proj-abcdefghijklmnop and " + "gsk" + "_ABCDEFGHIJKLMNOP Authorization: Bearer abc.def.ghijkl api_key=supersecret"
    out = redact(text)
    assert "abcdefghijklmnop" not in out and "ABCDEFGHIJKLMNOP" not in out
    assert "supersecret" not in out and "abc.def.ghijkl" not in out
    assert "sk-test-openai" not in repr(settings) and "sk-test-openai" not in settings.model_dump_json()

    tracer = Tracer(tmp_path / "traces", kind="query")
    with tracer.span("generate", note="used sk-live-1234567890abcdef"):
        pass
    raw = (tmp_path / "traces" / f"{tracer.run_id}.jsonl").read_text()
    assert "1234567890abcdef" not in raw and "[REDACTED]" in raw


def test_tracer_records_error_without_masking_it(tmp_path):
    tracer = Tracer(tmp_path / "t", kind="query", site_id="s1")
    with pytest.raises(RagError):
        with tracer.span("retrieve"):
            with tracer.span("dense"):
                raise RagError(ErrorCode.INDEX_INCOMPLETE, "no index")
    events = [json.loads(l) for l in (tmp_path / "t" / f"{tracer.run_id}.jsonl").read_text().splitlines()]
    inner, outer = events
    assert inner["parent_id"] == outer["span_id"]
    assert inner["error_code"] == "index_incomplete" and inner["site_id"] == "s1"


def test_tracer_write_failure_does_not_raise(tmp_path):
    blocker = tmp_path / "file"
    blocker.write_text("x")
    tracer = Tracer(blocker, kind="query")  # traces dir is a file -> writes fail
    with tracer.span("x"):
        pass
    assert tracer.write_failures == 1


# ---------------------------------------------------------------- accounting


def test_cost_arithmetic_with_cached_subset():
    p = Pricing()
    # gpt-4.1-mini: 0.40 in, 0.10 cached, 1.60 out per 1M; snapshot suffix resolves to base model
    cost = p.cost("openai", "gpt-4.1-mini-2025-04-14", 2000, 300, cached_input_tokens=1000)
    assert cost == pytest.approx((1000 * 0.40 + 1000 * 0.10 + 300 * 1.60) / 1e6)
    assert p.cost("groq", "openai/gpt-oss-120b", 1000, 1000) == pytest.approx((1000 * 0.15 + 1000 * 0.60) / 1e6)
    assert p.cost("openai", "unknown-model", 10, 10) is None  # unknown is not zero
    assert p.cost("openai", "gpt-4.1-mini", None, 10) is None


def test_ledger_summary_counts_unknown_and_retries(tmp_path):
    from website_rag.schemas import UsageEvent

    ledger = UsageLedger(tmp_path / "u.jsonl")
    base = dict(run_id="r", operation_id="op", phase="query", site_id="s", corpus_id="c",
                provider="openai", model="m", operation="generation", measurement="provider_reported")
    ledger.record(UsageEvent(event_id="1", attempt=1, outcome="error", error_code="rate_limited", cost_usd=None, **base))
    ledger.record(UsageEvent(event_id="2", attempt=2, outcome="success", input_tokens=100, output_tokens=10, cost_usd=0.001, **base))
    s = summarize(ledger.read_all())
    assert s["attempts"] == 2 and s["errors"] == 1 and s["unknown_cost_events"] == 1
    assert s["cost_usd"] == pytest.approx(0.001) and s["input_tokens"] == 100


# ---------------------------------------------------------------- provider errors


def _openai_err(status: int, body: dict | None, headers: dict | None = None):
    req = httpx2.Request("POST", "https://api.openai.com/v1/chat/completions")
    resp = httpx2.Response(status, request=req, headers={"x-request-id": "req_123", **(headers or {})})
    cls = {401: openai.AuthenticationError, 403: openai.PermissionDeniedError, 404: openai.NotFoundError,
           429: openai.RateLimitError, 400: openai.BadRequestError}.get(status, openai.InternalServerError)
    return cls("err", response=resp, body=(body or {}).get("error", body))


def _groq_err(status: int, body: dict, headers: dict | None = None):
    req = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    resp = httpx.Response(status, request=req, headers=headers or {})
    cls = {429: groq.RateLimitError, 413: groq.APIStatusError, 400: groq.BadRequestError}.get(status, groq.InternalServerError)
    return cls("err", response=resp, body=body)


@pytest.mark.parametrize(
    "status, body, expected",
    [
        (429, {"error": {"code": "credit_balance_exhausted", "type": "insufficient_quota"}}, ErrorCode.CREDITS_EXHAUSTED),
        (429, {"error": {"code": "insufficient_quota", "message": "You exceeded your current quota"}}, ErrorCode.QUOTA_EXCEEDED),
        (429, {"error": {"code": "project_spend_limit_exceeded"}}, ErrorCode.SPEND_LIMIT),
        (429, {"error": {"code": "rate_limit_exceeded", "type": "requests"}}, ErrorCode.RATE_LIMITED),
        (429, {"error": {"code": "slow_down", "type": "rate_limit_error"}}, ErrorCode.RATE_LIMITED),
        (401, {"error": {"code": "invalid_api_key"}}, ErrorCode.AUTH_INVALID),
        (403, {"error": {"code": "unsupported_country_region_territory"}}, ErrorCode.PERMISSION_DENIED),
        (404, {"error": {"code": "model_not_found"}}, ErrorCode.MODEL_UNAVAILABLE),
        (400, {"error": {"code": "context_length_exceeded"}}, ErrorCode.REQUEST_TOO_LARGE),
        (503, {"error": {"code": "server_is_overloaded"}}, ErrorCode.SERVER_ERROR),
    ],
)
def test_openai_error_classification(status, body, expected):
    failure = classify(_openai_err(status, body), "openai")
    assert failure.code == expected
    assert failure.request_id == "req_123"


def test_credit_exhaustion_message_is_exact_and_not_every_429():
    exhausted = classify(_openai_err(429, {"error": {"code": "credit_balance_exhausted"}}), "openai")
    assert exhausted.message == (
        "OpenAI API credits are exhausted. Please recharge your API billing account, "
        "or use the configured Groq provider."
    )
    assert not exhausted.retryable and exhausted.fallback_eligible
    plain = classify(_openai_err(429, {"error": {"code": "rate_limit_exceeded"}}), "openai")
    assert "credits" not in plain.message and plain.retryable


def test_groq_error_classification_and_retry_after():
    rl = classify(_groq_err(429, {"error": {"message": "Rate limit reached ... tokens per day (TPD)", "type": "tokens", "code": "rate_limit_exceeded"}}, {"retry-after": "900"}), "groq")
    assert rl.code == ErrorCode.RATE_LIMITED and rl.retry_after_s == 900 and "daily" in rl.message
    big = classify(_groq_err(413, {"error": {"message": "Request too large", "code": "rate_limit_exceeded"}}), "groq")
    assert big.code == ErrorCode.REQUEST_TOO_LARGE
    schema = classify(_groq_err(400, {"error": {"code": "json_validate_failed"}}), "groq")
    assert schema.code == ErrorCode.MALFORMED_RESPONSE


def test_transport_errors():
    req = httpx2.Request("POST", "https://api.openai.com/v1/x")
    assert classify(openai.APITimeoutError(request=req), "openai").code == ErrorCode.TIMEOUT
    assert classify(openai.APIConnectionError(request=req), "openai").code == ErrorCode.CONNECTION


# ---------------------------------------------------------------- retries and fallback


class FakeClock:
    def __init__(self):
        self.now = 0.0
        self.sleeps: list[float] = []

    def clock(self):
        return self.now

    def sleep(self, s):
        self.sleeps.append(s)
        self.now += s


def _policy(clock: FakeClock, attempts=3, budget=60.0, max_ra=20.0) -> RetryPolicy:
    return RetryPolicy(attempts, budget, max_ra, base_backoff_s=1.0, sleep=clock.sleep, clock=clock.clock)


def test_retries_are_bounded_and_honor_retry_after():
    clock = FakeClock()
    calls = []

    def fn():
        calls.append(1)
        raise _openai_err(429, {"error": {"code": "rate_limit_exceeded"}}, {"retry-after": "2"})

    records = []
    with pytest.raises(ProviderFailure) as err:
        call_with_retries(fn, "openai", "m", _policy(clock), records.append)
    assert err.value.code == ErrorCode.RATE_LIMITED
    assert len(calls) == 3 and clock.sleeps == [2.0, 2.0]
    assert [r.attempt for r in records] == [1, 2, 3]


def test_permanent_errors_are_not_retried():
    clock = FakeClock()
    calls = []

    def fn():
        calls.append(1)
        raise _openai_err(429, {"error": {"code": "credit_balance_exhausted"}})

    with pytest.raises(ProviderFailure):
        call_with_retries(fn, "openai", "m", _policy(clock), lambda r: None)
    assert len(calls) == 1 and clock.sleeps == []


def test_long_retry_after_is_not_waited():
    clock = FakeClock()
    with pytest.raises(ProviderFailure) as err:
        call_with_retries(
            lambda: (_ for _ in ()).throw(_groq_err(429, {"error": {"code": "rate_limit_exceeded"}}, {"retry-after": "600"})),
            "groq", "m", _policy(clock), lambda r: None)
    assert clock.sleeps == [] and "above the retry limit" in err.value.message


def test_retry_budget_stops_retries():
    clock = FakeClock()
    calls = []

    def fn():
        calls.append(1)
        raise _openai_err(503, {"error": {"code": "server_is_overloaded"}}, {"retry-after": "8"})

    with pytest.raises(ProviderFailure):
        call_with_retries(fn, "openai", "m", _policy(clock, attempts=5, budget=10), lambda r: None)
    assert len(calls) == 2  # 0 + 8 = 8 ok; next wait would exceed the 10s budget


def test_auto_fallback_once_and_no_ping_pong(settings):
    clock = FakeClock()
    calls: list[str] = []

    def call(provider):
        calls.append(provider)
        if provider == "openai":
            raise _openai_err(429, {"error": {"code": "credit_balance_exhausted"}})
        return "groq-answer"

    chain, notes = provider_chain("auto", settings)
    outcome = run_with_fallback(chain, settings, call, _policy(clock))
    assert outcome.result == "groq-answer" and outcome.provider == "groq"
    assert calls == ["openai", "groq"]
    assert "credits_exhausted" in outcome.fallback_reason


def test_auto_does_not_fallback_on_auth_error_and_explicit_mode_never_falls_back(settings):
    clock = FakeClock()
    calls: list[str] = []

    def call(provider):
        calls.append(provider)
        raise _openai_err(401, {"error": {"code": "invalid_api_key"}})

    with pytest.raises(ProviderFailure) as err:
        run_with_fallback(["openai", "groq"], settings, call, _policy(clock))
    assert err.value.code == ErrorCode.AUTH_INVALID and calls == ["openai"]

    calls.clear()

    def call2(provider):
        calls.append(provider)
        raise _openai_err(429, {"error": {"code": "credit_balance_exhausted"}})

    chain, _ = provider_chain("openai", settings)
    with pytest.raises(ProviderFailure):
        run_with_fallback(chain, settings, call2, _policy(clock))
    assert calls == ["openai"]


def test_provider_chain_missing_keys(tmp_path):
    s = Settings(_env_file=None, data_dir=tmp_path, GROQ_API_KEY="test-groq-key")
    chain, notes = provider_chain("auto", s)
    assert chain == ["groq"] and "OpenAI not configured" in notes[0]
    with pytest.raises(RagError) as err:
        provider_chain("openai", s)
    assert err.value.code == ErrorCode.PROVIDER_NOT_CONFIGURED
    with pytest.raises(RagError):
        provider_chain("auto", Settings(_env_file=None, data_dir=tmp_path))


def test_invalid_config_fails_clearly(tmp_path):
    with pytest.raises(ValueError):
        Settings(_env_file=None, data_dir=tmp_path, chunk_target_tokens=100, chunk_overlap_tokens=150)
    with pytest.raises(ValueError):
        Settings(_env_file=None, data_dir=tmp_path, provider="anthropic")
