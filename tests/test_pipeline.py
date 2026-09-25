"""Integration: ingestion, persistence, site isolation, graph outcomes, fallback, accounting."""

from __future__ import annotations

import json

import pytest

from website_rag.errors import ErrorCode, RagError
from website_rag.generate import build_messages
from website_rag.graph import Budget, QueryDeps, answer_question
from website_rag.index import CorpusStore, clear_caches, close_qdrant
from website_rag.ingest import ingest_site
from website_rag.providers import RetryPolicy
from website_rag.registry import SiteRegistry
from website_rag.retrieve import Retriever, select_context
from website_rag.schemas import CrawlLimits, SiteStatus
from website_rag.tracing import Tracer
from website_rag.usage import UsageLedger

from .helpers import SITE_A_PAGES, SITE_B_PAGES, HashEmbedder, draft, mock_transport, output
from .test_foundation import _openai_err

EMB = HashEmbedder()


@pytest.fixture
def env(settings, tmp_path):
    settings.embedding_model = EMB.model_name
    reg = SiteRegistry(settings.registry_path, defaults_file=None)
    limits = CrawlLimits(delay_s=0, max_pages=10, min_page_words=20)
    a, _ = reg.add("https://docs.a.test/docs/", "Site A", crawl=limits, check_network=False)
    b, _ = reg.add("https://docs.b.test/guide/", "Site B", crawl=limits, check_network=False)
    ledger = UsageLedger(settings.ledger_path)
    for site, pages in ((a, SITE_A_PAGES), (b, SITE_B_PAGES)):
        ingest_site(site, settings, reg, Tracer(settings.traces_dir, kind="ingestion"), ledger, embedder=EMB,
                    crawl_kwargs={"transport": mock_transport(pages), "check_network": False})
    yield settings, reg, ledger
    close_qdrant()
    clear_caches()


def deps(settings, reg, generator=None, **kw):
    return QueryDeps(settings=settings, registry=reg, ledger=UsageLedger(settings.ledger_path),
                     tracer=Tracer(settings.traces_dir), generator=generator,
                     retriever_factory=lambda site, store: Retriever(settings, site, store, embedder=EMB),
                     policy=RetryPolicy(3, 30, 20, sleep=lambda s: None), **kw)


def context_ids(result):
    return result.context_chunk_ids


# ------------------------------------------------------------------ ingestion / persistence


def test_ingestion_registry_state_and_reload(env):
    settings, reg, _ = env
    a, b = reg.get(1), reg.get(2)
    assert a.status == SiteStatus.READY and a.accepted_pages == 3 and a.embedding_model == EMB.model_name
    assert b.accepted_pages == 3
    close_qdrant()  # simulate a new process
    clear_caches()
    reloaded = SiteRegistry(settings.registry_path, defaults_file=None).get(1)
    store = CorpusStore(settings, reloaded.site_id, reloaded.active_corpus_id)
    manifest = store.verify_ready(reloaded, EMB.model_name)
    assert manifest.chunk_count == store.point_count() == reloaded.chunk_count
    chunks = store.load_chunks()
    assert all(c.site_id == reloaded.site_id and c.corpus_id == reloaded.active_corpus_id for c in chunks)
    retry = next(c for c in chunks if "WIDGET_RETRY_TIMES" in c.text)
    assert retry.anchor == "widget-retry-times" and retry.citation_url.endswith("settings.html#widget-retry-times")


def test_reingestion_is_idempotent_and_reuses_embeddings(env):
    settings, reg, ledger = env
    site = reg.get(1)
    old_corpus = site.active_corpus_id
    report = ingest_site(site, settings, reg, Tracer(None), ledger, embedder=EMB,
                         crawl_kwargs={"transport": mock_transport(SITE_A_PAGES), "check_network": False})
    assert report.embedded == 0 and report.reused_embeddings == report.chunk_count
    new = CorpusStore(settings, site.site_id, report.corpus_id)
    assert new.point_count() == report.chunk_count  # no duplicate points
    old_ids = {c.chunk_id for c in CorpusStore(settings, site.site_id, old_corpus).load_chunks()}
    assert {c.chunk_id for c in new.load_chunks()} == old_ids  # stable chunk identities across refresh


def test_failed_refresh_keeps_last_usable_index(env):
    settings, reg, ledger = env
    site = reg.get(1)
    before = site.active_corpus_id
    with pytest.raises(RagError) as err:
        ingest_site(site, settings, reg, Tracer(None), ledger, embedder=EMB,
                    crawl_kwargs={"transport": mock_transport(SITE_A_PAGES, fail_all=True), "check_network": False})
    assert err.value.code == ErrorCode.LOW_CONTENT
    after = SiteRegistry(settings.registry_path, defaults_file=None).get(1)
    assert after.active_corpus_id == before and after.status == SiteStatus.READY and after.last_error
    result = answer_question(deps(settings, reg, generator=lambda p, m: output(draft("insufficient_evidence", "", []))),
                             "widget color", 1, mode="dense", provider_mode="auto")
    assert result.status == "insufficient_evidence"  # still queryable


def test_incomplete_or_incompatible_indexes_are_refused(env):
    settings, reg, _ = env
    fresh, _ = reg.add("https://docs.c.test/x/", check_network=False)
    r = answer_question(deps(settings, reg), "anything", fresh.number, mode="dense", provider_mode="auto")
    assert r.status == "error" and r.error["code"] == "site_not_ready"

    site = reg.get(1)
    CorpusStore(settings, site.site_id, site.active_corpus_id).drop_collection()
    r = answer_question(deps(settings, reg), "anything", 1, mode="dense", provider_mode="auto")
    assert r.error["code"] == "index_incomplete"

    settings.embedding_model = "BAAI/bge-small-en-v1.5"
    r = answer_question(deps(settings, reg), "anything", 2, mode="dense", provider_mode="auto")
    assert r.error["code"] == "index_incompatible" and "Refusing to mix" in r.error["message"]


# ------------------------------------------------------------------ isolation


@pytest.mark.parametrize("mode", ["dense", "bm25", "hybrid"])
def test_answer_only_on_site_b_is_not_retrieved_on_site_a(env, mode):
    settings, reg, _ = env
    a, b = reg.get(1), reg.get(2)
    calls = []

    def gen(provider, messages):
        calls.append(messages)
        return output(draft("insufficient_evidence", "", [], missing="No frobnicator information."))

    r = answer_question(deps(settings, reg, gen), "What does the frobnicator command export?", 1, mode=mode, provider_mode="auto")
    assert r.status == "insufficient_evidence" and r.site_id == a.site_id
    assert r.answer.startswith("The indexed pages for Site A do not contain enough information")
    assert all(x.chunk.site_id == a.site_id for x in r.retrieved)
    assert not any("frobnicator" in x.chunk.text for x in r.retrieved)
    assert not r.citations
    if calls:
        assert "frobnicator command exports" not in str(calls[0][1].content)
        assert b.allowed_host not in str(calls[0][1].content)


def test_overlapping_contradictory_topic_uses_selected_site_only(env):
    settings, reg, _ = env
    a = reg.get(1)
    store = CorpusStore(settings, a.site_id, a.active_corpus_id)
    target = next(c for c in store.load_chunks() if "defaults to 3" in c.text)

    def gen(provider, messages):
        text = str(messages[1].content)
        assert "defaults to 7" not in text  # site B's contradictory value never reaches the model
        return output(draft("answered", "It defaults to 3 retries.",
                            [("WIDGET_RETRY_TIMES defaults to 3.", [(target.chunk_id, "The WIDGET_RETRY_TIMES setting defaults to 3 retries per widget.")])]))

    for mode in ("dense", "bm25", "hybrid"):
        r = answer_question(deps(settings, reg, gen), "What is the default WIDGET_RETRY_TIMES?", 1, mode=mode, provider_mode="auto")
        assert r.status == "answered", r.error
        assert [c.url for c in r.citations] == ["https://docs.a.test/docs/settings.html#widget-retry-times"]
        assert all(x.chunk.site_id == a.site_id for x in r.retrieved)


def test_citation_to_other_site_chunk_is_rejected(env):
    settings, reg, _ = env
    b = reg.get(2)
    b_chunk = next(c for c in CorpusStore(settings, b.site_id, b.active_corpus_id).load_chunks() if "defaults to 7" in c.text)

    def gen(provider, messages):
        return output(draft("answered", "It defaults to 7.", [("defaults to 7", [(b_chunk.chunk_id, "defaults to 7 retries per widget")])]))

    r = answer_question(deps(settings, reg, gen), "default WIDGET_RETRY_TIMES", 1, mode="dense", provider_mode="auto")
    assert r.status == "error" and r.error["code"] == "citation_invalid"
    assert r.answer == "" and not r.citations and "was not in the supplied evidence" in r.rejected_claims[0].reasons[0]


def test_tampered_lexical_index_with_foreign_chunk_fails_closed(env):
    settings, reg, _ = env
    a, b = reg.get(1), reg.get(2)
    a_store = CorpusStore(settings, a.site_id, a.active_corpus_id)
    foreign = CorpusStore(settings, b.site_id, b.active_corpus_id).load_chunks()[0]
    with (a_store.dir / "chunks.jsonl").open("a", encoding="utf-8") as fh:
        fh.write(foreign.model_dump_json() + "\n")
    clear_caches()
    r = answer_question(deps(settings, reg), "gizmos exported with tools", 1, mode="bm25", provider_mode="auto")
    assert r.status == "error" and r.error["code"] == "index_incompatible"


# ------------------------------------------------------------------ graph outcomes


def test_supported_answer_renders_metadata_urls_and_accounts_usage(env):
    settings, reg, _ = env
    a = reg.get(1)
    chunks = CorpusStore(settings, a.site_id, a.active_corpus_id).load_chunks()
    color = next(c for c in chunks if "WIDGET_COLOR" in c.text)

    def gen(provider, messages):
        return output(draft("answered", "Blue.", [("WIDGET_COLOR defaults to blue.",
                                                    [(color.chunk_id, "selects the paint used for widgets;  DEFAULT is blue")])]))

    d = deps(settings, reg, gen)
    r = answer_question(d, "What color are widgets by default?", 1, mode="hybrid", provider_mode="auto")
    assert r.status == "answered" and r.provider == "openai"
    assert r.citations[0].url == "https://docs.a.test/docs/settings.html#widget-color"
    assert r.citations[0].section == "Widget Settings > WIDGET_COLOR"
    gen_events = [e for e in d.ledger.read_all() if e.run_id == r.run_id and e.operation == "generation"]
    assert len(gen_events) == 1 and gen_events[0].input_tokens == 500
    assert r.usage["cost_usd"] == pytest.approx(gen_events[0].cost_usd) and r.usage["attempts"] == 1
    trace = [json.loads(l) for l in (settings.traces_dir / f"{r.run_id}.jsonl").read_text().splitlines()]
    stages = [e["stage"] for e in trace]
    for stage in ("validate", "retrieve", "context", "generate", "check_citations", "finalize", "query"):
        assert stage in stages
    assert all(e.get("site_id") == a.site_id for e in trace if e["stage"] in ("retrieve", "generate"))


def test_partially_invalid_claims_withhold_free_text(env):
    settings, reg, _ = env
    a = reg.get(1)
    color = next(c for c in CorpusStore(settings, a.site_id, a.active_corpus_id).load_chunks() if "WIDGET_COLOR" in c.text)

    def gen(provider, messages):
        return output(draft("answered", "Blue, and widgets are waterproof.", [
            ("Default is blue.", [(color.chunk_id, "default is blue")]),
            ("Widgets are waterproof.", [(color.chunk_id, "widgets are waterproof")]),
        ]))

    r = answer_question(deps(settings, reg, gen), "widget color", 1, mode="dense", provider_mode="auto")
    assert r.status == "partially_answered" and r.answer == ""
    assert [c.text for c in r.claims] == ["Default is blue."] and len(r.rejected_claims) == 1


def test_empty_retrieval_abstains_without_generation(env):
    settings, reg, _ = env
    called = []
    r = answer_question(deps(settings, reg, lambda p, m: called.append(1)), "zzqx qqzz", 1, mode="bm25", provider_mode="auto")
    assert r.status == "insufficient_evidence" and not called and r.provider is None


def test_fallback_to_groq_is_visible_and_counted_once_per_attempt(env):
    settings, reg, _ = env
    calls = []

    def gen(provider, messages):
        calls.append(provider)
        if provider == "openai":
            raise _openai_err(429, {"error": {"code": "credit_balance_exhausted"}})
        return output(draft("insufficient_evidence", "", []))

    d = deps(settings, reg, gen)
    r = answer_question(d, "widget color", 1, mode="dense", provider_mode="auto")
    assert calls == ["openai", "groq"] and r.provider == "groq" and "credits_exhausted" in r.fallback_reason
    assert [(a.provider, a.outcome) for a in r.provider_attempts] == [("openai", "error"), ("groq", "success")]
    events = [e for e in d.ledger.read_all() if e.run_id == r.run_id and e.operation == "generation"]
    assert len(events) == 2 and len({e.operation_id for e in events}) == 1
    assert events[0].cost_usd is None and events[0].error_code == "credits_exhausted"  # unknown, not zero
    assert r.usage["unknown_cost_events"] == 1


def test_explicit_provider_failure_is_an_operational_error_not_abstention(env):
    settings, reg, _ = env

    def gen(provider, messages):
        raise _openai_err(429, {"error": {"code": "credit_balance_exhausted"}})

    r = answer_question(deps(settings, reg, gen), "widget color", 1, mode="dense", provider_mode="openai")
    assert r.status == "error" and r.error["code"] == "credits_exhausted"
    assert r.error["message"].startswith("OpenAI API credits are exhausted.")
    assert r.retrieved  # evidence stage succeeded; failure stage is generation


def test_budget_cap_stops_generation(env):
    settings, reg, _ = env
    budget = Budget(max_requests=0)
    r = answer_question(deps(settings, reg, lambda p, m: output(draft("insufficient_evidence", "", [])), budget=budget),
                        "widget color", 1, mode="dense", provider_mode="auto")
    assert r.error["code"] == "budget_exceeded"


def test_context_budget_and_injection_escaping(env):
    settings, reg, _ = env
    a = reg.get(1)
    store = CorpusStore(settings, a.site_id, a.active_corpus_id)
    retrieved = Retriever(settings, a, store, embedder=EMB).search("build command widgets", "dense")
    smallest = min(r.chunk.token_count for r in retrieved)
    selected = select_context(retrieved, 6, smallest, a.site_id, store.corpus_id)
    assert len(selected) >= 1 and sum(r.chunk.token_count for r in selected) <= smallest
    cli_chunk = next(c for c in store.load_chunks() if "reveal the API key" in c.text)
    messages = build_messages("q", a, [cli_chunk])
    user = str(messages[1].content)
    assert user.count("</chunk>") == 1  # page text cannot close the evidence block early
    assert "untrusted data" in str(messages[0].content)
