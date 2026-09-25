"""Opt-in live checks (network, real providers). Run with: RAG_LIVE_TESTS=1 uv run pytest tests/test_live.py -q

They make at most one small generation call per configured provider. They require that
site 1 has been ingested (`rag ingest --site 1`) and keys are present in the local .env.
"""

from __future__ import annotations

import pytest

from website_rag.config import Settings
from website_rag.graph import QueryDeps, answer_question
from website_rag.index import close_qdrant
from website_rag.registry import SiteRegistry
from website_rag.tracing import Tracer
from website_rag.usage import UsageLedger

pytestmark = pytest.mark.live


@pytest.fixture(scope="module")
def live():
    s = Settings()
    reg = SiteRegistry(s.registry_path)
    if not reg.get(1).is_queryable:
        pytest.skip("site 1 is not ingested")
    yield s, reg
    close_qdrant()


@pytest.mark.parametrize("provider", ["openai", "groq"])
def test_live_provider_answers_with_valid_citations(live, provider):
    s, reg = live
    if s.key_for(provider) is None:
        pytest.skip(f"{provider} key not configured")
    deps = QueryDeps(settings=s, registry=reg, ledger=UsageLedger(s.ledger_path), tracer=Tracer(s.traces_dir), phase="query")
    r = answer_question(deps, "Which command creates a new Scrapy project?", 1, provider_mode=provider)
    assert r.status in ("answered", "partially_answered"), r.error
    assert r.provider == provider and r.citations
    assert all(c.url.startswith("https://docs.scrapy.org/en/2.19/") for c in r.citations)
    assert r.usage["input_tokens"] > 0 and r.usage["cost_usd"] > 0


def test_live_unanswerable_abstains(live):
    s, reg = live
    if s.openai_api_key is None and s.groq_api_key is None:
        pytest.skip("no provider configured")
    deps = QueryDeps(settings=s, registry=reg, ledger=UsageLedger(s.ledger_path), tracer=Tracer(s.traces_dir), phase="query")
    r = answer_question(deps, "What is the capital city of Australia?", 1)
    assert r.status == "insufficient_evidence", r.answer
