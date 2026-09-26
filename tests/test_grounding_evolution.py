"""Display-boundary and exact-passage regression tests."""
import io

import pytest
from rich.console import Console

from website_rag.citations import quote_in_text, validate_claims
from website_rag.evidence import evidence_spans
from website_rag.graph import answer_question
from website_rag.index import CorpusStore
from website_rag.render import answer_panel
from website_rag.schemas import AnswerSelection
from .helpers import draft, output
from .test_pipeline import env, deps


@pytest.mark.parametrize('quote,text', [
    ('FOO is enabled', 'foo is enabled'),
    ('is ... enabled', 'is not enabled'),
    ('return True', 'return true'),
    ('x == 1', 'x != 1'),
    ('the flag is enabled', 'the flag is not enabled'),
])
def test_material_quote_changes_fail(quote, text):
    assert not quote_in_text(quote, text)


def test_all_valid_claims_cannot_authorize_free_text_or_premise(env):
    settings, reg, _ = env
    site = reg.get(1)
    c = next(c for c in CorpusStore(settings, site.site_id, site.active_corpus_id).load_chunks() if 'default is blue' in c.text)
    d = draft('answered', 'Widgets are waterproof.', [('Default is blue.', [(c.chunk_id, 'default is blue')])],
              missing='Secret unsupported statement', premise='Widgets cost $99.')
    r = answer_question(deps(settings, reg, lambda p, m: output(d)), 'widget color', 1)
    assert r.status == 'answered' and r.answer == 'Default is blue. [1]'
    assert not r.premise_issue and not r.missing_information


def test_span_reconstruction_and_wrong_chunk_id(env):
    settings, reg, _ = env
    site = reg.get(1)
    chunks = CorpusStore(settings, site.site_id, site.active_corpus_id).load_chunks()
    c = next(c for c in chunks if 'default is blue' in c.text)
    span = next(s for s in evidence_spans(c) if 'default is blue' in s.text)
    d = AnswerSelection(status='answered', claims=[{'text': 'Default is blue.', 'evidence': [
        {'chunk_id': c.chunk_id, 'span_id': span.span_id}]}])
    accepted, citations, rejected = validate_claims(d, {c.chunk_id: c})
    assert accepted and not rejected
    assert citations[0].quote == c.text[span.start:span.end]
    d.claims[0].evidence[0].span_id = 'invented'
    assert not validate_claims(d, {c.chunk_id: c})[0]


@pytest.mark.parametrize('width', [80, 100, 140])
def test_partial_display_hides_rejected_and_deduplicates_sources(env, width):
    settings, reg, _ = env
    site = reg.get(1)
    c = next(c for c in CorpusStore(settings, site.site_id, site.active_corpus_id).load_chunks() if 'default is blue' in c.text)
    d = draft('answered', 'UNSAFE', [('Default is blue.', [(c.chunk_id, 'default is blue')]),
                                   ('REJECTEDSECRET', [(c.chunk_id, 'invented passage')])])
    r = answer_question(deps(settings, reg, lambda p, m: output(d)), 'widget color', 1)
    buffer = io.StringIO()
    Console(file=buffer, width=width, no_color=True).print(answer_panel(r))
    text = buffer.getvalue()
    assert 'Default is blue.' in text and 'REJECTEDSECRET' not in text and 'UNSAFE' not in text
    assert 'citation checks failed' in text and 'Not covered by the indexed pages' not in text
    assert text.count(c.citation_url) == 1 and 'Verified claims' not in text
    Console(file=buffer, width=width).print(answer_panel(r, explain=True))
    assert 'UNTRUSTED REJECTED CANDIDATE' in buffer.getvalue()
