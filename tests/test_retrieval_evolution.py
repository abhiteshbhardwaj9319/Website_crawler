from website_rag.retrieve import select_context, Retriever
from website_rag.index import CorpusStore
from website_rag.errors import RagError
import pytest
from .test_pipeline import env, EMB


def test_context_policies_respect_budget_and_site(env):
    settings, reg, _ = env
    site = reg.get(1)
    store = CorpusStore(settings, site.site_id, site.active_corpus_id)
    ranked = Retriever(settings, site, store, embedder=EMB).search('widget color retries', 'hybrid')
    for policy in ('ranked', 'diverse', 'neighbors'):
        ctx = select_context(ranked, 3, 500, site.site_id, store.corpus_id, policy, store.load_chunks())
        assert len(ctx) <= 3 and sum(c.chunk.token_count for c in ctx) <= 500
        assert len({c.chunk.chunk_id for c in ctx}) == len(ctx)
        corrupt = ranked[-1].model_copy(deep=True)
        corrupt.chunk.site_id = 'another-site'
        with pytest.raises(RagError):
            select_context(ranked[:1] + [corrupt], 1, 500, site.site_id, store.corpus_id, policy)


def test_rerank_depth_is_bounded_and_timings_recorded(env):
    settings, reg, _ = env
    settings.rerank_k = 2
    site = reg.get(1)
    store = CorpusStore(settings, site.site_id, site.active_corpus_id)
    class Reranker:
        def score(self, query, documents):
            assert len(documents) <= 2
            return list(range(len(documents)))
    retriever = Retriever(settings, site, store, embedder=EMB, reranker=Reranker())
    result = retriever.search('widget color retries', 'hybrid_rerank')
    assert result and 'rerank_ms' in retriever.timings and 'query_embedding_ms' in retriever.timings
