"""One fresh-process and one warm query, local only; not a statistical benchmark."""
import json
import time
from pathlib import Path

from website_rag.config import Settings
from website_rag.index import CorpusStore, close_qdrant
from website_rag.registry import SiteRegistry
from website_rag.retrieve import Retriever, select_context

s = Settings().model_copy(update={'candidate_k': 20, 'rerank_k': 20})
reg = SiteRegistry(s.registry_path)
site = reg.require_queryable(1)
rows = []
try:
    for label in ('fresh_process_cached_models_on_disk', 'warm_same_process'):
        start = time.perf_counter()
        store = CorpusStore(s, site.site_id, site.active_corpus_id)
        store.verify_ready(site, s.embedding_model)
        verify_ms = (time.perf_counter()-start)*1000
        retriever = Retriever(s, site, store)
        start = time.perf_counter()
        ranked = retriever.search('What command creates a new Scrapy project?', 'hybrid_rerank')
        search_ms = (time.perf_counter()-start)*1000
        start = time.perf_counter()
        context = select_context(ranked, 10, 3000, site.site_id, store.corpus_id)
        rows.append({'condition': label, 'verify_store_ms': round(verify_ms, 2),
                     'retrieval_ms': round(search_ms, 2), 'context_ms': round((time.perf_counter()-start)*1000, 2),
                     'stages_ms': retriever.timings, 'context_ids': [r.chunk.chunk_id for r in context]})
    report = {'question': 'What command creates a new Scrapy project?', 'corpus': store.corpus_id,
              'note': 'Disk/OS cache not cleared. Two observations; no network model download.', 'rows': rows}
    path = Path('eval/results/evolution-retrieval/cold-warm.json')
    if path.exists():
        raise SystemExit('Refusing to overwrite measured evidence.')
    path.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))
finally:
    close_qdrant()
