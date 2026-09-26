"""Bounded E1 capture; no secrets or raw crawl content are exported."""
import hashlib
import json
import time
from pathlib import Path

from rich.console import Console
from website_rag.config import Settings
from website_rag.evaluate import run_evaluation
from website_rag.generate import prompt_fingerprint
from website_rag.graph import Budget, QueryDeps, answer_question, load_run
from website_rag.index import CorpusStore, close_qdrant
from website_rag.registry import SiteRegistry
from website_rag.render import answer_panel
from website_rag.tracing import Tracer
from website_rag.usage import UsageLedger


def main():
    s = Settings()
    reg = SiteRegistry(s.registry_path)
    out = Path('eval/results/evolution-baseline')
    if (out / 'fingerprints.json').exists():
        raise SystemExit('Frozen baseline exists. Use a separately versioned capture; never overwrite it.')
    out.mkdir(parents=True, exist_ok=True)
    local = Path('artifacts/evolution')
    local.mkdir(parents=True, exist_ok=True)
    meta = {'prompt': prompt_fingerprint(), 'sites': [], 'saved_runs': [], 'live': []}
    for site in reg.list():
        store = CorpusStore(s, site.site_id, site.active_corpus_id)
        manifest = store.verify_ready(site, s.embedding_model)
        meta['sites'].append({'site': site.number, 'manifest': manifest.model_dump(mode='json'),
                              'chunks_sha256': hashlib.sha256((store.dir/'chunks.jsonl').read_bytes()).hexdigest(),
                              'input_definition_present': any(c.anchor == 'input' for c in store.load_chunks())})
    for rid in ['r-b8c282cd238248ee', 'r-50ca55c1fae84f43']:
        r = load_run(s, rid)
        meta['saved_runs'].append({'run_id': rid, 'status': r.status, 'usage': r.usage, 'latency_ms': r.latency_ms,
                                   'accepted_claims': len(r.claims), 'rejected_reasons': [x.reasons for x in r.rejected_claims]})
        for width in (80, 100, 140):
            with (local/f'{rid}-{width}-before.txt').open('w', encoding='utf-8') as f:
                Console(file=f, width=width, no_color=True).print(answer_panel(r))
    (out/'fingerprints.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')
    for split in ('dev', 'test', 'isolation'):
        run_evaluation(s, reg, None, split, ['dense', 'bm25', 'hybrid', 'hybrid_rerank'], None, True, str(out/split))
        print('retrieval complete:', split, flush=True)
    budget = Budget(min(3, s.eval_max_requests), min(.03, s.eval_max_cost_usd))
    for site, question in [(1, 'Which command creates a new Scrapy project?'), (2, 'what is python'),
                           (2, 'what is python command to get input')]:
        deps = QueryDeps(s, reg, UsageLedger(s.ledger_path), Tracer(s.traces_dir, kind='evaluation'),
                         phase='evaluation', budget=budget)
        r = answer_question(deps, question, site, provider_mode='openai')
        meta['live'].append({'site': site, 'question': question, 'run_id': r.run_id, 'status': r.status,
                             'error': r.error, 'usage': r.usage, 'latency_ms': r.latency_ms,
                             'claims': [c.model_dump() for c in r.claims],
                             'citations': [c.model_dump() for c in r.citations],
                             'rejected_reasons': [x.reasons for x in r.rejected_claims]})
        print('live:', site, r.status, r.run_id, flush=True)
    (out/'fingerprints.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')
    close_qdrant()


if __name__ == '__main__':
    main()
