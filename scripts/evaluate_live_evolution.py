"""Bounded, explicit live acceptance run; no fallback and no overwrite of evidence."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import time
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlsplit

from rich.console import Console

from website_rag.config import Settings
from website_rag.evaluate import load_questions, retrieval_metrics, score_answer, summarize_rows
from website_rag.generate import prompt_fingerprint
from website_rag.graph import Budget, QueryDeps, answer_question
from website_rag.index import CorpusStore, close_qdrant
from website_rag.registry import SiteRegistry
from website_rag.render import answer_panel
from website_rag.tracing import Tracer
from website_rag.usage import UsageLedger


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-live', action='store_true', required=True)
    args = parser.parse_args()
    assert args.run_live
    s = Settings().model_copy(update={'max_attempts_per_provider': 1, 'candidate_k': 20,
                                     'rerank_k': 20, 'context_policy': 'ranked'})
    reg = SiteRegistry(s.registry_path)
    out = Path('eval/results/evolution-live')
    out.mkdir(parents=True, exist_ok=True)
    target = out / 'answers.jsonl'
    if target.exists():
        raise SystemExit('Evidence exists; choose a new version rather than overwrite it.')
    corpora = {}
    for n, allowed in [(1, 'docs.scrapy.org'), (2, 'docs.python.org')]:
        site = reg.require_queryable(n)
        store = CorpusStore(s, site.site_id, site.active_corpus_id)
        store.verify_ready(site, s.embedding_model)
        chunks = store.load_chunks()
        assert all(urlsplit(c.source_url).hostname == allowed and c.site_id == site.site_id
                   and c.corpus_id == store.corpus_id for c in chunks)
        corpora[str(n)] = {'corpus_id': store.corpus_id, 'chunks': len(chunks),
                          'content_sha256': hashlib.sha256(''.join(c.text for c in chunks).encode()).hexdigest()}
    budget = Budget(min(40, s.eval_max_requests), min(.15, s.eval_max_cost_usd))
    meta = {'started_at': datetime.now(UTC).isoformat(), 'corpora': corpora, 'prompt': prompt_fingerprint(),
            'budget': {'max_requests': budget.max_requests, 'stop_after_known_cost_usd': budget.max_cost_usd},
            'settings': {'mode': 'hybrid_rerank', 'candidate_k': 20, 'rerank_k': 20,
                         'context_policy': 'ranked', 'evidence_token_budget': s.evidence_token_budget,
                         'context_max_chunks': s.context_max_chunks, 'embedding_model': s.embedding_model,
                         'reranker_model': s.reranker_model, 'openai_model': s.openai_model,
                         'groq_model': s.groq_model, 'max_attempts_per_provider': 1},
            'question_hashes': {name: hashlib.sha256(Path(f'eval/questions_{name}.jsonl').read_bytes()).hexdigest()
                                for name in ['holdout_v1', 'test', 'isolation']}}
    (out/'fingerprints.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')
    user_cases = [{'id': 'U01', 'site': 1, 'question': 'What command creates a new Scrapy project?',
                   'expected_behavior': 'answer'},
                  {'id': 'U02', 'site': 2, 'question': 'what is python', 'expected_behavior': 'answer'},
                  {'id': 'U03', 'site': 2, 'question': 'what is python command to get input',
                   'expected_behavior': 'answer'}]
    for c in user_cases:
        c.update(category='user', evidence_groups=[], required_pages=[])
    suites = [('holdout_v1', load_questions('holdout_v1'), 'openai'),
              ('test_observed', load_questions('test'), 'openai'),
              ('isolation', load_questions('isolation'), 'openai'), ('user', user_cases, 'openai')]
    if s.key_for('groq'):
        suites.append(('groq_independent', [user_cases[0], user_cases[2], load_questions('holdout_v1')[10]], 'groq'))
    rows = []
    summaries = []
    try:
        for split, cases, provider in suites:
            suite_rows = []
            for i, case in enumerate(cases):
                if provider == 'groq' and i:
                    time.sleep(35)  # bounded spacing; tier unknown, failures remain visible
                tracer = Tracer(s.traces_dir, kind='evaluation')
                result = answer_question(QueryDeps(s, reg, UsageLedger(s.ledger_path), tracer,
                                                   phase='evaluation', budget=budget),
                                         case['question'], case['site'], mode='hybrid_rerank', provider_mode=provider)
                context = [r for r in result.retrieved if r.chunk.chunk_id in result.context_chunk_ids]
                row = {'split': split, 'id': case['id'], 'category': case['category'], 'site': case['site'],
                       'question': case['question'], 'expected_behavior': case['expected_behavior'],
                       'run_id': result.run_id, 'corpus_id': result.corpus_id, 'answer': result.answer,
                       'claims': [c.model_dump() for c in result.claims],
                       'citations': [c.model_dump() for c in result.citations],
                       'rejected_claims': [c.model_dump() for c in result.rejected_claims],
                       'provider': result.provider, 'model': result.model, 'error': result.error,
                       'provider_attempts': [a.model_dump() for a in result.provider_attempts],
                       'fallback_reason': result.fallback_reason, 'usage': result.usage,
                       'latency_ms': result.latency_ms, **score_answer(case, result),
                       'retrieval': retrieval_metrics(case, result.retrieved, context),
                       'stage_ms': {e['stage']: e.get('duration_ms') for e in tracer.events if e['type'] == 'span'},
                       'leaks': [r.chunk.chunk_id for r in result.retrieved
                                 if r.chunk.site_id != result.site_id or r.chunk.corpus_id != result.corpus_id]}
                output = io.StringIO()
                start = time.perf_counter()
                Console(file=output, width=100, no_color=True).print(answer_panel(result))
                row['render_ms'] = round((time.perf_counter()-start)*1000, 2)
                with target.open('a', encoding='utf-8', newline='\n') as f:
                    f.write(json.dumps(row, ensure_ascii=False) + '\n')
                rows.append(row)
                suite_rows.append(row)
                print(split, case['id'], result.run_id, result.status, result.usage.get('cost_usd'), flush=True)
                if (result.error or {}).get('code') == 'budget_exceeded':
                    break
            summaries.append(summarize_rows(suite_rows, split, 'hybrid_rerank', False, provider))
            (out/'summary.json').write_text(json.dumps({'runs': summaries, 'requests': budget.requests,
                                                       'known_cost_usd': budget.cost_usd}, indent=2), encoding='utf-8')
    finally:
        close_qdrant()


if __name__ == '__main__':
    main()
