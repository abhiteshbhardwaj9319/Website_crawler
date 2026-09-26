"""Offline ablations; development selection precedes the separately invoked holdout."""
import argparse
import hashlib
import json
import statistics
import time
from pathlib import Path

from website_rag.config import Settings
from website_rag.evaluate import load_questions, retrieval_metrics, _group_hit_rank
from website_rag.generate import prompt_fingerprint
from website_rag.index import CorpusStore, close_qdrant
from website_rag.registry import SiteRegistry
from website_rag.retrieve import Retriever, select_context


def run(split: str, holdout: bool = False):
    s = Settings()
    reg = SiteRegistry(s.registry_path)
    out = Path('eval/results/evolution-retrieval')
    out.mkdir(parents=True, exist_ok=True)
    cases = load_questions(split)
    variants = [('dense20', 'dense', 20, 20, 'ranked'), ('bm2520', 'bm25', 20, 20, 'ranked'),
                ('hybrid20', 'hybrid', 20, 20, 'ranked'), ('rerank20', 'hybrid_rerank', 20, 20, 'ranked')]
    if not holdout:
        variants += [('hybrid40', 'hybrid', 40, 20, 'ranked'), ('rerank40', 'hybrid_rerank', 40, 40, 'ranked'),
                     ('rerank40_depth20', 'hybrid_rerank', 40, 20, 'ranked'),
                     ('hybrid_diverse', 'hybrid', 20, 20, 'diverse'),
                     ('hybrid_neighbors', 'hybrid', 20, 20, 'neighbors')]
    else:
        selection = json.loads((out/'selection.json').read_text())
        v = tuple(selection['variant'])
        if v not in variants:
            variants.append(v)
    stores = {n: CorpusStore(s, (site := reg.get(n)).site_id, site.active_corpus_id) for n in (1, 2)}
    for n, store in stores.items():
        store.verify_ready(reg.get(n), s.embedding_model)
    meta = {'split': split, 'questions_sha256': hashlib.sha256(Path(f'eval/questions_{split}.jsonl').read_bytes()).hexdigest(),
            'prompt': prompt_fingerprint(), 'corpora': {n: st.corpus_id for n, st in stores.items()},
            'embedding_model': s.embedding_model, 'reranker_model': s.reranker_model,
            'context_max_chunks': s.context_max_chunks, 'evidence_token_budget': s.evidence_token_budget,
            'scoring': 'exact contiguous quote, whitespace normalized; case preserved'}
    summaries = []
    for tag, mode, depth, rerank_depth, policy in variants:
        rows = []
        cfg = s.model_copy(update={'candidate_k': depth, 'rerank_k': rerank_depth})
        for case in cases:
            st = stores[case['site']]
            retriever = Retriever(cfg, reg.get(case['site']), st)
            t = time.perf_counter()
            ranked = retriever.search(case['question'], mode)
            retrieval_ms = (time.perf_counter() - t) * 1000
            t = time.perf_counter()
            ctx = select_context(ranked, cfg.context_max_chunks, cfg.evidence_token_budget, st.site_id, st.corpus_id,
                                 policy, st.load_chunks() if policy == 'neighbors' else None)
            context_ms = (time.perf_counter() - t) * 1000
            metrics = retrieval_metrics(case, ranked, ctx)
            stages = []
            for group in case.get('evidence_groups', []):
                stages.append('context' if _group_hit_rank(group, [r.chunk for r in ctx]) else
                              'context_budget' if _group_hit_rank(group, [r.chunk for r in ranked]) else
                              'candidate_miss' if _group_hit_rank(group, st.load_chunks()) else 'corpus_or_label_gap')
            rows.append({'id': case['id'], 'site': case['site'], 'question': case['question'], 'metrics': metrics,
                         'failure_stages': stages, 'retrieval_ms': round(retrieval_ms, 2),
                         'context_ms': round(context_ms, 2), 'timings_ms': retriever.timings,
                         'context_ids': [r.chunk.chunk_id for r in ctx],
                         'context_urls': [r.chunk.citation_url for r in ctx],
                         'isolation_ok': all(r.chunk.site_id == st.site_id and r.chunk.corpus_id == st.corpus_id for r in ranked + ctx)})
        hits = sum(r['metrics']['context_group_hits'] for r in rows)
        total = sum(r['metrics']['groups'] for r in rows)
        summary = {'variant': [tag, mode, depth, rerank_depth, policy], 'cases': len(rows), 'hits': hits, 'groups': total,
                   'context_recall': f'{hits}/{total}', 'mean_ms': round(statistics.mean(r['retrieval_ms'] for r in rows), 2),
                   'warm_mean_ms': round(statistics.mean(r['retrieval_ms'] for r in rows[1:]), 2),
                   'isolation_leaks': sum(not r['isolation_ok'] for r in rows)}
        (out/f'{split}-{tag}.json').write_text(json.dumps({'metadata':meta,'summary':summary,'rows':rows},indent=2),encoding='utf-8')
        summaries.append(summary)
        print(tag, summary['context_recall'], summary['mean_ms'], flush=True)
    (out/f'{split}-summary.json').write_text(json.dumps({'metadata':meta,'variants':summaries},indent=2),encoding='utf-8')
    close_qdrant()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--holdout', action='store_true')
    args = parser.parse_args()
    run('holdout_v1' if args.holdout else 'evolution_dev', args.holdout)
