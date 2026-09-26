"""Evaluation harness over the frozen question sets in eval/.

Retrieval metrics (answerable cases; an evidence group is "hit" when any retrieved chunk
from the labeled URL contains the labeled quote):
- evidence Recall@k for k in (3, 6, 10, 20): hit groups / labeled groups
- context evidence recall: hit groups among chunks actually sent to the model
- multi-page coverage: required pages present in context / required pages
Answer metrics (automatic; semantic support is reviewed manually, see docs/evaluation.md):
- behavior correctness per expected_behavior (answer / correct_premise / abstain)
- citation validity: validated claims / all generated claims
- gold-evidence citation: answered cases citing at least one gold evidence chunk
- isolation: no forbidden host in retrieved, context, or cited sources
Infrastructure errors are reported separately from quality failures.
"""

from __future__ import annotations

import hashlib
import json
import statistics
import time
from datetime import UTC, datetime
from pathlib import Path

from .citations import quote_in_text
from .config import PROJECT_ROOT, Settings
from .errors import ErrorCode, RagError
from .graph import Budget, QueryDeps, answer_question
from .index import CorpusStore
from .registry import SiteRegistry
from .retrieve import Retriever, select_context
from .schemas import ChunkRecord, RetrievedChunk
from .tracing import Tracer
from .usage import UsageLedger

EVAL_DIR = PROJECT_ROOT / "eval"
KS = (3, 6, 10, 20)
QUALITY_ERRORS = {str(ErrorCode.CITATION_INVALID), str(ErrorCode.MALFORMED_RESPONSE)}


def load_questions(split: str) -> list[dict]:
    path = EVAL_DIR / f"questions_{split}.jsonl"
    if not path.exists():
        raise RagError(ErrorCode.INVALID_INPUT, f"No question file for split '{split}'.", hint="Use test, dev, or isolation.")
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def file_sha256(split: str) -> str:
    return hashlib.sha256((EVAL_DIR / f"questions_{split}.jsonl").read_bytes()).hexdigest()


def _group_hit_rank(group: list[dict], ranked: list[ChunkRecord]) -> int | None:
    for rank, chunk in enumerate(ranked, start=1):
        for alt in group:
            if chunk.source_url == alt["url"] and quote_in_text(alt["quote"], chunk.text):
                return rank
    return None


def retrieval_metrics(case: dict, retrieved: list[RetrievedChunk], context: list[RetrievedChunk]) -> dict:
    groups = case.get("evidence_groups", [])
    ranked = [r.chunk for r in retrieved]
    ctx = [r.chunk for r in context]
    ranks = [_group_hit_rank(g, ranked) for g in groups]
    ctx_ranks = [_group_hit_rank(g, ctx) for g in groups]
    ctx_hits = [r is not None for r in ctx_ranks]
    hit_pages = {ctx[r - 1].source_url for r in ctx_ranks if r is not None}
    required = case.get("required_pages", [])
    ctx_pages = {c.source_url for c in ctx}
    out = {
        "groups": len(groups),
        "group_ranks": ranks,
        "context_group_hits": sum(ctx_hits),
        "required_pages": len(required),
        "required_pages_in_context": sum(1 for p in required if p in ctx_pages),
        "first_hit_rank": min((r for r in ranks if r), default=None),
        # every evidence group satisfied in context, by passages from >= 2 distinct pages
        "multi_page_complete": bool(groups) and all(ctx_hits) and len(hit_pages) >= 2,
        "evidence_pages_in_context": len(hit_pages),
    }
    for k in KS:
        out[f"hits@{k}"] = sum(1 for r in ranks if r is not None and r <= k)
    return out


def _forbidden_leaks(case: dict, retrieved, context, citations) -> list[str]:  # noqa: ANN001
    hosts = case.get("forbidden_hosts", [])
    leaks = []
    for label, urls in (("retrieved", [r.chunk.source_url for r in retrieved]),
                        ("context", [r.chunk.source_url for r in context]),
                        ("citations", [c.url for c in citations])):
        for url in urls:
            if any(f"//{h}/" in url for h in hosts):
                leaks.append(f"{label}:{url}")
    return leaks


def score_answer(case: dict, result) -> dict:  # noqa: ANN001
    expected = case["expected_behavior"]
    status = result.status
    error_code = (result.error or {}).get("code")
    infra_error = status == "error" and error_code not in QUALITY_ERRORS
    cited_ids = {c.chunk_id for c in result.citations}
    cited_chunks = [r.chunk for r in result.retrieved if r.chunk.chunk_id in cited_ids]
    gold_cited = sum(1 for g in case.get("evidence_groups", []) if _group_hit_rank(g, cited_chunks) is not None)
    total_claims = len(result.claims) + len(result.rejected_claims)
    if expected == "abstain":
        behavior_ok = status == "insufficient_evidence"
    elif expected == "correct_premise":
        # Structural proxy only; semantic correction is reviewed explicitly in the report.
        behavior_ok = status in ("answered", "partially_answered") and gold_cited > 0
    else:
        behavior_ok = status in ("answered", "partially_answered")
    return {
        "status": status,
        "error_code": error_code,
        "infra_error": infra_error,
        "behavior_ok": None if infra_error else behavior_ok,
        "inappropriate_abstention": expected == "answer" and status == "insufficient_evidence",
        "claims_total": total_claims,
        "claims_valid": len(result.claims),
        "gold_groups_cited": gold_cited,
    }


def run_evaluation(
    settings: Settings,
    registry: SiteRegistry,
    site_ref,  # noqa: ANN001
    split: str,
    modes: list[str],
    provider: str | None,
    retrieval_only: bool,
    out_dir: str | None,
    console=None,  # noqa: ANN001
) -> dict:
    cases = load_questions(split)
    if split != "isolation" and site_ref is not None:
        cases = [c for c in cases if str(c["site"]) == str(registry.get(site_ref).number)]
    out = Path(out_dir) if out_dir else EVAL_DIR / "results"
    out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
    budget = Budget(settings.eval_max_requests, settings.eval_max_cost_usd)
    summaries = []
    for mode in modes:
        tag = f"{stamp}_{split}_{mode}_{'retrieval' if retrieval_only else (provider or settings.provider)}"
        rows = []
        for case in cases:
            site = registry.require_queryable(case["site"])
            store = CorpusStore(settings, site.site_id, site.active_corpus_id)
            store.verify_ready(site, settings.embedding_model)
            row = {"id": case["id"], "category": case["category"], "site": site.number, "question": case["question"],
                   "expected_behavior": case["expected_behavior"], "mode": mode, "corpus_id": store.corpus_id}
            row['retrieval'] = retrieval_metrics(case, [], [])
            if retrieval_only:
                t0 = time.monotonic()
                retrieved = Retriever(settings, site, store).search(case["question"], mode)
                context = select_context(retrieved, settings.context_max_chunks, settings.evidence_token_budget,
                                         site.site_id, store.corpus_id, settings.context_policy,
                                         store.load_chunks() if settings.context_policy == 'neighbors' else None)
                row["retrieval_ms"] = int((time.monotonic() - t0) * 1000)
                citations = []
            else:
                deps = QueryDeps(settings=settings, registry=registry, ledger=UsageLedger(settings.ledger_path),
                                 tracer=Tracer(settings.traces_dir, kind="evaluation"), phase="evaluation", budget=budget)
                result = answer_question(deps, case["question"], site.number, mode=mode, provider_mode=provider)
                retrieved = result.retrieved
                context = [r for r in retrieved if r.chunk.chunk_id in set(result.context_chunk_ids)]
                citations = result.citations
                row.update(score_answer(case, result))
                row.update({
                    "run_id": result.run_id, "answer": result.answer, "premise_issue": result.premise_issue,
                    "missing_information": result.missing_information,
                    "claims": [c.model_dump() for c in result.claims],
                    "citations": [{"marker": c.marker, "url": c.url, "section": c.section, "quote": c.quote} for c in result.citations],
                    "rejected_claims": [r.model_dump() for r in result.rejected_claims],
                    "provider": result.provider, "model": result.model, "fallback_reason": result.fallback_reason,
                    "latency_ms": result.latency_ms, "usage": result.usage,
                })
                if row["error_code"] == str(ErrorCode.BUDGET_EXCEEDED):
                    rows.append(row)
                    if console:
                        console.print(f"[yellow]Budget cap reached at {case['id']}; stopping this run.[/]")
                    break
            row["retrieval"] = retrieval_metrics(case, retrieved, context)
            row["top_retrieved"] = [{"rank": r.rank, "url": r.chunk.source_url, "section": r.chunk.section_label,
                                     "score": round(r.score, 5), "chunk_id": r.chunk.chunk_id} for r in retrieved[:6]]
            row["context_pages"] = sorted({r.chunk.source_url for r in context})
            if case.get("forbidden_hosts"):
                row["leaks"] = _forbidden_leaks(case, retrieved, context, citations)
            rows.append(row)
            if console:
                status = row.get("status", "retrieval")
                console.print(f"  {case['id']:<4} {mode:<14} {status:<22} hits@6={row['retrieval'].get('hits@6')}/{row['retrieval']['groups']}")
        summary = summarize_rows(rows, split, mode, retrieval_only, provider or settings.provider)
        summary["questions_sha256"] = file_sha256(split)
        summary["settings"] = {"candidate_k": settings.candidate_k, "context_max_chunks": settings.context_max_chunks,
                               'rerank_k': settings.rerank_k, 'context_policy': settings.context_policy,
                               "evidence_token_budget": settings.evidence_token_budget, "embedding_model": settings.embedding_model,
                               "reranker_model": settings.reranker_model, "rrf_k": settings.rrf_k,
                               "openai_model": settings.openai_model, "groq_model": settings.groq_model}
        from .generate import prompt_fingerprint
        summary['prompt'] = prompt_fingerprint()
        summary["results_file"] = f"{tag}.jsonl"
        (out / f"{tag}.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False, default=str) for r in rows) + "\n", encoding="utf-8")
        (out / f"{tag}_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        summaries.append(summary)
        if console:
            console.print(format_summary(summary))
    return {"runs": summaries}


def summarize_rows(rows: list[dict], split: str, mode: str, retrieval_only: bool, provider: str) -> dict:
    answerable = [r for r in rows if r["expected_behavior"] != "abstain" and r["retrieval"]["groups"]]
    groups = sum(r["retrieval"]["groups"] for r in answerable)
    s: dict = {"split": split, "mode": mode, "provider_mode": None if retrieval_only else provider,
               "cases": len(rows), "answerable_cases": len(answerable), "evidence_groups": groups}
    for k in KS:
        s[f"recall@{k}"] = f"{sum(r['retrieval'][f'hits@{k}'] for r in answerable)}/{groups}"
    s["context_recall"] = f"{sum(r['retrieval']['context_group_hits'] for r in answerable)}/{groups}"
    mp = [r for r in rows if r["category"] == "multi_page"]
    s["multi_page_labeled_pages_in_context"] = f"{sum(r['retrieval']['required_pages_in_context'] for r in mp)}/{sum(r['retrieval']['required_pages'] for r in mp)}"
    s["multi_page_complete"] = f"{sum(1 for r in mp if r['retrieval']['multi_page_complete'])}/{len(mp)}"
    rr = [1 / r["retrieval"]["first_hit_rank"] if r["retrieval"]["first_hit_rank"] else 0 for r in answerable]
    s["mrr@20"] = round(sum(rr) / len(rr), 3) if rr else None
    if "leaks" in (rows[0] if rows else {}):
        s["isolation_leaks"] = sum(len(r.get("leaks", [])) for r in rows)
    if retrieval_only:
        s["retrieval_ms_mean"] = round(statistics.mean(r["retrieval_ms"] for r in rows), 1) if rows else None
        return s
    scored = [r for r in rows if not r.get("infra_error")]
    s["infra_errors"] = sum(1 for r in rows if r.get("infra_error"))
    s["behavior_correct"] = f"{sum(1 for r in scored if r['behavior_ok'])}/{len(scored)}"
    by_cat: dict[str, list] = {}
    for r in scored:
        by_cat.setdefault(r["category"], []).append(r["behavior_ok"])
    s["behavior_by_category"] = {k: f"{sum(v)}/{len(v)}" for k, v in by_cat.items()}
    abst = [r for r in scored if r["expected_behavior"] == "abstain"]
    s["correct_abstention"] = f"{sum(1 for r in abst if r['status'] == 'insufficient_evidence')}/{len(abst)}"
    ans = [r for r in scored if r["expected_behavior"] == "answer"]
    s["inappropriate_abstention"] = f"{sum(1 for r in ans if r['inappropriate_abstention'])}/{len(ans)}"
    s["citation_validity"] = f"{sum(r['claims_valid'] for r in scored)}/{sum(r['claims_total'] for r in scored)}"
    cited = [r for r in scored if r["expected_behavior"] != "abstain" and r["retrieval"]["groups"]]
    s["gold_evidence_cited"] = f"{sum(r['gold_groups_cited'] for r in cited)}/{sum(r['retrieval']['groups'] for r in cited)}"
    lat = [r["latency_ms"] for r in rows if r.get("latency_ms")]
    s["latency_ms"] = {"mean": round(statistics.mean(lat)), "min": min(lat), "max": max(lat)} if lat else None
    s["providers"] = sorted({f"{r.get('provider')}/{r.get('model')}" for r in rows if r.get("provider")})
    s["fallbacks"] = sum(1 for r in rows if r.get("fallback_reason"))
    s["tokens"] = {"input": sum((r.get("usage") or {}).get("input_tokens", 0) for r in rows),
                   "output": sum((r.get("usage") or {}).get("output_tokens", 0) for r in rows)}
    s["cost_usd"] = round(sum((r.get("usage") or {}).get("cost_usd", 0.0) for r in rows), 6)
    s["unknown_cost_events"] = sum((r.get("usage") or {}).get("unknown_cost_events", 0) for r in rows)
    return s


def format_summary(s: dict) -> str:
    keys = [k for k in s if k not in ("settings", "results_file", "questions_sha256")]
    return "\n".join(f"  {k}: {s[k]}" for k in keys)
