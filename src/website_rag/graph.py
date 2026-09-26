"""Bounded LangGraph query workflow.

    validate ──(error)──────────────────────────────────────────┐
       │                                                        │
    retrieve ──(error)──────────────────────────────────────────┤
       │                                                        │
    context ──(no evidence)──> abstain ─────────────────────────┤
       │                                                        │
    generate ──(provider error, after bounded retry/fallback)───┤
       │                                                        │
    check_citations ────────────────────────────────────────> finalize

The graph fixes the transitions; nothing loops. A normal answer uses one query embedding
and one generation call (plus bounded transport retries / one visible fallback). The
graph adds explicit, inspectable state and routing; it does not by itself improve accuracy.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Callable, TypedDict

from langgraph.graph import END, START, StateGraph

from .chunk import count_tokens
from .config import ProviderMode, ProviderName, RetrievalMode, Settings
from .citations import validate_claims
from .errors import ErrorCode, RagError
from .generate import (
    PROMPT_VERSION,
    GenerationOutput,
    build_chat_model,
    build_messages,
    generate_once,
    prompt_fingerprint,
)
from .index import CorpusStore
from .providers import AttemptRecord, ProviderFailure, RetryPolicy, provider_chain, run_with_fallback
from .registry import SiteRegistry
from .retrieve import Retriever, select_context
from .schemas import AnswerResult, ProviderAttempt, RetrievedChunk, SiteRecord, UsageEvent
from .tracing import Tracer, new_id
from .usage import Pricing, UsageLedger, summarize

MAX_QUESTION_CHARS = 2000


@dataclass
class Budget:
    """Request/spend cap for batch runs (evaluation). None disables a cap."""

    max_requests: int | None = None
    max_cost_usd: float | None = None
    requests: int = 0
    cost_usd: float = 0.0

    def check(self) -> None:
        if self.max_requests is not None and self.requests >= self.max_requests:
            raise RagError(ErrorCode.BUDGET_EXCEEDED, f"Request cap of {self.max_requests} reached; stopping.",
                           hint="Raise RAG_EVAL_MAX_REQUESTS to continue.", stage="generate")
        if self.max_cost_usd is not None and self.cost_usd >= self.max_cost_usd:
            raise RagError(ErrorCode.BUDGET_EXCEEDED, f"Spend cap of ${self.max_cost_usd:.2f} reached; stopping.",
                           hint="Raise RAG_EVAL_MAX_COST_USD to continue.", stage="generate")


GeneratorFn = Callable[[ProviderName, list], GenerationOutput]


@dataclass
class QueryDeps:
    settings: Settings
    registry: SiteRegistry
    ledger: UsageLedger
    tracer: Tracer
    phase: str = "query"
    generator: GeneratorFn | None = None  # injected in tests; default calls the real provider
    retriever_factory: Callable[[SiteRecord, CorpusStore], Retriever] | None = None
    policy: RetryPolicy | None = None
    budget: Budget | None = None
    pricing: Pricing = field(default_factory=Pricing)
    base_urls: dict[str, str] = field(default_factory=dict)  # test servers only


class QueryState(TypedDict, total=False):
    question: str
    site_ref: Any
    mode: RetrievalMode
    provider_mode: ProviderMode
    site: SiteRecord
    store: CorpusStore
    retrieved: list[RetrievedChunk]
    context: list[RetrievedChunk]
    generation: GenerationOutput
    provider: str
    model: str
    fallback_reason: str | None
    attempts: list[ProviderAttempt]
    result: AnswerResult
    error: RagError | None
    started: float


def build_graph(deps: QueryDeps):  # noqa: ANN201
    s, tracer, pricing = deps.settings, deps.tracer, deps.pricing

    def record_usage(**kw: Any) -> None:
        ok = deps.ledger.record(UsageEvent(event_id=new_id("u-"), run_id=tracer.run_id, phase=deps.phase, **kw))
        if not ok:
            tracer.event("accounting_write_failed", operation=kw.get("operation"))

    # ---------------------------------------------------------------- nodes
    def validate(state: QueryState) -> QueryState:
        with tracer.span("validate") as span:
            try:
                question = (state.get("question") or "").strip()
                if not question:
                    raise RagError(ErrorCode.INVALID_INPUT, "The question is empty.", stage="validate")
                if len(question) > MAX_QUESTION_CHARS:
                    raise RagError(ErrorCode.INVALID_INPUT, f"The question exceeds {MAX_QUESTION_CHARS} characters.", stage="validate")
                site = deps.registry.require_queryable(state.get("site_ref"))
                store = CorpusStore(s, site.site_id, site.active_corpus_id)  # type: ignore[arg-type]
                store.verify_ready(site, s.embedding_model)
                tracer.bind(site_id=site.site_id, corpus_id=store.corpus_id)
                span.set(site_number=site.number, site_id=site.site_id, corpus_id=store.corpus_id,
                         retrieval_mode=state["mode"], provider_mode=state["provider_mode"])
                return {"question": question, "site": site, "store": store}
            except RagError as err:
                err.stage = err.stage or "validate"
                span.fail(str(err.code), err.message)
                return {"error": err}

    def retrieve(state: QueryState) -> QueryState:
        with tracer.span("retrieve", mode=state["mode"], k=s.candidate_k) as span:
            try:
                factory = deps.retriever_factory or (lambda site, store: Retriever(s, site, store))
                retriever = factory(state["site"], state["store"])
                t0 = time.monotonic()
                retrieved = retriever.search(state["question"], state["mode"])
                elapsed = int((time.monotonic() - t0) * 1000)
                emb_model = s.embedding_model
                if state["mode"] != "bm25":
                    record_usage(operation_id=new_id("op-"), attempt=1, site_id=state["site"].site_id,
                                 corpus_id=state["store"].corpus_id, provider="local", model=emb_model,
                                 operation="embedding", input_tokens=count_tokens(state["question"]),
                                 output_tokens=0, measurement="local_count", cost_usd=0.0,
                                 pricing_version=pricing.version, outcome="success", latency_ms=elapsed)
                span.set(timings_ms=retriever.timings, results=[{"chunk_id": r.chunk.chunk_id, "rank": r.rank, "score": round(r.score, 5),
                                   "components": r.component_ranks, "page": r.chunk.source_url}
                                  for r in retrieved[:10]], count=len(retrieved))
                return {"retrieved": retrieved}
            except RagError as err:
                span.fail(str(err.code), err.message)
                return {"error": err}
            except Exception as exc:  # noqa: BLE001 - vector store / model runtime failure
                err = RagError(ErrorCode.INTERNAL, f"Retrieval failed: {type(exc).__name__}: {exc}", stage="retrieve")
                span.fail(str(err.code), err.message)
                return {"error": err}

    def context(state: QueryState) -> QueryState:
        with tracer.span("context", max_chunks=s.context_max_chunks, token_budget=s.evidence_token_budget) as span:
            try:
                selected = select_context(state["retrieved"], s.context_max_chunks, s.evidence_token_budget,
                                          state["site"].site_id, state["store"].corpus_id, s.context_policy,
                                          state['store'].load_chunks() if s.context_policy == 'neighbors' else None)
            except RagError as err:
                span.fail(str(err.code), err.message)
                return {"error": err}
            span.set(chunk_ids=[r.chunk.chunk_id for r in selected],
                     pages=sorted({r.chunk.source_url for r in selected}),
                     tokens=sum(r.chunk.token_count for r in selected))
            known = {r.chunk.chunk_id for r in state['retrieved']}
            return {"context": selected, 'retrieved': state['retrieved'] + [r for r in selected if r.chunk.chunk_id not in known]}

    def abstain(state: QueryState) -> QueryState:
        tracer.event("abstain_without_generation", reason="no evidence retrieved")
        return {}

    def generate(state: QueryState) -> QueryState:
        with tracer.span("generate", prompt=prompt_fingerprint()) as span:
            attempts: list[ProviderAttempt] = []
            op_id = new_id("op-")
            try:
                if deps.budget:
                    deps.budget.check()
                chain, notes = provider_chain(state["provider_mode"], s)
                for note in notes:
                    tracer.event("provider_skipped", note=note)
                messages = build_messages(state["question"], state["site"], [r.chunk for r in state["context"]])
                span.set(chain=chain, input_chars=sum(len(str(m.content)) for m in messages))

                def call(provider: ProviderName) -> GenerationOutput:
                    if deps.budget:
                        deps.budget.check()
                        deps.budget.requests += 1
                    if deps.generator:
                        return deps.generator(provider, messages)
                    model = build_chat_model(provider, s, base_url=deps.base_urls.get(provider))
                    return generate_once(model, messages)

                def on_attempt(rec: AttemptRecord) -> None:
                    usage = (rec.result.usage if rec.result else (rec.error.usage if rec.error else None)) or {}
                    cost = pricing.cost(rec.provider, rec.model, usage.get("input_tokens"),
                                        usage.get("output_tokens"), usage.get("cached_input_tokens"))
                    if deps.budget and cost:
                        deps.budget.cost_usd += cost
                    record_usage(operation_id=op_id, attempt=len(attempts) + 1, site_id=state["site"].site_id,
                                 corpus_id=state["store"].corpus_id, provider=rec.provider, model=rec.model,
                                 operation="generation", input_tokens=usage.get("input_tokens"),
                                 cached_input_tokens=usage.get("cached_input_tokens"),
                                 output_tokens=usage.get("output_tokens"), reasoning_tokens=usage.get("reasoning_tokens"),
                                 measurement="provider_reported" if usage.get("input_tokens") is not None else "unknown",
                                 cost_usd=cost, pricing_version=pricing.version, outcome=rec.outcome,
                                 error_code=str(rec.error.code) if rec.error else None, latency_ms=rec.latency_ms,
                                 request_id=(rec.error.request_id if rec.error else rec.result.response_id))
                    attempts.append(ProviderAttempt(
                        provider=rec.provider, model=rec.model, attempt=rec.attempt, outcome=rec.outcome,
                        error_code=str(rec.error.code) if rec.error else None,
                        message=rec.error.message if rec.error else None, latency_ms=rec.latency_ms,
                        request_id=(rec.error.request_id if rec.error else rec.result.response_id),
                        retry_after_s=rec.error.retry_after_s if rec.error else None,
                    ))
                    tracer.event("provider_attempt", provider=rec.provider, model=rec.model, attempt=rec.attempt,
                                 outcome=rec.outcome, error_code=str(rec.error.code) if rec.error else None,
                                 latency_ms=rec.latency_ms, usage=usage, cost_usd=cost)

                policy = deps.policy or RetryPolicy.from_settings(s)
                outcome = run_with_fallback(chain, s, call, policy, on_attempt)
                if outcome.fallback_reason:
                    tracer.event("provider_fallback", reason=outcome.fallback_reason)
                span.set(provider=outcome.provider, model=outcome.model, attempts=len(attempts),
                         fallback_reason=outcome.fallback_reason, status=outcome.result.draft.status)
                return {"generation": outcome.result, "provider": outcome.provider,
                        "model": outcome.result.model or outcome.model,
                        "fallback_reason": outcome.fallback_reason, "attempts": attempts}
            except ProviderFailure as failure:
                err = failure.to_rag_error()
                span.fail(str(err.code), err.message)
                return {"error": err, "attempts": attempts,
                        "fallback_reason": getattr(failure, "fallback_reason", None)}
            except RagError as err:
                span.fail(str(err.code), err.message)
                return {"error": err, "attempts": attempts}

    def check_citations(state: QueryState) -> QueryState:
        with tracer.span("check_citations") as span:
            draft = state["generation"].draft
            ctx = {r.chunk.chunk_id: r.chunk for r in state["context"]}
            claims, citations, rejected = validate_claims(draft, ctx)
            span.set(draft_status=draft.status, claims=len(draft.claims), valid=len(claims), rejected=len(rejected),
                     rejected_reasons=[r.reasons for r in rejected])
            result, err = _partial_result(state, draft, claims, citations, rejected)
            return {"result": result, "error": err} if err else {"result": result}

    def finalize(state: QueryState) -> QueryState:
        with tracer.span("finalize") as span:
            result = state.get("result") or _empty_result(state)
            err = state.get("error")
            if err is not None:
                result.status = "error"
                result.error = err.to_dict()
                result.answer = ""
            result.provider_attempts = state.get("attempts", [])
            result.fallback_reason = state.get("fallback_reason")
            run_events = [e for e in deps.ledger.session_events if e.run_id == tracer.run_id]
            # LLM tokens and local-model tokens are reported separately; they are not comparable.
            result.usage = summarize([e for e in run_events if e.provider != "local"])
            result.usage["local_embedding_tokens"] = sum(
                e.input_tokens or 0 for e in run_events if e.provider == "local")
            result.usage["accounting_write_failures"] = deps.ledger.write_failures
            result.latency_ms = int((time.monotonic() - state["started"]) * 1000)
            span.set(status=result.status, error_code=(err.code if err else None),
                     citations=len(result.citations), cost_usd=result.usage["cost_usd"],
                     unknown_cost_events=result.usage["unknown_cost_events"])
            return {"result": result}

    def _empty_result(state: QueryState) -> AnswerResult:
        site: SiteRecord | None = state.get("site")
        store = state.get("store")
        result = AnswerResult(
            run_id=tracer.run_id, question=state.get("question", ""), site_id=site.site_id if site else None,
            site_number=site.number if site else None, site_name=site.display_name if site else None,
            corpus_id=store.corpus_id if store else None, status="insufficient_evidence",
            retrieval_mode=state.get("mode"), retrieved=state.get("retrieved", []),
            context_chunk_ids=[r.chunk.chunk_id for r in state.get("context", [])],
            prompt_version=PROMPT_VERSION,
        )
        if site and not state.get("error"):
            result.missing_information = "No passages were retrieved for this question."
            result.answer = insufficient_message(site)
        return result

    def _partial_result(state, draft, claims, citations, rejected):  # noqa: ANN001, ANN202
        result = _empty_result(state)
        err: RagError | None = None
        site = state["site"]
        result.provider, result.model = state.get("provider"), state.get("model")
        result.claims, result.citations, result.rejected_claims = claims, citations, rejected
        # Model-written auxiliary prose is not evidence and must never reach normal output.
        result.missing_information = ""
        result.premise_issue = ""
        supported_answer = "\n\n".join(
            c.text + " " + "".join(f"[{m}]" for m in c.citations) for c in claims
        )
        if draft.status == "insufficient_evidence":
            result.status = "insufficient_evidence"
            result.answer = insufficient_message(site)
            result.claims, result.citations = [], []
        elif claims and not rejected:
            result.status = draft.status
            result.answer = supported_answer
            if draft.status == "partially_answered":
                result.missing_information = "The retrieved evidence supports only part of the requested answer."
        elif claims and rejected:
            # Some claims failed validation: withhold the free-text answer, show only verified claims.
            result.status = "partially_answered"
            result.answer = supported_answer
            result.missing_information = (
                f"{len(rejected)} generated statement(s) were withheld because their citations failed validation.")
        else:
            err = RagError(
                ErrorCode.CITATION_INVALID,
                "The generated answer could not be verified against the retrieved evidence, so it was withheld.",
                hint="Inspect `rag trace <run-id>` for the rejected citations.",
                stage="check_citations",
            )
            result.status = "error"
            result.answer = ""
        return result, err

    # ---------------------------------------------------------------- wiring
    def route_error(next_node: str) -> Callable[[QueryState], str]:
        return lambda state: "finalize" if state.get("error") else next_node

    graph = StateGraph(QueryState)
    for name, fn in [("validate", validate), ("retrieve", retrieve), ("context", context), ("abstain", abstain),
                     ("generate", generate), ("check_citations", check_citations), ("finalize", finalize)]:
        graph.add_node(name, fn)
    graph.add_edge(START, "validate")
    graph.add_conditional_edges("validate", route_error("retrieve"), ["retrieve", "finalize"])
    graph.add_conditional_edges("retrieve", route_error("context"), ["context", "finalize"])
    graph.add_conditional_edges(
        "context",
        lambda st: "finalize" if st.get("error") else ("generate" if st.get("context") else "abstain"),
        ["generate", "abstain", "finalize"],
    )
    graph.add_edge("abstain", "finalize")
    graph.add_conditional_edges("generate", route_error("check_citations"), ["check_citations", "finalize"])
    graph.add_edge("check_citations", "finalize")
    graph.add_edge("finalize", END)
    return graph.compile()


def insufficient_message(site: SiteRecord) -> str:
    return (f"The indexed pages for {site.display_name} do not contain enough information to answer this question.")


def answer_question(
    deps: QueryDeps,
    question: str,
    site_ref: Any = None,
    mode: RetrievalMode | None = None,
    provider_mode: ProviderMode | None = None,
) -> AnswerResult:
    app = build_graph(deps)
    state: QueryState = {
        "question": question, "site_ref": site_ref, "mode": mode or deps.settings.retrieval_mode,
        "provider_mode": provider_mode or deps.settings.provider, "started": time.monotonic(),
    }
    with deps.tracer.span("query", question_chars=len(question or "")):
        final = app.invoke(state)
    result: AnswerResult = final["result"]
    save_run(deps.settings, result)
    return result


def save_run(settings: Settings, result: AnswerResult) -> None:
    try:
        settings.runs_dir.mkdir(parents=True, exist_ok=True)
        (settings.runs_dir / f"{result.run_id}.json").write_text(result.model_dump_json(indent=2), encoding="utf-8")
    except OSError:
        result.usage["run_record_write_failed"] = True


def load_run(settings: Settings, run_id: str) -> AnswerResult:
    path = settings.runs_dir / f"{run_id}.json"
    if not path.exists():
        raise RagError(ErrorCode.INVALID_INPUT, f"No saved run '{run_id}'.", hint="Run IDs are shown after each answer.")
    return AnswerResult.model_validate(json.loads(path.read_text(encoding="utf-8")))
