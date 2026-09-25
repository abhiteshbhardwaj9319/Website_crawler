"""Provider selection, error classification, bounded retries, and auto fallback.

Design:
- SDK-level retries are disabled (max_retries=0); this module owns every retry so attempts
  are counted exactly once and cannot multiply (SDK x application).
- Errors are classified from HTTP status + provider error `code`/`type`, never from status
  alone: a 429 can be a temporary rate limit, a confirmed credit exhaustion, a spend limit,
  or an ambiguous quota error, and each needs a different action.
- Permanent failures (billing, auth, bad request) are never retried.
- Auto mode tries OpenAI first, then falls back once to Groq for availability/quota/
  transient failures. There is no path back to OpenAI (no ping-pong).
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from typing import Any, Callable, TypeVar

from .config import ProviderMode, ProviderName, Settings
from .errors import ErrorCode, RagError

T = TypeVar("T")

PROVIDER_LABELS = {"openai": "OpenAI", "groq": "Groq"}

RETRYABLE = {ErrorCode.RATE_LIMITED, ErrorCode.SERVER_ERROR, ErrorCode.TIMEOUT, ErrorCode.CONNECTION}
FALLBACK_ELIGIBLE = RETRYABLE | {
    ErrorCode.CREDITS_EXHAUSTED,
    ErrorCode.QUOTA_EXCEEDED,
    ErrorCode.SPEND_LIMIT,
    ErrorCode.MODEL_UNAVAILABLE,
}


@dataclass
class ProviderFailure(Exception):
    code: ErrorCode
    provider: str
    message: str
    hint: str | None = None
    status: int | None = None
    provider_code: str | None = None
    retry_after_s: float | None = None
    request_id: str | None = None
    usage: dict | None = None  # token usage when the call completed but failed afterwards

    def __post_init__(self) -> None:
        super().__init__(self.message)

    @property
    def retryable(self) -> bool:
        return self.code in RETRYABLE

    @property
    def fallback_eligible(self) -> bool:
        return self.code in FALLBACK_ELIGIBLE

    def to_rag_error(self, stage: str = "generate") -> RagError:
        return RagError(
            self.code,
            self.message,
            hint=self.hint,
            stage=stage,
            details={
                "provider": self.provider,
                "status": self.status,
                "provider_code": self.provider_code,
                "request_id": self.request_id,
            },
        )


def _error_fields(exc: BaseException) -> tuple[str | None, str | None, str]:
    """(code, type, message) from an SDK exception body in either OpenAI or Groq shape."""
    code = getattr(exc, "code", None)
    etype = getattr(exc, "type", None)
    message = ""
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        inner = body.get("error", body)
        if isinstance(inner, dict):
            code = code or inner.get("code")
            etype = etype or inner.get("type")
            message = str(inner.get("message") or "")
    return (str(code) if code else None, str(etype) if etype else None, message or str(exc))


def _retry_after(exc: BaseException) -> float | None:
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    if not headers:
        return None
    value = headers.get("retry-after")
    if value is None:
        ms = headers.get("retry-after-ms")
        return float(ms) / 1000 if ms else None
    try:
        return max(0.0, float(value))
    except ValueError:
        return None


def _request_id(exc: BaseException) -> str | None:
    rid = getattr(exc, "request_id", None)
    if rid:
        return str(rid)
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    return headers.get("x-request-id") if headers else None


def classify(exc: BaseException, provider: str) -> ProviderFailure:
    """Map any provider/SDK exception to a ProviderFailure with an actionable message."""
    if isinstance(exc, ProviderFailure):
        return exc
    label = PROVIDER_LABELS.get(provider, provider)
    name = type(exc).__name__
    status = getattr(exc, "status_code", None)
    code, etype, detail = _error_fields(exc)
    retry_after = _retry_after(exc)
    request_id = _request_id(exc)

    def fail(ecode: ErrorCode, message: str, hint: str | None = None) -> ProviderFailure:
        return ProviderFailure(
            ecode, provider, message, hint, status, code or etype, retry_after, request_id,
            getattr(exc, "usage", None),
        )

    # Transport failures (no HTTP status). Timeout is a subclass of connection error.
    if name == "APITimeoutError" or isinstance(exc, TimeoutError):
        return fail(ErrorCode.TIMEOUT, f"{label} request timed out.", "Retry, or raise RAG_REQUEST_TIMEOUT_S.")
    if name == "APIConnectionError" or isinstance(exc, ConnectionError):
        return fail(ErrorCode.CONNECTION, f"Could not connect to {label}.", "Check your network connection.")

    if status is None:
        # Parsing / schema problems raised by LangChain or pydantic after a successful call.
        if name in {"OutputParserException", "ValidationError", "JSONDecodeError", "MalformedResponse"}:
            return fail(ErrorCode.MALFORMED_RESPONSE, f"{label} returned a response that did not match the answer schema.")
        return fail(ErrorCode.INTERNAL, f"Unexpected {label} client error ({name}).")

    lowered = f"{code or ''} {etype or ''} {detail}".lower()

    if status == 401:
        return fail(
            ErrorCode.AUTH_INVALID,
            f"{label} rejected the API key (invalid, revoked, or wrong organization).",
            f"Update {provider.upper()}_API_KEY in your local .env.",
        )
    if status == 403:
        return fail(
            ErrorCode.PERMISSION_DENIED,
            f"{label} denied access to this model or region.",
            "Check project permissions / model access in the provider console.",
        )
    if status == 404 or code in {"model_not_found", "model_decommissioned"}:
        return fail(
            ErrorCode.MODEL_UNAVAILABLE,
            f"{label} model is unavailable to this key.",
            "Check the configured model ID (RAG_OPENAI_MODEL / RAG_GROQ_MODEL).",
        )
    if status == 413 or code in {"context_length_exceeded", "request_too_large"}:
        return fail(
            ErrorCode.REQUEST_TOO_LARGE,
            f"The request is too large for {label} (context or per-request token limit).",
            "Lower RAG_EVIDENCE_TOKEN_BUDGET or RAG_CONTEXT_MAX_CHUNKS.",
        )
    if status == 429:
        if code == "credit_balance_exhausted":
            return fail(
                ErrorCode.CREDITS_EXHAUSTED,
                "OpenAI API credits are exhausted. Please recharge your API billing account, "
                "or use the configured Groq provider." if provider == "openai"
                else f"{label} credits are exhausted.",
                "Use --provider groq, or --provider auto to fall back automatically.",
            )
        if code in {"organization_spend_limit_exceeded", "project_spend_limit_exceeded",
                    "organization_usage_limit_exceeded"}:
            return fail(
                ErrorCode.SPEND_LIMIT,
                f"{label} spend/usage limit reached ({code}).",
                "Raise the limit in the provider's billing settings or wait for the monthly reset.",
            )
        if code == "insufficient_quota" or "exceeded your current quota" in lowered:
            return fail(
                ErrorCode.QUOTA_EXCEEDED,
                f"{label} reports the quota for this key is exceeded.",
                "Check API billing, remaining credits, and usage limits in the provider console.",
            )
        return fail(
            ErrorCode.RATE_LIMITED,
            f"{label} rate limit reached" + (" (daily token quota)." if "per day" in lowered else "."),
            "Wait and retry; free tiers have low per-minute/day limits.",
        )
    if status in {500, 502, 503, 504} or status >= 500:
        return fail(ErrorCode.SERVER_ERROR, f"{label} server error or overload (HTTP {status}).", "Retry shortly.")
    if status == 400 and (code == "json_validate_failed" or "json" in lowered and "schema" in lowered):
        return fail(ErrorCode.MALFORMED_RESPONSE, f"{label} could not produce output matching the answer schema.")
    return fail(ErrorCode.BAD_REQUEST, f"{label} rejected the request (HTTP {status}: {code or 'bad request'}).")


@dataclass
class AttemptRecord:
    provider: str
    model: str
    attempt: int
    outcome: str
    latency_ms: int
    error: ProviderFailure | None = None
    result: Any = None


@dataclass
class RetryPolicy:
    max_attempts: int
    budget_s: float
    max_retry_after_s: float
    base_backoff_s: float = 1.0
    sleep: Callable[[float], None] = time.sleep
    clock: Callable[[], float] = time.monotonic

    @classmethod
    def from_settings(cls, s: Settings, **overrides: Any) -> "RetryPolicy":
        return cls(s.max_attempts_per_provider, s.retry_budget_s, s.max_retry_after_s, **overrides)

    def backoff(self, attempt: int) -> float:
        return self.base_backoff_s * (2 ** (attempt - 1)) * (0.8 + 0.4 * random.random())


def call_with_retries(
    fn: Callable[[], T],
    provider: str,
    model: str,
    policy: RetryPolicy,
    on_attempt: Callable[[AttemptRecord], None],
) -> T:
    """Invoke fn with bounded retries for transient failures only."""
    deadline = policy.clock() + policy.budget_s
    attempt = 0
    while True:
        attempt += 1
        started = policy.clock()
        try:
            result = fn()
        except Exception as exc:  # classified below; non-provider bugs surface as INTERNAL
            failure = classify(exc, provider)
            latency = int((policy.clock() - started) * 1000)
            on_attempt(AttemptRecord(provider, model, attempt, "error", latency, failure))
            if not failure.retryable or attempt >= policy.max_attempts:
                raise failure from exc
            if failure.retry_after_s is not None and failure.retry_after_s > policy.max_retry_after_s:
                failure.message += f" Provider asked to wait {failure.retry_after_s:.0f}s, above the retry limit."
                raise failure from exc
            wait = failure.retry_after_s if failure.retry_after_s is not None else policy.backoff(attempt)
            if policy.clock() + wait > deadline:
                raise failure from exc
            policy.sleep(wait)
            continue
        latency = int((policy.clock() - started) * 1000)
        on_attempt(AttemptRecord(provider, model, attempt, "success", latency, result=result))
        return result


def provider_chain(mode: ProviderMode, settings: Settings) -> tuple[list[ProviderName], list[str]]:
    """Ordered providers to try, plus notes explaining any skipped provider."""
    notes: list[str] = []
    if mode in ("openai", "groq"):
        if settings.key_for(mode) is None:
            raise RagError(
                ErrorCode.PROVIDER_NOT_CONFIGURED,
                f"{PROVIDER_LABELS[mode]} is selected but {mode.upper()}_API_KEY is not set.",
                hint="Add it to your local .env (see .env.example), or use --provider auto.",
                stage="provider_selection",
            )
        return [mode], notes
    chain: list[ProviderName] = []
    for p in ("openai", "groq"):
        if settings.key_for(p) is None:
            notes.append(f"{PROVIDER_LABELS[p]} not configured ({p.upper()}_API_KEY missing)")
        else:
            chain.append(p)
    if not chain:
        raise RagError(
            ErrorCode.PROVIDER_NOT_CONFIGURED,
            "No generation provider is configured.",
            hint="Set OPENAI_API_KEY and/or GROQ_API_KEY in your local .env (see .env.example).",
            stage="provider_selection",
        )
    return chain, notes


@dataclass
class ChainOutcome:
    result: Any
    provider: str
    model: str
    attempts: list[AttemptRecord] = field(default_factory=list)
    fallback_reason: str | None = None


def run_with_fallback(
    chain: list[ProviderName],
    settings: Settings,
    call: Callable[[ProviderName], Any],
    policy: RetryPolicy,
    on_attempt: Callable[[AttemptRecord], None] | None = None,
) -> ChainOutcome:
    """Try providers in order; fall back at most once per provider, never back again."""
    attempts: list[AttemptRecord] = []
    fallback_reason: str | None = None

    def record(rec: AttemptRecord) -> None:
        attempts.append(rec)
        if on_attempt:
            on_attempt(rec)

    last_failure: ProviderFailure | None = None
    for index, provider in enumerate(chain):
        model = settings.model_for(provider)
        try:
            result = call_with_retries(lambda: call(provider), provider, model, policy, record)
            return ChainOutcome(result, provider, model, attempts, fallback_reason)
        except ProviderFailure as failure:
            last_failure = failure
            has_next = index + 1 < len(chain)
            if has_next and failure.fallback_eligible:
                nxt = PROVIDER_LABELS[chain[index + 1]]
                fallback_reason = f"{PROVIDER_LABELS[provider]} failed ({failure.code}); falling back to {nxt}."
                continue
            failure.attempts = attempts  # type: ignore[attr-defined]
            failure.fallback_reason = fallback_reason  # type: ignore[attr-defined]
            raise
    assert last_failure is not None
    raise last_failure
