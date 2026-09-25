"""Typed application errors.

Every expected failure carries a stable code, the stage where it happened, a user-facing
message, and a next step. The CLI renders these without a traceback; unexpected
exceptions are still reported with a sanitized message and the trace ID.
"""

from __future__ import annotations

from enum import StrEnum


class ErrorCode(StrEnum):
    # Configuration / user input
    CONFIG_INVALID = "config_invalid"
    INVALID_INPUT = "invalid_input"
    SITE_NOT_FOUND = "site_not_found"
    SITE_NOT_READY = "site_not_ready"
    INDEX_INCOMPATIBLE = "index_incompatible"
    INDEX_INCOMPLETE = "index_incomplete"
    # Crawling
    URL_REJECTED = "url_rejected"
    ROBOTS_DISALLOWED = "robots_disallowed"
    FETCH_FAILED = "fetch_failed"
    LOW_CONTENT = "low_content"
    # Providers
    PROVIDER_NOT_CONFIGURED = "provider_not_configured"
    AUTH_INVALID = "auth_invalid"
    PERMISSION_DENIED = "permission_denied"
    MODEL_UNAVAILABLE = "model_unavailable"
    CREDITS_EXHAUSTED = "credits_exhausted"
    QUOTA_EXCEEDED = "quota_exceeded"
    SPEND_LIMIT = "spend_limit"
    RATE_LIMITED = "rate_limited"
    SERVER_ERROR = "server_error"
    TIMEOUT = "timeout"
    CONNECTION = "connection_error"
    REQUEST_TOO_LARGE = "request_too_large"
    BAD_REQUEST = "bad_request"
    MALFORMED_RESPONSE = "malformed_response"
    BUDGET_EXCEEDED = "budget_exceeded"
    # Answer validation
    CITATION_INVALID = "citation_invalid"
    # Catch-all
    INTERNAL = "internal_error"


class RagError(Exception):
    """An expected, explainable failure."""

    def __init__(
        self,
        code: ErrorCode,
        message: str,
        *,
        hint: str | None = None,
        stage: str | None = None,
        details: dict | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.hint = hint
        self.stage = stage
        self.details = details or {}

    def to_dict(self) -> dict:
        return {
            "code": str(self.code),
            "message": self.message,
            "hint": self.hint,
            "stage": self.stage,
            "details": self.details,
        }
