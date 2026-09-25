"""Grounded answer generation through LangChain chat-model integrations.

One structured-output call per provider attempt: `with_structured_output(AnswerDraft,
method="json_schema", strict=True, include_raw=True)`. Both configured models support
strict JSON-schema output (OpenAI gpt-4.1-mini; Groq openai/gpt-oss-120b). `include_raw`
keeps the raw message so provider-reported token usage is captured even when parsing fails.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import ValidationError

from .config import ProviderName, Settings
from .schemas import AnswerDraft, ChunkRecord, SiteRecord

PROMPT_VERSION = "answer_v1"
PROMPT_PATH = Path(__file__).parent / "prompts" / f"{PROMPT_VERSION}.md"


def system_prompt() -> str:
    return PROMPT_PATH.read_text(encoding="utf-8")


def prompt_fingerprint() -> str:
    return f"{PROMPT_VERSION}:{hashlib.sha256(system_prompt().encode()).hexdigest()[:12]}"


def _escape(text: str) -> str:
    return text.replace("</chunk", "</ chunk").replace("<chunk", "< chunk")


def build_messages(question: str, site: SiteRecord, context: list[ChunkRecord]) -> list:
    evidence = "\n".join(
        f'<chunk id="{c.chunk_id}" page_title="{_escape(c.title)}" section="{_escape(c.section_label)}">\n'
        f"{_escape(c.text)}\n</chunk>"
        for c in context
    )
    user = (
        f"Website: {site.display_name} ({site.allowed_host}{site.allowed_path_prefix}); "
        f"{site.accepted_pages} pages indexed. Only the evidence chunks below were retrieved for this question.\n\n"
        f"<evidence>\n{evidence}\n</evidence>\n\n"
        f"<question>\n{question}\n</question>"
    )
    return [SystemMessage(content=system_prompt()), HumanMessage(content=user)]


class MalformedResponse(Exception):
    def __init__(self, message: str, usage: dict | None = None) -> None:
        super().__init__(message)
        self.usage = usage


@dataclass
class GenerationOutput:
    draft: AnswerDraft
    usage: dict
    model: str
    response_id: str | None


def build_chat_model(provider: ProviderName, settings: Settings, base_url: str | None = None) -> BaseChatModel:
    key = settings.key_for(provider)
    common = dict(
        model=settings.model_for(provider),
        api_key=key,
        temperature=settings.temperature,
        max_tokens=settings.max_output_tokens,
        timeout=settings.request_timeout_s,
        max_retries=0,  # the application owns retries (providers.call_with_retries)
    )
    if provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(**common, base_url=base_url)
    from langchain_groq import ChatGroq

    return ChatGroq(**common, base_url=base_url, reasoning_effort=settings.groq_reasoning_effort)


def extract_usage(raw) -> dict:  # noqa: ANN001
    meta = getattr(raw, "usage_metadata", None) or {}
    details_in = meta.get("input_token_details") or {}
    details_out = meta.get("output_token_details") or {}
    return {
        "input_tokens": meta.get("input_tokens"),
        "output_tokens": meta.get("output_tokens"),
        "cached_input_tokens": details_in.get("cache_read"),
        "reasoning_tokens": details_out.get("reasoning"),
    }


def generate_once(model: BaseChatModel, messages: list) -> GenerationOutput:
    structured = model.with_structured_output(AnswerDraft, method="json_schema", strict=True, include_raw=True)
    try:
        out = structured.invoke(messages)
    except (ValidationError, ValueError) as exc:
        if type(exc).__module__.startswith(("openai", "groq", "httpx")):
            raise
        # The SDK's own parser rejected the content; the raw message (and its usage) is not returned.
        raise MalformedResponse(f"Response did not match the answer schema ({type(exc).__name__}).", None) from exc
    raw = out.get("raw")
    usage = extract_usage(raw)
    parsed = out.get("parsed")
    if parsed is None:
        err = out.get("parsing_error")
        raise MalformedResponse(f"Response did not match the answer schema ({type(err).__name__ if err else 'empty'}).", usage)
    meta = getattr(raw, "response_metadata", {}) or {}
    return GenerationOutput(
        draft=parsed, usage=usage, model=str(meta.get("model_name") or meta.get("model") or ""),
        response_id=getattr(raw, "id", None) or meta.get("id"),
    )
