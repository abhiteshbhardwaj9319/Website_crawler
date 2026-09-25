"""Local structured tracing.

Each run (one ask, one ingestion, one evaluation) writes JSONL events to
`data/traces/<run_id>.jsonl`. Events carry trace/span/parent IDs, the selected site and
corpus, stage, timing, outcome, and error code. Credentials are redacted from every string,
and full prompts/page bodies are never written by default.

A failure to write a trace never masks the original exception: write errors are counted
and surfaced through `Tracer.write_failures`.
"""

from __future__ import annotations

import json
import re
import time
import traceback
import uuid
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterator

_SECRET_PATTERNS = [
    re.compile(r"sk-[A-Za-z0-9_\-]{8,}"),
    re.compile(r"gsk_[A-Za-z0-9]{8,}"),
    re.compile(r"(?i)bearer\s+[A-Za-z0-9._\-]{8,}"),
    re.compile(r"(?i)(api[_-]?key|authorization)(['\"]?\s*[:=]\s*['\"]?)[^\s'\",}]{6,}"),
]


def redact(text: str) -> str:
    for pattern in _SECRET_PATTERNS:
        if pattern.groups >= 2:
            text = pattern.sub(lambda m: f"{m.group(1)}{m.group(2)}[REDACTED]", text)
        else:
            text = pattern.sub("[REDACTED]", text)
    return text


def sanitize(value: Any) -> Any:
    if isinstance(value, str):
        return redact(value)
    if isinstance(value, dict):
        return {str(k): sanitize(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [sanitize(v) for v in value]
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    return redact(str(value))


def new_id(prefix: str = "") -> str:
    return f"{prefix}{uuid.uuid4().hex[:16]}"


class Span:
    def __init__(self, tracer: "Tracer", name: str, parent_id: str | None, attrs: dict) -> None:
        self.tracer = tracer
        self.name = name
        self.span_id = new_id("s-")
        self.parent_id = parent_id
        self.attrs = dict(attrs)
        self.status = "ok"
        self.error_code: str | None = None
        self.error_message: str | None = None

    def set(self, **attrs: Any) -> None:
        self.attrs.update(attrs)

    def fail(self, code: str, message: str) -> None:
        self.status = "error"
        self.error_code = code
        self.error_message = message


class Tracer:
    def __init__(
        self,
        traces_dir: Path | None,
        run_id: str | None = None,
        kind: str = "query",
        debug: bool = False,
        **context: Any,
    ) -> None:
        self.run_id = run_id or new_id("r-")
        self.trace_id = self.run_id
        self.kind = kind
        self.debug = debug
        self.context = context  # e.g. site_id / corpus_id
        self.path = traces_dir / f"{self.run_id}.jsonl" if traces_dir else None
        self._stack: list[Span] = []
        self.write_failures = 0
        self.events: list[dict] = []  # in-memory copy (used by tests and CLI)

    def bind(self, **context: Any) -> None:
        self.context.update(context)

    def _write(self, event: dict) -> None:
        event = sanitize(event)
        self.events.append(event)
        if self.path is None:
            return
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(event, ensure_ascii=False, default=str) + "\n")
        except OSError:
            self.write_failures += 1

    def event(self, name: str, **attrs: Any) -> None:
        parent = self._stack[-1].span_id if self._stack else None
        self._write(
            {
                "type": "event",
                "ts": datetime.now(UTC).isoformat(),
                "trace_id": self.trace_id,
                "run_id": self.run_id,
                "kind": self.kind,
                "span_id": new_id("e-"),
                "parent_id": parent,
                "stage": name,
                **self.context,
                "attrs": attrs,
            }
        )

    @contextmanager
    def span(self, name: str, **attrs: Any) -> Iterator[Span]:
        parent = self._stack[-1].span_id if self._stack else None
        span = Span(self, name, parent, attrs)
        self._stack.append(span)
        started = time.perf_counter()
        start_ts = datetime.now(UTC).isoformat()
        exc_info: BaseException | None = None
        try:
            yield span
        except BaseException as exc:  # recorded, then re-raised unchanged
            exc_info = exc
            if span.status == "ok":
                code = str(getattr(exc, "code", type(exc).__name__))
                span.fail(code, str(getattr(exc, "message", exc)))
            raise
        finally:
            self._stack.pop()
            event = {
                "type": "span",
                "ts": start_ts,
                "trace_id": self.trace_id,
                "run_id": self.run_id,
                "kind": self.kind,
                "span_id": span.span_id,
                "parent_id": span.parent_id,
                "stage": name,
                **self.context,
                "duration_ms": int((time.perf_counter() - started) * 1000),
                "status": span.status,
                "error_code": span.error_code,
                "error_message": span.error_message,
                "attrs": span.attrs,
            }
            if exc_info is not None and self.debug:
                event["stack"] = "".join(traceback.format_exception(exc_info))[-4000:]
            try:
                self._write(event)
            except Exception:  # never mask the original exception
                self.write_failures += 1


def load_trace(traces_dir: Path, run_id: str) -> list[dict]:
    path = traces_dir / f"{run_id}.jsonl"
    if not path.exists():
        return []
    events = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                events.append({"type": "corrupt_line", "raw": line[:200]})
    return events
