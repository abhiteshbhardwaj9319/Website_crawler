"""Usage accounting: dated prices, per-call cost arithmetic, and a JSONL ledger.

Each billable (or locally computed) call attempt is one UsageEvent. Retries and fallback
attempts are separate events that share an operation_id, so they are counted once each
and never double-counted by graph callbacks. Unknown usage yields cost None, not zero.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from .config import CONFIG_DIR
from .schemas import UsageEvent

PRICING_FILE = CONFIG_DIR / "pricing.json"
_SNAPSHOT_SUFFIX = re.compile(r"-\d{4}-\d{2}-\d{2}$")


@dataclass(frozen=True)
class ModelRate:
    key: str
    input: float
    cached_input: float | None
    output: float
    source: str


class Pricing:
    def __init__(self, path: Path = PRICING_FILE) -> None:
        raw = json.loads(path.read_text(encoding="utf-8"))
        self.version: str = raw["pricing_version"]
        self.currency: str = raw["currency"]
        self.rates = {
            key: ModelRate(key, v["input"], v.get("cached_input"), v["output"], v["source"])
            for key, v in raw["models"].items()
        }

    def rate_for(self, provider: str, model: str) -> ModelRate | None:
        for candidate in (model, _SNAPSHOT_SUFFIX.sub("", model)):
            rate = self.rates.get(f"{provider}:{candidate}")
            if rate:
                return rate
        return None

    def cost(
        self,
        provider: str,
        model: str,
        input_tokens: int | None,
        output_tokens: int | None,
        cached_input_tokens: int | None = 0,
    ) -> float | None:
        """USD cost. Cached tokens are a subset of input tokens (OpenAI/Groq schema)."""
        rate = self.rate_for(provider, model)
        if rate is None or input_tokens is None or output_tokens is None:
            return None
        cached = min(cached_input_tokens or 0, input_tokens)
        cached_rate = rate.cached_input if rate.cached_input is not None else rate.input
        return (
            (input_tokens - cached) * rate.input + cached * cached_rate + output_tokens * rate.output
        ) / 1_000_000


class UsageLedger:
    def __init__(self, path: Path | None) -> None:
        self.path = path
        self.session_events: list[UsageEvent] = []
        self.write_failures = 0

    def record(self, event: UsageEvent) -> bool:
        """Append an event. Returns False (and counts the failure) if persistence fails."""
        self.session_events.append(event)
        if self.path is None:
            return True
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(event.model_dump_json() + "\n")
            return True
        except OSError:
            self.write_failures += 1
            return False

    def read_all(self) -> list[UsageEvent]:
        if self.path is None or not self.path.exists():
            return []
        events = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                events.append(UsageEvent.model_validate_json(line))
        return events


def summarize(events: list[UsageEvent]) -> dict:
    """Totals per run: tokens, attempts, known cost, and count of unknown-cost events."""
    total = {
        "attempts": 0, "errors": 0, "input_tokens": 0, "cached_input_tokens": 0,
        "output_tokens": 0, "reasoning_tokens": 0, "cost_usd": 0.0, "unknown_cost_events": 0,
    }
    for e in events:
        total["attempts"] += 1
        total["errors"] += e.outcome == "error"
        total["input_tokens"] += e.input_tokens or 0
        total["cached_input_tokens"] += e.cached_input_tokens or 0
        total["output_tokens"] += e.output_tokens or 0
        total["reasoning_tokens"] += e.reasoning_tokens or 0
        if e.cost_usd is None:
            total["unknown_cost_events"] += 1
        else:
            total["cost_usd"] += e.cost_usd
    return total


def group_summary(events: list[UsageEvent]) -> list[dict]:
    """Aggregate by (phase, provider, model, operation) for the cost report."""
    groups: dict[tuple, list[UsageEvent]] = defaultdict(list)
    for e in events:
        groups[(e.phase, e.provider, e.model, e.operation)].append(e)
    rows = []
    for (phase, provider, model, operation), evs in sorted(groups.items()):
        s = summarize(evs)
        s.update(
            phase=phase, provider=provider, model=model, operation=operation,
            runs=len({e.run_id for e in evs}),
        )
        rows.append(s)
    return rows
