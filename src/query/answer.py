"""Phase 6 support - the full answer chain behind one call.

``answer_question`` is the only function the UI needs. Keeping the chain here
rather than in ``app.py`` means the whole pipeline - guard, retrieval,
generation, formatting - is testable without starting Streamlit, and the UI
cannot accidentally bypass a guardrail by forgetting a step.

Every path out of this function returns a :class:`FormattedAnswer`, so the UI
renders answered, refused, out-of-scope, PII-blocked and unknown replies through
one code path and they look identical.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any

from src.query import config
from src.query.format import FormattedAnswer, format_answer, format_blocked
from src.query.generate import generate_answer
from src.query.guard import run_guards
from src.query.llm import LLMClient
from src.query.retrieve import detect_scheme_detail, retrieve

logger = logging.getLogger("mf_faq.answer")


@dataclass
class AnswerOutcome:
    """A finished reply plus the diagnostics the UI shows in its footer."""

    formatted: FormattedAnswer
    blocked: bool
    guard_kind: str
    scheme: str | None
    intent: str
    top_distance: float
    latency_ms: float
    above_threshold: bool = False

    @property
    def text(self) -> str:
        return self.formatted.text

    @property
    def source_url(self) -> str:
        return self.formatted.source_url

    @property
    def fetched_at(self) -> str:
        return self.formatted.fetched_at

    @property
    def sentences(self) -> int:
        return self.formatted.sentences


def answer_question(
    message: str,
    *,
    collection: Any | None = None,
    llm: LLMClient | None = None,
) -> AnswerOutcome:
    """Run guard -> retrieve -> generate -> format and return one outcome.

    Nothing is persisted and the message text is never logged; only coarse
    labels reach ``logger`` from the layers below.
    """
    started = time.perf_counter()
    message = (message or "").strip()

    if not message:
        out = format_blocked(
            "Please type a question about one of these five HDFC Mutual Fund "
            "schemes."
        )
        return AnswerOutcome(
            formatted=out, blocked=True, guard_kind="empty",
            scheme=None, intent="none", top_distance=float("inf"), latency_ms=0.0,
        )

    # 1. Guards first. A block never reaches the retriever.
    detection = detect_scheme_detail(message)
    verdict = run_guards(message, detection)
    if verdict.blocked:
        out = format_blocked(verdict.message, link=verdict.link)
        return AnswerOutcome(
            formatted=out,
            blocked=True,
            guard_kind=verdict.kind,
            scheme=detection.scheme,
            intent="blocked",
            top_distance=float("inf"),
            latency_ms=(time.perf_counter() - started) * 1000.0,
        )

    # 2. Retrieval, filtered to the detected scheme plus the general chunks.
    result = retrieve(message, detection.scheme, collection=collection)
    latency_ms = (time.perf_counter() - started) * 1000.0

    if not result.confident:
        # FR-8: below the distance threshold, or no hits at all.
        out = format_blocked(
            config.UNKNOWN_TEMPLATE.format(
                scheme=detection.scheme or "HDFC Mutual Fund"
            ),
            link=config.REFUSAL_LINK,
        )
        return AnswerOutcome(
            formatted=out,
            blocked=True,
            guard_kind="unknown",
            scheme=detection.scheme,
            intent=result.intent,
            top_distance=result.best_distance,
            latency_ms=latency_ms,
            above_threshold=result.above_threshold,
        )

    # 3. Generation, then 4. compliance formatting.
    generation = generate_answer(message, result.chunks, llm)
    out = format_answer(
        generation,
        scheme_for_unknown=detection.scheme or "",
    )
    return AnswerOutcome(
        formatted=out,
        blocked=not out.cited,
        guard_kind="ok" if out.cited else "uncited",
        scheme=detection.scheme,
        intent=result.intent,
        top_distance=result.best_distance,
        latency_ms=latency_ms,
    )
