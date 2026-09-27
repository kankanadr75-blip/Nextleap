"""Phase 5c - prompt construction and answer generation.

The model is asked for two things: the answer text and the id of the chunk it
used. It is never asked for a URL, and the id is validated against the retrieved
set before it goes any further - an unknown id is rejected, not trusted.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from src.query import config
from src.query.llm import SYSTEM_PROMPT, LLMClient, get_llm
from src.query.retrieve import RetrievedChunk

logger = logging.getLogger("mf_faq.generate")


@dataclass
class GenerationResult:
    """The model's proposal, before any compliance formatting."""

    answer: str
    source_chunk_id: str
    provider: str
    valid_chunk_id: bool
    candidates: list[str] = field(default_factory=list)

    @property
    def needs_fallback(self) -> bool:
        """True when the model returned something we cannot cite.

        Either no id, or an id that was not in the retrieved set. ``format.py``
        refuses to attach a citation it cannot verify, so this routes to the
        FR-8 "not in my sources" reply instead.
        """
        return not self.valid_chunk_id


def build_user_turn(question: str, chunks: list[RetrievedChunk]) -> str:
    """Render the retrieved chunks as a numbered, cited CONTEXT block.

    Each block carries its id, type and source URL. The URL is given to the model
    only as context to read; the system prompt forbids it from echoing one, and
    ``format.py`` strips any it does.
    """
    lines = [
        "CONTEXT:",
        "",
    ]
    for i, chunk in enumerate(chunks, start=1):
        lines.append(
            f"[{i}] chunk_id: {chunk.id} type: {chunk.chunk_type}\n"
            f"source_url: {chunk.source_url}\n"
            f"{chunk.document}"
        )
        lines.append("")

    lines.append(f"QUESTION: {question}")
    lines.append("")
    lines.append(
        "Answer using only the CONTEXT above, then return the JSON object "
        "with your answer and the chunk_id you used."
    )
    return "\n".join(lines)


def generate_answer(
    question: str,
    chunks: list[RetrievedChunk],
    llm: LLMClient | None = None,
) -> GenerationResult:
    """Ask the model for an answer, then validate the chunk id it claims.

    Validation is by membership, not by plausibility: ``source_chunk_id`` must be
    the id of a chunk we actually retrieved. Anything else is marked invalid so
    the caller can fall back, which is what stops a hallucinated citation from
    becoming a real link.
    """
    client = llm or get_llm()
    if not chunks:
        return GenerationResult(
            answer="", source_chunk_id="", provider=client.name, valid_chunk_id=False
        )

    user_turn = build_user_turn(question, chunks)
    payload: dict[str, Any] = {}
    try:
        payload = client.complete_json(SYSTEM_PROMPT, user_turn) or {}
    except Exception as exc:  # noqa: BLE001 - never raise into the chat path
        logger.warning("generation failed (%s); treating as no answer",
                       type(exc).__name__)
        payload = {}

    answer = str(payload.get("answer", "") or "").strip()
    claimed_id = str(payload.get("source_chunk_id", "") or "").strip()

    valid_ids = [c.id for c in chunks]
    valid = claimed_id in valid_ids
    if not valid:
        logger.info("generation chunk_id rejected (provider=%s)", client.name)

    # Never log the answer text or the question.
    logger.info("generation provider=%s chunks=%d valid_chunk_id=%s",
                client.name, len(chunks), valid)
    return GenerationResult(
        answer=answer,
        source_chunk_id=claimed_id,
        provider=client.name,
        valid_chunk_id=valid,
        candidates=valid_ids,
    )
