"""Phase 4 - Retrieval.

Turns a user question into a ranked, scheme-filtered list of chunks.

Two defences keep the five near-identical scheme pages apart, and **both are
required** - removing either degrades accuracy:

1. the scheme name is inside every chunk's context header, so it is part of the
   embedded vector; and
2. a ``scheme`` metadata filter restricts the search to one scheme plus the
   scheme-agnostic ``general`` chunks.

Distances are **cosine distances in [0, 2]** - lower is closer. A distance above
``config.DISTANCE_THRESHOLD`` means we do not have a confident match and the
caller should emit the FR-8 "not in my sources" reply.

Logging policy: this module logs the intent label and latency only. Message text
is never logged or stored (FR-7 and the no-persistence NFR).
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Literal

from src.ingest.embed_store import embed_query, get_collection
from src.query import config

logger = logging.getLogger("mf_faq.retrieve")

AMBIGUOUS = "ambiguous"
SchemeGuess = Literal["unique", "ambiguous", "none"]


# --------------------------------------------------------------------------
# Scheme detection
# --------------------------------------------------------------------------


@dataclass
class SchemeDetection:
    """Outcome of matching a question against the alias map."""

    kind: SchemeGuess
    scheme: str | None = None
    candidates: list[str] = field(default_factory=list)
    matched_alias: str | None = None
    hdfc_generic: bool = False

    @property
    def needs_disambiguation(self) -> bool:
        return self.kind == "ambiguous"


def _alias_pattern(alias: str) -> re.Pattern[str]:
    """Word-boundary match so 'small cap' cannot match inside another word."""
    return re.compile(rf"(?<!\w){re.escape(alias)}(?!\w)", re.IGNORECASE)


# Pre-compiled once; longest alias first so 'small cap fund' beats 'small cap'.
_ALIAS_PATTERNS: list[tuple[str, str, re.Pattern[str]]] = sorted(
    ((slug, alias, _alias_pattern(alias)) for alias, slug in config.SCHEME_ALIASES.items()),
    key=lambda item: len(item[1]),
    reverse=True,
)

_HDFC_MENTION_RE = re.compile(r"(?<!\w)hdfc(?!\w)", re.IGNORECASE)
_HDFC_GENERIC_RE = re.compile(
    r"(?<!\w)hdfc\s+(?:mutual\s+fund|fund|scheme|mf)(?!\w)", re.IGNORECASE
)

# Questions that are inherently scheme-agnostic. Without this, "which RTA holds
# my HDFC Mutual Fund folio" would be treated as an ambiguous *scheme* reference
# and sent to the disambiguation prompt, when the user is really asking about
# statements - retrievable from the `general` chunks with no scheme at all.
_SCHEME_AGNOSTIC_RE = re.compile(
    r"(?<!\w)("
    r"statements?|documents?|download|tax|folios?|rta|registrar|"
    r"capital[\s-]*gains?|cas|consolidated|"
    r"itr|xml|account\s+statement|nfo|kim|sid|factsheets?|"
    r"investor|education|grievance|"
    r"how\s+do\s+i|how\s+can\s+i|where\s+can\s+i|where\s+do\s+i"
    r")(?!\w)",
    re.IGNORECASE,
)


def detect_scheme_detail(text: str) -> SchemeDetection:
    """Classify a question as a unique scheme, ambiguous, or unconstrained.

    FR-9: a bare reference such as "HDFC fund" is ambiguous and must trigger the
    disambiguation question rather than a guess.
    """
    if not text or not text.strip():
        return SchemeDetection(kind="none")

    slugs: dict[str, str] = {}  # slug -> the alias that matched
    for slug, alias, pattern in _ALIAS_PATTERNS:
        if pattern.search(text):
            slugs.setdefault(slug, alias)

    if len(slugs) == 1:
        slug, alias = next(iter(slugs.items()))
        return SchemeDetection(kind="unique", scheme=slug, candidates=[slug], matched_alias=alias)

    if len(slugs) > 1:
        return SchemeDetection(
            kind="ambiguous",
            candidates=sorted(slugs),
            matched_alias=None,
        )

    # No specific alias matched. A scheme-agnostic question ("...my HDFC Mutual
    # Fund folio") is not an ambiguous scheme reference, so leave it
    # unconstrained and let the `general` chunks answer it.
    if _SCHEME_AGNOSTIC_RE.search(text):
        return SchemeDetection(kind="none")

    # Otherwise, if the user said "HDFC ..." at all, we know the AMC but not the
    # scheme, so ask which one (FR-9).
    if _HDFC_GENERIC_RE.search(text) or _HDFC_MENTION_RE.search(text):
        return SchemeDetection(kind="ambiguous", candidates=[], hdfc_generic=True)

    return SchemeDetection(kind="none")


def detect_scheme(text: str) -> str | None:
    """Documented simple form: the slug, ``"ambiguous"``, or ``None``."""
    detection = detect_scheme_detail(text)
    if detection.kind == "unique":
        return detection.scheme
    if detection.kind == "ambiguous":
        return AMBIGUOUS
    return None


# --------------------------------------------------------------------------
# Retrieval
# --------------------------------------------------------------------------


@dataclass
class RetrievedChunk:
    """One search hit, with the metadata needed to build a citation."""

    id: str
    document: str
    distance: float
    scheme: str
    attribute: str
    source_url: str
    source_type: str
    fetched_at: str
    chunk_type: str

    @classmethod
    def from_chroma(
        cls, chunk_id: str, document: str, metadata: dict[str, Any], distance: float
    ) -> "RetrievedChunk":
        return cls(
            id=chunk_id,
            document=document,
            distance=float(distance),
            scheme=str(metadata.get("scheme", "")),
            attribute=str(metadata.get("attribute", "")),
            source_url=str(metadata.get("source_url", "")),
            source_type=str(metadata.get("source_type", "")),
            fetched_at=str(metadata.get("fetched_at", "")),
            chunk_type=str(metadata.get("chunk_type", "")),
        )

    @property
    def score(self) -> float:
        """Cosine *similarity* in [-1, 1]. Convenience inverse of distance."""
        return 1.0 - self.distance


@dataclass
class RetrievalResult:
    """Ranked hits plus the confidence decision."""

    query_scheme: str | None
    chunks: list[RetrievedChunk]
    best_distance: float
    latency_ms: float
    filtered: bool

    @property
    def above_threshold(self) -> bool:
        return self.best_distance > config.DISTANCE_THRESHOLD

    @property
    def confident(self) -> bool:
        """True when we have a hit and it is within the distance threshold."""
        return bool(self.chunks) and not self.above_threshold

    @property
    def intent(self) -> str:
        """Coarse intent label for logs. Never derived from message text."""
        if not self.chunks:
            return "no_match"
        return self.chunks[0].attribute or "unknown"

    @property
    def best(self) -> RetrievedChunk | None:
        return self.chunks[0] if self.chunks else None


def build_where(scheme: str | None) -> dict[str, Any] | None:
    """Metadata filter for a detected scheme, always including ``general``.

    Returns None when no scheme was detected, so the search spans the whole
    collection (needed for scheme-agnostic questions such as statement
    downloads).
    """
    if not scheme or scheme == AMBIGUOUS:
        return None
    return {"scheme": {"$in": [scheme, config.GENERAL_SCHEME]}}


def retrieve(
    query: str,
    scheme: str | None = None,
    *,
    top_k: int | None = None,
    collection: Any | None = None,
) -> RetrievalResult:
    """Search the collection and return ranked hits with a confidence verdict.

    ``scheme`` of None means "no filter". Pass ``AMBIGUOUS`` to also get no
    filter; the caller is expected to have asked the user which scheme by then.
    """
    started = time.perf_counter()
    k = top_k or config.TOP_K
    col = collection if collection is not None else get_collection()

    embedding = embed_query(query)
    where = build_where(scheme)

    kwargs: dict[str, Any] = {
        "query_embeddings": [embedding],
        "n_results": k,
        "include": ["documents", "metadatas", "distances"],
    }
    if where is not None:
        kwargs["where"] = where

    raw = col.query(**kwargs)

    ids = (raw.get("ids") or [[]])[0]
    documents = (raw.get("documents") or [[]])[0]
    metadatas = (raw.get("metadatas") or [[]])[0]
    distances = (raw.get("distances") or [[]])[0]

    chunks = [
        RetrievedChunk.from_chroma(cid, doc, meta or {}, dist)
        for cid, doc, meta, dist in zip(ids, documents, metadatas, distances)
    ]

    best = chunks[0].distance if chunks else float("inf")
    latency_ms = (time.perf_counter() - started) * 1000.0

    result = RetrievalResult(
        query_scheme=scheme,
        chunks=chunks,
        best_distance=best,
        latency_ms=latency_ms,
        filtered=where is not None,
    )

    # Intent label and latency only - never the message text.
    logger.info(
        "intent=%s scheme=%s hits=%d best_distance=%.4f above_threshold=%s latency_ms=%.1f",
        result.intent,
        scheme or "-",
        len(chunks),
        result.best_distance,
        result.above_threshold,
        latency_ms,
    )
    return result
