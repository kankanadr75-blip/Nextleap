"""Phase 2b - Chunking.

Turns cleaned scheme records into small, atomic, self-describing chunks.

Why this shape
--------------
* **One attribute per chunk.** Most questions target a single attribute
  ("exit load of X"). One attribute per chunk gives the sharpest possible
  retrieval match, and the five Groww pages use near-identical wording, so
  mixing attributes in one chunk blurs which scheme/attribute actually matched.
* **Scheme name in every chunk.** Required for the same reason, reinforced by
  the ``scheme`` metadata filter at query time.
* **Hard cap at 200 tokens.** all-MiniLM-L6-v2 silently truncates beyond 256
  word-pieces, so an over-long chunk loses its tail with no error raised.
  [verified] 384-dim, 256 word-piece limit.
* **Returns are never indexed.** Phase 1 already refused to extract them; this
  module emits only allow-listed attributes, so FR-6 holds structurally.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Callable, Iterable

from src.ingest.clean import (
    clean_record,
    format_lock_in,
    format_money,
    format_percent,
    record_get,
)
from src.query import config

# Human-readable section labels used in the context header.
SOURCE_TYPE_LABELS: dict[str, str] = {
    "groww": "Groww scheme page",
    "amc_sid": "HDFC AMC SID",
    "amc_kim": "HDFC AMC KIM",
    "amc_factsheet": "HDFC AMC factsheet",
    "amfi": "AMFI investor page",
    "sebi": "SEBI investor page",
    "rta": "RTA investor portal",
}

MANUAL_SOURCE_TYPE = "rta"

# Display name used for scheme-agnostic (general) chunks, e.g. statement guides.
GENERAL_DISPLAY = "HDFC Mutual Fund schemes"


@dataclass
class Chunk:
    """One indexable unit. Mirrors the Chroma schema in architecture.md."""

    id: str
    document: str
    scheme: str
    attribute: str
    source_url: str
    source_type: str
    fetched_at: str
    chunk_type: str  # fact_card | prose | table | steps

    def embed_text(self) -> str:
        """The text actually handed to the embedding model.

        ``document`` states only verified facts, so it stays the citation text
        and the LLM's source material. This method appends a short paraphrase
        line (``config.ATTRIBUTE_ALIASES``) so the *vector* also covers how users
        actually phrase things ("charges for managing" for expense ratio, "who
        looks after" for fund manager).

        The split matters: folding the synonyms into ``document`` would push
        search keywords into the prompt and the quoted citation, and would risk
        the LLM restating a paraphrase as if it were a sourced fact. Kept
        separate, the synonyms can never be cited.
        """
        hint = config.ATTRIBUTE_ALIASES.get(self.attribute, "").strip()
        if not hint:
            return self.document
        return f"{self.document} Also described as: {hint}."

    def chroma_metadata(self) -> dict[str, str | int | float | bool]:
        """Metadata for Chroma.

        Every value is coerced to a scalar: ChromaDB rejects None, lists and
        dicts in metadata, which is a hard runtime error rather than a warning.
        """
        return {
            "scheme": str(self.scheme or config.GENERAL_SCHEME),
            "attribute": str(self.attribute or "other"),
            "source_url": str(self.source_url or ""),
            "source_type": str(self.source_type or "other"),
            "fetched_at": str(self.fetched_at or ""),
            "chunk_type": str(self.chunk_type or "other"),
        }


# --------------------------------------------------------------------------
# Tokenisation
# --------------------------------------------------------------------------


@lru_cache(maxsize=1)
def _tokenizer():  # type: ignore[no-untyped-def]
    """The real MiniLM tokenizer, if transformers can load it.

    Using the actual tokenizer is the only way to assert the 256 word-piece
    limit honestly; a whitespace approximation would undercount sub-words.
    """
    try:
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(config.EMBED_MODEL)
    except Exception:  # noqa: BLE001 - offline / not installed
        return None


def count_tokens(text: str) -> int:
    """Word-piece count via the real tokenizer, else a word-count fallback."""
    tok = _tokenizer()
    if tok is not None:
        try:
            return len(tok.encode(text, add_special_tokens=False))
        except Exception:  # noqa: BLE001
            pass
    return max(1, len(text.split()))


# --------------------------------------------------------------------------
# Context header + ids
# --------------------------------------------------------------------------


def context_header(display_name: str, section: str, source_type: str) -> str:
    """The ``[Scheme] | [Section] | [Source type]`` prefix.

    Prepended to the chunk text *and* embedded with it, so the scheme name is
    part of the vector. This is what keeps near-identical scheme pages apart.
    """
    label = SOURCE_TYPE_LABELS.get(source_type, source_type)
    return f"{display_name} | {section} | {label}"


def make_id(source_url: str, attribute: str, scheme: str, index: int) -> str:
    """Stable id so re-running ingestion upserts instead of duplicating."""
    raw = f"{source_url}#{attribute}#{scheme}"
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]
    return f"{digest}_{index:03d}"


def build_document(display_name: str, section: str, source_type: str, body: str) -> str:
    return f"{context_header(display_name, section, source_type)}. {collapse(body)}"


def collapse(text: str) -> str:
    return " ".join(str(text).split()).strip()


# --------------------------------------------------------------------------
# Fact-card builders
# --------------------------------------------------------------------------

# attribute -> (section label, builder(record, display_name) -> body | None)
_FACT_CARDS: list[tuple[str, str, Callable[[dict[str, Any], str], str | None]]] = [
    (
        "expense_ratio",
        "Expense ratio",
        lambda rec, _name: (
            f"the expense ratio is {format_percent(record_get(rec, 'expense_ratio'))} per annum."
            if format_percent(record_get(rec, "expense_ratio"))
            else None
        ),
    ),
    (
        "base_expense_ratio",
        "Base expense ratio",
        lambda rec, _name: (
            f"the base expense ratio (before the additional fund-expense charge) is "
            f"{format_percent(record_get(rec, 'base_expense_ratio'))} per annum."
            if format_percent(record_get(rec, "base_expense_ratio"))
            else None
        ),
    ),
    (
        "exit_load",
        "Exit load",
        lambda rec, _name: _exit_load_card(rec),
    ),
    (
        "min_sip",
        "Minimum SIP",
        lambda rec, _name: (
            f"the minimum SIP amount is {format_money(record_get(rec, 'min_sip_investment'))}."
            if format_money(record_get(rec, "min_sip_investment"))
            else None
        ),
    ),
    (
        "min_lumpsum",
        "Minimum lump-sum",
        lambda rec, _name: (
            f"the minimum lump-sum investment is "
            f"{format_money(record_get(rec, 'min_investment_amount'))}."
            if format_money(record_get(rec, "min_investment_amount"))
            else None
        ),
    ),
    (
        "lock_in",
        "Lock-in",
        lambda rec, _name: (
            f"the lock-in period is {format_lock_in(rec)} from the date of each investment."
            if format_lock_in(rec)
            else None
        ),
    ),
    (
        "riskometer",
        "Riskometer",
        lambda rec, _name: (
            f"the riskometer level is {record_get(rec, 'nfo_risk')}."
            if record_get(rec, "nfo_risk")
            else None
        ),
    ),
    (
        "benchmark",
        "Benchmark",
        lambda rec, _name: _benchmark_card(rec),
    ),
    (
        "fund_manager",
        "Fund manager",
        lambda rec, _name: _fund_manager_card(rec),
    ),
    (
        "category",
        "Fund category",
        lambda rec, _name: _category_card(rec),
    ),
    (
        "rta",
        "Registrar and transfer agent",
        lambda rec, _name: (
            f"the registrar and transfer agent (RTA) is {record_get(rec, 'rta_details.rta_name')}."
            if record_get(rec, "rta_details.rta_name")
            else None
        ),
    ),
    (
        "tax_status",
        "Tax status",
        lambda rec, _name: (
            "the fund has ELSS tax-saving status, so units are eligible for a deduction under "
            "section 80C subject to the applicable lock-in."
            if record_get(rec, "sub_category") == "ELSS"
            or record_get(rec, "additional_details.lock_in_yrs")
            else None
        ),
    ),
]


def _exit_load_card(rec: dict[str, Any]) -> str | None:
    """Phrase the exit load faithfully, without reinterpreting it.

    Two shapes occur in the source data:

    * a bare value - ``"1% if redeemed within 1 year"``, ``"Nil"`` -> "is X"
    * a conditions clause - the balanced-advantage scheme reads
      ``"for units in excess of 15% of the investment, 1% will be charged..."``
      [verified]. Forcing that into "the exit load is for units..." is
      ungrammatical, so a lowercase-initial value is presented as a clause
      instead. The wording is never re-derived or summarised.
    """
    value = record_get(rec, "exit_load")
    if not value:
        return None
    value = str(value).strip()
    if value[:1].islower():
        return f"the exit load applies as follows: {value}."
    return f"the exit load is {value}."


def _benchmark_card(rec: dict[str, Any]) -> str | None:
    benchmark = record_get(rec, "benchmark")
    if not benchmark:
        return None
    name = record_get(rec, "benchmark_name")
    if name and name != benchmark:
        return f"the benchmark index is {benchmark} ({name})."
    return f"the benchmark index is {benchmark}."


def _fund_manager_card(rec: dict[str, Any]) -> str | None:
    """The scheme's fund manager, from the singular ``fund_manager`` field only.

    [verified] ``fund_manager_details[].person_name`` is deliberately ignored:
    it is a cross-scheme list (Dhruv Muchhal appears in all five schemes) that
    omits the page's own stated fund manager in four of five cases. Trusting it
    would produce confidently wrong answers to "who manages this fund".
    """
    single = record_get(rec, "fund_manager")
    return f"the fund manager is {single}." if single else None


def _category_card(rec: dict[str, Any]) -> str | None:
    category = record_get(rec, "category")
    sub = record_get(rec, "sub_category")
    if category and sub:
        return f"the scheme category is {category} - {sub}."
    if category:
        return f"the scheme category is {category}."
    if sub:
        return f"the scheme sub-category is {sub}."
    return None


# --------------------------------------------------------------------------
# Prose splitting
# --------------------------------------------------------------------------


def split_prose(text: str, max_tokens: int, overlap: int) -> list[str]:
    """Split long prose on sentence boundaries, respecting the token cap.

    Never splits mid-sentence. When a single sentence exceeds the cap it is
    emitted whole rather than truncated, because silently cutting a sentence
    produces a misleading chunk.
    """
    text = collapse(text)
    if not text:
        return []
    if count_tokens(text) <= max_tokens:
        return [text]

    sentences = _split_sentences(text)
    chunks: list[str] = []
    current: list[str] = []
    for sentence in sentences:
        candidate = " ".join(current + [sentence])
        if current and count_tokens(candidate) > max_tokens:
            chunks.append(" ".join(current))
            # Rebuild the overlap tail from the sentences we are about to drop.
            tail: list[str] = []
            for prev in reversed(current):
                if count_tokens(" ".join(tail + [prev])) > overlap:
                    break
                tail.insert(0, prev)
            current = tail + [sentence]
        else:
            current.append(sentence)
    if current:
        chunks.append(" ".join(current))
    return chunks


_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z₹(])")


def _split_sentences(text: str) -> list[str]:
    parts = [p.strip() for p in _SENTENCE_RE.split(text) if p.strip()]
    return parts or [text]


# --------------------------------------------------------------------------
# Builders
# --------------------------------------------------------------------------


def chunks_from_record(record: dict[str, Any], fetched_at: str) -> list[Chunk]:
    """Build every chunk for one cleaned scheme record."""
    rec = clean_record(record)
    slug = rec.get("scheme_slug", "")
    display = rec.get("display_name") or config.display_name(slug)
    url = rec.get("source_url", "")
    chunks: list[Chunk] = []

    for attribute, section, builder in _FACT_CARDS:
        body = builder(rec, display)
        if not body:
            continue
        sentence = f"{display}: {collapse(body)}"
        document = build_document(display, section, "groww", sentence)
        if count_tokens(document) > config.FACT_CARD_MAX_TOKENS:
            # Should not happen for one-attribute cards; fail loudly rather than
            # ship a chunk whose tail MiniLM would truncate.
            raise ValueError(
                f"fact card too long ({count_tokens(document)} tokens): {attribute}/{slug}"
            )
        chunks.append(
            Chunk(
                id=make_id(url, attribute, slug, 0),
                document=document,
                scheme=slug,
                attribute=attribute,
                source_url=url,
                source_type="groww",
                fetched_at=fetched_at,
                chunk_type="fact_card",
            )
        )

    # Scheme objective / description, as bounded prose.
    description = rec.get("description")
    if description:
        for i, piece in enumerate(split_prose(description, config.PROSE_CHUNK_TOKENS, config.PROSE_CHUNK_OVERLAP)):
            document = build_document(display, "Scheme objective", "groww", piece)
            chunks.append(
                Chunk(
                    id=make_id(url, "objective", slug, i),
                    document=document,
                    scheme=slug,
                    attribute="objective",
                    source_url=url,
                    source_type="groww",
                    fetched_at=fetched_at,
                    chunk_type="prose",
                )
            )

    return chunks


def chunks_from_manual(fetched_at: str) -> list[Chunk]:
    """Build ``steps`` chunks for the curated statement-download procedures."""
    path = config.MANUAL_DIR / "statement_steps.json"
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    entries: Iterable[dict[str, Any]] = payload.get("entries", [])
    chunks: list[Chunk] = []

    for entry in entries:
        title = collapse(entry.get("title", ""))
        steps = [collapse(s) for s in entry.get("steps", []) if collapse(s)]
        if not title or not steps:
            continue
        url = entry.get("source_url", "")
        source_type = entry.get("source_type", MANUAL_SOURCE_TYPE)
        attribute = entry.get("attribute", "statement_download")

        # The title already appears in the context header, so the body starts at
        # step 1 rather than repeating it.
        body = " ".join(f"Step {i}: {s}" for i, s in enumerate(steps, start=1))
        # Keep each procedure whole; only split *between* steps if too long.
        pieces = _split_steps(body, config.PROSE_CHUNK_TOKENS)
        for i, piece in enumerate(pieces):
            document = build_document(
                GENERAL_DISPLAY, title, source_type, piece
            )
            chunks.append(
                Chunk(
                    id=make_id(url, f"{attribute}:{entry.get('id', i)}", config.GENERAL_SCHEME, i),
                    document=document,
                    scheme=config.GENERAL_SCHEME,
                    attribute=attribute,
                    source_url=url,
                    source_type=source_type,
                    fetched_at=fetched_at,
                    chunk_type="steps",
                )
            )
    return chunks


def _split_steps(body: str, max_tokens: int) -> list[str]:
    """Split a numbered step list only at step boundaries."""
    if count_tokens(body) <= max_tokens:
        return [body]
    parts = re.split(r"(?=Step \d+: )", body)
    parts = [p.strip() for p in parts if p.strip()]
    pieces: list[str] = []
    current: list[str] = []
    for part in parts:
        if current and count_tokens(" ".join(current + [part])) > max_tokens:
            pieces.append(" ".join(current))
            current = [part]
        else:
            current.append(part)
    if current:
        pieces.append(" ".join(current))
    return pieces


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------


def build_chunks(*, verbose: bool = True) -> list[Chunk]:
    """Read records.json, clean it, and emit every chunk.

    Order-independent and idempotent: ids are derived from
    (source_url, attribute, scheme), so re-running produces the same ids.
    """
    if not config.RECORDS_JSON.exists():
        raise FileNotFoundError(
            f"{config.RECORDS_JSON} not found - run the load stage first"
        )

    payload = json.loads(config.RECORDS_JSON.read_text(encoding="utf-8"))
    fetched_at = payload.get("fetched_at", "")

    chunks: list[Chunk] = []
    for record in payload.get("records", []):
        chunks.extend(chunks_from_record(record, fetched_at))
    chunks.extend(chunks_from_manual(fetched_at))

    _assert_token_caps(chunks)
    if verbose:
        _print_summary(chunks)
    return chunks


def _assert_token_caps(chunks: list[Chunk]) -> None:
    """No chunk may exceed the 256 word-piece MiniLM truncation point."""
    limit = 256
    offenders = [
        (c.id, count_tokens(c.document))
        for c in chunks
        if count_tokens(c.document) > limit
    ]
    if offenders:
        detail = ", ".join(f"{cid} ({n} tokens)" for cid, n in offenders[:5])
        raise ValueError(
            f"{len(offenders)} chunk(s) exceed the {limit} word-piece MiniLM "
            f"limit and would be silently truncated: {detail}"
        )


def write_chunks(chunks: list[Chunk]) -> None:
    """Write chunks.jsonl, one JSON object per line."""
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    with config.CHUNKS_JSONL.open("w", encoding="utf-8") as fh:
        for chunk in chunks:
            fh.write(json.dumps(chunk.chroma_metadata() | {
                "id": chunk.id,
                "document": chunk.document,
            }, ensure_ascii=False) + "\n")


def _print_summary(chunks: list[Chunk]) -> None:
    by_scheme: dict[str, int] = {}
    by_type: dict[str, int] = {}
    for c in chunks:
        by_scheme[c.scheme] = by_scheme.get(c.scheme, 0) + 1
        by_type[c.chunk_type] = by_type.get(c.chunk_type, 0) + 1

    print(f"  total chunks : {len(chunks)}")
    print("  per scheme   :")
    for slug in [*config.CORPUS_SLUGS, config.GENERAL_SCHEME]:
        print(f"      {slug:<26} {by_scheme.get(slug, 0)}")
    print("  per type     :")
    for kind, n in sorted(by_type.items()):
        print(f"      {kind:<26} {n}")

    tokens = [count_tokens(c.document) for c in chunks]
    if tokens:
        print(f"  tokens       : min {min(tokens)}, median "
              f"{sorted(tokens)[len(tokens)//2]}, max {max(tokens)} (cap 256)")
    print(f"  chunks jsonl : {config.CHUNKS_JSONL}")


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Phase 2: clean and chunk.")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    chunks = build_chunks(verbose=not args.quiet)
    write_chunks(chunks)
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
