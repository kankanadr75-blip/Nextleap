"""Phase 5d - compliance formatting.

Nothing the model produced is trusted. Length, citations, links and return
figures are all re-derived here from code-side state:

* the citation comes from a chunk map built out of ``chunks.jsonl`` - the model
  only names a chunk, so it cannot invent a URL;
* any URL the model emitted is stripped before the one verified URL is appended;
* return percentages are removed, but **fee percentages are kept**;
* the answer is hard-trimmed to 3 sentences by a real sentence splitter.

On the percentage rule
----------------------
The PRD says to strip "any ``%``". Applied literally that breaks three of the
core supported intents: the expense ratio, the base expense ratio and the exit
load are all percentages, and answering without them would be wrong. So the rule
is applied **by context**: a percentage is stripped only when it sits in a
return/performance context, and a fee percentage is left alone. ``strip_performance_percentages``
is separated out and unit-tested precisely because this is a deliberate
narrowing of the written rule, not an oversight.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from src.query import config
from src.query.generate import GenerationResult
from src.query.retrieve import RetrievedChunk

logger = logging.getLogger("mf_faq.format")

URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)

# A percentage figure: 12%, 12.5 %, 3.2%
PERCENT_FIGURE_RE = re.compile(r"\b\d+(?:\.\d+)?\s?(?:%|per\s*cent\b|percent\b)", re.IGNORECASE)

# Return vocabulary that makes a nearby percentage a performance claim.
#
# Deliberately excludes growth / gain / loss / rise / rose. [bug caught by
# tests/test_guardrails.py] Every corpus scheme is named "... - Direct Growth",
# so matching "growth" classified all five expense-ratio answers as return
# statements and the stripper deleted the percentage from every answer. Only
# unambiguous return terms belong here.
_RETURN_CONTEXT_RE = re.compile(
    r"\b(returns?|returned|cagr|nav|performance|profit|appreciation|"
    r"grew|yield|since inception|value of my investment|worth now|"
    r"p\.?a\.? return|annualised return|annualized return)\b",
    re.IGNORECASE,
)

# Attributes whose percentage figures are *sourced facts*, not performance claims.
# The corpus stores no returns at all (Phase 1's allow-list blocks their
# extraction), so a percentage on one of these attributes is definitionally a
# fee and must survive the stripper.
FEE_PERCENT_ATTRIBUTES: frozenset[str] = frozenset(
    {"expense_ratio", "base_expense_ratio", "exit_load"}
)

# Abbreviations whose trailing period must not end a sentence. Without this,
# "Rs. 100" splits into two sentences and the 3-sentence cap silently truncates
# a correct answer.
_ABBREVIATIONS: frozenset[str] = frozenset(
    {
        "rs", "no", "vs", "etc", "ie", "eg", "approx", "amc", "rta", "nav",
        "ter", "sebi", "amfi", "sid", "kim", "pdf", "xml", "csv", "sip",
        "nfo", "p.a", "pa", "ltd", "inc", "m.p", "mp",
    }
)

_SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[.!?])\s+(?=[^\s])")

FOOTER_PREFIX = "Last updated from sources: "
SOURCE_PREFIX = "Source: "


# --------------------------------------------------------------------------
# Chunk map (code-side citation truth)
# --------------------------------------------------------------------------


@dataclass
class Citation:
    """Everything needed to cite a chunk, resolved from our own records."""

    chunk_id: str
    scheme: str
    attribute: str
    source_url: str
    source_type: str
    fetched_at: str


@lru_cache(maxsize=1)
def chunk_citations() -> dict[str, Citation]:
    """id -> Citation, loaded from chunks.jsonl.

    The model never supplies a link. It names a chunk id; the link comes from
    here, so a fabricated citation is structurally impossible.
    """
    path = config.CHUNKS_JSONL
    if not path.exists():
        return {}
    citations: dict[str, Citation] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        citations[row["id"]] = Citation(
            chunk_id=row["id"],
            scheme=row.get("scheme", ""),
            attribute=row.get("attribute", ""),
            source_url=row.get("source_url", ""),
            source_type=row.get("source_type", ""),
            fetched_at=row.get("fetched_at", ""),
        )
    return citations


# --------------------------------------------------------------------------
# Text scrubbing
# --------------------------------------------------------------------------


def strip_urls(text: str) -> str:
    """Remove every URL from model output, then tidy the seam it leaves."""
    cleaned = URL_RE.sub("", text)
    cleaned = re.sub(r"\s+([.,;:])", r"\1", cleaned)
    return re.sub(r"\s{2,}", " ", cleaned).strip()


def strip_performance_percentages(
    text: str, attribute: str = ""
) -> tuple[str, int]:
    """Remove percentages stated as returns. Returns (text, n_removed).

    A percentage is a performance figure only when a return word appears in the
    same clause. Fee percentages ("the expense ratio is 1.03% per annum") have
    no return word and are preserved - stripping them would delete the answer.

    ``attribute`` is the cited chunk's attribute. On a fee attribute a
    percentage is a sourced fact and is never stripped, which makes the rule
    robust even if the surrounding wording is unusual.

    ``(text, 0)`` for a fee percentage, ``(text_without_it, 1)`` for a return
    figure.
    """
    if attribute in FEE_PERCENT_ATTRIBUTES:
        return text, 0

    kept: list[str] = []
    removed = 0
    for clause in re.split(r"(?<=[.;!?])\s+", text):
        if PERCENT_FIGURE_RE.search(clause) and _RETURN_CONTEXT_RE.search(clause):
            without = PERCENT_FIGURE_RE.sub("", clause).strip()
            removed += len(PERCENT_FIGURE_RE.findall(clause))
            if without:
                kept.append(without)
        else:
            kept.append(clause.strip())
    out = " ".join(p for p in kept if p).strip()
    if removed and out and not out.endswith((".", "!", "?")):
        out += "."
    return out, removed


def _strip_percentages_preserving_lines(text: str, attribute: str = "") -> str:
    """Percentage-strip line by line, so authored line breaks survive.

    ``strip_performance_percentages`` is a single-paragraph reflow: it splits on
    whitespace and rejoins with a single space, which flattens every newline.

    [bug] Applying it to guard copy destroyed the line break in a template. Worse,
    whether the break survived depended on the punctuation in front of it -
    only ``.`` ``;`` ``!`` ``?`` are split points, so ``"...facts for are:"``
    kept its break while ``"...do you mean?"`` lost it. The disambiguation menu
    rendered as "Which of these five... do you mean? 1. HDFC Large Cap Fund",
    with the first option swallowed into the question.

    Guard copy is multi-line by design, so the transform is applied per line
    instead. Generated answers keep the reflowing version: they are one paragraph
    by the time they reach the formatter, and the 3-sentence cap has already
    re-joined them.
    """
    return "\n".join(
        strip_performance_percentages(line, attribute)[0] for line in text.split("\n")
    )


def split_sentences(text: str) -> list[str]:
    """Split into sentences without breaking decimals or abbreviations.

    ``split('.')`` is unusable here: it shreds "0.77%", "Rs. 100" and "Step 1."
    This splits only on terminal punctuation followed by whitespace, then rejoins
    fragments that end in a known abbreviation or a single-letter initial.
    """
    if not text.strip():
        return []

    parts = [p.strip() for p in _SENTENCE_BOUNDARY_RE.split(text.strip()) if p.strip()]

    merged: list[str] = []
    for part in parts:
        if merged and _ends_with_abbreviation(merged[-1]):
            merged[-1] = f"{merged[-1]} {part}"
        else:
            merged.append(part)
    return merged


def _ends_with_abbreviation(fragment: str) -> bool:
    """True if ``fragment`` ends mid-thought (abbreviation, initial, number)."""
    if not fragment.endswith((".", "!", "?")):
        return False
    body = fragment.rstrip(".!?")
    if not body:
        return False
    last = body.split()[-1] if body.split() else ""
    # A decimal like "0.77" - the period sits between digits, so it never
    # reaches here, but guard anyway.
    if re.fullmatch(r"\d+", last):
        return False
    # Strip the dots before comparing: "i.e" must match the "ie" entry, and "p"
    # must match the single-initial rule.
    normalised = last.lower().replace(".", "").replace(",", "")
    return normalised in _ABBREVIATIONS or len(normalised) == 1


def truncate_to_sentences(text: str, max_sentences: int | None = None) -> tuple[str, int]:
    """Hard-trim to ``max_sentences``. Returns (text, sentences_kept)."""
    limit = max_sentences or config.MAX_SENTENCES
    sentences = split_sentences(text)
    if not sentences:
        return text.strip(), 0
    kept = sentences[:limit]
    return " ".join(kept).strip(), len(kept)


# --------------------------------------------------------------------------
# Final formatting
# --------------------------------------------------------------------------


@dataclass
class FormattedAnswer:
    """The user-visible reply, after every compliance rule has been applied."""

    text: str
    source_url: str
    fetched_at: str
    chunk_id: str
    sentences: int
    cited: bool
    note: str = ""

    @property
    def footer(self) -> str:
        return f"{FOOTER_PREFIX}{self.fetched_at}"


def _url_allowed(url: str) -> bool:
    if not url.startswith("http"):
        return False
    host = url.split("//", 1)[1].split("/", 1)[0].lower()
    return host in config._citation_domains()


def format_answer(
    generation: GenerationResult,
    *,
    scheme_for_unknown: str = "",
    max_sentences: int | None = None,
) -> FormattedAnswer:
    """Turn a generation result into a compliant reply, or an FR-8 refusal.

    The order is the point: strip the model's URL, strip return percentages,
    trim to 3 sentences, then attach exactly one citation resolved from the
    code-side map. A citation is only ever one URL from ``sources.csv``.
    """
    limit = max_sentences or config.MAX_SENTENCES

    def unknown(note: str) -> FormattedAnswer:
        """The FR-8 path: say we do not have it, cite the official source.

        Exactly one URL either way - the refusal link comes from config, not from
        the model, so it is as uncitable-by-hallucination as a real citation.
        """
        body, kept = truncate_to_sentences(
            config.UNKNOWN_TEMPLATE.format(
                scheme=scheme_for_unknown or "HDFC Mutual Fund"
            ),
            limit,
        )
        return FormattedAnswer(
            text=f"{body}\n{SOURCE_PREFIX}{config.REFUSAL_LINK}",
            source_url=config.REFUSAL_LINK,
            fetched_at="",
            chunk_id="",
            sentences=kept,
            cited=False,
            note=note,
        )

    if generation.needs_fallback or not generation.answer.strip():
        return unknown("no_valid_chunk_id")

    citation = chunk_citations().get(generation.source_chunk_id)
    if citation is None:
        return unknown("unknown_chunk_id")
    if not _url_allowed(citation.source_url):
        # A chunk id we recognise but whose URL is not in the allow-list is
        # treated as uncitable rather than printed.
        logger.warning("citation rejected: url not in allow-list")
        return unknown("url_not_allowlisted")

    body = strip_urls(generation.answer)
    # The cited attribute decides whether a percentage is a fee or a return.
    body, _ = strip_performance_percentages(body, attribute=citation.attribute)
    body, kept = truncate_to_sentences(body, limit)

    if not body.strip():
        return unknown("empty_after_scrub")

    # Exactly one URL, in the reply, from our own records.
    text = f"{body}\n{SOURCE_PREFIX}{citation.source_url}"
    if citation.fetched_at:
        text = f"{text}\n{FOOTER_PREFIX}{citation.fetched_at}"

    return FormattedAnswer(
        text=text,
        source_url=citation.source_url,
        fetched_at=citation.fetched_at,
        chunk_id=citation.chunk_id,
        sentences=kept,
        cited=True,
    )


def format_blocked(message: str, link: str = "", fetched_at: str = "") -> FormattedAnswer:
    """Render a guard refusal for display.

    Deliberately **not** sentence-truncated. The 3-sentence cap governs
    *generated* answers; guard copy is hand-authored, already compliant, and its
    line structure is load-bearing. [bug found by tests/demo_answers.py] capping
    the disambiguation menu collapsed "Which of these five... 1. ... 5." into
    "Which of these five... 1. HDFC Large Cap Fund 2." - the prompt asked the
    user to choose and then listed only two options.

    Percentages are still stripped as a safety net, and exactly one ``Source:``
    line is appended - the templates carry no link of their own. The strip runs
    per line so a template's numbered list keeps its line breaks; see
    ``_strip_percentages_preserving_lines``.
    """
    body = _strip_percentages_preserving_lines(message).strip()
    text = body
    if link:
        text = f"{text}\n{SOURCE_PREFIX}{link}"
    if fetched_at:
        text = f"{text}\n{FOOTER_PREFIX}{fetched_at}"
    return FormattedAnswer(
        text=text,
        source_url=link,
        fetched_at=fetched_at,
        chunk_id="",
        # Not a generated answer, so the cap does not apply.
        sentences=0,
        cited=bool(link),
        note="blocked",
    )


def answered_text_only(formatted: FormattedAnswer) -> str:
    """The reply with the ``Source:`` and footer lines removed.

    Used by the test suite to count sentences honestly - the PRD's 3-sentence cap
    applies to the answer, not to the citation block.
    """
    lines = [
        line
        for line in formatted.text.splitlines()
        if not line.startswith((SOURCE_PREFIX, FOOTER_PREFIX))
    ]
    return "\n".join(lines).strip()
