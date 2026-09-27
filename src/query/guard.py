"""Phase 5a - guardrails.

Every refusal and block is decided here, in code, **before** retrieval runs. The
prompt asks the model to behave; this module is what makes that true.

Order is load-bearing: ``PII -> out-of-scope -> ambiguity -> performance ->
advice``. PII is first because it is the only irreversible harm - once a PAN or
an OTP is in the message it must not reach the retriever, the model, or a log
file. Everything downstream of a PII hit is moot.

A blocked question never reaches Chroma, never reaches the LLM, and logs nothing
but a coarse label. Message text is never logged (FR-7, no-persistence NFR).
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Literal

from src.query import config
from src.query.retrieve import AMBIGUOUS, SchemeDetection

logger = logging.getLogger("mf_faq.guard")

GuardKind = Literal["pii", "out_of_scope", "ambiguous", "performance", "advice", "ok"]


@dataclass
class GuardResult:
    """Outcome of the guard chain.

    ``blocked`` True means the assistant replies with ``message`` and stops.
    ``ok`` means the question is answerable and retrieval may run.
    """

    kind: GuardKind
    message: str = ""
    link: str = ""
    candidates: list[str] | None = None

    @property
    def blocked(self) -> bool:
        return self.kind != "ok"

    @property
    def label(self) -> str:
        return f"guard:{self.kind}"


OK = GuardResult(kind="ok")


# --------------------------------------------------------------------------
# 1. PII (FR-7)
# --------------------------------------------------------------------------

# Ordered most-specific first, because several patterns can match the same
# string and the first hit names the kind. A 12-digit run next to "folio" is
# reported as a folio, not as an Aadhaar number - the block is identical either
# way, only the label differs.
PII_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("pan", re.compile(r"(?<![A-Za-z0-9])[A-Za-z]{5}[0-9]{4}[A-Za-z](?![A-Za-z0-9])")),
    (
        "folio_or_account",
        re.compile(r"(?i)(?:folio|account|a/c|acc\.?\s*(?:no|number))[^\d]{0,20}\d{6,}"),
    ),
    ("email", re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")),
    ("otp", re.compile(r"(?i)otp[^0-9]{0,20}\d{4,8}|\d{4,8}[^0-9]{0,20}otp")),
    # 12 digits, optionally spaced. All-same-digit runs are excluded because
    # 000000000000 / 111111111111 are placeholders, not Aadhaar numbers.
    (
        "aadhaar",
        re.compile(
            r"(?<!\d)(?!(?:0{12}|1{12}|2{12}|3{12}|4{12}|5{12}|6{12}|7{12}|8{12}|9{12}))"
            r"\d{4}[\s-]?\d{4}[\s-]?\d{4}(?!\d)"
        ),
    ),
    ("mobile", re.compile(r"(?<![\d-])(?:\+?91[-\s]?)?[6-9]\d{9}(?![\d-])")),
)


def detect_pii(text: str) -> str | None:
    """Return the PII kind found in ``text``, or None. First match wins.

    The caller must not log or store ``text`` when this returns non-None.
    """
    if not text:
        return None
    for kind, pattern in PII_PATTERNS:
        if pattern.search(text):
            return kind
    return None


# --------------------------------------------------------------------------
# 2. Out of scope (FR-10)
# --------------------------------------------------------------------------

_OTHER_AMC_RE = tuple(re.compile(p, re.IGNORECASE) for p in config.OTHER_AMC_PATTERNS)
_HDFC_SCHEME_RE = re.compile(config.HDFC_SCHEME_RE, re.IGNORECASE)
_HDFC_MENTION_RE = re.compile(r"(?<!\w)hdfc(?!\w)", re.IGNORECASE)

# Words that carry no scheme identity, dropped before comparing a fund name to
# the alias table.
_NAME_STOPWORDS: frozenset[str] = frozenset({"fund", "funds", "the", "of", "direct", "plan"})

# alias -> frozenset of its identifying tokens, e.g. "large cap" -> {"large","cap"}
# and "equity fund" -> {"equity"}. Built from the same table retrieval uses, so
# the two can never disagree about which names we cover.
_ALIAS_TOKEN_SETS: tuple[frozenset[str], ...] = tuple(
    frozenset(
        token
        for token in alias.lower().split()
        if token not in _NAME_STOPWORDS
    )
    for alias in config.SCHEME_ALIASES
    if any(t not in _NAME_STOPWORDS for t in alias.lower().split())
)


def _name_tokens(text: str) -> frozenset[str]:
    return frozenset(
        token for token in re.split(r"[^\w]+", text.lower()) if token
        and token not in _NAME_STOPWORDS
    )


def check_out_of_scope(text: str) -> GuardResult:
    """Reject other AMCs and HDFC schemes outside the five.

    An HDFC scheme we do not carry is *not* the same as an ambiguous reference:
    "HDFC Mid Cap Fund" names a real fund we cannot answer for, so it gets the
    scope message, not "which of the five do you mean?".

    A name is in scope only when its identifying tokens **exactly** match one of
    our aliases. A looser subset test gets this wrong in both directions - it
    rejects our own five ("HDFC Large Cap Fund", where "cap" is not itself a
    corpus token) and it would accept "HDFC Small Cap Value Fund", a different
    scheme that happens to contain a known alias.
    """
    if not text:
        return OK

    for pattern in _OTHER_AMC_RE:
        if pattern.search(text):
            return GuardResult(
                kind="out_of_scope",
                message=config.OUT_OF_SCOPE_TEMPLATE,
                link=config.REFUSAL_LINK,
            )

    for match in _HDFC_SCHEME_RE.finditer(text):
        middle = match.group(1).strip()

        # "HDFC Mutual Fund" names the AMC, not a scheme. The ambiguity guard
        # owns that case; treating it as an unknown scheme would wrongly refuse
        # questions like "which RTA holds my HDFC Mutual Fund folio".
        if not middle or "mutual" in middle.lower():
            continue

        tokens = _name_tokens(middle)
        if not tokens:
            continue

        if any(tokens == alias for alias in _ALIAS_TOKEN_SETS):
            continue  # one of our five

        # A compound name that contains two different known aliases ("HDFC Large
        # and Small Cap Fund") is a user naming several schemes, not an unknown
        # one. Leave it to the ambiguity guard, which asks which they meant.
        contained = {alias for alias in _ALIAS_TOKEN_SETS if alias <= tokens}
        if len({frozenset(alias) for alias in contained}) > 1:
            continue

        return GuardResult(
            kind="out_of_scope",
            message=config.OUT_OF_SCOPE_TEMPLATE,
            link=config.REFUSAL_LINK,
        )

    return OK


# --------------------------------------------------------------------------
# 3. Ambiguity (FR-9)
# --------------------------------------------------------------------------


def check_ambiguity(detection: SchemeDetection | str | None) -> GuardResult:
    """Ask which scheme when the reference does not identify one.

    Multiple matches ("large cap or small cap") and a bare "HDFC fund" both land
    here, because both leave the scheme unidentified.
    """
    if detection is None:
        return OK
    is_ambiguous = (
        (isinstance(detection, str) and detection == AMBIGUOUS)
        or (isinstance(detection, SchemeDetection) and detection.needs_disambiguation)
    )
    if not is_ambiguous:
        return OK
    candidates = (
        [] if isinstance(detection, str) else list(detection.candidates or [])
    )
    return GuardResult(
        kind="ambiguous",
        message=config.DISAMBIGUATION_TEMPLATE,
        link=config.REFUSAL_LINK,
        candidates=candidates,
    )


# --------------------------------------------------------------------------
# 4. Performance (FR-6)
# --------------------------------------------------------------------------

_PERFORMANCE_RE = tuple(re.compile(p, re.IGNORECASE) for p in config.PERFORMANCE_PATTERNS)


def check_performance(text: str) -> GuardResult:
    """Refuse return/NAV/performance questions without quoting any number.

    The corpus never holds these figures (Phase 1's allow-list blocks their
    extraction), so there is nothing to answer with. The reply points at the
    official factsheet and carries no figure itself.
    """
    if not text:
        return OK
    for pattern in _PERFORMANCE_RE:
        if pattern.search(text):
            return GuardResult(
                kind="performance",
                # No link in the template body; format_blocked appends the single
                # "Source:" line, so printing it here too would double the URL.
                message=config.PERFORMANCE_REFUSAL_TEMPLATE,
                link=config.PERFORMANCE_SOURCE_URL,
            )
    return OK


# --------------------------------------------------------------------------
# 5. Advice (FR-5)
# --------------------------------------------------------------------------

_ADVICE_RE = tuple(re.compile(p, re.IGNORECASE) for p in config.ADVICE_PATTERNS)


def check_advice(text: str) -> GuardResult:
    """Refuse advice and recommendation questions with the verbatim template.

    Verified by ``tests/test_guardrails.py`` to contain no ``%`` figure: a
    refusal that leaked "12.5%" would fail the check mid-sentence.
    """
    if not text:
        return OK
    for pattern in _ADVICE_RE:
        if pattern.search(text):
            return GuardResult(
                kind="advice",
                message=config.REFUSAL_TEMPLATE,
                link=config.REFUSAL_LINK,
            )
    return OK


# --------------------------------------------------------------------------
# 6. Attribute coverage (FR-8, extended)
# --------------------------------------------------------------------------

_UNCOVERED_RE = tuple(
    re.compile(p, re.IGNORECASE) for p in config.UNCOVERED_ATTRIBUTE_PATTERNS
)


def check_coverage(
    text: str, detection: SchemeDetection | str | None = None
) -> GuardResult:
    """Refuse a question about a fact the corpus does not hold.

    The corpus covers 14 attributes. A question about a 15th thing has nothing
    to answer with, but retrieval will still hand back a nearby chunk and the
    generator will state it confidently with a real citation - which reads as a
    sourced fact and is simply wrong.

    [bug found by tests/demo_ui.py] "What is the ticker symbol of HDFC Flexi Cap
    Fund?" answered "the benchmark index is NIFTY 500 TRI", top-1 at distance
    0.343, well inside the 0.65 threshold. Distance cannot catch it: the query
    shares the scheme name with every chunk in that scheme, so all of them score
    close together.

    Runs last in the chain, so the more specific refusals win. A performance
    question gets the factsheet pointer, an advice question gets the adviser
    pointer, and only an otherwise-answerable question about an uncovered fact
    lands here.
    """
    if not text:
        return OK
    for pattern in _UNCOVERED_RE:
        if pattern.search(text):
            scheme = _scheme_label(detection)
            return GuardResult(
                kind="unknown",
                message=config.UNKNOWN_TEMPLATE.format(scheme=scheme),
                link=config.REFUSAL_LINK,
            )
    return OK


def _scheme_label(detection: SchemeDetection | str | None) -> str:
    """A human-readable scheme name for the refusal copy, never blank."""
    if isinstance(detection, str):
        return config.base_name(detection)
    slug = getattr(detection, "scheme", None)
    if slug:
        return config.base_name(slug)
    return "HDFC Mutual Fund"


# --------------------------------------------------------------------------
# Chain
# --------------------------------------------------------------------------


def run_guards(
    message: str,
    detection: SchemeDetection | str | None = None,
) -> GuardResult:
    """Run the full chain in the mandated order and return the first block.

    A PII hit short-circuits immediately - no later check runs, and the caller
    must not persist the message.
    """
    pii_kind = detect_pii(message)
    if pii_kind:
        # Deliberately logs the *kind* of PII, never the value or the message.
        logger.info("guard=pii pii_kind=%s action=blocked_no_retrieval", pii_kind)
        return GuardResult(kind="pii", message=config.PII_BLOCK_MESSAGE)

    # Mandated order: PII -> out-of-scope -> ambiguity -> performance -> advice,
    # then coverage. Ambiguity deliberately precedes advice, so "which is better,
    # large cap or small cap" resolves to FR-9 (ask which scheme) rather than a
    # refusal. Coverage is last so the specific refusals win over the generic one.
    checks = (
        check_out_of_scope,
        check_ambiguity,
        check_performance,
        check_advice,
        check_coverage,
    )
    for check in checks:
        if check is check_ambiguity:
            result = check(detection)
        elif check is check_coverage:
            result = check(message, detection)
        else:
            result = check(message)
        if result.blocked:
            logger.info("guard=%s action=blocked", result.kind)
            return result

    logger.info("guard=ok action=proceed")
    return OK
