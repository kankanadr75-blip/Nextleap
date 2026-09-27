"""Phase 2a - Cleaning.

Normalises the raw records produced by Phase 1 so that chunk text is stable and
retrieval-friendly.

The normalisations here exist because the source data is genuinely inconsistent
[verified 27 Sep 2026]:

* ``nfo_risk`` arrives as both ``"Moderately High"`` and
  ``"Moderately High Riskometer"`` depending on the scheme.
* ``exit_load`` contains embedded newlines, e.g.
  ``"Exit load of 1% if redeemed within 1 year\\r\\n"``.
* The balanced-advantage exit-load string has a missing space after a comma:
  ``"...of the investment,1% will be charged..."``.
* ``super_category`` is not a usable display name (see config.DISPLAY_NAMES).
"""

from __future__ import annotations

import re
from typing import Any

from src.query import config

# The six riskometer levels defined by SEBI's product-labelling framework.
SEBI_RISK_LEVELS: tuple[str, ...] = (
    "Low",
    "Low to Moderate",
    "Moderate",
    "Moderately High",
    "High",
    "Very High",
)

_WS_RE = re.compile(r"\s+")
_MISSING_SPACE_AFTER_COMMA_RE = re.compile(r",(?=\S)")
_RISK_SUFFIX_RE = re.compile(r"\s*risk[-\s]?ometer\s*$", re.I)
_LEADING_EXIT_LOAD_RE = re.compile(r"^\s*exit\s+load\s*(?:of|is|:)?\s*", re.I)

_META_KEYS = ("scheme_slug", "display_name", "source_url")


def collapse_ws(value: Any) -> Any:
    """Collapse all whitespace runs (incl. \\r\\n) into single spaces."""
    if not isinstance(value, str):
        return value
    return _WS_RE.sub(" ", value).strip()


def fix_punctuation(value: Any) -> Any:
    """Restore the missing space in strings like ``investment,1%``."""
    if not isinstance(value, str):
        return value
    return _MISSING_SPACE_AFTER_COMMA_RE.sub(", ", value)


def normalise_risk(value: Any) -> str | None:
    """Map a raw ``nfo_risk`` string onto one of the six SEBI levels.

    Strips a trailing "Riskometer" and then matches the six canonical levels
    case-insensitively. Returns None for empty input, and the cleaned string
    unchanged when it matches nothing (so an unexpected new level is visible in
    the chunk text rather than silently mangled).
    """
    text = collapse_ws(value)
    if not text:
        return None
    text = _RISK_SUFFIX_RE.sub("", str(text)).strip()
    if not text:
        return None
    for level in SEBI_RISK_LEVELS:
        if text.casefold() == level.casefold():
            return level
    return text


def format_lock_in(record: dict[str, Any]) -> str | None:
    """Render the lock-in from the flat ``lock_in.*`` keys.

    Returns None when the total is zero, so no "lock-in: none" card is emitted
    for schemes that have no lock-in. [verified] only the ELSS scheme has one.
    """
    years = _as_int(record.get("lock_in.years"))
    months = _as_int(record.get("lock_in.months"))
    days = _as_int(record.get("lock_in.days"))
    if years is None and months is None and days is None:
        return None
    parts: list[str] = []
    if years:
        parts.append(f"{years} year" + ("s" if years != 1 else ""))
    if months:
        parts.append(f"{months} month" + ("s" if months != 1 else ""))
    if days:
        parts.append(f"{days} day" + ("s" if days != 1 else ""))
    if not parts:
        return None
    if len(parts) == 1:
        return parts[0]
    return ", ".join(parts[:-1]) + " and " + parts[-1]


def format_money(value: Any) -> str | None:
    """``500`` -> ``"₹500"``. Indian digit grouping is left as-is."""
    amount = _as_number(value)
    if amount is None:
        return None
    if isinstance(amount, float) and not amount.is_integer():
        return f"₹{amount:,.2f}"
    return f"₹{int(amount):,}"


def format_percent(value: Any) -> str | None:
    """``"0.78"`` -> ``"0.78%"``. Expense ratios and exit loads are fees, not
    returns, so this is safe content for a facts-only assistant."""
    number = _as_number(value)
    if number is None:
        return None
    text = f"{number:g}"
    return f"{text}%"


def clean_exit_load(value: Any) -> str | None:
    """Normalise an exit-load string and drop the redundant leading label."""
    text = collapse_ws(fix_punctuation(value))
    if not text:
        return None
    text = _LEADING_EXIT_LOAD_RE.sub("", str(text)).strip()
    if not text:
        return None
    # Drop trailing sentence punctuation; the fact card supplies its own.
    text = text.rstrip(" .").strip()
    if not text:
        return None
    # "Nil" / "NIL" / "Not applicable" all mean the same thing to a user. The
    # card reads "the exit load is X", so keep the value bare.
    if text.casefold() in ("nil", "na", "not applicable", "none"):
        return "Nil"
    return text


def record_get(record: dict[str, Any], dotted: str, default: Any = None) -> Any:
    """Read a flat dot-path key, tolerating absence."""
    value = record.get(dotted, default)
    return default if value is None else value


def clean_record(record: dict[str, Any]) -> dict[str, Any]:
    """Return a normalised copy of one Phase 1 record.

    Applies whitespace/punctuation fixes to every string, canonicalises the
    riskometer level, and forces the curated display name.
    """
    cleaned: dict[str, Any] = {}
    for key, value in record.items():
        if key in ("lock_in.years", "lock_in.months", "lock_in.days"):
            number = _as_int(value)
            if number is not None:
                cleaned[key] = number
            continue
        if key == "nfo_risk":
            risk = normalise_risk(value)
            if risk:
                cleaned[key] = risk
            continue
        if key == "exit_load":
            exit_load = clean_exit_load(value)
            if exit_load:
                cleaned[key] = exit_load
            continue
        if key == "display_name":
            cleaned[key] = config.display_name(record.get("scheme_slug", ""))
            continue
        if key == "description":
            cleaned[key] = collapse_ws(fix_punctuation(value))
            continue
        if isinstance(value, list):
            # e.g. fund_manager_details.person_name
            cleaned[key] = [collapse_ws(fix_punctuation(v)) for v in value]
            continue
        cleaned[key] = collapse_ws(fix_punctuation(value))

    # Guarantee the display name even if the record lacked it.
    slug = record.get("scheme_slug")
    if slug:
        cleaned["display_name"] = config.display_name(slug)
    return cleaned


def _as_int(value: Any) -> int | None:
    number = _as_number(value)
    if number is None:
        return None
    return int(number)


def _as_number(value: Any) -> float | int | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return value
    text = str(value).strip()
    match = re.search(r"-?\d+(?:\.\d+)?", text.replace(",", ""))
    if not match:
        return None
    number = float(match.group(0))
    return int(number) if number.is_integer() else number
