"""Phase 2 acceptance checks (implementation.md)."""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from src.ingest.chunk import count_tokens
from src.query import config

failures: list[str] = []


def check(label: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}" + (f" - {detail}" if detail else ""))
    if not ok:
        failures.append(label)


rows = [
    json.loads(line)
    for line in config.CHUNKS_JSONL.read_text(encoding="utf-8").splitlines()
    if line.strip()
]
by_id = {r["id"]: r for r in rows}

print("== 1. chunk counts ==")
check("chunks.jsonl non-empty", len(rows) > 0, f"{len(rows)} chunks")
fact_cards = [r for r in rows if r["chunk_type"] == "fact_card"]
check("fact_card >= 9 per scheme (x5)", len(fact_cards) >= 45, f"{len(fact_cards)} fact cards")
for slug in config.CORPUS_SLUGS:
    n = len([r for r in fact_cards if r["scheme"] == slug])
    check(f"{slug} has >= 9 fact cards", n >= 9, f"{n}")
general = [r for r in rows if r["scheme"] == config.GENERAL_SCHEME]
check("statement/steps chunks exist", len(general) > 0, f"{len(general)} general chunks")
check(
    "statement chunks carry attribute=statement_download",
    all(r["attribute"] == "statement_download" for r in general),
)

print()
print("== 2. balance across schemes (a lopsided count means extraction broke) ==")
per_scheme = {slug: len([r for r in rows if r["scheme"] == slug]) for slug in config.CORPUS_SLUGS}
spread = max(per_scheme.values()) - min(per_scheme.values())
check("per-scheme chunk counts within 2 of each other", spread <= 2, f"spread={spread} {per_scheme}")

print()
print("== 3. FR-6: no return/NAV/holdings content indexed ==")
BANNED_TERMS = ["return1y", "return3y", "SIP Return", "holdings", "NAV", "CAGR", "XIRR", "AUM"]
blob = "\n".join(r["document"] for r in rows)
for term in BANNED_TERMS:
    check(f"'{term}' absent from every chunk", term.lower() not in blob.lower())
check(
    "no scheme AUM leaked",
    "15991" not in blob and "crore" not in blob.lower(),
)

print()
print("== 4. token caps (MiniLM 256 word-piece silent truncation) ==")
tokens = {r["id"]: count_tokens(r["document"]) for r in rows}
over_256 = {i: n for i, n in tokens.items() if n > 256}
check("no chunk exceeds 256 word-pieces", not over_256, str(over_256))
prose_over = {
    r["id"]: tokens[r["id"]]
    for r in rows
    if r["chunk_type"] in ("prose", "steps") and tokens[r["id"]] > config.PROSE_CHUNK_TOKENS
}
check(
    "no prose/steps chunk exceeds 200 tokens",
    not prose_over,
    str(prose_over),
)
fc = [tokens[r["id"]] for r in rows if r["chunk_type"] == "fact_card"]
check(
    "fact cards within 30-80 tokens",
    all(config.FACT_CARD_MIN_TOKENS <= n <= config.FACT_CARD_MAX_TOKENS for n in fc),
    f"min={min(fc)} max={max(fc)}",
)

print()
print("== 5. chunk schema (architecture.md section 5) ==")
REQUIRED = ["id", "document", "scheme", "attribute", "source_url", "source_type", "fetched_at", "chunk_type"]
for field in REQUIRED:
    check(f"every chunk has '{field}'", all(field in r for r in rows))
check("all ids unique", len(by_id) == len(rows), f"{len(by_id)} unique / {len(rows)}")
check(
    "no metadata value is None",
    all(v is not None for r in rows for v in r.values()),
)
check(
    "every source_url is https",
    all(str(r["source_url"]).startswith("https://") for r in rows),
)
check(
    "every chunk carries a fetched_at date",
    all(str(r["fetched_at"])[:2] == "20" for r in rows),
)
check(
    "source_url is in the allow-list / known set",
    all(
        any(host in r["source_url"] for host in ("groww.in", "amfiindia.com", "sebi.gov.in", "camsonline.com", "kfintech.com", "hdfcfund.com"))
        for r in rows
    ),
)

print()
print("== 6. context header on every chunk ==")
header_ok = []
for r in rows:
    head = r["document"].split(" | ")
    header_ok.append(len(head) >= 3 and head[0].startswith("HDFC"))
check("every document starts with '<Scheme> | <Section> | <Source type>'", all(header_ok))
check(
    "every scheme chunk header names its own scheme",
    all(
        config.display_name(r["scheme"]) in r["document"]
        for r in rows
        if r["scheme"] in config.DISPLAY_NAMES
    ),
)

print()
print("== 7. correctness of the fact cards ==")
EXPECTED_CARDS = {
    "hdfc-elss-tax-saver": {
        "expense_ratio": "1.21%",
        "min_sip": "₹500",
        "lock_in": "3 years",
        "riskometer": "Moderately High",
        "benchmark": "NIFTY 500 TRI",
        "fund_manager": "Vinay Kulkarni",
    },
    "hdfc-large-cap": {
        "expense_ratio": "1.03%",
        "min_sip": "₹100",
        "riskometer": "Moderately High",
        "fund_manager": "Prashant Jain",
        "benchmark": "NIFTY 100 TRI",
    },
    "hdfc-small-cap": {
        "expense_ratio": "0.78%",
        "fund_manager": "Chirag Setalvad",
        "benchmark": "BSE 250 SmallCap TRI",
    },
    "hdfc-flexi-cap": {"expense_ratio": "0.77%", "fund_manager": "Prashant Jain"},
    "hdfc-balanced-advantage": {
        "expense_ratio": "0.78%",
        "fund_manager": "Srinivas Rao Ravuri",
    },
}
for slug, expects in EXPECTED_CARDS.items():
    for attribute, needle in expects.items():
        match = [
            r
            for r in rows
            if r["scheme"] == slug and r["attribute"] == attribute
        ]
        ok = bool(match) and needle in match[0]["document"]
        check(f"{slug}/{attribute} mentions {needle!r}", ok,
              (match[0]["document"][:90] + "...") if match else "card missing")

print()
print("== 8. verified data-quality guard: fund_manager_details must NOT be used ==")
check(
    "no chunk names a cross-scheme manager as this fund's manager",
    not any("Dhruv Muchhal" in r["document"] for r in rows),
    "Dhruv Muchhal leaked into a fund_manager card" if any("Dhruv Muchhal" in r["document"] for r in rows) else "",
)
check(
    "'the fund manager is' (singular) used for all 5 schemes",
    len([r for r in rows if r["attribute"] == "fund_manager" and "the fund manager is" in r["document"]]) == 5,
)

print()
print("== 9. lock-in only where it exists ==")
lock_schemes = {r["scheme"] for r in rows if r["attribute"] == "lock_in"}
check(
    "lock_in card exists only for the ELSS scheme",
    lock_schemes == {"hdfc-elss-tax-saver"},
    str(lock_schemes),
)

print()
print("== 10. normalisation applied ==")
risk_cards = [r for r in rows if r["attribute"] == "riskometer"]
check(
    "no riskometer card says 'Riskometer' twice / leaks the raw suffix",
    all("Moderately High Riskometer" not in r["document"] for r in risk_cards),
)
bal = [r for r in rows if r["scheme"] == "hdfc-balanced-advantage" and r["attribute"] == "exit_load"]
check(
    "balanced-advantage exit load has a space after the comma",
    bool(bal) and "investment, 1%" in bal[0]["document"],
    bal[0]["document"][-90:] if bal else "missing",
)
check(
    "no chunk contains a stray newline or carriage return",
    all("\n" not in r["document"] and "\r" not in r["document"] for r in rows),
)

print()
if failures:
    print(f"FAILED {len(failures)} check(s):")
    for f in failures:
        print(f"  - {f}")
    sys.exit(1)
print("ALL PHASE 2 ACCEPTANCE CHECKS PASSED")
