"""Phase 1 acceptance checks (implementation.md)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ingest.load import (
    ALLOWED_FIELDS,
    BANNED_KEY_SUBSTRINGS,
    assert_no_banned_keys,
    read_sources,
)
from src.query import config

failures: list[str] = []


def check(label: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}" + (f" - {detail}" if detail else ""))
    if not ok:
        failures.append(label)


print("== 1. sources.csv: 5 corpus rows all fetch_status=ok ==")
rows = read_sources()
corpus = [r for r in rows if r.is_corpus]
check("corpus row count == 5", len(corpus) == 5, f"got {len(corpus)}")
for r in corpus:
    check(
        f"{r.scheme_slug} status=ok and has fetched_at",
        r.fetch_status == "ok" and bool(r.fetched_at),
        f"status={r.fetch_status} fetched_at={r.fetched_at}",
    )
citation = [r for r in rows if not r.is_corpus]
check("citation_only rows preserved", len(citation) == 8, f"got {len(citation)}")
check(
    "citation_only rows untouched by loader",
    all(r.fetch_status in ("ok", "blocked_403", "unreachable") for r in citation),
)

print()
print("== 2. data/raw: 5 non-trivial HTML files ==")
raws = sorted(config.RAW_DIR.glob("*.html"))
check("raw file count == 5", len(raws) == 5, f"got {len(raws)}")
for p in raws:
    size = p.stat().st_size
    check(f"{p.name} > 100 KB", size > 100_000, f"{size/1024:.0f} KB")

print()
print("== 3. records.json: no return/nav/holdings keys survive (FR-6) ==")
payload = json.loads(config.RECORDS_JSON.read_text(encoding="utf-8"))
records = payload["records"]
check("record count == 5", len(records) == 5, f"got {len(records)}")

all_keys: set[str] = set()
for rec in records:
    all_keys |= set(rec)

banned_hits = [k for k in all_keys if any(b in k.lower() for b in BANNED_KEY_SUBSTRINGS)]
check("no banned key substrings in any record", not banned_hits, str(banned_hits))

extra = sorted(k for k in all_keys if k not in ALLOWED_FIELDS and k not in
               ("scheme_slug", "display_name", "source_url"))
check("every extracted field is on the allow-list", not extra, str(extra))

try:
    assert_no_banned_keys({"nav": 1.0})
    check("assert_no_banned_keys actually raises", False, "did not raise")
except ValueError:
    check("assert_no_banned_keys actually raises", True)

print()
print("== 4. extracted facts match the verified snapshot ==")
EXPECTED = {
    "hdfc-large-cap": {"expense_ratio": "1.03", "min_sip_investment": 100, "nfo_risk": "Moderately High Riskometer", "benchmark": "NIFTY 100 TRI", "fund_manager": "Prashant Jain"},
    "hdfc-flexi-cap": {"expense_ratio": "0.77", "min_sip_investment": 100, "nfo_risk": "Moderately High", "benchmark": "NIFTY 500 TRI", "fund_manager": "Prashant Jain"},
    "hdfc-elss-tax-saver": {"expense_ratio": "1.21", "min_sip_investment": 500, "nfo_risk": "Moderately High", "benchmark": "NIFTY 500 TRI", "fund_manager": "Vinay Kulkarni", "lock_in.years": 3},
    "hdfc-small-cap": {"expense_ratio": "0.78", "min_sip_investment": 100, "nfo_risk": "Moderately High Riskometer", "benchmark": "BSE 250 SmallCap TRI", "fund_manager": "Chirag Setalvad"},
    "hdfc-balanced-advantage": {"expense_ratio": "0.78", "min_sip_investment": 100, "nfo_risk": "Moderately High Riskometer", "benchmark": "NIFTY 50 Hybrid Composite Debt 50:50 Index", "fund_manager": "Srinivas Rao Ravuri"},
}
by_slug = {r["scheme_slug"]: r for r in records}
for slug, exp in EXPECTED.items():
    rec = by_slug.get(slug, {})
    for key, want in exp.items():
        got = rec.get(key)
        check(f"{slug}.{key} == {want!r}", got == want, f"got {got!r}")

print()
print("== 5. guardrails ==")
from src.ingest.load import domain_allowed, robots_allows

check(
    "allow-list permits groww.in",
    domain_allowed("https://groww.in/mutual-funds/hdfc-small-cap-fund-direct-growth"),
)
check(
    "allow-list blocks hdfcfund.com",
    not domain_allowed("https://www.hdfcfund.com/our-funds/equity/large-cap"),
)
check(
    "allow-list blocks sebi.gov.in",
    not domain_allowed("https://investor.sebi.gov.in/index.html"),
)
ok, reason = robots_allows(
    "https://groww.in/mutual-funds/hdfc-small-cap-fund-direct-growth"
)
check("robots.txt permits the 5 corpus paths", ok, reason)
ok2, reason2 = robots_allows("https://groww.in/mutual-funds/compare/anything")
check("robots.txt blocks /mutual-funds/compare/*", not ok2, reason2)

print()
if failures:
    print(f"FAILED {len(failures)} check(s):")
    for f in failures:
        print(f"  - {f}")
    sys.exit(1)
print("ALL PHASE 1 ACCEPTANCE CHECKS PASSED")
