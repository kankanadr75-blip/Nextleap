"""Adversarial retrieval probe: paraphrases that avoid the chunk's own wording.

The Phase 4 golden set is close paraphrases of the fact cards, so a 100% hit
rate there may just mean the eval is leaky. These queries deliberately avoid the
indexed vocabulary ("expense ratio", "exit load", "minimum SIP") to estimate
retrieval quality on phrasing a real user would not copy from our own chunks.
"""

from __future__ import annotations

import io
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from src.ingest.embed_store import embed_query
from src.query.retrieve import detect_scheme, retrieve

HARD = [
    # (query, expected_scheme, expected_attribute, what makes it hard)
    ("charges for managing hdfc flexi cap", "hdfc-flexi-cap", "expense_ratio", "avoids 'expense ratio'"),
    ("annual fee taken from hdfc equity fund", "hdfc-flexi-cap", "expense_ratio", "synonym for expense ratio"),
    ("what happens if i sell small cap units before a year", "hdfc-small-cap", "exit_load", "avoids 'exit load'"),
    ("penalty for redeeming hdfc large cap early", "hdfc-large-cap", "exit_load", "avoids 'exit load'"),
    ("how much do i need to start an sip in elss", "hdfc-elss-tax-saver", "min_sip", "avoids 'minimum SIP'"),
    ("smallest monthly instalment for tax saver fund", "hdfc-elss-tax-saver", "min_sip", "avoids 'minimum SIP'"),
    ("who looks after the balanced advantage fund", "hdfc-balanced-advantage", "fund_manager", "avoids 'fund manager'"),
    ("what index is small cap measured against", "hdfc-small-cap", "benchmark", "avoids 'benchmark'"),
    ("risk level of the flexi cap scheme", "hdfc-flexi-cap", "riskometer", "avoids 'riskometer'"),
    ("how long must i hold elss units", "hdfc-elss-tax-saver", "lock_in", "avoids 'lock-in'"),
    ("expence ratio of hdfc smal cap", "hdfc-small-cap", "expense_ratio", "typos"),
    ("exit load hdfc balance advantage", "hdfc-balanced-advantage", "exit_load", "typo + partial name"),
    ("one time investment minimum for flexi cap", "hdfc-flexi-cap", "min_lumpsum", "avoids 'lump-sum'"),
    ("what kind of scheme is hdfc small cap", "hdfc-small-cap", "category", "avoids 'category'"),
]


def main() -> int:
    embed_query("warmup")
    print("=" * 104)
    print("ADVERSARIAL RETRIEVAL PROBE (paraphrases avoiding the indexed wording)")
    print("=" * 104)
    print()
    top1 = topk = 0
    wrong_scheme = 0
    latencies = []
    misses = []

    for query, want_scheme, want_attr, why in HARD:
        scheme = detect_scheme(query)
        if scheme == "ambiguous":
            print(f"  ASK  {query}")
            misses.append((query, "scheme detection returned ambiguous", why))
            continue
        res = retrieve(query, scheme)
        latencies.append(res.latency_ms)

        rank = None
        for i, c in enumerate(res.chunks, start=1):
            if c.scheme == want_scheme and c.attribute == want_attr:
                rank = i
                break
        if rank == 1:
            top1 += 1
        if rank:
            topk += 1
        else:
            best = res.best
            if best and best.scheme != want_scheme:
                wrong_scheme += 1
            misses.append(
                (query, f"got {best.attribute}/{best.scheme}" if best else "no hits", why)
            )

        verdict = f"#{rank}" if rank else "MISS"
        best = res.best
        got = f"{best.attribute}/{best.scheme}" if best else "-"
        print(f"  {verdict:>5}  d={res.best_distance:.4f}  got {got:<42} want {want_attr}/{want_scheme}")

    n = len(HARD)
    print()
    print(f"top-1 : {top1}/{n} = {top1/n*100:.1f}%")
    print(f"top-5 : {topk}/{n} = {topk/n*100:.1f}%   (PRD target >= 85%)")
    if wrong_scheme:
        print(f"wrong-scheme top hits: {wrong_scheme}  <- worse than a near miss")
    if latencies:
        print(f"latency median: {statistics.median(latencies):.0f} ms")
    if misses:
        print()
        print("MISSES:")
        for q, got, why in misses:
            print(f"  - {q}")
            print(f"      got: {got}   (hard because it {why})")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
