"""Phase 4 evaluation harness.

    python tests/eval_retrieval.py

Prints a per-query rank table and reports top-1 / top-5 hit rates against
tests/golden.json. A case counts as a hit only when a returned chunk matches on
BOTH expected_scheme and expected_attribute (and expected_source_url when the
case pins one) - matching the scheme alone would let a wrong-attribute chunk
score as a pass.

Also sweeps the distance threshold so the 0.6 starting value can be justified
with data rather than left as a guess.
"""

from __future__ import annotations

import io
import json
import statistics
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from src.query import config
from src.query.retrieve import detect_scheme, retrieve
from src.ingest.embed_store import embed_query

GOLDEN = Path(__file__).resolve().parent / "golden.json"


def load_cases() -> list[dict]:
    return json.loads(GOLDEN.read_text(encoding="utf-8"))["cases"]


def rank_of_expected(result, case: dict) -> int | None:
    """1-based rank of the chunk matching this case, or None if absent."""
    for i, chunk in enumerate(result.chunks, start=1):
        if chunk.scheme != case["expected_scheme"]:
            continue
        if chunk.attribute != case["expected_attribute"]:
            continue
        want_url = case.get("expected_source_url")
        if want_url and chunk.source_url != want_url:
            continue
        return i
    return None


def main() -> int:
    cases = load_cases()
    # The `hard` group deliberately avoids the indexed vocabulary ("charges for
    # managing" instead of "expense ratio"). It is the honest measure; the
    # `factual` group is close paraphrase and scores near 100% by construction.
    # `hard_typo` cases must trigger FR-9 rather than guess a scheme.
    retrieval_cases = [c for c in cases if c.get("group") != "hard_typo"]
    typo_cases = [c for c in cases if c.get("group") == "hard_typo"]

    print(LINE := "=" * 104)
    print("PHASE 4 RETRIEVAL EVALUATION")
    print(LINE)
    print(f"model       : {config.EMBED_MODEL}")
    print(f"collection  : {config.COLLECTION_NAME} (cosine)")
    print(f"top_k       : {config.TOP_K}")
    print(f"threshold   : {config.DISTANCE_THRESHOLD}")
    print(f"cases       : {len(retrieval_cases)} retrieval + {len(typo_cases)} typo")
    print(f"groups      : {dict(Counter(c.get('group', '?') for c in retrieval_cases))}")

    # Warm the model and the Chroma client before timing. [verified] the cold
    # load costs ~22 s, which would otherwise show up as one bogus outlier and
    # hide the real steady-state latency.
    cold_started = time.perf_counter()
    embed_query("warmup")
    warmup_s = time.perf_counter() - cold_started
    print(f"cold start  : {warmup_s:.1f} s (one-time model + client load, "
          f"paid once at app start)")
    print()

    rows = []
    top1 = topk = 0
    latencies: list[float] = []
    wrong_scheme = 0
    per_group: dict[str, list[tuple[int, int]]] = {}

    for case in retrieval_cases:
        query = case["query"]
        scheme = detect_scheme(query)
        group = case.get("group", "?")
        # An ambiguous detection means the assistant must ask, not retrieve.
        if scheme == "ambiguous":
            rows.append((query, case, "ASK", 0.0, None, None, group))
            per_group.setdefault(group, []).append((0, 0))
            continue

        result = retrieve(query, scheme)
        latencies.append(result.latency_ms)
        rank = rank_of_expected(result, case)
        if rank == 1:
            top1 += 1
        if rank is not None:
            topk += 1
        elif result.chunks and result.chunks[0].scheme != case["expected_scheme"]:
            wrong_scheme += 1
        per_group.setdefault(group, []).append((1 if rank == 1 else 0, 1 if rank else 0))

        best = result.best
        rows.append(
            (
                query,
                case,
                f"#{rank}" if rank else "MISS",
                result.best_distance,
                best.attribute if best else "-",
                best.scheme if best else "-",
                group,
            )
        )

    print(f"{'query':<52} {'rank':>5} {'dist':>7}  {'grp':<11} top attribute / scheme")
    print("-" * 104)
    for query, case, verdict, dist, attribute, scheme, group in rows:
        q = query if len(query) <= 51 else query[:48] + "..."
        print(f"{q:<52} {verdict:>5} {dist:>7.4f}  {group:<11} {attribute} / {scheme}")
    print()

    n = len(retrieval_cases)
    print(f"top-1 hit rate : {top1}/{n} = {top1/n*100:.1f}%")
    print(f"top-{config.TOP_K} hit rate : {topk}/{n} = {topk/n*100:.1f}%   "
          f"(PRD target >= 85%)")
    print()
    print("  per group - 'hard' avoids the indexed vocabulary, so it is the")
    print("  figure the 85% target should be judged against:")
    for group in sorted(per_group):
        scores = per_group[group]
        g1 = sum(s[0] for s in scores)
        gk = sum(s[1] for s in scores)
        flag = "  <- honest number" if group == "hard" else ""
        print(f"    {group:<14} top-1 {g1}/{len(scores)}  top-{config.TOP_K} {gk}/{len(scores)}"
              f" = {gk/len(scores)*100:5.1f}%{flag}")
    print()

    # Typos must produce the FR-9 question, never a confidently wrong scheme.
    typo_ok = 0
    if typo_cases:
        print("TYPO HANDLING (must ask, never guess a scheme)")
        for case in typo_cases:
            got = detect_scheme(case["query"])
            ok = got == "ambiguous"
            typo_ok += ok
            print(f"  [{'PASS' if ok else 'FAIL'}] {case['query']:<44} -> {got}")
        print()
    if wrong_scheme:
        print(f"wrong-scheme top hits: {wrong_scheme}  <- the near-identical-page risk")
    if latencies:
        print(f"latency median: {statistics.median(latencies):.0f} ms  "
              f"max {max(latencies):.0f} ms  (PRD target < 3000 ms, steady state)")
    print()

    # ---- threshold sweep -------------------------------------------------
    print("THRESHOLD SWEEP (is 0.6 the right cutoff?)")
    print("-" * 104)
    print("  A threshold must do two jobs: keep real answers, reject unknowns.")
    print("  'kept' counts cases whose best distance is at or below the cutoff;")
    print("  'missed' counts golden answers the cutoff would wrongly reject.")
    print()
    print(f"  {'threshold':>10} {'kept':>6} {'missed':>8}   note")
    all_best = []
    for case in retrieval_cases:
        scheme = detect_scheme(case["query"])
        if scheme == "ambiguous":
            continue
        all_best.append(retrieve(case["query"], scheme).best_distance)
    for threshold in [0.3, 0.4, 0.5, 0.55, 0.6, 0.65, 0.7, 0.8, 0.9]:
        kept = sum(1 for d in all_best if d <= threshold)
        missed = len(all_best) - kept
        note = ""
        if missed == 0:
            note = "keeps every golden answer"
        print(f"  {threshold:>10.2f} {kept:>6} {missed:>8}   {note}")
    print()
    worst = max(all_best) if all_best else 0.0
    print(f"  worst golden-answer distance: {worst:.4f}")
    print(f"  current threshold           : {config.DISTANCE_THRESHOLD}")
    if worst < config.DISTANCE_THRESHOLD:
        print("  -> current threshold keeps all golden answers with headroom")
    else:
        print("  -> current threshold would REJECT a real answer; lower it")
    print()

    # ---- scheme detection ------------------------------------------------
    detection_cases = json.loads(GOLDEN.read_text(encoding="utf-8"))["scheme_detection_cases"]
    det_ok = 0
    print("SCHEME DETECTION")
    print("-" * 104)
    for case in detection_cases:
        got = detect_scheme(case["query"])
        want = case["expected"]
        ok = got == want
        det_ok += ok
        print(f"  [{'PASS' if ok else 'FAIL'}] {case['query']:<48} -> {got}"
              + ("" if ok else f"  (expected {want})"))
    print()
    print(f"scheme detection: {det_ok}/{len(detection_cases)} = "
          f"{det_ok/len(detection_cases)*100:.1f}%")
    print()
    print(LINE)

    # The 85% target is judged against the `hard` group only. Scoring it on the
    # close-paraphrase cases would report ~100% and hide the real limit.
    hard_scores = per_group.get("hard", [])
    hard_topk = sum(s[1] for s in hard_scores)
    hard_rate = hard_topk / len(hard_scores) if hard_scores else 0.0
    typo_passed = typo_ok == len(typo_cases)

    passed = hard_rate >= 0.85 and det_ok == len(detection_cases) and typo_passed
    print(f"gate: hard top-{config.TOP_K} >= 85%   -> {hard_rate*100:.1f}%")
    print(f"gate: scheme detection complete      -> {det_ok}/{len(detection_cases)}")
    print(f"gate: typos ask, never guess         -> {typo_ok}/{len(typo_cases)}")
    print("RESULT:", "PASS" if passed else "NEEDS WORK")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
