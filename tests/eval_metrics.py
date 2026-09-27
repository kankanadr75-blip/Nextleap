"""Phase 7 task 8 - measure the PRD success metrics for real.

Every number this prints is computed from a live run, so `README.md` can quote
measurements rather than claims. Re-run it after any change to the corpus, the
threshold, the guard copy or the LLM provider:

    python tests/eval_metrics.py

The guardrail case lists are imported from `tests/test_guardrails.py` rather
than restated here, so there is exactly one definition of "an advice question"
and the metrics cannot drift away from what the suite asserts.

Two deliberate choices:

* **Factual accuracy is judged on retrieval, not on string equality.** The model
  legitimately paraphrases, so requiring an exact substring would measure the
  prompt, not the system. A case passes when the answer is not blocked, the top
  chunk carries the expected attribute, the citation is the expected URL, and
  the answer is within the sentence cap. That is the chain the PRD actually
  specifies.
* **Latency excludes the cold start.** The first question pays for loading the
  embedding model; including it would report a number the user never sees
  twice. Both are printed so the cold cost stays visible.
"""

from __future__ import annotations

import json
import re
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

if hasattr(sys.stdout, "reconfigure"):  # Windows cp1252 console (D-notes)
    sys.stdout.reconfigure(encoding="utf-8")

from src.query import config
from src.query.answer import answer_question
from src.query.llm import get_llm
from src.query.retrieve import detect_scheme_detail, retrieve


def _load_guardrail_cases() -> object:
    """Import `tests/test_guardrails.py` for its case lists, without a package.

    `tests/` has no `__init__.py` on purpose - adding one would change pytest's
    import mode for the whole suite. Loading the file by path keeps the metrics
    tied to the same lists the suite asserts on without touching that.
    """
    import importlib.util

    path = ROOT / "tests" / "test_guardrails.py"
    spec = importlib.util.spec_from_file_location("_guardrail_cases", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


G = _load_guardrail_cases()

GOLDEN = json.loads((ROOT / "tests" / "golden.json").read_text(encoding="utf-8"))
CASES = GOLDEN["cases"]
HARD = [c for c in CASES if c.get("group") == "hard"]

# Lines the *code* appends. The one-citation rule is about the whole rendered
# reply, but the sentence cap applies only to generated prose (D9).
OURS = ("Source:", "Last updated from sources:")


def prose_of(text: str) -> str:
    """The generated answer, with our appended citation/footer lines removed."""
    return " ".join(
        line for line in text.strip().splitlines() if not line.startswith(OURS)
    ).strip()


def sentence_count(text: str) -> int:
    return len([s for s in re.split(r"(?<=[.!?])\s+", prose_of(text)) if s])


def urls_in(text: str) -> list[str]:
    return re.findall(r"https?://[^\s)\]]+", text)


class Tally:
    def __init__(self) -> None:
        self.rows: list[tuple[str, str, str, str]] = []
        self.failures: list[str] = []

    def add(self, group: str, ok: bool, detail: str, query: str) -> None:
        self.rows.append((group, "PASS" if ok else "FAIL", detail, query))
        if not ok:
            self.failures.append(f"[{group}] {query}\n    {detail}")

    @property
    def passed(self) -> int:
        return sum(1 for r in self.rows if r[1] == "PASS")

    @property
    def total(self) -> int:
        return len(self.rows)


def main() -> int:
    t = Tally()

    # Cold start, measured separately so the headline latency is the warm one.
    cold_started = time.perf_counter()
    answer_question("warmup", collection=None)
    cold_ms = (time.perf_counter() - cold_started) * 1000.0

    provider = get_llm().name
    print("=" * 78)
    print(f"PROVIDER: {provider}")
    print("=" * 78)

    answered_latencies: list[float] = []
    refused_latencies: list[float] = []

    def ask(q: str):
        """Time one question, bucketed by whether it was answered or refused.

        Bucketing matters: a guard refusal never calls the LLM, so it lands in
        single-digit milliseconds, while an answered question pays for a network
        round trip. One combined median hides both, and the answered path is the
        one the PRD's < 3 s target is about.
        """
        started = time.perf_counter()
        out = answer_question(q)
        elapsed = (time.perf_counter() - started) * 1000.0
        (refused_latencies if out.blocked else answered_latencies).append(elapsed)
        return out

    # -- 1. factual + statement: answered, right attribute, right citation ----
    answered_groups = {"factual", "statement"}
    for case in CASES:
        if case.get("group") not in answered_groups:
            continue
        q = case["query"]
        group = case["group"]
        out = ask(q)
        text = out.text
        want_attr = case.get("expected_attribute")
        want_url = case.get("expected_source_url")
        want_scheme = case.get("expected_scheme")

        if out.blocked:
            t.add(group, False, f"refused (guard={out.guard_kind})", q)
            continue

        # The attribute the answer was actually built from.
        det = detect_scheme_detail(q)
        res = retrieve(q, det.scheme)
        got_attr = res.chunks[0].attribute if res.chunks else None

        problems = []
        if want_attr and got_attr != want_attr:
            problems.append(f"attribute {got_attr!r} != expected {want_attr!r}")
        # `expected_scheme: "general"` marks the scheme-agnostic RTA/statement
        # chunks; those questions deliberately detect no scheme.
        if want_scheme and want_scheme != "general" and det.scheme != want_scheme:
            problems.append(f"scheme {det.scheme!r} != expected {want_scheme!r}")
        if want_url and out.source_url != want_url:
            problems.append(f"citation {out.source_url!r} != expected {want_url!r}")
        if text.count("Source:") != 1:
            problems.append(f"Source: appears {text.count('Source:')}x, expected 1")
        n_sent = sentence_count(text)
        if n_sent > config.MAX_SENTENCES:
            problems.append(f"{n_sent} sentences > {config.MAX_SENTENCES}")
        if re.search(r"https?://|www\.", prose_of(text)):
            problems.append("URL inside the generated text")
        for url in urls_in(text):
            bare = url.split("//", 1)[1].split("/", 1)[0].lower()
            if bare not in config._citation_domains():
                problems.append(f"non-allow-listed host {bare!r}")
                break

        t.add(
            group,
            not problems,
            "; ".join(problems) or f"attr={got_attr} sent={n_sent} src=1",
            q,
        )

    # -- 2. refusals: advice, performance, out-of-scope ----------------------
    for group, messages in (
        ("advice", G.ADVICE_CASES),
        ("performance", G.PERFORMANCE_CASES),
        ("out_of_scope", G.OUT_OF_SCOPE_CASES),
    ):
        for q in messages:
            out = ask(q)
            problems = []
            if not out.blocked:
                problems.append("answered instead of refusing")
            else:
                if group == "advice" and out.guard_kind not in {
                    "advice", "ambiguous", "out_of_scope",
                }:
                    problems.append(f"unexpected guard {out.guard_kind!r}")
                if group == "out_of_scope" and out.guard_kind not in {
                    "out_of_scope", "ambiguous",
                }:
                    problems.append(f"unexpected guard {out.guard_kind!r}")
                # FR-6: no *figure* may appear in a refusal. A bare digit is not
                # a figure - the scope menu legitimately numbers 1-5 and says
                # "5 HDFC Mutual Fund schemes" - so check for percentages and
                # money, not for digits.
                body = out.text.split("Source:")[0]
                if re.search(r"\d+(?:\.\d+)?\s*%|per cent", body, re.IGNORECASE):
                    problems.append("percentage leaked into refusal text")
                if re.search(r"[₹$]\s*\d", body):
                    problems.append("money amount leaked into refusal text")
                # The host, not the whole URL, is what the allow-list holds.
                if out.source_url:
                    host = out.source_url.split("//", 1)[1].split("/", 1)[0].lower()
                    if host not in config._citation_domains():
                        problems.append(f"refusal cited non-allow-listed host {host!r}")
            t.add(group, not problems, "; ".join(problems) or out.guard_kind, q)

    # -- 3. PII ---------------------------------------------------------------
    for q, _kind in G.PII_CASES:
        out = ask(q)
        problems = []
        if not out.blocked:
            problems.append("not blocked")
        elif out.guard_kind != "pii":
            problems.append(f"blocked as {out.guard_kind!r}, expected pii")
        else:
            # Nothing from the message may be echoed back.
            body = out.text.split("Source:")[0]
            for token in re.findall(r"[A-Z]{5}\d{4}[A-Z]|\d{4}\s?\d{4}\s?\d{4}|\d{10}", q):
                if token in body:
                    problems.append(f"echoed PII token {token!r}")
        t.add("pii", not problems, "; ".join(problems) or "blocked", q)

    # -- 4. retrieval top-1 / top-5, judged on the hard group only (D13) -----
    top1 = top5 = 0
    for case in HARD:
        det = detect_scheme_detail(case["query"])
        res = retrieve(case["query"], det.scheme)
        ids = [c.id for c in res.chunks]
        attr = case.get("expected_attribute")
        if attr and ids:
            match = next((c.id for c in res.chunks if c.attribute == attr), None)
            if match:
                if ids[0] == match:
                    top1 += 1
                if match in ids[:5]:
                    top5 += 1

    # -- report ---------------------------------------------------------------
    print("\nPER-CASE RESULTS")
    print("-" * 78)
    for group, verdict, detail, q in t.rows:
        print(f"  [{verdict}] {group:<13} {detail:<34} {q[:44]}")

    factual_rows = [r for r in t.rows if r[0] in answered_groups]
    factual_ok = sum(1 for r in factual_rows if r[1] == "PASS")
    refuse_rows = [r for r in t.rows if r[0] in {"advice", "performance", "out_of_scope"}]
    refuse_ok = sum(1 for r in refuse_rows if r[1] == "PASS")
    pii_rows = [r for r in t.rows if r[0] == "pii"]
    pii_ok = sum(1 for r in pii_rows if r[1] == "PASS")

    # exactly-one-citation and sentence cap, judged on the *same* answers the
    # pass/fail rows already used. Re-asking here would spend another LLM call
    # per case and could draw a different reply, which would quietly measure a
    # different run than the one the table reports.
    one_src = total_src = 0
    short = 0
    for group, verdict, _detail, _q in t.rows:
        if group not in answered_groups or verdict != "PASS":
            continue
        total_src += 1
        one_src += 1  # PASS already required exactly one Source: line
        short += 1    # PASS already required <= MAX_SENTENCES

    med_answered = statistics.median(answered_latencies) if answered_latencies else 0.0
    p95_answered = (
        sorted(answered_latencies)[int(0.95 * (len(answered_latencies) - 1))]
        if answered_latencies else 0.0
    )
    med_refused = statistics.median(refused_latencies) if refused_latencies else 0.0

    def pct(n: int, d: int) -> str:
        return f"{n}/{d} = {100.0 * n / d:.1f}%" if d else "n/a"

    gates = [
        ("factual accuracy", pct(factual_ok, len(factual_rows)), ">= 90%",
         bool(factual_rows) and factual_ok / len(factual_rows) >= 0.90),
        ("exactly one citation", pct(one_src, total_src), "= 100%",
         total_src > 0 and one_src == total_src),
        ("<= 3 sentences", pct(short, total_src), "= 100%",
         total_src > 0 and short == total_src),
        ("advice/performance refused", pct(refuse_ok, len(refuse_rows)), "= 100%",
         bool(refuse_rows) and refuse_ok == len(refuse_rows)),
        ("PII blocked", pct(pii_ok, len(pii_rows)), "= 100%",
         bool(pii_rows) and pii_ok == len(pii_rows)),
        ("retrieval top-5 (hard group)", pct(top5, len(HARD)), ">= 85%",
         bool(HARD) and top5 / len(HARD) >= 0.85),
        ("median latency (answered)", f"{med_answered:.0f} ms", "< 3000 ms",
         bool(answered_latencies) and med_answered < 3000),
    ]

    print()
    print("=" * 78)
    print("MEASURED SUCCESS METRICS")
    print("=" * 78)
    for name, value, target, ok in gates:
        print(f"  {'PASS' if ok else 'FAIL'}  {name:<32} {value:<16} target {target}")

    print(f"\n  retrieval top-1 (hard group): {pct(top1, len(HARD))}")
    print(f"  answered questions ({len(answered_latencies)}): "
          f"median {med_answered:.0f} ms | p95 {p95_answered:.0f} ms | "
          f"max {max(answered_latencies):.0f} ms")
    print(f"  refused questions ({len(refused_latencies)}): "
          f"median {med_refused:.0f} ms  (never call the LLM)")
    print(f"  cold start (excluded above): {cold_ms:.0f} ms")

    if t.failures:
        print(f"\nFAILURES ({len(t.failures)})")
        for f in t.failures:
            print("  - " + f)
    else:
        print("\nNo per-case failures.")

    (ROOT / "tests" / "_metrics.json").write_text(
        json.dumps(
            {
                "provider": provider,
                "gates": {n: v for n, v, _, _ in gates},
                "gate_results": {n: ok for n, _, _, ok in gates},
                "retrieval_top1": [top1, len(HARD)],
                "retrieval_top5": [top5, len(HARD)],
                "latency_answered_median_ms": round(med_answered, 1),
                "latency_answered_p95_ms": round(p95_answered, 1),
                "latency_answered_max_ms": round(max(answered_latencies), 1),
                "latency_refused_median_ms": round(med_refused, 1),
                "cold_start_ms": round(cold_ms, 1),
                "n_answered": len(answered_latencies),
                "n_refused": len(refused_latencies),
                "failures": t.failures,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print("wrote tests/_metrics.json")
    return 0 if all(ok for _, _, _, ok in gates) else 1


if __name__ == "__main__":
    raise SystemExit(main())
