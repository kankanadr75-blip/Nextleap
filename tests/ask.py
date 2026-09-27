"""Ask the retriever a single question and show what it actually returned.

    python tests/ask.py "expense ratio of flexi cap"
    python tests/ask.py "who looks after the fund" --top-k 8
    python tests/ask.py --scheme hdfc-elss "charges for managing"
    python tests/ask.py                      # interactive REPL

The batch harnesses (``eval_retrieval.py``, ``probe_hard_queries.py``) answer
"how good is retrieval overall". This one answers "why did *this* query rank
*that* chunk first", which is the question you actually have when one phrasing
misbehaves. It shows the ranked hits with cosine distances, the distance
threshold they are judged against, and - when the query is also a case in
``golden.json`` - whether top-1 hit the expected attribute.

Two things this deliberately still does when a guard would have blocked:

* it runs the guard chain and reports the verdict, and
* it runs retrieval anyway and marks the result ``(pipeline refuses)``.

That is the point. When a query is blocked you usually want to know whether
retrieval *would* have found the right chunk - if it would, the guard is too
eager; if it would not, the phrasing is the problem. Suppressing the retrieval
output would hide the half you came to look at.

First run loads the embedding model (~30 s cold), then every query is ~40 ms.
"""

from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Scheme text and the rupee sign are not cp1252-representable, and printing an
# en dash to a default Windows console raises UnicodeEncodeError mid-report.
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from src.ingest.embed_store import embed_query, get_collection
from src.query import config
from src.query.guard import run_guards
from src.query.retrieve import detect_scheme_detail, retrieve

WIDTH = 78
BULLET = "•"


def _rule(char: str = "-") -> str:
    return char * WIDTH


def _kind(detection) -> str:
    kind = detection.kind
    return getattr(kind, "value", str(kind))


def _golden_case(query: str) -> dict | None:
    """The golden.json case for this exact query, if there is one."""
    path = ROOT / "tests" / "golden.json"
    if not path.exists():
        return None
    for case in json.loads(path.read_text(encoding="utf-8")).get("cases", []):
        if case.get("query", "").strip().lower() == query.strip().lower():
            return case
    return None


def ask(
    query: str,
    *,
    top_k: int | None = None,
    threshold: float | None = None,
    scheme: str | None = None,
    full: bool = False,
    collection=None,
) -> dict:
    """Retrieve for one query and return a report dict."""
    detection = detect_scheme_detail(query)
    verdict = run_guards(query, detection)

    # scheme_override lets you ask a bare question against a specific scheme.
    target = scheme or detection.scheme
    result = retrieve(query, target, top_k=top_k, collection=collection)

    limit = config.DISTANCE_THRESHOLD if threshold is None else threshold
    case = _golden_case(query)
    expected = (case or {}).get("expected_attribute")

    top1 = result.chunks[0].attribute if result.chunks else None
    hits = [c for c in result.chunks if c.distance <= limit]
    correct = None if expected is None else top1 == expected

    return {
        "query": query,
        "guard": verdict.kind,
        "detected_scheme": detection.scheme,
        "detection_kind": _kind(detection),
        "matched_alias": detection.matched_alias,
        "searched_scheme": target,
        "filtered": result.filtered,
        "threshold": limit,
        "latency_ms": result.latency_ms,
        "best_distance": result.best_distance,
        "confident": bool(hits),
        "above_threshold": result.best_distance > limit,
        "would_be_blocked": verdict.blocked,
        "golden_group": (case or {}).get("group"),
        "expected_attribute": expected,
        "top1_attribute": top1,
        "top1_correct": correct,
        "hits": [
            {
                "rank": i + 1,
                "id": c.id,
                "distance": c.distance,
                "attribute": c.attribute,
                "scheme": c.scheme,
                "chunk_type": c.chunk_type,
                "within_threshold": c.distance <= limit,
                "source_url": c.source_url,
                "text": c.document,
            }
            for i, c in enumerate(result.chunks)
        ],
        "full": full,
    }


def render(report: dict) -> str:
    out: list[str] = []
    add = out.append

    add(f"\n{_rule('=')}")
    add(f"QUERY  {report['query']}")
    add(_rule("="))

    guard = report["guard"]
    if report["would_be_blocked"]:
        add(f"  guard      {guard}  ->  the pipeline refuses before retrieval")
    else:
        add(f"  guard      {guard}")

    scheme = report["searched_scheme"] or "(none detected -> searched all)"
    alias = f"  alias={report['matched_alias']!r}" if report["matched_alias"] else ""
    add(f"  scheme     {scheme}  ({report['detection_kind']}){alias}")
    if report["filtered"]:
        add("              filtered to that scheme + general chunks")

    verdict = "CONFIDENT" if report["confident"] else "NOT CONFIDENT -> FR-8"
    best = report["best_distance"]
    add(
        f"  threshold  {report['threshold']:.2f}   "
        f"top-1 {best:.3f}   {verdict}"
    )
    add(f"  latency    {report['latency_ms']:.0f} ms")

    if report["golden_group"]:
        add(
            f"  golden     group={report['golden_group']}  "
            f"expect={report['expected_attribute']}"
        )

    hits = report["hits"]
    if not hits:
        add("\n  (no chunks returned at all)")
        return "\n".join(out)

    add("")
    for hit in hits:
        flag = "ok " if hit["within_threshold"] else "far"
        add(
            f"  {hit['rank']}. d={hit['distance']:.3f} [{flag}] "
            f"{hit['attribute']:<20} {hit['scheme']}"
        )
        text = hit["text"] if report["full"] else _clip(hit["text"], WIDTH - 10)
        add(f"     {text}")
        if report["full"]:
            add(f"     id={hit['id']}  type={hit['chunk_type']}")
            add(f"     src={hit['source_url']}")

    correct = report["top1_correct"]
    if correct is True:
        add(f"\n  PASS  top-1 is the expected attribute ({report['expected_attribute']})")
    elif correct is False:
        add(
            f"\n  FAIL  top-1 is {report['top1_attribute']!r}, "
            f"expected {report['expected_attribute']!r}"
        )
        add("        either the threshold is too loose or the two attributes collide")
    else:
        add("\n  (not a golden.json case, so nothing to check top-1 against)")

    if report["would_be_blocked"]:
        add(
            "  note: retrieval above is what the pipeline would NOT have used, "
            "shown for diagnosis"
        )
    return "\n".join(out)


def _clip(text: str, width: int) -> str:
    text = " ".join(text.split())
    return text if len(text) <= width else text[: width - 1] + "…"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Ask the retriever a question and inspect the ranked hits.",
    )
    parser.add_argument("queries", nargs="*", help="one or more queries")
    parser.add_argument("--top-k", type=int, default=None, help=f"default {config.TOP_K}")
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help=f"distance threshold, default {config.DISTANCE_THRESHOLD}",
    )
    parser.add_argument(
        "--scheme",
        default=None,
        help="force the scheme slug instead of detecting one (e.g. hdfc-elss)",
    )
    parser.add_argument(
        "--full", action="store_true", help="print whole chunks, ids and source urls"
    )
    parser.add_argument("--json", action="store_true", help="emit JSON instead of a report")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="exit 1 if any query falls outside the threshold",
    )
    args = parser.parse_args(argv)

    if args.scheme and args.scheme not in config.CORPUS_SLUGS:
        parser.error(
            f"unknown scheme {args.scheme!r}; choose from "
            + ", ".join(config.CORPUS_SLUGS)
        )

    print("Loading the embedding model and index (first run only)...", flush=True)
    embed_query("warmup")
    collection = get_collection()
    total = collection.count()
    print(f"Ready. {total} chunks indexed, top_k={args.top_k or config.TOP_K}, "
          f"threshold={args.threshold or config.DISTANCE_THRESHOLD}")

    common = {
        "top_k": args.top_k,
        "threshold": args.threshold,
        "scheme": args.scheme,
        "full": args.full,
        "collection": collection,
    }

    queries = args.queries
    interactive = not queries
    reports: list[dict] = []

    def run_one(query: str) -> dict:
        report = ask(query, **common)
        reports.append(report)
        if not args.json:
            print(render(report))
        return report

    if interactive:
        print("\nType a question, or 'exit' to quit.\n")
        while True:
            try:
                query = input("> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if query.lower() in {"exit", "quit", "q"}:
                break
            if query:
                run_one(query)

    for query in queries:
        run_one(query)

    if args.json:
        print(json.dumps(reports, indent=2, ensure_ascii=False))

    if not reports:
        print("\nNo queries. Example: python tests/ask.py \"expense ratio of flexi cap\"")
        return 0

    confident = sum(1 for r in reports if r["confident"])
    graded = [r for r in reports if r["top1_correct"] is not None]
    correct = sum(1 for r in graded if r["top1_correct"])
    print(f"\n{_rule('=')}")
    print(
        f"{len(reports)} queries: {confident} within threshold, "
        f"{len(reports) - confident} outside"
    )
    if graded:
        print(f"golden cases: {correct}/{len(graded)} top-1 correct")
    elif not args.json:
        print("(none of these queries are in golden.json, so no pass/fail above)")

    if args.strict and confident != len(reports):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
