"""Phase 5 end-to-end demo - runs fully offline with the stub LLM.

    python tests/demo_answers.py

Shows the full chain (guard -> retrieve -> generate -> format) for one question
from each supported category, so the output can be eyeballed for the PRD rules:
<=3 sentences, exactly one citation, the fetched_at footer, and no advice,
no returns, no PII echoed back.
"""

from __future__ import annotations

import io
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from src.ingest.embed_store import embed_query
from src.query import config
from src.query.format import format_answer, format_blocked, answered_text_only, split_sentences
from src.query.generate import generate_answer
from src.query.guard import run_guards
from src.query.llm import get_llm
from src.query.retrieve import detect_scheme_detail, retrieve

QUESTIONS = [
    ("factual", "What is the expense ratio of HDFC Flexi Cap Fund?"),
    ("factual", "What is the lock-in period for HDFC ELSS Tax Saver?"),
    ("factual", "Who manages HDFC Small Cap Fund?"),
    ("factual", "minimum SIP for HDFC Large Cap Fund"),
    ("statement", "How do I download my capital-gains statement?"),
    ("ambiguity", "HDFC fund"),
    ("out-of-scope", "What is the expense ratio of HDFC Mid Cap Fund?"),
    ("out-of-scope", "Tell me about SBI Bluechip Fund"),
    ("performance", "What is the 1 year return of HDFC Large Cap Fund?"),
    ("advice", "Should I invest in HDFC Small Cap Fund?"),
    ("pii", "my PAN is ABCDE1234F, what is the expense ratio?"),
]


def main() -> int:
    embed_query("warmup")
    llm = get_llm()
    print(f"provider: {llm.name}   (no API key required)")
    print("=" * 96)

    for label, question in QUESTIONS:
        detection = detect_scheme_detail(question)
        verdict = run_guards(question, detection)

        if verdict.blocked:
            out = format_blocked(verdict.message, link=verdict.link)
            print(f"\n[{label}] {question}")
            print(f"  -> guard: {verdict.kind}")
            for line in out.text.splitlines():
                print(f"  | {line}")
            assert "ABCDE1234F" not in out.text, "PII echoed back!"
            continue

        result = retrieve(question, detection.scheme)
        if not result.confident:
            out = format_blocked(
                config.UNKNOWN_TEMPLATE.format(
                    scheme=detection.scheme or "HDFC Mutual Fund"
                ),
                link=config.REFUSAL_LINK,
            )
            print(f"\n[{label}] {question}")
            print(f"  -> below threshold (d={result.best_distance:.3f}), FR-8")
            for line in out.text.splitlines():
                print(f"  | {line}")
            continue

        generation = generate_answer(question, result.chunks, llm)
        out = format_answer(generation, scheme_for_unknown=detection.scheme or "")

        print(f"\n[{label}] {question}")
        print(f"  -> cited={out.cited} sentences={out.sentences} "
              f"provider={generation.provider} note={out.note or '-'}")
        for line in out.text.splitlines():
            print(f"  | {line}")

        # Assert the PRD rules on real output rather than trusting the code path.
        urls = re.findall(r"https?://[^\s)]+", out.text)
        assert len(urls) == 1, f"expected 1 URL, got {urls}"
        host = urls[0].split("//", 1)[1].split("/", 1)[0]
        assert host in config._citation_domains(), f"{host} not allow-listed"
        assert out.sentences <= config.MAX_SENTENCES
        assert len(split_sentences(answered_text_only(out))) <= config.MAX_SENTENCES
        if out.cited:
            assert out.footer in out.text

    print("\n" + "=" * 96)
    print("all PRD rules held on every answered reply: 1 URL (allow-listed), "
          "<=3 sentences, fetched_at footer")
    return 0


if __name__ == "__main__":
    sys.exit(main())
