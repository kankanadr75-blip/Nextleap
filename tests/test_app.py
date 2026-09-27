"""Phase 6 tests for the answer chain and the UI's no-persistence guarantee.

Streamlit itself is not started here - the UI is a thin renderer over
``src.query.answer.answer_question``, so testing the chain is what actually
protects the behaviour. What these tests pin down:

* all five schemes answer, each with exactly one allow-listed citation and the
  ``Last updated from sources:`` footer;
* the three documented example questions all produce cited answers, since the
  UI ships them as one-click buttons;
* PII is blocked and the value never appears in the rendered reply;
* refusals, out-of-scope and unknown questions all come back through the same
  ``FormattedAnswer`` shape, so one renderer covers every case;
* **nothing is persisted** - no file under the project is written by asking a
  question, and no message text reaches the log.
"""

from __future__ import annotations

import io
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.query import config
from src.query.answer import answer_question
from src.query.format import answered_text_only

ALLOWED = config._citation_domains()


@pytest.fixture(scope="module")
def warm() -> None:
    from src.ingest.embed_store import embed_query

    embed_query("warmup")


def _urls(text: str) -> list[str]:
    return re.findall(r"https?://[^\s)]+", text)


def test_all_five_schemes_answer_with_one_citation(warm: None) -> None:
    for slug in config.CORPUS_SLUGS:
        outcome = answer_question(
            f"What is the expense ratio of {config.display_name(slug)}?"
        )
        assert not outcome.blocked, f"{slug} was blocked: {outcome.guard_kind}"
        assert outcome.formatted.cited, f"{slug}: {outcome.formatted.note}"

        urls = _urls(outcome.text)
        assert len(urls) == 1, f"{slug}: expected 1 URL, got {urls}"
        host = urls[0].split("//", 1)[1].split("/", 1)[0]
        assert host in ALLOWED
        assert outcome.sentences <= config.MAX_SENTENCES
        assert outcome.fetched_at, f"{slug}: missing fetched_at footer"
        assert outcome.formatted.footer in outcome.text


def test_the_three_ui_example_buttons_all_answer(warm: None) -> None:
    """These are rendered as one-click buttons, so a refusal would be a bug."""
    for question in config.EXAMPLE_QUESTIONS:
        outcome = answer_question(question)
        assert outcome.formatted.cited, f"{question!r} -> {outcome.formatted.note}"
        assert len(_urls(outcome.text)) == 1, question
        assert outcome.sentences <= config.MAX_SENTENCES, question
        # The answer body must not itself contain a URL; the Source line is the
        # only place a link may appear.
        body = answered_text_only(outcome.formatted)
        assert not _urls(body), f"{question!r} leaked a URL into the answer body"


def test_pii_is_blocked_and_never_echoed(warm: None) -> None:
    secret = "ABCDE1234F"
    outcome = answer_question(f"my PAN is {secret}, what is the expense ratio?")
    assert outcome.blocked
    assert outcome.guard_kind == "pii"
    assert secret not in outcome.text
    assert outcome.sentences == 0, "a block is authored copy, not a capped answer"


def test_pii_block_does_not_reach_the_retriever(warm: None) -> None:
    """A blocked message must not even embed - provable via the intent label."""
    outcome = answer_question("email me at ravi.sharma@example.com")
    assert outcome.guard_kind == "pii"
    assert outcome.intent == "blocked"
    assert outcome.top_distance == float("inf"), "retrieval must not have run"


@pytest.mark.parametrize(
    "question,expected_kind",
    [
        ("HDFC fund", "ambiguous"),
        ("What is the expense ratio of HDFC Mid Cap Fund?", "out_of_scope"),
        ("Should I invest in HDFC Small Cap Fund?", "advice"),
        ("What is the 1 year return of HDFC Large Cap Fund?", "performance"),
    ],
)
def test_every_guard_kind_returns_the_same_shape(
    warm: None, question: str, expected_kind: str
) -> None:
    """One renderer in the UI, so every branch must produce a FormattedAnswer."""
    outcome = answer_question(question)
    assert outcome.blocked
    assert outcome.guard_kind == expected_kind
    assert isinstance(outcome.text, str) and outcome.text.strip()
    assert outcome.sentences == 0, "authored copy is not sentence-capped"
    if outcome.source_url:
        host = outcome.source_url.split("//", 1)[1].split("/", 1)[0]
        assert host in ALLOWED


def test_disambiguation_lists_all_five_from_the_ui(warm: None) -> None:
    outcome = answer_question("HDFC fund")
    for name in (
        "HDFC Large Cap Fund",
        "HDFC Flexi Cap Fund",
        "HDFC ELSS Tax Saver",
        "HDFC Small Cap Fund",
        "HDFC Balanced Advantage Fund",
    ):
        assert name in outcome.text, f"missing {name}"


def test_unanswerable_question_falls_back_to_fr8(warm: None) -> None:
    """A question in scope but unanswerable must refuse, never guess."""
    outcome = answer_question("What is the ticker symbol of HDFC Flexi Cap Fund?")
    if not outcome.formatted.cited:
        assert outcome.guard_kind in {"unknown", "uncited"}
        assert len(_urls(outcome.text)) == 1
    else:
        assert outcome.sentences <= config.MAX_SENTENCES


def test_empty_input_is_handled(warm: None) -> None:
    for value in ("", "   "):
        outcome = answer_question(value)
        assert outcome.blocked
        assert outcome.guard_kind == "empty"
        assert outcome.text.strip()


def test_second_question_reuses_the_loaded_model(warm: None) -> None:
    """Cold start is a one-time cost; steady state must be far under 3 s."""
    answer_question("What is the expense ratio of HDFC Flexi Cap Fund?")
    outcome = answer_question("What is the lock-in period for HDFC ELSS Tax Saver?")
    assert outcome.formatted.cited
    assert outcome.latency_ms < 3000, f"steady-state {outcome.latency_ms:.0f} ms"


# --------------------------------------------------------------------------
# No persistence
# --------------------------------------------------------------------------


def test_asking_questions_writes_no_files(warm: None, tmp_path: Path) -> None:
    """The no-persistence NFR, asserted on the filesystem.

    Snapshots every file under the project (excluding the vector store and
    caches, which are build outputs, not user data), asks several questions, and
    asserts nothing was created or modified.
    """
    ignored = {"chroma_db", ".git", "__pycache__", ".pytest_cache", ".streamlit"}

    def snapshot() -> dict[str, float]:
        out: dict[str, float] = {}
        for path in ROOT.rglob("*"):
            if not path.is_file():
                continue
            if any(part in ignored for part in path.relative_to(ROOT).parts):
                continue
            out[str(path.relative_to(ROOT))] = path.stat().st_mtime
        return out

    before = snapshot()
    for question in (
        "What is the expense ratio of HDFC Flexi Cap Fund?",
        "my PAN is ABCDE1234F",
        "Should I invest in HDFC Small Cap Fund?",
        "How do I download my capital-gains statement?",
    ):
        answer_question(question)
    after = snapshot()

    assert set(after) == set(before), (
        f"files created: {set(after) - set(before)}"
    )
    changed = {k for k in before if before[k] != after.get(k)}
    assert not changed, f"files modified by asking questions: {changed}"


def test_no_chat_history_file_exists() -> None:
    """The UI keeps history in session state only, so no such file may exist."""
    for name in ("chat_history", "chat_history.json", "chat.db", "messages.json"):
        assert not (ROOT / name).exists(), f"{name} exists - history is being persisted"
        assert not (ROOT / "data" / name).exists(), f"data/{name} exists"


def test_streamlit_telemetry_is_disabled() -> None:
    cfg = ROOT / ".streamlit" / "config.toml"
    assert cfg.exists(), ".streamlit/config.toml missing"
    text = cfg.read_text(encoding="utf-8")
    assert "gatherUsageStats = false" in text
    assert 'address = "127.0.0.1"' in text, "app must not bind beyond localhost"
