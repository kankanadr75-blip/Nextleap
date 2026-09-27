"""Phase 6 UI test - executes app.py itself with Streamlit's headless harness.

`streamlit run app.py` returning HTTP 200 proves nothing: the root response is
the static shell, and the script only executes when a websocket client
connects. ``streamlit.testing.v1.AppTest`` runs the real script and exposes the
rendered widget tree, so these assertions are about what a user would actually
see.

Covered: the verbatim welcome line, the five scope chips, three working example
buttons that each produce a cited <=3-sentence answer, the persistent
facts-only note, the PII input hint, and a blocked PII reply that echoes
nothing back.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from streamlit.testing.v1 import AppTest

from src.query import config

APP = ROOT / "app.py"
ALLOWED = config._citation_domains()
TIMEOUT = 300


def _all_text(at: AppTest) -> str:
    """Every rendered string in the app.

    ``at.markdown`` already includes markdown inside chat_message containers, so
    this must not also walk the bubbles - doing both counts every line twice.
    """
    parts: list[str] = []
    for element in list(at.markdown) + list(at.caption) + list(at.title) + list(at.info):
        parts.append(str(getattr(element, "value", "")))
    return "\n".join(parts)


def _assistant_text(at: AppTest) -> str:
    """Only the assistant's replies, for asserting what the model side emitted."""
    return "\n".join(
        str(getattr(el, "value", ""))
        for bubble in at.chat_message
        if getattr(bubble, "name", "") == "assistant"
        for el in list(bubble.markdown) + list(bubble.caption)
    )


def _urls(text: str) -> list[str]:
    """Distinct URLs in rendered markdown.

    A markdown link ``[label](target)`` is one link, not two, so the label is
    collapsed first - otherwise the visible URL and the href both match.
    """
    text = re.sub(r"\[([^\]]*)\]\((https?://[^)]+)\)", r"\1", text)
    return sorted(set(re.findall(r"https?://[^\s)\]]+", text)))


@pytest.fixture(scope="module")
def app() -> AppTest:
    at = AppTest.from_file(str(APP), default_timeout=TIMEOUT).run()
    assert not at.exception, f"app.py raised on load: {at.exception}"
    return at


def test_app_loads_without_exception(app: AppTest) -> None:
    assert not app.exception
    assert app.title[0].value == "HDFC Mutual Fund FAQ Assistant"


def test_welcome_line_is_verbatim(app: AppTest) -> None:
    assert config.WELCOME_LINE in _all_text(app)
    assert (
        "Hi! I answer factual questions about 5 HDFC Mutual Fund schemes, "
        "with a source for every answer." in _all_text(app)
    )


def test_scope_row_shows_all_five_schemes(app: AppTest) -> None:
    text = _all_text(app)
    for slug in config.CORPUS_SLUGS:
        # Chips drop the plan suffix, so match the distinguishing name only.
        base = config.display_name(slug).split(" – ")[0]
        assert base in text, f"scope chip missing {base}"


def test_three_example_buttons_exist(app: AppTest) -> None:
    labels = [b.label for b in app.button]
    assert len(labels) == 3, f"expected 3 example buttons, got {labels}"
    for question in config.EXAMPLE_QUESTIONS:
        assert question in labels, f"missing example button {question!r}"


def test_facts_only_note_and_pii_hint_are_persistent(app: AppTest) -> None:
    text = _all_text(app)
    assert config.FACTS_ONLY_NOTE in text
    assert config.PII_INPUT_HINT in text


@pytest.mark.parametrize("index", [0, 1, 2])
def test_each_example_button_produces_a_cited_answer(index: int) -> None:
    """The documented acceptance check: all three buttons answer with a source."""
    question = config.EXAMPLE_QUESTIONS[index]
    at = AppTest.from_file(str(APP), default_timeout=TIMEOUT).run()
    at.button[index].click().run()
    assert not at.exception, f"{question!r} raised: {at.exception}"

    text = _all_text(at)
    urls = _urls(text)
    assert len(urls) == 1, f"{question!r}: expected 1 URL, got {urls}"
    host = urls[0].split("//", 1)[1].split("/", 1)[0]
    assert host in ALLOWED, f"{question!r}: {host} not allow-listed"

    outcome = at.session_state["messages"][-1]["outcome"]
    assert f"Last updated from sources: {outcome.fetched_at}" in text
    assert outcome.formatted.cited
    assert outcome.sentences <= config.MAX_SENTENCES
    # The user's question is echoed in the transcript.
    assert question in text
    # Exactly one Source line, not one per rerun.
    assert text.count("Source: ") == 1, "the source link was rendered more than once"


def test_pii_typed_into_the_ui_is_blocked_and_not_echoed() -> None:
    secret = "ABCDE1234F"
    at = AppTest.from_file(str(APP), default_timeout=TIMEOUT).run()
    at.chat_input[0].set_value(f"my PAN is {secret}, what is the expense ratio?").run()
    assert not at.exception, at.exception

    outcome = at.session_state["messages"][-1]["outcome"]
    assert outcome.blocked and outcome.guard_kind == "pii"

    text = _all_text(at)
    assert "personal data" in text

    # The user's own message legitimately shows in the transcript, but the
    # assistant's reply must not contain it: the PII value never reaches the
    # retriever or the model, so it cannot come back out.
    assistant_text = _assistant_text(at)
    assert assistant_text.strip(), "no assistant reply was rendered"
    assert secret not in assistant_text, "the PAN was echoed back by the assistant"


def test_no_history_is_persisted_by_a_ui_exchange() -> None:
    """Session state only - asking a question must not write anything to disk."""
    ignored = {"chroma_db", ".git", "__pycache__", ".pytest_cache", ".streamlit"}

    def snapshot() -> set[str]:
        return {
            str(p.relative_to(ROOT))
            for p in ROOT.rglob("*")
            if p.is_file()
            and not any(part in ignored for part in p.relative_to(ROOT).parts)
        }

    before = snapshot()
    at = AppTest.from_file(str(APP), default_timeout=TIMEOUT).run()
    at.button[0].click().run()
    at.chat_input[0].set_value("Who manages HDFC Small Cap Fund?").run()
    assert not at.exception
    assert len(at.session_state["messages"]) == 2, "history lives in session state"
    assert snapshot() == before, "the UI wrote a file to disk"
