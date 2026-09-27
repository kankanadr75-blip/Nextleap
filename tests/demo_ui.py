"""Phase 6 demo - render the UI and print it as a user would see it.

Unit tests assert properties, not prose. Guard copy and the answer layout are
exactly the kind of thing that passes every assertion and still reads wrong -
a duplicated link, a mangled sentence, a chip showing a truncated name. This
drives the real ``app.py`` through Streamlit's headless harness and prints the
rendered element tree, so the output can be read the way a user reads it.

    python tests/demo_ui.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Scheme names contain an en dash. On a default Windows console (cp1252/cp437)
# printing one raises UnicodeEncodeError and kills the run mid-demo, which looks
# exactly like an app failure. Force UTF-8 output instead.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from streamlit.testing.v1 import AppTest

from src.query import config

APP = ROOT / "app.py"
TIMEOUT = 300

SCRIPT = [
    # One per scheme, so a single silent gap in the corpus cannot hide.
    "What is the expense ratio of HDFC Large Cap Fund?",
    "What is the expense ratio of HDFC Flexi Cap Fund?",
    "What is the lock-in period for HDFC ELSS Tax Saver?",
    "Who manages HDFC Small Cap Fund?",
    "What is the minimum SIP for HDFC Balanced Advantage Fund?",
    # The two remaining example questions, by click.
    "How do I download my capital-gains statement?",
    # Every guard branch.
    "Should I invest in HDFC Small Cap Fund?",
    "What is the 1 year return of HDFC Large Cap Fund?",
    "What is the NAV of HDFC Flexi Cap Fund?",
    "What is the expense ratio of HDFC Mid Cap Fund?",
    "HDFC fund",
    "my PAN is ABCDE1234F and my phone is 9876543210, what is the exit load?",
    "What is the ticker symbol of HDFC Flexi Cap Fund?",
]


def _rule(char: str = "-", width: int = 78) -> str:
    return char * width


def _show(label: str, question: str, reply: str, caption: str) -> None:
    print(f"\n{_rule('=')}")
    print(f"USER  ({label}): {question}")
    print(_rule("="))
    # A source link and the date are rendered as their own elements, so print
    # them the way the browser lays them out: body, then Source, then date.
    for line in reply.splitlines():
        print(f"  {line}" if line.strip() else "")
    if caption:
        print(f"  ({caption})")


def main() -> int:
    at = AppTest.from_file(str(APP), default_timeout=TIMEOUT).run()
    if at.exception:
        print(f"app.py failed to load:\n{at.exception}")
        return 1

    # First-run state: the elements a user sees before typing anything.
    print(_rule("="))
    print("FIRST LOAD - what the user sees before asking anything")
    print(_rule("="))
    print(f"  {at.title[0].value}")
    for element in at.markdown:
        print(f"  {element.value}")
    for caption in at.caption[: len(config.CORPUS_SLUGS)]:
        print(f"  [chip] {caption.value}")
    print("  [buttons]")
    for button in at.button:
        print(f"    {button.label}")
    print(f"  {at.caption[-2].value}")
    print(f"  {at.caption[-1].value}")

    sent = 0
    for question in SCRIPT:
        at = AppTest.from_file(str(APP), default_timeout=TIMEOUT).run()
        at.chat_input[0].set_value(question).run()
        if at.exception:
            print(f"\n!! {question!r} raised: {at.exception}")
            return 1

        outcome = at.session_state["messages"][-1]["outcome"]
        assistant_bubble = next(
            b for b in at.chat_message if getattr(b, "name", "") == "assistant"
        )
        reply = "\n".join(str(m.value) for m in assistant_bubble.markdown)
        caption = " ".join(str(c.value) for c in assistant_bubble.caption)

        label = f"answered/{outcome.sentences}s" if not outcome.blocked else outcome.guard_kind
        _show(label, question, reply, caption)
        sent += 1

    print(f"\n{_rule('=')}")
    print(f"{sent} exchanges rendered, no exceptions, no persistence")

    # A multi-turn transcript, to confirm the history replays without
    # duplicating a turn and that the new reply is drawn exactly once.
    print(_rule("="))
    print("MULTI-TURN - does the transcript replay cleanly?")
    print(_rule("="))
    at = AppTest.from_file(str(APP), default_timeout=TIMEOUT).run()
    at.button[0].click().run()
    at.chat_input[0].set_value("Who manages HDFC Small Cap Fund?").run()
    at.button[1].click().run()
    if at.exception:
        print(f"multi-turn raised: {at.exception}")
        return 1
    print(f"  turns stored: {len(at.session_state['messages'])}")
    for i, bubble in enumerate(at.chat_message):
        body = " | ".join(str(m.value) for m in bubble.markdown)
        print(f"  [{i}] {bubble.name:>9}: {body[:96]}")

    # One Source line per assistant turn, and no turn drawn twice.
    sources = [str(t.value) for t in at.markdown if str(t.value).startswith("Source: ")]
    turns = len(at.session_state["messages"])
    print(f"  assistant turns: {turns}, Source lines: {len(sources)}")
    if len(sources) != turns:
        print(f"  !! expected exactly one Source line per turn, got {len(sources)}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
