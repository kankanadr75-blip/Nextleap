"""HDFC Mutual Fund FAQ Assistant - chat UI.

    streamlit run app.py

Runs entirely locally. The only network call this project ever makes is to an
LLM provider, and only if ``LLM_PROVIDER`` is set; by default the offline
``StubLLM`` answers from the indexed chunks, so the app works with no
credentials.

Privacy properties, both deliberate:

* **No chat-history database.** History lives in ``st.session_state`` and dies
  with the browser session. Nothing is written to disk, and no layer logs
  message text - the guards, retriever and generator log coarse labels only.
* **Telemetry off** in ``.streamlit/config.toml``, so nothing leaves the machine.

The model and the Chroma client are loaded once per server process via
``@st.cache_resource``; a cold start costs ~30 s, after which answers are fast.
"""

from __future__ import annotations

import streamlit as st

from src.query import config
from src.query.answer import AnswerOutcome, answer_question

st.set_page_config(
    page_title="HDFC Mutual Fund FAQ Assistant",
    page_icon="📊",
    layout="centered",
)

@st.cache_resource(show_spinner="Loading the local model and index (first run only, ~30 s)...")
def warm_resources() -> object:
    """Load the embedding model and open the Chroma collection exactly once.

    ``cached_resource`` rather than ``cached_data`` because these are stateful
    clients, not data - they must be shared, not copied per rerun.
    """
    from src.ingest.embed_store import embed_query, get_collection

    embed_query("warmup")
    return get_collection()


def _allowlisted(url: str) -> bool:
    """Only allow-listed hosts become clickable links.

    ``format.py`` already enforces this, so this is a second, independent gate
    on the rendering path - model output should never be able to inject a link
    into the UI even if a future refactor loosens the formatter.
    """
    if not url.startswith("http"):
        return False
    host = url.split("//", 1)[1].split("/", 1)[0].lower()
    return host in config._citation_domains()


def _render(outcome: AnswerOutcome) -> None:
    """One layout for every reply kind: answered, refused, blocked or unknown.

    Answer text, then a clickable ``Source:`` link, then the
    ``Last updated from sources:`` date in smaller text.
    """
    formatted = outcome.formatted
    st.markdown(formatted.text.split("\nSource: ")[0].strip() or formatted.text)

    if outcome.source_url:
        if _allowlisted(outcome.source_url):
            st.markdown(f'Source: [{outcome.source_url}]({outcome.source_url})')
        else:
            # Should be unreachable: format.py refuses non-allow-listed URLs.
            st.markdown("Source: unavailable")

    if outcome.fetched_at:
        st.caption(f"Last updated from sources: {outcome.fetched_at}")


# --------------------------------------------------------------------------
# Header
# --------------------------------------------------------------------------

st.title("HDFC Mutual Fund FAQ Assistant")
st.markdown(config.WELCOME_LINE)

# Scope chips: the five schemes this assistant covers, and nothing else.
st.markdown("**Scope — these 5 schemes only**")
chips = st.columns(5)
for col, slug in zip(chips, config.CORPUS_SLUGS):
    col.caption(config.base_name(slug))

# --------------------------------------------------------------------------
# Example questions
# --------------------------------------------------------------------------

st.markdown("**Try one of these**")
example_cols = st.columns(len(config.EXAMPLE_QUESTIONS))
for col, question in zip(example_cols, config.EXAMPLE_QUESTIONS):
    if col.button(question, key=f"example_{question[:24]}", use_container_width=True):
        st.session_state["pending"] = question

# Read `pending` only now. A button click handler runs while the buttons render,
# so a read at the top of the script would always see the *previous* run's value
# and the example buttons would lag a rerun behind.
# `st.session_state` is not a MutableMapping in Streamlit 1.45, so this is an
# explicit get rather than a `.pop()`; the key is deleted once it is answered.
_pending = st.session_state.get("pending")

# --------------------------------------------------------------------------
# Chat history - session state only, never persisted (see module docstring)
# --------------------------------------------------------------------------
#
# Streamlit re-executes this whole script on every interaction, so earlier turns
# are replayed from `messages` and the new one is drawn inline. The invariant
# that makes each turn render exactly once is that `messages` only ever holds
# *completed* exchanges: a new turn is appended after its answer is drawn, never
# before. A `st.rerun()` here would re-enter the script and draw the same reply
# a second time via the loop above - which is exactly what it did.

st.session_state.setdefault("messages", [])

for entry in st.session_state["messages"]:
    if entry["outcome"] is None:
        # Unreachable under the invariant above, but a browser session that
        # predates it can still hold a half-written turn, and a crash loop here
        # would be far worse than skipping one bubble.
        continue
    with st.chat_message("user"):
        st.markdown(entry["question"])
    with st.chat_message("assistant"):
        _render(entry["outcome"])

typed = st.chat_input(
    placeholder="Ask a factual question about one of these five schemes"
)
message = typed or _pending

if _pending is not None:
    # Consume the clicked example button, so the same question is not re-sent on
    # the next interaction.
    del st.session_state["pending"]

if message:
    with st.chat_message("user"):
        st.markdown(message)

    with st.chat_message("assistant"):
        with st.spinner("Checking sources..."):
            outcome = answer_question(message, collection=warm_resources())
        _render(outcome)

    # An outcome held in session state, not a database row.
    st.session_state["messages"].append({"question": message, "outcome": outcome})

# --------------------------------------------------------------------------
# Persistent notices
# --------------------------------------------------------------------------

st.divider()
st.markdown(f"**{config.FACTS_ONLY_NOTE}**")
st.caption(config.PII_INPUT_HINT)
st.caption(
    "Answers are drawn from indexed scheme documents and always cite a source. "
    "I don't have return, NAV or performance figures."
)
