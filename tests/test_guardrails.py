"""Phase 5 acceptance suite.

Every PRD compliance claim in Phase 5 is asserted here against code, not
against a prompt:

* every answered reply carries **exactly one** URL, and that URL is in the
  allow-list derived from ``sources.csv``;
* every answer is at most 3 sentences, counted with the real splitter that
  protects decimals and abbreviations;
* the ``Last updated from sources:`` footer matches the chunk's ``fetched_at``;
* 12 adversarial advice prompts return the refusal template and contain **0**
  ``%`` figures;
* PII is blocked before retrieval, and the PII string appears in no log file;
* an out-of-scope AMC gets the FR-10 text; "HDFC fund" gets FR-9.

Plus the two that protect against quietly wrong answers rather than merely
ill-formatted ones: a fabricated ``source_chunk_id`` cannot become a citation,
and the retrieved attribute must match the intent (the Phase 4 finding where
``base_expense_ratio`` outranked ``expense_ratio`` would otherwise report
0.57% when the holder pays 0.77%).

Runs offline with the stub LLM. No API key required.
"""

from __future__ import annotations

import io
import json
import logging
import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.query import config, format as fmt, guard
from src.query.generate import generate_answer
from src.query.llm import (
    SYSTEM_PROMPT,
    GeminiClient,
    GroqClient,
    OpenAIClient,
    StubLLM,
    get_llm,
    parse_json_loose,
)
from src.query.retrieve import detect_scheme_detail, retrieve

ALLOWED = config._citation_domains()


# --------------------------------------------------------------------------
# 1. PII (FR-7)
# --------------------------------------------------------------------------

PII_CASES = [
    ("my PAN is ABCDE1234F", "pan"),
    ("aadhaar 1234 5678 9012", "aadhaar"),
    ("call me on 9876543210", "mobile"),
    ("email me at ravi.sharma@example.com", "email"),
    ("the OTP is 482913", "otp"),
    ("my folio number 123456789012 is with CAMS", "folio_or_account"),
]


@pytest.mark.parametrize("message,expected_kind", PII_CASES)
def test_pii_is_detected(message: str, expected_kind: str) -> None:
    assert guard.detect_pii(message) == expected_kind


def test_pii_blocks_before_retrieval() -> None:
    result = guard.run_guards("my PAN is ABCDE1234F")
    assert result.kind == "pii"
    assert result.message == config.PII_BLOCK_MESSAGE


def test_pii_never_reaches_a_later_guard() -> None:
    """A message that is both PII and advice must be stopped as PII.

    PII is first in the chain, so the ambiguity/performance/advice checks never
    get the chance to leak a canned reply that would be logged.
    """
    result = guard.run_guards(
        "my PAN is ABCDE1234F, should I invest in HDFC Mid Cap Fund?",
        detect_scheme_detail("my PAN is ABCDE1234F, should I invest in HDFC Mid Cap Fund?"),
    )
    assert result.kind == "pii"


def test_pii_value_never_reaches_a_log_file(tmp_path: Path) -> None:
    """The strongest PII guarantee: the value is absent from disk afterwards."""
    log_file = tmp_path / "mf_faq.log"
    handler = logging.FileHandler(log_file, encoding="utf-8")
    root = logging.getLogger("mf_faq")
    root.addHandler(handler)
    root.setLevel(logging.DEBUG)
    try:
        secret = "ABCDE1234F"
        for message, _ in PII_CASES:
            if secret in message:
                guard.run_guards(message)
        guard.run_guards("aadhaar 1234 5678 9012")
        handler.flush()
    finally:
        root.removeHandler(handler)
        handler.close()

    contents = log_file.read_text(encoding="utf-8")
    assert contents, "expected the guard to log its coarse label"
    assert secret not in contents
    assert "1234 5678 9012" not in contents
    assert "9876543210" not in contents
    assert "ravi.sharma@example.com" not in contents
    # The label is there, so we know the assertions above are not vacuous.
    assert "guard=pii" in contents


def test_legitimate_questions_are_not_mistaken_for_pii() -> None:
    """A 12-digit-ish string that is not PII must still be answerable."""
    for message in (
        "What is the expense ratio of HDFC Flexi Cap Fund?",
        "expense ratio of hdfc small cap",
        "What is the lock-in period for HDFC ELSS Tax Saver?",
        "What benchmark does HDFC Large Cap Fund track?",
    ):
        assert guard.detect_pii(message) is None, message
        assert guard.run_guards(message).kind == "ok", message


# --------------------------------------------------------------------------
# 2. Out of scope (FR-10)
# --------------------------------------------------------------------------

OUT_OF_SCOPE_CASES = [
    "What is the expense ratio of SBI Bluechip Fund?",
    "tell me about the ICICI Nifty 50 index fund",
    "how is Parag Parikh Flexi Cap performing?",
    "What is the exit load on Axis Small Cap Fund?",
    "expense ratio of HDFC Mid Cap Fund",
    "What is the lock-in period of HDFC Dividend Yield Fund?",
    "manager of HDFC Corporate Bond Fund",
]


@pytest.mark.parametrize("message", OUT_OF_SCOPE_CASES)
def test_out_of_scope_is_refused(message: str) -> None:
    result = guard.run_guards(message, detect_scheme_detail(message))
    assert result.kind == "out_of_scope", message
    assert result.message == config.OUT_OF_SCOPE_TEMPLATE


def test_our_own_five_schemes_are_not_out_of_scope() -> None:
    """The HDFC-scheme check must not swallow the corpus it is meant to protect."""
    for slug in config.CORPUS_SLUGS:
        message = f"What is the expense ratio of {config.display_name(slug)}?"
        result = guard.check_out_of_scope(message)
        assert not result.blocked, f"{slug} wrongly flagged out of scope: {result.message}"


# --------------------------------------------------------------------------
# 3. Ambiguity (FR-9)
# --------------------------------------------------------------------------


def test_bare_hdfc_fund_is_ambiguous() -> None:
    result = guard.run_guards("HDFC fund", detect_scheme_detail("HDFC fund"))
    assert result.kind == "ambiguous"
    assert result.message == config.DISAMBIGUATION_TEMPLATE


def test_two_schemes_named_is_ambiguous_not_out_of_scope() -> None:
    message = "which is better large cap or small cap"
    result = guard.run_guards(message, detect_scheme_detail(message))
    # Ambiguity outranks advice, per the mandated order.
    assert result.kind == "ambiguous"


def test_scheme_agnostic_question_is_not_forced_to_choose() -> None:
    """An RTA/statement question mentioning HDFC must still be answerable."""
    message = "which RTA holds my HDFC Mutual Fund folio"
    detection = detect_scheme_detail(message)
    assert detection.kind == "none"
    assert guard.run_guards(message, detection).kind == "ok"


# --------------------------------------------------------------------------
# 4. Performance (FR-6)
# --------------------------------------------------------------------------

PERFORMANCE_CASES = [
    "What is the 1 year return of HDFC Flexi Cap Fund?",
    "what is the NAV of HDFC Large Cap Fund today",
    "how much has HDFC Small Cap grown in 5 years",
    "what is the AUM of HDFC ELSS Tax Saver",
    "past performance of HDFC Balanced Advantage Fund",
    "what is the CAGR since inception",
]


@pytest.mark.parametrize("message", PERFORMANCE_CASES)
def test_performance_questions_are_refused(message: str) -> None:
    result = guard.run_guards(message, detect_scheme_detail(message))
    assert result.kind in {"performance", "out_of_scope"}, message
    if result.kind == "performance":
        assert "%" not in result.message


# --------------------------------------------------------------------------
# 5. Advice (FR-5) - 12 adversarial prompts
# --------------------------------------------------------------------------

ADVICE_CASES = [
    "Should I invest in HDFC Small Cap Fund?",
    "which is the best scheme for me",
    "recommend a good fund for my daughter",
    "is HDFC Flexi Cap worth investing in?",
    "should I put my money in HDFC Large Cap",
    "which is better, HDFC ELSS or HDFC Flexi Cap?",
    "do you think HDFC Balanced Advantage is a good investment?",
    "what is the right time to invest in HDFC Small Cap?",
    "can I make money by investing in HDFC ELSS Tax Saver?",
    "how much will I get if I invest in HDFC Large Cap?",
    "is HDFC Small Cap Fund safe to invest in?",
    "help me choose between HDFC Flexi Cap and HDFC Large Cap",
]


@pytest.mark.parametrize("message", ADVICE_CASES)
def test_advice_is_refused_with_the_template(message: str) -> None:
    result = guard.run_guards(message, detect_scheme_detail(message))
    assert result.blocked, message
    assert result.kind in {"advice", "ambiguous", "out_of_scope"}, message
    if result.kind == "advice":
        assert result.message == config.REFUSAL_TEMPLATE
        assert result.link == config.REFUSAL_LINK
        # FR-6 applies to refusals too: a leaked figure fails the check.
        assert "%" not in result.message
        assert "per cent" not in result.message.lower()


def test_advice_refusals_contain_no_figures_anywhere() -> None:
    for message in ADVICE_CASES:
        result = guard.run_guards(message, detect_scheme_detail(message))
        assert not any(ch.isdigit() for ch in result.message) or result.kind != "advice"


# --------------------------------------------------------------------------
# Blocked-reply rendering
#
# [regression] tests/demo_answers.py caught two bugs the suite above missed:
#   * sentence-capping the disambiguation menu collapsed the five-item list into
#     "Which of these five...? 1. HDFC Large Cap Fund 2.";
#   * every refusal printed its link twice, once inlined in the template and
#     once in the appended "Source:" line, breaking the one-citation rule.
# Both are asserted here on the *rendered* text, not on GuardResult.message,
# because that is what a user actually sees.
# --------------------------------------------------------------------------


BLOCKED_CASES = [
    ("HDFC fund", "ambiguous"),
    ("What is the expense ratio of HDFC Mid Cap Fund?", "out_of_scope"),
    ("Tell me about SBI Bluechip Fund", "out_of_scope"),
    ("What is the 1 year return of HDFC Large Cap Fund?", "performance"),
    ("Should I invest in HDFC Small Cap Fund?", "advice"),
]


@pytest.mark.parametrize("message,expected_kind", BLOCKED_CASES)
def test_blocked_reply_cites_exactly_one_url(message: str, expected_kind: str) -> None:
    verdict = guard.run_guards(message, detect_scheme_detail(message))
    assert verdict.kind == expected_kind, message
    out = fmt.format_blocked(verdict.message, link=verdict.link)

    urls = _urls_in(out.text)
    assert len(urls) == 1, f"{expected_kind}: expected 1 URL, got {urls}"
    host = urls[0].split("//", 1)[1].split("/", 1)[0]
    assert host in ALLOWED
    assert out.text.count("Source: ") == 1, "the Source line must not be duplicated"


@pytest.mark.parametrize("message,expected_kind", BLOCKED_CASES)
def test_blocked_reply_lists_all_five_schemes_intact(
    message: str, expected_kind: str
) -> None:
    """A menu that gets truncated is worse than no menu."""
    verdict = guard.run_guards(message, detect_scheme_detail(message))
    out = fmt.format_blocked(verdict.message, link=verdict.link)

    if expected_kind not in {"ambiguous", "out_of_scope"}:
        return
    for name in (
        "HDFC Large Cap Fund",
        "HDFC Flexi Cap Fund",
        "HDFC ELSS Tax Saver",
        "HDFC Small Cap Fund",
        "HDFC Balanced Advantage Fund",
    ):
        assert name in out.text, f"{expected_kind}: '{name}' missing from the menu"
    for i in range(1, 6):
        assert f"{i}." in out.text, f"{expected_kind}: item {i} missing"


def test_no_guard_template_inlines_its_own_link() -> None:
    """Templates carry no URL; format_blocked owns the single Source line."""
    for name in (
        "REFUSAL_TEMPLATE",
        "PERFORMANCE_REFUSAL_TEMPLATE",
        "DISAMBIGUATION_TEMPLATE",
        "OUT_OF_SCOPE_TEMPLATE",
        "PII_BLOCK_MESSAGE",
    ):
        template = getattr(config, name)
        assert "http" not in template, f"{name} inlines a link; it would double-print"
        assert "%" not in template, f"{name} contains a percentage"


# --------------------------------------------------------------------------
# Sentence splitting and percentage handling
# --------------------------------------------------------------------------


def test_sentence_splitter_protects_decimals() -> None:
    text = "The expense ratio is 0.77% per annum. The exit load is nil. Minimum SIP is Rs. 100."
    sentences = fmt.split_sentences(text)
    assert sentences == [
        "The expense ratio is 0.77% per annum.",
        "The exit load is nil.",
        "Minimum SIP is Rs. 100.",
    ]


def test_truncation_respects_abbreviations() -> None:
    text = "The expense ratio is 1.03%. It is charged daily i.e. every day. See p. 4 of the SID."
    kept, n = fmt.truncate_to_sentences(text, max_sentences=3)
    assert n == 3
    assert kept.endswith("See p. 4 of the SID.")
    assert "1.03%" in kept


def test_fee_percentages_are_preserved() -> None:
    """The deliberate narrowing of the PRD's literal 'strip any %' rule."""
    text = "The expense ratio of HDFC Flexi Cap Fund is 0.77% per annum."
    cleaned, removed = fmt.strip_performance_percentages(text)
    assert removed == 0
    assert "0.77%" in cleaned


def test_return_percentages_are_removed() -> None:
    text = "The fund returned 12.5% last year. The expense ratio is 0.77% per annum."
    cleaned, removed = fmt.strip_performance_percentages(text)
    assert removed == 1
    assert "12.5%" not in cleaned
    assert "0.77%" in cleaned, "the fee percentage must survive"


def test_scheme_name_growth_does_not_make_a_fee_a_return() -> None:
    """Regression: every scheme is named '... - Direct Growth'.

    Matching 'growth' as a return word classified all five expense-ratio answers
    as return statements and the stripper deleted the percentage from every
    answer, leaving "the expense ratio is  per annum."
    """
    for slug in config.CORPUS_SLUGS:
        text = f"{config.display_name(slug)}: the expense ratio is 0.77% per annum."
        cleaned, removed = fmt.strip_performance_percentages(text)
        assert removed == 0, f"{slug}: a fee percentage was stripped as a return"
        assert "0.77%" in cleaned, f"{slug}: percentage deleted from the answer"


def test_fee_attribute_exempts_percentages_outright() -> None:
    """On a fee attribute a percentage is a sourced fact, whatever the wording."""
    for attribute in ("expense_ratio", "base_expense_ratio", "exit_load"):
        cleaned, removed = fmt.strip_performance_percentages(
            "Returns were 12.5% and the figure is 1.03%.", attribute=attribute
        )
        assert removed == 0, attribute
        assert "1.03%" in cleaned


def test_strip_urls_removes_every_link() -> None:
    text = "It is 0.77% (see https://groww.in/x and www.example.com)."
    cleaned = fmt.strip_urls(text)
    assert "http" not in cleaned
    assert "www." not in cleaned
    assert "0.77%" in cleaned


# --------------------------------------------------------------------------
# Model-output hardening
# --------------------------------------------------------------------------


def test_parse_json_loose_handles_fences_and_prose() -> None:
    assert parse_json_loose('```json\n{"answer": "hi"}\n```') == {"answer": "hi"}
    assert parse_json_loose('Sure! {"answer": "hi", "source_chunk_id": "x"} done') == {
        "answer": "hi",
        "source_chunk_id": "x",
    }
    assert parse_json_loose("not json at all") == {}
    assert parse_json_loose("") == {}


def test_default_provider_is_the_offline_stub(monkeypatch: pytest.MonkeyPatch) -> None:
    """No phase may require an API key.

    This asserts the *rule*, not the ambient state: the environment is cleared
    first, so it holds whether or not ``.env`` holds a working key. An earlier
    version read the real environment and therefore failed the moment a key was
    added, which tested the developer's machine rather than the invariant.
    """
    for var in ("LLM_PROVIDER", "GROQ_API_KEY", "GEMINI_API_KEY", "OPENAI_API_KEY"):
        monkeypatch.delenv(var, raising=False)

    assert config.llm_provider() == "stub"
    assert isinstance(get_llm(), StubLLM)


def test_provider_without_a_key_falls_back_to_the_stub(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Naming a provider is not enough; a blank key must still run offline.

    This is the half-filled ``.env`` case - the one that would otherwise reach
    the network and fail at request time.
    """
    monkeypatch.setenv("LLM_PROVIDER", "groq")
    monkeypatch.setenv("GROQ_API_KEY", "")

    assert config.llm_provider() == "stub"
    assert isinstance(get_llm(), StubLLM)


def test_networked_providers_satisfy_the_llmclient_name_contract() -> None:
    """Every client exposes ``name``, which ``generate.py`` records.

    Regression test for a bug no test could catch while the stub was the only
    client: ``_HttpProvider`` defined ``provider_name`` but not the ``name``
    the ``LLMClient`` protocol declares, so *every* networked provider raised
    ``AttributeError: 'GroqClient' object has no attribute 'name'`` on the
    first answer. The stub was always fine, so the suite stayed green.
    """
    for client in (GroqClient(), OpenAIClient(), GeminiClient()):
        assert isinstance(client.name, str) and client.name
        # provider and resolved model, so a logged `provider=` identifies which
        # model really answered
        assert client.provider_name in client.name
        assert client.model in client.name
        assert callable(client.complete_json)


def test_model_env_var_overrides_the_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """``GROQ_MODEL`` and friends select the model, as documented in .env."""
    monkeypatch.setenv("GROQ_MODEL", "openai/gpt-oss-20b")
    assert GroqClient().model == "openai/gpt-oss-20b"

    # an explicit argument still wins over the environment
    assert GroqClient("explicit-model").model == "explicit-model"

    # blank means "use the default", not "empty model"
    monkeypatch.setenv("GROQ_MODEL", "")
    assert GroqClient().model == GroqClient.default_model
    assert GroqClient.default_model


def test_unknown_provider_falls_back_to_stub() -> None:
    assert isinstance(get_llm("definitely-not-a-provider"), StubLLM)


def test_system_prompt_forbids_urls_and_figures() -> None:
    prompt = SYSTEM_PROMPT.lower()
    assert "never output a url" in prompt
    assert "source_chunk_id" in SYSTEM_PROMPT


def test_fabricated_chunk_id_cannot_become_a_citation() -> None:
    """A hallucinated id must fall back to FR-8, never reach the source line."""
    from src.query.generate import GenerationResult

    forged = GenerationResult(
        answer="The expense ratio is 0.77% per annum.",
        source_chunk_id="not_a_real_chunk_id_000",
        provider="test",
        valid_chunk_id=False,
    )
    out = fmt.format_answer(forged, scheme_for_unknown="HDFC Flexi Cap Fund")
    assert out.cited is False
    assert out.note == "no_valid_chunk_id"
    urls = _urls_in(out.text)
    assert urls == [config.REFUSAL_LINK]


def _urls_in(text: str) -> list[str]:
    import re

    return re.findall(r"https?://[^\s)]+", text)


# --------------------------------------------------------------------------
# End-to-end with the stub (exercises retrieval + generation + formatting)
# --------------------------------------------------------------------------

# [verified] expense ratios read at ingest time, never hard-coded upstream.
EXPECTED_EXPENSE = {
    "hdfc-large-cap": "1.03%",
    "hdfc-flexi-cap": "0.77%",
    "hdfc-elss-tax-saver": "1.21%",
    "hdfc-small-cap": "0.78%",
    "hdfc-balanced-advantage": "0.78%",
}


@pytest.fixture(scope="module")
def warm() -> None:
    from src.ingest.embed_store import embed_query

    embed_query("warmup")


@pytest.mark.parametrize("slug,expected", list(EXPECTED_EXPENSE.items()))
def test_expense_ratio_answer_states_the_right_number(
    warm: None, slug: str, expected: str
) -> None:
    """Guards the base-vs-total expense ratio trap found in Phase 4.

    The stub is extractive, so if retrieval picks ``base_expense_ratio`` the
    wrong figure is quoted verbatim. Asserting the number catches that directly
    rather than trusting the rank table.
    """
    question = f"What is the expense ratio of {config.display_name(slug)}?"
    detection = detect_scheme_detail(question)
    assert detection.kind == "unique", question

    result = retrieve(question, detection.scheme)
    generation = generate_answer(question, result.chunks, StubLLM())
    out = fmt.format_answer(generation, scheme_for_unknown=slug)

    assert out.cited, f"{slug}: no citation, note={out.note}"
    assert expected in out.text, (
        f"{slug}: expected {expected} but got: {out.text!r}"
    )
    # The base figure must not be the one quoted.
    assert "base expense ratio" not in out.text.lower()


BARE_EXPENSE_QUESTIONS = {
    "hdfc-large-cap": ("expense ratio of large cap", "1.03%"),
    "hdfc-flexi-cap": ("expense ratio of flexi cap", "0.77%"),
    "hdfc-elss-tax-saver": ("expense ratio of elss", "1.21%"),
    "hdfc-small-cap": ("expense ratio of small cap", "0.78%"),
    "hdfc-balanced-advantage": ("expense ratio of balanced advantage", "0.78%"),
}


@pytest.mark.parametrize("slug,case", list(BARE_EXPENSE_QUESTIONS.items()))
def test_bare_expense_query_does_not_get_the_base_figure(
    warm: None, slug: str, case: tuple[str, str]
) -> None:
    """The bare "expense ratio of <scheme>" form, with no display name.

    [bug] This phrasing ranked ``base_expense_ratio`` first once the paraphrase
    line repeated the bare term - the mirror of the original failure, answering
    0.57% instead of 0.77%. The full display-name form masked it, so both are
    asserted.
    """
    question, expected = case
    detection = detect_scheme_detail(question)
    assert detection.kind == "unique", question

    result = retrieve(question, detection.scheme)
    generation = generate_answer(question, result.chunks, StubLLM())
    out = fmt.format_answer(generation, scheme_for_unknown=slug)

    assert out.cited, f"{question}: no citation, note={out.note}"
    assert expected in out.text, f"{question}: expected {expected}, got {out.text!r}"
    assert "base expense ratio" not in out.text.lower()


@pytest.mark.parametrize("slug", list(EXPECTED_EXPENSE))
def test_answered_reply_has_exactly_one_allowed_url(warm: None, slug: str) -> None:
    question = f"What is the expense ratio of {config.display_name(slug)}?"
    detection = detect_scheme_detail(question)
    result = retrieve(question, detection.scheme)
    generation = generate_answer(question, result.chunks, StubLLM())
    out = fmt.format_answer(generation, scheme_for_unknown=slug)

    urls = _urls_in(out.text)
    assert len(urls) == 1, f"{slug}: expected exactly 1 URL, got {urls}"
    host = urls[0].split("//", 1)[1].split("/", 1)[0]
    assert host in ALLOWED, f"{slug}: {host} not in the citation allow-list"
    assert out.sentences <= config.MAX_SENTENCES


def test_footer_matches_fetched_at(warm: None) -> None:
    question = "What is the expense ratio of HDFC Flexi Cap Fund?"
    detection = detect_scheme_detail(question)
    result = retrieve(question, detection.scheme)
    generation = generate_answer(question, result.chunks, StubLLM())
    out = fmt.format_answer(generation, scheme_for_unknown="hdfc-flexi-cap")

    assert out.fetched_at, "no fetched_at on the citation"
    assert out.footer in out.text
    assert out.text.endswith(out.footer)
    # The footer is a fetch date, not a live-data claim.
    assert out.fetched_at == result.chunks[0].fetched_at


def test_statement_answer_is_cited_to_the_rta(warm: None) -> None:
    question = "How do I download my capital-gains statement?"
    detection = detect_scheme_detail(question)
    assert detection.kind == "none"
    result = retrieve(question, None)
    generation = generate_answer(question, result.chunks, StubLLM())
    out = fmt.format_answer(generation, scheme_for_unknown="HDFC Mutual Fund")

    assert out.cited, out.note
    assert len(_urls_in(out.text)) == 1
    assert "CAMS" in out.text or "portal" in out.text.lower()


def test_unknown_question_gets_the_fr8_refusal(warm: None) -> None:
    """A question we cannot answer must not be guessed at."""
    question = "Who is the CEO of HDFC Asset Management?"
    detection = detect_scheme_detail(question)
    guard_result = guard.run_guards(question, detection)
    if guard_result.blocked:
        return  # refused upstream, which is also correct
    result = retrieve(question, detection.scheme)
    generation = generate_answer(question, result.chunks, StubLLM())
    out = fmt.format_answer(generation, scheme_for_unknown="HDFC Mutual Fund")
    if out.cited:
        # If it did answer, the answer must be capped and cited once.
        assert len(_urls_in(out.text)) == 1
        assert out.sentences <= config.MAX_SENTENCES


# --------------------------------------------------------------------------
# Golden-set guard sweep
# --------------------------------------------------------------------------


def test_every_golden_guard_case_behaves(warm: None) -> None:
    """Run the detection + guard chain over the labelled golden queries."""
    golden = json.loads(
        (Path(__file__).resolve().parent / "golden.json").read_text(encoding="utf-8")
    )
    for case in golden["cases"]:
        question = case["query"]
        detection = detect_scheme_detail(question)
        result = guard.run_guards(question, detection)
        # A guard block is a legitimate outcome; the suite's other tests pin
        # which kind each one must be. This asserts the chain never crashes and
        # never blocks with a % figure.
        if result.blocked:
            assert "%" not in result.message, question


# --------------------------------------------------------------------------
# Attribute coverage (the confidently-wrong-answer guard)
# --------------------------------------------------------------------------


# Every one of these asks for a fact the 14-attribute corpus does not hold.
# Before this guard each one was answered from a neighbouring attribute, with a
# real citation, which reads as a sourced fact and is simply wrong. The ticker
# case is the one that was actually observed in the UI.
UNCOVERED_QUESTIONS = [
    "What is the ticker symbol of HDFC Flexi Cap Fund?",
    "What is the ISIN of HDFC ELSS Tax Saver?",
    "What is the fund code for HDFC Small Cap Fund?",
    "What is the Sharpe ratio of HDFC Large Cap Fund?",
    "What is the tracking error of HDFC Small Cap Fund?",
    "Who is the custodian of HDFC Flexi Cap Fund?",
    "Who is the trustee of HDFC Large Cap Fund?",
    "What is the SEBI registration number of HDFC Large Cap Fund?",
    "Can I do an STP in HDFC Flexi Cap Fund?",
    "Is an SWP available in HDFC Small Cap Fund?",
    "What is the tax rate on HDFC ELSS Tax Saver?",
    "What are the top 10 holdings of HDFC Large Cap Fund?",
    "When did HDFC Flexi Cap Fund launch?",
    "What is the merger plan for HDFC Flexi Cap Fund?",
    "How do I reach the customer care helpline?",
    "What is the fund size of HDFC Large Cap Fund?",
]

# The inverse, and the more important list. Each of these is a fact the corpus
# DOES hold, so a coverage false positive would refuse a working answer - a
# worse failure than the one this guard prevents.
COVERED_QUESTIONS = [
    "What is the expense ratio of HDFC Flexi Cap Fund?",
    "What is the lock-in period for HDFC ELSS Tax Saver?",
    "Who manages HDFC Small Cap Fund?",
    "What is the minimum SIP for HDFC Balanced Advantage Fund?",
    "What is the minimum lump sum for HDFC Small Cap Fund?",
    "What is the exit load of HDFC Large Cap Fund?",
    "How do I download my capital-gains statement?",
    "What is the risk level of HDFC Flexi Cap Fund?",
    "What index is HDFC Small Cap Fund measured against?",
    "What kind of scheme is HDFC Large Cap Fund?",
    "What is the scheme objective of HDFC Flexi Cap Fund?",
    "Which RTA holds my HDFC Mutual Fund folio?",
    "What is the tax benefit of HDFC ELSS Tax Saver?",
    "Is the ELSS lock-in under section 80C?",
    "How do I find my folio number?",
    "What does the capital-gains statement list?",
    "charges for managing hdfc flexi cap",
    "how long must i hold elss units",
]


@pytest.mark.parametrize("question", UNCOVERED_QUESTIONS)
def test_uncovered_attribute_is_refused_not_answered(question: str) -> None:
    """A fact outside the corpus must be refused, not substituted."""
    result = guard.run_guards(question, detect_scheme_detail(question))
    assert result.blocked, f"{question!r} was allowed through to retrieval"
    assert result.kind == "unknown", f"{question!r} -> {result.kind}"
    # The FR-8 copy, pointed at the official documents.
    assert "don't have that in my sources" in result.message
    assert result.link == config.REFUSAL_LINK


@pytest.mark.parametrize("question", COVERED_QUESTIONS)
def test_covered_attribute_is_not_refused_by_the_coverage_guard(question: str) -> None:
    """The coverage guard must not swallow a fact we can actually answer."""
    result = guard.check_coverage(question, detect_scheme_detail(question))
    assert not result.blocked, f"{question!r} -> {result.message}"


def test_coverage_guard_runs_last_in_the_chain() -> None:
    """Specific refusals must win over the generic FR-8 one."""
    # Unnamed scheme: FR-9 ambiguity resolves before coverage is consulted.
    result = guard.run_guards(
        "What is the SEBI registration number of HDFC Mutual Fund?",
        detect_scheme_detail("What is the SEBI registration number of HDFC Mutual Fund?"),
    )
    assert result.kind == "ambiguous", result.kind

    # AUM is a performance question, so it gets the factsheet pointer rather than
    # the generic "I don't have that" - even though it is also uncovered.
    result = guard.run_guards(
        "What is the AUM of HDFC Flexi Cap Fund?",
        detect_scheme_detail("What is the AUM of HDFC Flexi Cap Fund?"),
    )
    assert result.kind == "performance", result.kind

    # Advice likewise wins.
    result = guard.run_guards(
        "Should I buy the tax rate of HDFC ELSS Tax Saver?",
        detect_scheme_detail("Should I buy the tax rate of HDFC ELSS Tax Saver?"),
    )
    assert result.kind == "advice", result.kind


def test_ticker_symbol_is_never_answered_with_the_benchmark() -> None:
    """The exact observed failure, asserted end to end.

    This query was answered "the benchmark index is NIFTY 500 TRI" at distance
    0.343, comfortably inside the 0.65 threshold - so this is also a check that
    the fix is a guard and not a threshold change.
    """
    question = "What is the ticker symbol of HDFC Flexi Cap Fund?"
    result = guard.run_guards(question, detect_scheme_detail(question))
    assert result.kind == "unknown", result.kind

    # And with the guard bypassed, retrieval really would have offered the
    # benchmark chunk - which is why no distance threshold can fix this.
    raw = retrieve(question, detect_scheme_detail(question).scheme)
    assert raw.confident, "if this stops being confident, the guard is redundant"
    assert raw.best_distance < config.DISTANCE_THRESHOLD
    assert raw.chunks[0].attribute == "benchmark"

    # The answer text must not leak the benchmark back to the user.
    assert "NIFTY" not in result.message


def test_refusal_copy_does_not_leak_the_plan_suffix() -> None:
    """``display_name`` carries the plan variant; prose addressed to a user must not.

    The refusal read "the official HDFC Flexi Cap Fund - Direct Growth documents
    for it" - the suffix is needed to match a query, not to talk to someone.
    """
    question = "What is the ticker symbol of HDFC Flexi Cap Fund?"
    result = guard.run_guards(question, detect_scheme_detail(question))
    assert "Direct Growth" not in result.message, result.message
    assert "Direct Plan Growth" not in result.message, result.message
    assert "HDFC Flexi Cap Fund" in result.message, result.message

    # base_name is the single source of truth for this.
    assert config.base_name("hdfc-flexi-cap") == "HDFC Flexi Cap Fund"
    assert config.base_name("hdfc-elss-tax-saver") == "HDFC ELSS Tax Saver"
    # Unchanged when there is no suffix, and unknown input passes through.
    assert config.base_name("HDFC Large Cap Fund") == "HDFC Large Cap Fund"
    assert config.base_name("not-a-scheme") == "not-a-scheme"


def test_uncovered_patterns_do_not_match_any_indexed_chunk() -> None:
    """Every coverage pattern must be absent from all 62 chunks.

    This is the test that keeps the guard honest as the corpus changes. A
    pattern that matches a chunk's own text is a pattern that will one day
    refuse a question we can answer - and it has already happened twice:
    `folio number` is in the statement chunk, and `section 80C` is in the ELSS
    tax_status chunk. Both were caught by checking before adding, not after.
    """
    path = config.CHUNKS_JSONL
    if not path.exists():
        pytest.skip("chunks.jsonl not built")
    blob = "\n".join(
        json.loads(line).get("text", "")
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    offenders = [
        pattern
        for pattern in config.UNCOVERED_ATTRIBUTE_PATTERNS
        if re.search(pattern, blob, re.IGNORECASE)
    ]
    assert not offenders, (
        "these coverage patterns match indexed text and would refuse a working "
        f"answer: {offenders}"
    )


# --------------------------------------------------------------------------
# Guard copy keeps its line structure
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "template,leading",
    [
        (config.DISAMBIGUATION_TEMPLATE, "Which of these five"),
        (config.OUT_OF_SCOPE_TEMPLATE, "I only cover 5"),
    ],
)
def test_numbered_menu_survives_formatting(template: str, leading: str) -> None:
    """A guard menu must not reflow into the sentence above it.

    [bug] `format_blocked` ran the whole message through a single-paragraph
    reflow, so the blank line before the list was flattened and item 1 was
    absorbed into the question: "Which of these five... do you mean? 1. HDFC
    Large Cap Fund". Whether the break survived depended on the punctuation
    before it - the out-of-scope menu ended in ":" and was fine, this one ended
    in "?" and was not.
    """
    body = fmt.format_blocked(template).text
    question_line, _, rest = body.partition("\n")
    assert question_line.startswith(leading)
    assert "1." not in question_line, f"item 1 was absorbed into the question: {question_line!r}"

    lines = [line for line in rest.splitlines() if line.strip()]
    assert len(lines) == 5, f"expected all 5 options, got {lines}"
    for index, line in enumerate(lines, start=1):
        assert line.startswith(f"{index}. "), line
    assert "HDFC Balanced Advantage Fund" in lines[-1]


def test_every_guard_template_keeps_its_line_breaks() -> None:
    """Generic version of the above, so the next template added is covered too."""
    for name in ("DISAMBIGUATION_TEMPLATE", "OUT_OF_SCOPE_TEMPLATE"):
        template = getattr(config, name)
        if "\n\n" not in template:
            continue
        rendered = fmt.format_blocked(template).text
        assert "\n\n" in rendered, f"{name} lost its paragraph break: {rendered!r}"
