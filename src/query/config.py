"""Central configuration for the Mutual Fund FAQ Assistant.

Every tunable lives here so that no magic value is scattered through the code.
Values marked [verified] were confirmed by live probes on 27 Sep 2026.
"""

from __future__ import annotations

import os
from pathlib import Path

# Quieten HF/transformers noise (symlink warning on Windows, progress bars,
# unauthenticated-rate-limit notice). Must happen before transformers loads.
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
MANUAL_DIR = DATA_DIR / "manual"
PROCESSED_DIR = DATA_DIR / "processed"
CHROMA_DIR = ROOT / "chroma_db"

SOURCES_CSV = ROOT / "sources.csv"
RECORDS_JSON = PROCESSED_DIR / "records.json"
CHUNKS_JSONL = PROCESSED_DIR / "chunks.jsonl"

for _d in (RAW_DIR, MANUAL_DIR, PROCESSED_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------
# Vector store / retrieval
# --------------------------------------------------------------------------

COLLECTION_NAME = "hdfc_mf_faq"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # [verified] 384-dim, loads locally
EMBED_BATCH_SIZE = 32
EMBED_DIM = 384
CHROMA_SPACE = "cosine"  # PRD: cosine distance

TOP_K = 5

# Tuned in Phase 4 against tests/golden.json, not guessed.
#
# The PRD suggested 0.6 as a starting point. At the time of tuning 0.6 would have
# refused a real answer - "How do I get my tax statement for income tax filing?"
# sat at 0.6044. Since the per-attribute paraphrase lines below were added, the
# worst golden-answer distance fell to 0.5876, so 0.6 would also work now. 0.65
# is kept deliberately: paraphrase distance is the noisiest number in the system,
# and the headroom costs only a few extra unknowns reaching the LLM, which the
# FR-8 check catches anyway. Missing a valid answer is the worse failure.
#
# Re-run `python tests/eval_retrieval.py` if the corpus or the synonyms change.
DISTANCE_THRESHOLD = 0.65
MAX_SENTENCES = 3

# Search-time paraphrases, one per attribute, appended to the text sent to the
# embedding model ONLY (never to the stored document, the citation, or the LLM
# prompt).
#
# Why this exists: [verified in Phase 4] the fact cards state each attribute in
# its official wording only, so a user asking "charges for managing" or "who
# looks after the fund" or "how much to start a monthly instalment" missed the
# obvious chunk. On a 14-query adversarial probe the hit rate was 57% top-1 /
# 78.6% top-5 - below the 85% target - even though the leaky paraphrase set
# scored 100%. Embedding the synonym line teaches the vector the paraphrase
# space without adding a single fact to the corpus.
#
# Two rules these entries must respect:
#   1. No numbers. A synonym line that carried a figure could put an unverified
#      number into the vector for a chunk that does not state it.
#   2. No overlap between sibling attributes. `expense_ratio` and
#      `base_expense_ratio` already collide in the source text; sharing
#      synonyms there is how a query about fees ends up answering 0.57% when the
#      holder actually pays 0.77%. The two entries below are deliberately
#      contrasted ("total"/"holder pays" vs "base component only").
ATTRIBUTE_ALIASES: dict[str, str] = {
    "expense_ratio": (
        "the expense ratio, the total expense ratio a holder pays, also called the "
        "management fee, annual fund charges, the ongoing fee taken from the scheme"
    ),
    # No "expense ratio" in this line. [bug] Repeating the bare term here made a
    # plain "expense ratio of flexi cap" rank the *base* card first - the exact
    # mirror of the original failure, answering 0.57% instead of 0.77%. The word
    # still appears in the base card's own document, where it is a sourced fact;
    # what matters is not adding a second pull toward it.
    "base_expense_ratio": (
        "only the base component, the basic figure before the additional "
        "fund-expense charge, not the total a holder pays"
    ),
    "exit_load": (
        "the charge on redemption, the penalty for selling early, the withdrawal "
        "charge, what happens if units are sold before the exit load period ends"
    ),
    "min_sip": (
        "the monthly instalment, the amount needed to start a recurring monthly "
        "investment, the smallest monthly payment, how much to begin an SIP"
    ),
    "min_lumpsum": (
        "the one-time investment amount, a single payment, the minimum needed to "
        "buy units once without a monthly instalment"
    ),
    "lock_in": (
        "how long units must be held, the minimum holding period, the period "
        "before which units cannot be sold, the lock up duration"
    ),
    "riskometer": (
        "the risk level and risk rating, how risky the scheme is, the risk "
        "category assigned by the riskometer"
    ),
    "benchmark": (
        "which index the scheme is measured against, the reference or market "
        "index used for comparison, the index tracked"
    ),
    "fund_manager": (
        "who manages the fund, who looks after the scheme, the portfolio "
        "manager, the name of the fund manager in charge"
    ),
    "category": (
        "what type or kind of scheme this is, which fund category it belongs to, "
        "the scheme classification"
    ),
    "objective": (
        "the investment objective, what the scheme seeks to achieve, its aim and "
        "mandate, what the fund is designed to do"
    ),
    "tax_status": (
        "the tax benefit and section 80C deduction, whether the scheme is tax "
        "saving, eligibility for tax exemption"
    ),
    "rta": (
        "the registrar and transfer agent, which RTA holds the units and "
        "maintains the investor records"
    ),
    "statement_download": (
        "how to get or download an account statement, the capital gains tax "
        "report, the consolidated account statement, the RTA investor portal"
    ),
}
PROSE_CHUNK_TOKENS = 200  # keep well under MiniLM's 256 word-piece silent truncation
PROSE_CHUNK_OVERLAP = 30
FACT_CARD_MIN_TOKENS = 30
FACT_CARD_MAX_TOKENS = 80

# --------------------------------------------------------------------------
# Schemes
# --------------------------------------------------------------------------

# Curated display names. Do NOT derive these from the scraped `super_category`
# field: [verified] it yields values like "HDFC Flexi Cap Direct Plan-Growth".
DISPLAY_NAMES: dict[str, str] = {
    "hdfc-large-cap": "HDFC Large Cap Fund \u2013 Direct Growth",
    "hdfc-flexi-cap": "HDFC Flexi Cap Fund \u2013 Direct Growth",
    "hdfc-elss-tax-saver": "HDFC ELSS Tax Saver \u2013 Direct Plan Growth",
    "hdfc-small-cap": "HDFC Small Cap Fund \u2013 Direct Growth",
    "hdfc-balanced-advantage": "HDFC Balanced Advantage Fund \u2013 Direct Growth",
}

# Alias -> slug. Matched longest-first against the lowercased user message.
# [verified] "flexi cap" is the former "HDFC Equity Fund", hence both aliases.
SCHEME_ALIASES: dict[str, str] = {
    # large cap
    "hdfc large cap": "hdfc-large-cap",
    "large cap fund": "hdfc-large-cap",
    "large cap": "hdfc-large-cap",
    # flexi cap
    "hdfc flexi cap": "hdfc-flexi-cap",
    "flexi cap fund": "hdfc-flexi-cap",
    "flexi cap": "hdfc-flexi-cap",
    "hdfc equity fund": "hdfc-flexi-cap",
    "equity fund": "hdfc-flexi-cap",
    # elss
    "hdfc elss": "hdfc-elss-tax-saver",
    "elss tax saver": "hdfc-elss-tax-saver",
    "tax saver": "hdfc-elss-tax-saver",
    "elss": "hdfc-elss-tax-saver",
    # small cap
    "hdfc small cap": "hdfc-small-cap",
    "small cap fund": "hdfc-small-cap",
    "small cap": "hdfc-small-cap",
    # balanced advantage
    "hdfc balanced advantage": "hdfc-balanced-advantage",
    "balanced advantage fund": "hdfc-balanced-advantage",
    "balanced advantage": "hdfc-balanced-advantage",
}

GENERAL_SCHEME = "general"

# The five corpus URLs, in the order given by the PRD.
CORPUS_SLUGS: list[str] = list(DISPLAY_NAMES)

# --------------------------------------------------------------------------
# Fetching
# --------------------------------------------------------------------------

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)
ROBOTS_USER_AGENT = "*"

# Domain allow-list: only these hosts may ever be fetched (guardrail).
ALLOWED_DOMAINS: frozenset[str] = frozenset(
    {
        "groww.in",
        "www.groww.in",
        "www.amfiindia.com",
        "amfiindia.com",
    }
)

REQUEST_DELAY_SECONDS = 1.0  # PRD: one request per second
REQUEST_TIMEOUT_SECONDS = 40
CACHE_MAX_AGE_HOURS = 24

# LLM call budget. The SDK default is a 10-minute timeout with 2 retries, which
# means a stalled provider can hang the chat path for ~30 min before the stub
# fallback catches it. 20 s with a single retry keeps the worst case bounded
# while leaving room for one slow-but-succeeding call.
LLM_TIMEOUT_SECONDS = 20
LLM_MAX_RETRIES = 1

# [verified] TLS interception breaks the default CA bundle in this environment.
VERIFY_TLS = False

# Tier labels used in sources.csv
TIER_CORPUS = "corpus"
TIER_CITATION_ONLY = "citation_only"

SOURCE_TYPES = {
    "groww",
    "amc_sid",
    "amc_kim",
    "amc_factsheet",
    "amfi",
    "sebi",
    "rta",
}

# --------------------------------------------------------------------------
# Guardrail copy (verbatim from the PRD)
# --------------------------------------------------------------------------

REFUSAL_LINK = "https://www.amfiindia.com/investor"
REFUSAL_SECONDARY_LINK = "https://investor.sebi.gov.in/index.html"

# No {link} in the template body. `format.py` appends exactly one
# "Source: <url>" line, so inlining the link here as well would print every
# refusal with two URLs and break the one-citation rule. The link is carried on
# GuardResult.link instead.
REFUSAL_TEMPLATE = (
    "I can only share factual information about these schemes, not investment advice. "
    "For help deciding what suits you, please see this investor education resource or "
    "consult a SEBI-registered adviser."
)

PII_BLOCK_MESSAGE = (
    "Please don't share personal data such as PAN, Aadhaar, account or folio numbers, "
    "OTPs, email addresses or phone numbers. I can't accept or store those, and I don't "
    "need them to answer factual questions about these schemes."
)

# The link is NOT inlined here. format.py always appends exactly one
# "Source: <url>" line, so embedding it in the sentence too would print the URL
# twice and break the one-citation rule.
UNKNOWN_TEMPLATE = (
    "I don't have that in my sources. You can check the official {scheme} "
    "documents for it."
)

WELCOME_LINE = (
    "Hi! I answer factual questions about 5 HDFC Mutual Fund schemes, "
    "with a source for every answer."
)

EXAMPLE_QUESTIONS = [
    "What is the expense ratio of HDFC Flexi Cap Fund?",
    "What is the lock-in period for HDFC ELSS Tax Saver?",
    "How do I download my capital-gains statement?",
]

FACTS_ONLY_NOTE = "Facts-only. No investment advice."
PII_INPUT_HINT = "Please don't share PAN, Aadhaar, account numbers, OTPs, email or phone."

# --------------------------------------------------------------------------
# Guard patterns (Phase 5)
# --------------------------------------------------------------------------

# Phrases, not bare words. A single word like "best" or "risk" produces constant
# false positives on legitimate questions ("which is the best time..." is advice,
# but "riskometer" is a core supported fact). Multi-word phrases keep FR-5
# refusals from swallowing answerable questions.
ADVICE_PATTERNS: tuple[str, ...] = (
    r"should\s+i\b", r"should\s+we\b", r"should\s+my\b", r"would\s+you\b",
    r"do\s+you\s+think", r"your\s+opinion", r"what\s+do\s+you\s+recommend",
    r"\brecommend", r"\bsuggest", r"which\s+is\s+better", r"better\s+to\s+(?:buy|invest|choose)",
    r"\bbest\s+(?:scheme|fund|option|choice)", r"is\s+it\s+worth", r"worth\s+investing",
    r"worth\s+it\b", r"good\s+(?:investment|buy|to\s+buy)", r"is\s+it\s+safe",
    # The scheme name often sits between "is" and the adjective, so these match
    # the verb pair directly rather than a fixed "is it <adj>" shape.
    r"\bsafe\s+to\s+(?:invest|buy)", r"\brisky\s+to\s+(?:invest|buy)",
    r"\bworth\s+my\s+money", r"\bshould\s+i\s+have\s+invested",
    r"can\s+i\s+make\s+money", r"how\s+much\s+will\s+i", r"will\s+i\s+(?:get|earn|make)",
    r"\ballocat", r"\bportfolio\b", r"suitable\s+for\s+me", r"which\s+should\s+i",
    r"help\s+me\s+choose", r"right\s+time\s+to\s+invest", r"now\s+a\s+good\s+time",
    r"pros\s+and\s+cons", r"is\s+it\s+too\s+(?:risky|safe)",
)

# FR-6: returns, NAV and performance figures. These are never indexed (the
# allow-list in load.py prevents extraction), so any request for them can only
# be answered with a refusal plus a pointer to the official factsheet.
PERFORMANCE_PATTERNS: tuple[str, ...] = (
    r"\breturns?\b", r"\bcagr\b", r"\bnav\b", r"\baum\b", r"\bperformance\b",
    r"past\s+performance", r"since\s+inception", r"\byield\b", r"\bprofit\b",
    # "how much has it grown" needs the gap left open for an inserted scheme
    # name, so match the verb on its own rather than the whole phrase.
    r"\bgrown\b", r"\brisen\b", r"\bdropped\b", r"\bfell\b", r"\bappreciat",
    r"growth\s+rate", r"worth\s+now\b", r"is\s+it\s+up\s+or\s+down",
    r"how\s+(?:much|many)\s+(?:has|have)\s+(?:it|they|the\s+\w+)\s+\w+",
)

# Other AMCs. A mention means the question is out of scope (FR-10). Matched as
# whole words with a trailing boundary so "sbi" cannot fire inside other text.
OTHER_AMC_PATTERNS: tuple[str, ...] = (
    r"sbi\s+(?:mutual\s+fund|fund|bluechip|flexi|small|large|mid)",
    r"icici\s*(?:directi|mutual|pru)?\s*(?:fund|bluechip|flexi|nifty|small|large|mid|value)",
    r"axis\s+(?:mutual\s+fund|fund|bluechip|flexi|small|large|mid|longterm)",
    r"kotak\s+(?:mutual\s+fund|fund|flexi|small|large|mid|opportunit)",
    r"nippon\s*(?:india)?\s*(?:mutual\s+fund|fund|small|large|mid|flexi|index)",
    r"parag\s+parikh", r"motilal\s+oswal", r"dsp\s+(?:mutual\s+fund|fund|small|large|mid|flexi)",
    r"hdfamf", r"bandhan\s+mf", r"baroda\s+pioneer", r"canara\s+pioneer",
    r"pnb\s+imf", r"union\s+(?:mutual|asset)", r"indus(?:ind\s+)?(?:l\s+indus)?\s*mf",
    r"sbimf", r"tata\s+mf", r"navi\s+mf", r"quant\s+mf", r"jupiter\s+mf",
)

# HDFC schemes that exist but are outside the five-scheme corpus. Caught by
# matching an "HDFC <words> fund" shape and then checking the middle against the
# corpus, so new HDFC schemes are recognised as out of scope without a code
# change here.
HDFC_SCHEME_RE = r"hdfc\s+((?:[a-z]+\s+){0,3}?)funds?\b"

# Facts the corpus does not hold, for which a request must be refused rather than
# answered from a neighbouring attribute.
#
# Why a guard and not a distance. [bug found by tests/demo_ui.py] "What is the
# ticker symbol of HDFC Flexi Cap Fund?" was answered "the benchmark index is
# NIFTY 500 TRI" - a different attribute, stated confidently and cited. Retrieval
# cannot catch this: the query shares the scheme name with every chunk in that
# scheme, so all of them score close, and top-1 landed at distance 0.343, well
# inside the 0.65 threshold. No threshold that still admits real questions also
# rejects this. A wrong sourced fact is worse than a refusal, so coverage is
# checked before the model ever sees a chunk.
#
# The corpus holds exactly 14 attributes (run `python tests/inspect_chunks.py`):
# expense_ratio, base_expense_ratio, exit_load, min_sip, min_lumpsum, lock_in,
# riskometer, benchmark, fund_manager, category, rta, statement_download,
# objective, tax_status. Every term below was verified absent from all 62 chunks.
#
# Terms deliberately NOT here, because they DO appear in the corpus and guarding
# them would refuse a working answer:
#   dividend, factsheet, KIM, SID  - all in the statement_download chunk
#   section 80C, 80C               - in the ELSS tax_status chunk
#   AMC                            - in the statement_download chunk
#   \bAUM\b, \bNAV\b, returns      - already in PERFORMANCE_PATTERNS above
# Word boundaries are load-bearing: bare `ISIN` matches "instruments" and bare
# `closure` matches "disclosures", both of which are in the corpus.
UNCOVERED_ATTRIBUTE_PATTERNS: tuple[str, ...] = (
    # Identity codes. None of these are indexed, and each one is a plausible
    # thing for a user to ask for.
    #
    # `folio number` is deliberately NOT here. The statement_download chunk
    # answers "how do I find my folio number?", and a user *supplying* one is
    # already stopped by the PII guard, which is the correct place for it.
    r"\bticker\b", r"\bsymbol\b", r"\bISIN\b", r"\bfund\s+code\b",
    r"\bapplication\s+number\b",
    # Risk ratios, which sit right next to the covered `riskometer` attribute and
    # are the most likely thing to be answered with the risk level instead.
    r"\bsharpe\b", r"\bsortino\b", r"\btracking\s+error\b", r"\bvolatility\b",
    r"\balpha\b", r"\bbeta\b", r"\bstandard\s+deviation\b",
    # Fund entities and registration details.
    r"\btrustee\b", r"\bcustodian\b", r"\bsebi\s+registration\b",
    r"\bregistration\s+number\b", r"\bCIN\b", r"\bPROT\s+number\b",
    # Documents.
    r"\bprospectus\b", r"\baddendum\b", r"\bkey\s+investor\b",
    r"statement\s+of\s+additional",
    # Tax mechanics. The covered `tax_status` attribute says a unit is eligible
    # for a deduction under section 80C - it does not give rates, so "what tax
    # will I pay" must not be answered with eligibility.
    r"\btax\s+rate\b", r"\btax\s+implication", r"\btax\s+harvesting",
    r"how\s+much\s+tax", r"\bwithdraw(?:al)?\s+tax\b", r"\bTDS\b",
    # Portfolio actions and availability. These are also investment advice, but
    # answering "is STP available" with a neighbouring fact is the failure here.
    r"\bSWP\b", r"\bSTP\b", r"\bgoal\s+index\b", r"\bTARGET\s+100\b",
    # Fund lifecycle. Matched on the verb, not on "launch date" - "when did the
    # scheme launch?" has no "date" in it and slipped through. Boundaries are
    # load-bearing here too: "open-ended" is a real fund category, and bare
    # `closure` matches inside "disclosures", which is in the corpus.
    r"\bNFO\b", r"\bopen(?:ing)?\s+date\b", r"\bopens?\s+for\b",
    r"\blaunch(?:ed|ing)?\b", r"\bsubscription\s+(?:opens|closes|period)\b",
    r"\bmerg(?:e|er|ing)\b", r"\brenam", r"\bclosure\s+date\b",
    # Portfolio disclosures - never indexed by design (no holdings in the corpus).
    r"\bholdings\b", r"\bportfolio\s+composition\b", r"\bsector\s+allocation\b",
    r"\btop\s+\d+\s+stocks?\b",
    # Fund size and contact routing. AUM is handled by PERFORMANCE_PATTERNS.
    r"\bfund\s+size\b", r"size\s+of\s+the\s+fund", r"\bcorpus\b",
    r"\bhelpline\b", r"\bhotline\b", r"\btoll\s+free\b", r"\bcustomer\s+care\b",
    r"\bbranch\s+office\b", r"\bpostal\s+address\b",
)

DISAMBIGUATION_TEMPLATE = (
    "Which of these five HDFC Mutual Fund schemes do you mean?\n"
    "\n"
    "1. HDFC Large Cap Fund\n"
    "2. HDFC Flexi Cap Fund\n"
    "3. HDFC ELSS Tax Saver\n"
    "4. HDFC Small Cap Fund\n"
    "5. HDFC Balanced Advantage Fund"
)

OUT_OF_SCOPE_TEMPLATE = (
    "I only cover 5 HDFC Mutual Fund schemes, so I can't answer questions about "
    "other funds or other asset managers. The schemes I have facts for are:\n"
    "\n"
    "1. HDFC Large Cap Fund\n"
    "2. HDFC Flexi Cap Fund\n"
    "3. HDFC ELSS Tax Saver\n"
    "4. HDFC Small Cap Fund\n"
    "5. HDFC Balanced Advantage Fund"
)

# No {link} here either - format.py appends the single "Source:" line.
PERFORMANCE_REFUSAL_TEMPLATE = (
    "I don't have return, NAV or performance figures for these schemes - my sources "
    "cover static scheme facts only. The official monthly factsheet carries those "
    "numbers."
)

PERFORMANCE_SOURCE_URL = "https://www.hdfcfund.com/"


def _citation_domains() -> frozenset[str]:
    """Domains a citation may point at, derived from sources.csv.

    Deliberately wider than ``ALLOWED_DOMAINS``: that list governs what we may
    *fetch*, while this one governs what we may *display*. Citing a page we did
    not scrape (an RTA portal, the AMFI investor hub) is required by the PRD's
    "exactly one citation" rule, and is not a fetch.
    """
    if not SOURCES_CSV.exists():
        return frozenset()
    hosts: set[str] = set()
    for line in SOURCES_CSV.read_text(encoding="utf-8").splitlines()[1:]:
        for field in line.split(","):
            field = field.strip()
            if field.startswith("http"):
                hosts.add(field.split("//", 1)[1].split("/", 1)[0].lower())
    return frozenset(hosts)



# --------------------------------------------------------------------------
# Minimal .env loader (no extra dependency)
# --------------------------------------------------------------------------


def load_dotenv(path: Path | None = None) -> dict[str, str]:
    """Load KEY=VALUE pairs from .env without overriding real env vars."""
    env_path = path or (ROOT / ".env")
    loaded: dict[str, str] = {}
    if not env_path.exists():
        return loaded
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip("'\"")
        loaded[key] = value
        os.environ.setdefault(key, value)
    return loaded


load_dotenv()


def llm_provider() -> str:
    """Active LLM provider. Falls back to the offline stub when no key exists."""
    provider = os.environ.get("LLM_PROVIDER", "").strip().lower()
    if provider and provider != "stub":
        key_var = {
            "groq": "GROQ_API_KEY",
            "gemini": "GEMINI_API_KEY",
            "openai": "OPENAI_API_KEY",
        }.get(provider)
        if key_var and os.environ.get(key_var):
            return provider
    return "stub"


def display_name(slug: str) -> str:
    return DISPLAY_NAMES.get(slug, slug)


def base_name(slug_or_name: str) -> str:
    """The scheme name without its plan/growth suffix.

    ``display_name`` carries the plan variant because it is needed to match a
    query and to disambiguate two chunks of the same scheme. It reads badly
    anywhere a human is addressed - "the official HDFC Flexi Cap Fund - Direct
    Growth documents" - so scope chips and refusal copy use this instead.
    """
    name = display_name(slug_or_name) if slug_or_name in DISPLAY_NAMES else slug_or_name
    return name.split(" \u2013 ")[0].strip() or name
