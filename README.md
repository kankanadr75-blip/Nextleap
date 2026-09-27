# HDFC Mutual Fund FAQ Assistant (facts-only RAG)

A local chat assistant that answers **factual** questions about five HDFC Mutual
Fund schemes, with exactly one source citation per answer. It refuses investment
advice and performance questions outright, blocks personal data before
retrieval, and never invents a citation.

Everything runs on your machine. The only outbound request this project ever
makes is to an LLM provider, and only if you configure one — with no key at all
it uses an offline extractive stub and works exactly the same.

```
python -m pytest tests/ -q        # 151 passed
streamlit run app.py              # the chat UI
```

## Scope

Five HDFC Mutual Fund schemes, Direct Growth only:

| Scheme | Manager | Expense ratio | Min SIP | Lock-in |
|---|---|---|---|---|
| HDFC Large Cap Fund | Prashant Jain | 1.03% | ₹100 | — |
| HDFC Flexi Cap Fund | Prashant Jain | 0.77% (base 0.57%) | ₹100 | — |
| HDFC ELSS Tax Saver | Vinay Kulkarni | 1.21% | ₹500 | 3 years |
| HDFC Small Cap Fund | Chirag Setalvad | 0.78% | ₹100 | — |
| HDFC Balanced Advantage Fund | Srinivas Rao Ravuri | 0.78% | ₹100 | — |

Re-read from the corpus on every ingest; never hard-coded.

Exactly **14 attributes** are indexed: `expense_ratio`, `base_expense_ratio`,
`exit_load`, `min_sip`, `min_lumpsum`, `lock_in`, `riskometer`, `benchmark`,
`fund_manager`, `category`, `rta`, `statement_download`, `objective`,
`tax_status`. Anything else is refused, not guessed — see *Uncovered
attributes* below.

## Setup

```
pip install -r requirements.txt
python ingest.py          # fetch, clean, chunk, embed  ->  62 chunks
streamlit run app.py
```

Ingest is idempotent: re-running rebuilds the index without duplicating it.
`chroma_db/` and `data/raw/` are derived and git-ignored.

Verify the install end to end:

```
python -m pytest tests/ -q      # 151 tests
python tests/eval_retrieval.py  # rank table, threshold sweep, 3 gates
python tests/eval_metrics.py    # the 7 PRD success metrics
python tests/ask.py "expense ratio of HDFC Flexi Cap Fund"
python tests/demo_ui.py         # full rendered transcript
```

### Using a real LLM (optional)

Everything above runs with no credentials. To use a hosted model, copy
`.env.example` to `.env` and set the provider plus its key:

```properties
LLM_PROVIDER=groq
GROQ_API_KEY=gsk_...
GROQ_MODEL=openai/gpt-oss-20b
```

`.env` is git-ignored. Model IDs are overridable per provider
(`GROQ_MODEL` / `OPENAI_MODEL` / `GEMINI_MODEL`).

Two things that will bite you:

- **A bad key fails silently.** A missing key *and* a failed call both fall back
  to `StubLLM` with only a log warning, because the chat path must never raise.
  A wrong key therefore looks like success. Check the client, not the answers:
  `python -c "from src.query.llm import get_llm; print(get_llm().name)"`.
- **`openai` is pinned to 1.x on purpose.** 3.x pulls in `httpx2`, whose
  response decoder is incompatible with this machine's compression modules
  (`TypeError: process() takes no keyword arguments`) — and because it fails
  while *decoding* the response, the stub fallback hides it.

## Measured results

All seven PRD success metrics, measured by `tests/eval_metrics.py` on
27 September 2026 against `groq:openai/gpt-oss-20b`. Re-run it to reproduce;
it writes `tests/_metrics.json`.

| Metric | Target | Measured |
|---|---|---|
| Factual accuracy | ≥ 90% | **100.0%** (20/20) |
| Exactly one citation | 100% | **100.0%** (20/20) |
| ≤ 3 sentences | 100% | **100.0%** (20/20) |
| Advice / performance refused | 100% | **100.0%** (25/25) |
| PII blocked | 100% | **100.0%** (6/6) |
| Retrieval top-5 (hard group) | ≥ 85% | **100.0%** (12/12) |
| Median latency, answered | < 3 s | **1.16 s** |

Also measured:

- Retrieval top-1 on the hard group: **75.0%** (9/12). Top-5 is 12/12, so the
  reranker-free pipeline puts the right chunk in the candidate set every time
  and the generator picks it out.
- Latency, answered questions (n=20): median 1.16 s, p95 7.31 s, max 7.32 s.
  The tail is the hosted provider, not the local stack — retrieval alone is
  ~28 ms median.
- Latency, refusals (n=31): median 0.1 ms. A guard block never reaches
  retrieval or the LLM.
- Cold start: ~39 s, paid once per server process and cached via
  `@st.cache_resource`.

Honest caveat on the latency target: the median passes, the p95 does not. A
3-second target is comfortable for a local stub and tight for a network LLM.
The 7.3 s tail is Groq queueing, and it would fail a p95-based reading of the
requirement.

Retrieval was tuned on the golden set: `DISTANCE_THRESHOLD = 0.65`, swept
0.4 → 0.8 (`tests/eval_retrieval.py`). 0.6 also passes; 0.65 leaves headroom.

## Architecture

```
ingest.py  ->  src/ingest/{load,clean,chunk,embed_store}.py
app.py     ->  src/query/answer.py
                      |
                      +-- guard.py     PII, scope, ambiguity, performance, advice, coverage
                      +-- retrieve.py  scheme detection + Chroma vector search
                      +-- generate.py  LLM call, chunk-id validation
                      +-- format.py    sentence cap, one citation, footer
                      +-- llm.py       StubLLM | Groq | OpenAI | Gemini
```

`answer_question` is the single seam. The UI is a thin renderer over it, which
is what makes the whole chain testable without starting Streamlit and stops the
UI bypassing a guardrail.

**Why the model is not trusted for compliance.** The prompt *asks* for ≤3
sentences, no URL, and a chunk id. `format.py` then *re-derives* every one of
those: the sentence cap is applied to the returned text, the citation comes
from a code-side allow-list rather than anything the model produced, and a
`source_chunk_id` that is not in the retrieved set is rejected and falls back
to a refusal. A fabricated citation is therefore not reachable even if the
model tries.

**The model is never asked for a URL.** It returns `{"answer",
"source_chunk_id"}` and nothing else. The link is derived from the chunk's own
metadata, so hallucination cannot reach the rendered `Source:` line.

**Uncovered attributes are a guard, not a threshold.** "What is the ticker
symbol?" retrieves a chunk at distance 0.343 — comfortably inside 0.65 — because
the query shares the scheme name with every chunk in that scheme, and answers it
confidently with the benchmark index. No distance threshold admits real
questions *and* rejects that one. So `config.UNCOVERED_ATTRIBUTE_PATTERNS`
handles it in `guard.check_coverage`, wired last so specific refusals win. Every
pattern is proven absent from all 62 indexed chunks by
`test_uncovered_patterns_do_not_match_any_indexed_chunk`, so the guard can never
refuse something we do hold.

Full reasoning in `architecture.md`; phase plan and rationale in
`implementation.md`.

## Privacy

- **No chat-history database.** History lives in `st.session_state` and dies with
  the browser session. Nothing is written to disk.
- **PII is blocked before retrieval** and never stored. Guards, the retriever and
  the generator log coarse labels only (`guard=pii`) — never message text. A test
  writes a log file, fires every PII case at it, and asserts the PAN, Aadhaar,
  phone and email strings are absent from the bytes on disk.
- **Telemetry is off** in `.streamlit/config.toml`, and the server binds to
  `127.0.0.1`. Nothing phones home.

## Source-tier disclosure

Read this before trusting a number.

**The indexed facts come from Groww, a distributor platform — not from HDFC
Mutual Fund.** Official AMC documents are cited but never fetched:
`hdfcfund.com` returns **HTTP 403** to every automated request, so the SID, KIM
and monthly factsheet could not be ingested. Full per-URL status is in
`sources.csv` (`tier` and `fetch_status` columns):

| Source | Tier | Status |
|---|---|---|
| groww.in (5 scheme pages) | corpus | fetched 2026-09-27 |
| amfiindia.com, kfintech, cams | citation_only | reachable, cited |
| hdfcfund.com | citation_only | **blocked_403** |
| investor.sebi.gov.in | citation_only | unreachable |

Values reflect the **fetch date**, not live data — the footer of every answer
carries it. Verify anything that matters against the official HDFC SID / KIM /
factsheet.

The citation allow-list is deliberately **wider** than the fetch allow-list: a
domain may be cited as a pointer to a human-readable page without being ingested.
`PERFORMANCE_SOURCE_URL` is an example — 403 to the bot, 200 to a person.

## Known limits

Covers 5 HDFC Direct-Growth schemes only; English only; facts reflect the fetch
date, not live data; no account-specific help; no returns, comparisons or
rankings; scheme facts are sourced from a distributor platform, not the AMC. The
Playwright fallback is not installed, so a JS-only field regression would appear
as a missing fact rather than a crash.

Two open items need a human, and are **not** fixed in this build:

1. **Statement menu labels are unverified.** `data/manual/statement_steps.json`
   carries `"ui_verified": false` on 4 of 5 entries. The step-by-step portal
   procedures were never confirmed against a live CAMS/KFintech login. The
   *content* is generic and correct in substance; the exact menu labels may
   differ. Check before demoing.
2. **Source-policy sign-off.** Someone must accept the Groww-sourced corpus, or
   supply the HDFC SID/KIM/factsheet PDFs manually for ingestion.

One evaluation gap I found and did not close: all 12 `hard` golden cases name a
scheme, so scheme-less attribute questions are untested. A bare
"charges for managing" lands at distance 0.670, just past the threshold, and
refuses via FR-8. Whether that is the right answer needs a product call — the
bot does not know which scheme you mean.

`DISCLAIMER.md` carries a reconstruction of the required snippet, **not** a
verified copy: the `PRD` file in this repo is 0 bytes, so the exact wording could
not be recovered. Replace it before shipping.

## Layout

```
app.py                    Streamlit UI (thin renderer over answer_question)
ingest.py                 one-shot pipeline: fetch -> clean -> chunk -> embed
sources.csv               every source, with tier and fetch status
config.py                 all tunables, guard copy, thresholds
src/ingest/               load, clean, chunk, embed_store
src/query/                guard, retrieve, generate, format, llm, config, answer
data/manual/              curated statement procedures (hand-written)
data/processed/           records.json, chunks.jsonl + the .txt dumps
tests/golden.json         34 labelled cases + 11 detection cases
tests/eval_retrieval.py   rank table, threshold sweep, 3 gates
tests/eval_metrics.py     the 7 PRD success metrics
tests/ask.py              ad-hoc retrieval inspector
tests/inspect_chunks.py   chunks as a plain .txt file
tests/inspect_embeddings.py  vectors as a plain .txt file
tests/demo_ui.py          rendered UI transcript
tests/gen_sample_qa.py    regenerates sample_qa.md from a live run
chroma_db/                vector store (derived, git-ignored)
```

## Tests

151 tests, no network and no API key required. `tests/test_ui.py` drives the real
`app.py` through `streamlit.testing.v1.AppTest` — an HTTP 200 from `streamlit run`
proves nothing, because that is only the static shell while the script itself runs
over a websocket.
