# Implementation Plan — Mutual Fund FAQ Assistant

Build order for Cursor. Read `architecture.md` first — it holds the locked decisions (D1–D6),
the verified field map and the exact chunk schema. This file is the *how* and the *when*.

**Ground rules**

1. Work phases in order. Do not start Phase N+1 until Phase N's exit criteria pass.
2. Every phase ends with a runnable check. Paste the actual output into the phase notes.
3. Findings marked **[verified]** come from live probes on 27 Sep 2026. Trust them; don't re-probe.
4. No phase may require an LLM API key. The default `StubLLM` keeps everything offline and testable.
5. Never hard-code a scheme value (expense ratio, exit load…). Always read it at ingest time.
   `architecture.md` §3 has a snapshot table for *sanity-checking only*.

**Environment (verified)**

| Already installed | Must install |
|---|---|
| chromadb 1.5.9, sentence-transformers 6.1.0, streamlit 1.45.1, beautifulsoup4 4.12.3, requests 2.32.3, pytest 8.3.4 | pdfplumber, (optional) playwright |

`all-MiniLM-L6-v2` downloads fine (384-dim, ~22 s cold). Set
`HF_HUB_DISABLE_SYMLINKS_WARNING=1` to silence the Windows symlink warning.

---

## Phase 0 — Scaffold and config

**Goal:** a repo that imports cleanly and runs with zero credentials.

**Files**
- `requirements.txt` (pin the installed versions above; add `pdfplumber`)
- `.env.example` — `LLM_PROVIDER=stub`, `GROQ_API_KEY=`, `GEMINI_API_KEY=`, `OPENAI_API_KEY=`
- `.gitignore` — `.env`, `data/raw/`, `chroma_db/`, `__pycache__/`
- `src/query/config.py` — `COLLECTION_NAME = "hdfc_mf_faq"`, `EMBED_MODEL`, `TOP_K = 5`,
  `DISTANCE_THRESHOLD = 0.6`, `MAX_SENTENCES = 3`, `DISPLAY_NAMES` map (slug → clean name),
  `SCHEME_ALIASES`. Load `.env` manually (no extra dependency).
- `src/ingest/__init__.py`, `src/query/__init__.py`

**Tasks**
1. Create the tree from `architecture.md` §9.
2. `config.py` must expose `DISPLAY_NAMES` for all five slugs. Do **not** derive display names
   from `super_category` — **[verified]** it yields `"HDFC Flexi Cap Direct Plan-Growth"`.
3. Add a `python -c "import src.query.config"` smoke check.

**Exit criteria**
- `pip install -r requirements.txt` succeeds.
- `python -c "import src.query.config as c; print(c.COLLECTION_NAME, len(c.DISPLAY_NAMES))"`
  prints `hdfc_mf_faq 5`.
- `git status` shows `.env` ignored.

---

## Phase 1 — Loading (`src/ingest/load.py`)

**Goal:** fetch the 5 scheme pages, cache raw HTML, and emit a clean structured record each.

**Files**
- `sources.csv` — schema: `scheme_slug,display_name,source_url,source_type,tier,fetch_status,fetched_at`
- `src/ingest/load.py`

**Tasks**
1. `sources.csv`: the 5 required Groww URLs as `tier=corpus`. Add citation-only rows
   (`tier=citation_only`) for: `https://www.amfiindia.com/investor`,
   `https://www.amfiindia.com/investor/knowledge-center-info?zoneName=riskInMutualFunds`,
   `https://investor.sebi.gov.in/index.html`, `https://investor.sebi.gov.in/riskometer.html`,
   `https://www.hdfcfund.com/`, `https://mfs.kfintech.com/investor/`,
   `https://investor.kfintech.com/`, `https://www.camsonline.com/`.
   Pre-fill `fetch_status` from the **[verified]** probe results — `hdfcfund.com` and SEBI are
   `blocked_403` / `unreachable`; the rest `ok`. Do not let the loader overwrite a
   `citation_only` row.
2. Implement a domain **allow-list**. Refuse any URL whose host isn't in it (guardrail: "only
   listed domains are fetched").
3. Fetch with `robots.txt` respected — parse it with `urllib.robotparser`, cache the result,
   and **skip + log** any URL the rules disallow. **[verified]** all 5 target paths are allowed;
   `/mutual-funds/compare/*` is not, so never probe comparison URLs.
4. Rate limit: ≥1 s between requests. Set a browser-like `User-Agent`. Cache raw HTML to
   `data/raw/<slug>.html`; if the cache exists and is <24 h old, reuse it (dev loop speed).
5. Parse `__NEXT_DATA__`: `json.loads` of the `<script id="__NEXT_DATA__" type="application/json">`
   payload → `props.pageProps.mfServerSideData`.
6. Apply an **allow-list** of fields (`architecture.md` §3). Copy only those keys.
   Never pass the raw dict downstream.
7. Record `fetched_at` (ISO date) per page; write it back to `sources.csv`.
8. Graceful degradation: on a network/parse failure, keep the previous cache, mark the row
   `fetch_status=stale`, and continue. Ingestion must never wipe a good corpus because one
   page 403'd.

**Acceptance checks**
- `python ingest.py --fetch-only` → 5 rows, all `fetch_status=ok`.
- `data/raw/` has 5 non-trivial HTML files (**[verified]** 450 KB–815 KB each).
- Assert no `return`/`nav`/`holdings` key survives into the extracted record.

**Pitfalls**
- `requests` against these hosts needs `verify=False` or a CA-bundle fix; the sandbox has no
  working chain. Keep it behind one `make_session()` helper.
- The flexi-cap `exit_load` string ends with `\r\n` — collapse whitespace in Phase 2, not here.

**Exit criteria:** all 5 records extracted, allow-list enforced, `sources.csv` has real
`fetched_at` values.

---

## Phase 2 — Cleaning and chunking (`src/ingest/clean.py`, `chunk.py`)

**Goal:** normalised, atomic, MiniLM-safe chunks in `data/processed/chunks.jsonl`.

**Files**
- `src/ingest/clean.py`
- `src/ingest/chunk.py`
- `data/manual/statement_steps.json`  ← **do not skip**
- `data/processed/chunks.jsonl`

**Tasks — `clean.py`**
1. `normalise_risk(v)` — strip a trailing `" Riskometer"`, then map to one of the six SEBI
   levels. **[verified]** raw values are inconsistent (`"Moderately High Riskometer"` on Large
   Cap vs `"Moderately High"` on Flexi Cap).
2. `collapse_ws(v)` — strip `\r\n`, collapse runs of whitespace, trim.
3. `format_lock_in(d)` — return `None` when `years+months+days == 0`, else
   `"3 years"`. **[verified]** only ELSS has a non-zero lock-in.
4. `format_money(n)` — `500` → `"₹500"`.
5. Apply `DISPLAY_NAMES` from config for every scheme-bearing string.

**Tasks — `chunk.py`**
6. Build one `fact_card` per (scheme × attribute) for: `expense_ratio`, `exit_load`, `min_sip`,
   `min_lumpsum`, `lock_in`, `riskometer`, `benchmark`, `fund_manager`, `category`.
   Each is **one sentence, 30–80 tokens**, e.g.
   `"HDFC ELSS Tax Saver – Direct Plan Growth: lock-in period is 3 years from each investment date."`
   Skip a card when the source value is absent — never emit an empty or hedged card.
7. Add a `category` card using `category` + `sub_category` (e.g. "Equity – Small Cap").
8. Emit one card for `fund_manager` per manager listed in `fund_manager_details[].person_name`
   when that list is present, falling back to the single `fund_manager` string.
9. Prepend the context header before embedding, exactly:
   `"<display name> | <Section> | <Source type>"`
   e.g. `"HDFC ELSS Tax Saver – Direct Plan Growth | Lock-in | Groww scheme page"`.
10. Hard cap every chunk at **200 tokens** with a 30-token overlap for prose. Assert at build
    time that no chunk exceeds the 256 word-piece MiniLM limit — **[verified]** MiniLM truncates
    silently past 256 word-pieces, so an over-long chunk loses its tail with no error.
11. `data/manual/statement_steps.json`: hand-write the statement / capital-gains download
    procedures. **[verified]** these are *not* in the scheme JSON — only `rta_details.rta_name`
    is. Each entry `{title, steps[], source_url, source_type: "rta"}`, chunked as
    `chunk_type: "steps"`, `scheme: "general"`, `attribute: "statement_download"`. Cite
    `https://mfs.kfintech.com/investor/` or `https://www.camsonline.com/`.
    Cover at minimum: consolidated account statement, capital-gains statement, and tax report
    / 26G-ish documents, HDFC's RTA being **CAMS** (**[verified]**).
12. `id` = `sha1(source_url + "#" + attribute + "#" + scheme)[:12] + f"_{n:03d}"` — stable so
    re-running upserts rather than duplicates.
13. Write `chunks.jsonl`. Then **hand-inspect it** (the PRD requires this).

**Acceptance checks**
- `python ingest.py` → `chunks.jsonl` exists; print counts per `scheme`.
- Expect **≥ 9 fact cards × 5 schemes**, plus statement chunks. Balanced counts across schemes
  (a scheme with 3 cards means extraction silently broke).
- `grep -c` for banned terms in `chunks.jsonl` → `0` for `return1y`, `SIP Return`, `holdings`,
  `NAV`. This is the FR-6 guarantee; make it a test, not a one-off check.
- `python -c` over `chunks.jsonl` asserting no chunk > 200 tokens and no metadata `None`.

**Pitfalls**
- `fund_manager_details[*].funds_managed` lists 42 unrelated schemes per manager. Indexing it
  wrecks retrieval for every manager question. Excluded by the allow-list — keep it that way.
- One attribute per chunk is what makes retrieval sharp. Do not merge expense ratio + exit load
  into one chunk "to save tokens".

**Exit criteria:** `chunks.jsonl` hand-verified; ~50+ chunks; zero banned terms; zero over-long
chunks; all five schemes represented.

---

## Phase 3 — Embedding and storage (`src/ingest/embed_store.py`)

**Goal:** a populated, idempotent ChromaDB collection.

**Files**
- `src/ingest/embed_store.py`

**Tasks**
1. `PersistentClient(path="chroma_db")`;
   `get_or_create_collection("hdfc_mf_faq", metadata={"hnsw:space": "cosine"})`.
2. Load MiniLM **once** via a module-level cache (`functools.lru_cache`) and
   `encode(..., normalize_embeddings=True, batch_size=32)`. Expect 384-dim.
3. Embed `"<header> <chunk text>"` together — the header is part of the embedded string.
4. `clean_meta()` before upsert. **Chroma rejects `None`, lists and dicts in metadata** —
   coerce to `str | int | float | bool`. This is a guaranteed runtime error if missed.
5. `upsert(ids=, documents=, metadatas=, embeddings=)`. Re-running must replace, not duplicate.
6. Print per-scheme counts and the collection total.

**Acceptance checks**
- Run `ingest.py` twice → identical total count. Proves idempotency.
- `col.count()` matches the `chunks.jsonl` line count.
- A sample embedding round-trips at 384 dims.

**Pitfalls**
- First model load costs ~22 s. Load it once at app start, not per query — the <3 s latency
  target depends on it.
- Chroma's default space is L2. Forgetting `hnsw:space: cosine` makes the 0.6 threshold
  meaningless.

**Exit criteria:** collection populated, idempotent, cosine-configured, per-scheme counts printed.

---

## Phase 4 — Retrieval (`src/query/retrieve.py`)

**Goal:** correct chunk in the top-5 for the golden set, with a working confidence cutoff.

**Files**
- `src/query/retrieve.py`

**Tasks**
1. `detect_scheme(text)` via `SCHEME_ALIASES`. Return the slug, `None`, or `"ambiguous"`.
   Bare "HDFC fund" with no distinguishing word → `"ambiguous"` (FR-9).
   **[verified]** aliases needed: `flexi cap` / `equity fund` / `hdfc equity` → `hdfc-flexi-cap`;
   `tax saver` / `elss` → `hdfc-elss-tax-saver`; plus `large cap`, `small cap`,
   `balanced advantage`.
2. `retrieve(query, scheme)`:
   `col.query(query_embeddings=[emb], n_results=TOP_K, where={"scheme": {"$in": [scheme, "general"]}})`
   when a scheme is detected; no `where` otherwise.
3. Return `(chunks, best_distance)`.
4. Confidence gate: `best_distance > DISTANCE_THRESHOLD` (start `0.6`) → the caller emits FR-8.
5. Log `intent` + `latency_ms` only. **Never log message text** (FR-7 / no-persistence NFR).

**Acceptance checks**
- A harness that runs 20 golden queries and prints `rank of expected chunk` per query.
- Sanity spot-checks by hand: "exit load HDFC Small Cap" → the Small Cap exit-load card at
  rank 1; "expense ratio Flexi Cap" → the Flexi Cap expense card at rank 1.
- Confirm the `general` statement chunks are reachable for a question with no scheme name.

**Pitfalls**
- The five pages use near-identical wording. Both defences are required: the scheme name inside
  the chunk header **and** the metadata filter. Removing either degrades accuracy.
- `distances` are cosine distances in `[0, 2]`, not similarities. Higher = worse.

**Exit criteria:** top-5 hit rate measured on the golden set, threshold justified, per-query
rank table produced.

---

## Phase 5 — Generation and guardrails

**Goal:** ≤3-sentence, single-citation, refusal-correct answers — enforced in code.

**Files**
- `src/query/guard.py`
- `src/query/llm.py`
- `src/query/generate.py`
- `src/query/format.py`
- `tests/test_guardrails.py`

### 5a — `guard.py` (runs first; short-circuits before retrieval)

1. **PII (FR-7)** — PAN `[A-Z]{5}[0-9]{4}[A-Z]`; Aadhaar `\b\d{4}\s?\d{4}\s?\d{4}\b`;
   mobile `(\+91)?[6-9]\d{9}`; email; OTP 4–8 digits near "OTP"; folio/account long digit runs
   near "folio"/"account". On hit: fixed reply asking the user not to share personal data,
   **return immediately, log nothing, store nothing.**
2. **Out-of-scope (FR-10)** — detect a non-HDFC AMC or a scheme outside the five →
   state the assistant covers only these five.
3. **Ambiguity (FR-9)** — `"ambiguous"` from Phase 4 → one question listing the five schemes.
4. **Performance (FR-6)** — return/NAV phrasing → one-line pointer to the official factsheet,
   no numbers.
5. **Advice (FR-5)** — "should I", "better", "recommend", "best", "invest in", "returns will",
   "which is good", "worth it", "allocation", "portfolio" → verbatim refusal template with the
   AMFI link.
6. Order matters: **PII → out-of-scope → ambiguity → performance → advice.** PII first, always.

### 5b — `llm.py` (provider-agnostic adapter)

7. `class LLMClient(Protocol): def complete_json(self, system: str, user: str) -> dict: ...`
8. `StubLLM` — deterministic, offline, **extractive**: take the top chunk, compose an answer from
   its own sentences. Default when no key is set. This is what makes the demo, the tests and
   `sample_qa.md` work with zero credentials.
9. `GroqClient`, `GeminiClient`, `OpenAIClient` — temperature 0, JSON mode. Select via
   `LLM_PROVIDER`; unknown/absent → `StubLLM` + a one-line warning.
10. Robust `json.loads` (strip code fences, fall back to the first `{...}` block). Never raise
    into the chat path — on failure, fall back to the stub.

### 5c — `generate.py`

11. System prompt: answer **only** from the supplied context; ≤3 sentences; no advice, opinions,
    comparisons or return figures; say so if the context lacks the answer; return JSON
    `{answer, source_chunk_id}`; **never output a URL**.
12. Build the user turn from the retrieved chunks: header, text, `source_url`, `source_chunk_id`.
13. Validate `source_chunk_id` against the retrieved set. Reject unknown IDs.

### 5d — `format.py` (all compliance lives here)

14. Resolve `source_chunk_id` → `source_url` + `fetched_at` **from a code-side chunk map**.
    The model never supplies the link, so it cannot be invented.
15. Strip any `http(s)://…` the model emitted.
16. Strip `%` / percent return figures.
17. Hard-trim to 3 sentences (sentence splitter, not `split('.')` — protect decimals and
    abbreviations).
18. Append `Source: <url>` and `Last updated from sources: <date>` (**[verified]** `fetched_at`
    drives this; it is a fetch date, not a live-data claim).

**Acceptance checks** — `pytest tests/ -q` all green, including:
- every answered response has exactly **1** URL, and it is in the allow-list;
- every answer ≤ 3 sentences (excluding `Source:` and footer lines);
- footer present and matching `fetched_at`;
- 10+ adversarial advice prompts → refusal template, **0** `%` figures;
- 3+ PII prompts → blocked, and the PII string appears nowhere in any log file;
- out-of-scope AMC → FR-10 text;
- "HDFC fund" → FR-9 disambiguation.

**Pitfalls**
- Do not trust the model for length, citations or refusals. Every one of those is re-derived in
  `format.py`. Prompt-only compliance will fail the metrics.
- Refusals must contain no percent figures — a leaked "12.5%" fails the check even mid-sentence.

**Exit criteria:** full guardrail suite green with the stub; same suite green with a real
provider if a key is added.

---

## Phase 6 — UI (`app.py`)

**Files**
- `app.py`

**Tasks**
1. One chat page. Welcome line verbatim:
   *"Hi! I answer factual questions about 5 HDFC Mutual Fund schemes, with a source for every answer."*
2. Three clickable example buttons that send the query:
   - "What is the expense ratio of HDFC Flexi Cap Fund?"
   - "What is the lock-in period for HDFC ELSS Tax Saver?"
   - "How do I download my capital-gains statement?"
3. Scope chip row with the five scheme names.
4. Persistent note under the input: *"Facts-only. No investment advice."*
5. Input hint: *"Please don't share PAN, Aadhaar, account numbers, OTPs, email or phone."*
6. Answer layout: answer text → `Source: <link>` (clickable) → `Last updated from sources: <date>`
   in smaller text.
7. **No chat-history database.** Keep history in Streamlit session state only; persist nothing.
8. Load the embedding model and the Chroma client **once** via `@st.cache_resource`.
9. Render guard responses (PII, refusal, out-of-scope, unknown) through the same formatter so
   every message gets identical layout.

**Acceptance checks**
- `streamlit run app.py` → the three buttons each produce a cited ≤3-sentence answer.
- PII typed into the box → blocked, nothing stored.
- Cold start loads the model once; second message is fast.

**Exit criteria:** all five schemes answerable from the UI, every answer cited, no persistence.

---

## Phase 7 — Evaluation and tuning

**Files**
- `tests/golden.json` — 25–30 labelled cases
- `tests/test_golden.py`, `tests/eval_retrieval.py`
- `sample_qa.md`, `README.md`, `DISCLAIMER.md`

**Tasks**
1. Build the golden set: 15 factual (3 per scheme across attributes), 3 statement/tax-document,
   5 advice/performance (must refuse), 3 PII (must block), 2 out-of-scope.
   Each factual case stores `query`, `expected_scheme`, `expected_attribute`,
   `expected_source_url`, `expected_answer_contains`.
2. `eval_retrieval.py` → top-1 and top-5 hit rates, plus the rank table from Phase 4.
3. **Tune the threshold** on the golden set: sweep `0.4 → 0.8`, pick the value maximising
   top-5 hit rate without letting unknowns through. Record the chosen value and the sweep table.
4. Chunk-size comparison: 128 / 200 / 256 prose tokens. Keep the best top-5 hit rate; **break
   ties toward the smaller chunk** (per the PRD). Record the numbers.
5. `sample_qa.md` — 5–10 real queries with the assistant's **actual** answers and links,
   generated by running the app, not written by hand.
6. `README.md` — setup, scope (AMC + 5 schemes), architecture summary, known limits
   (`architecture.md` §10), and the **source-tier disclosure**.
7. `DISCLAIMER.md` — the PRD snippet verbatim.
8. Final `pytest tests/ -q` + the six PRD success metrics, each with a measured number:
   factual accuracy ≥ 90%, exactly-one-citation 100%, ≤3 sentences 100%, advice refused 100%,
   PII blocked 100%, retrieval top-5 ≥ 85%, median latency < 3 s.
9. Measure median latency over ≥20 queries and record it.

**Exit criteria:** every success metric has a measured number in the README; `sample_qa.md`
contains real output; the six deliverables exist.

---

## Deliverables checklist

- [ ] Working prototype — `streamlit run app.py` (or ≤3-min demo video)
- [ ] `sources.csv` — 5 scheme URLs + tier-2 official URLs, with scheme / type / fetched date
- [ ] `README.md` — setup, scope, architecture, known limits
- [ ] `sample_qa.md` — 5–10 queries with actual answers and links
- [ ] `DISCLAIMER.md` — UI disclaimer snippet
- [ ] `architecture.md`, `implementation.md` (this plan)

## Definition of done

`python ingest.py` runs clean from scratch → `streamlit run app.py` answers every golden
question with one citation → `pytest tests/ -q` green with no API key → all seven success
metrics measured and recorded.

---

## Open items for the human

1. **LLM provider** — adapter is ready; add a key to `.env` and set `LLM_PROVIDER` to move off
   the stub. Groq is the best fit for the <3 s target.
2. **Source-policy sign-off** — the corpus is Groww-sourced with official links citation-only,
   per D2. Someone must accept that trade-off, or supply HDFC SID/KIM/factsheet PDFs manually
   (**[verified]** hdfcfund.com cannot be fetched automatically).
3. **Refusal link** — AMFI Investor Corner is the default; SEBI Investor is the secondary.
   Confirm which is the house standard.
4. **Deployment target** — Streamlit Community Cloud is the easy path; the Hugging Face Spaces
   route needs the Chroma directory committed or rebuilt at boot.
