# Architecture — Mutual Fund FAQ Assistant (Facts-Only RAG)

Companion to `PRD`. This document locks the technical design. `implementation.md` is the
phase-by-phase build order that realises it.

All findings marked **[verified]** were confirmed by running probes against the live sites on
27 Sep 2026. Do not re-litigate them; build on them.

---

## 1. Locked decisions

| # | Decision | Rationale |
|---|---|---|
| D1 | Facts are extracted from **Groww `__NEXT_DATA__` JSON**, not scraped HTML | [verified] All 5 pages return 200 and embed a complete structured record. HTML scraping would be fragile; JSON is exact. |
| D2 | **hdfcfund.com / SEBI links are citation-only metadata — never fetched** | [verified] hdfcfund.com returns 403 on every path (WAF); sebi.gov.in resets the connection. Both are stored as URLs in `sources.csv` and surfaced to users, but the corpus is built from what is actually fetchable. |
| D3 | **LLM sits behind a provider-agnostic adapter** with a deterministic offline stub | [verified] No API key exists in this environment. Every phase and the whole test suite must pass with zero keys. |
| D4 | **Streamlit** for the UI | Already installed (1.45.1). Gradio is not. |
| D5 | AMFI is the refusal link; SEBI is the secondary | [verified] `amfiindia.com/investor` returns 200. |
| D6 | Returns/NAV/holdings are **structurally excluded at extraction time** | Enforces FR-6 in code, not by prompt. A field never extracted cannot leak. |

### Source-policy honesty

The PRD's "official sources only" rule conflicts with its own required 5-URL Groww deliverable.
This build resolves it transparently rather than pretending:

- **Corpus tier (fetched):** Groww scheme pages — the 5 required URLs.
- **Citation tier (link-only):** HDFC AMC, AMFI, SEBI, RTA portals.
- Every `sources.csv` row carries a `tier` column (`corpus` | `citation_only`) and a
  `fetch_status` column (`ok` | `blocked_403` | `unreachable`).
- README and `sample_qa.md` must state plainly that scheme facts are sourced from a
  distributor platform, and name the official document a user should verify against.

---

## 2. Verified environment

| Item | Finding |
|---|---|
| Python | 3.13.5 (anaconda base), pip 25.1 |
| Installed | chromadb 1.5.9, sentence-transformers 6.1.0, streamlit 1.45.1, beautifulsoup4 4.12.3, requests 2.32.3, pytest 8.3.4 |
| **Missing** | pdfplumber, playwright, all LLM SDKs → install in Phase 0 |
| Embedding model | `all-MiniLM-L6-v2` **downloads OK**, 384-dim, ~22 s cold load |
| Windows quirk | HF cache warns symlinks unsupported; set `HF_HUB_DISABLE_SYMLINKS_WARNING=1` |
| API keys | **None** for OpenAI/Anthropic/Groq/Gemini/OpenRouter. `api.openai.com` and `api.groq.com` are network-reachable (401/404 without key) |

### robots.txt (verified)

- `groww.in` — **`/mutual-funds/<slug>` is ALLOWED.** Disallowed: `/dashboard/`, `/onboarding/`,
  `/pages/*`, `/v1/api/*`, `/mutual-funds/filter?*`, `/mutual-funds/compare/*`,
  `/mutual-funds/user/`, `/stocks/positions/`, `/stocks/user/`, `/report/`, `/user/`,
  `/api/time/server`.
  → All 5 target URLs are compliant. Never fetch `/mutual-funds/compare/*`.
- `amfiindia.com` — Disallow `/admin/`, `/login/`, `/search/`. Rest allowed.
- `hdfcfund.com` — robots.txt itself returns 403.
- `sebi.gov.in` — connection reset from this machine.

Fetch policy: allow-list of exact domains, ≥1 s between requests, `robots.txt` checked at
runtime, raw HTML cached to `data/raw/`.

---

## 3. Verified field map (the single source of truth for extraction)

All values read from `__NEXT_DATA__ → props.pageProps.mfServerSideData`.

| Attribute | Field | Type | Example |
|---|---|---|---|
| Display name | `scheme_name` | str | `HDFC ELSS Tax Saver Fund Direct Plan Growth` |
| Category | `category` / `sub_category` | str | `Equity` / `ELSS` |
| **Expense ratio** | `expense_ratio` | str | `"1.21"` |
| Base expense ratio | `base_expense_ratio` | str | `"0.97"` |
| **Exit load** | `exit_load` | str | `Exit load of 1% if redeemed within 1 year` |
| **Min SIP** | `min_sip_investment` | int | `500` |
| Min lumpsum | `min_investment_amount` | int | `500` |
| SIP/lumpsum allowed | `sip_allowed`, `lumpsum_allowed` | bool | `True` |
| **Lock-in** | `lock_in` `{years,months,days}` | dict | `{"years":3,"months":0,"days":0}` |
| **Riskometer** | `nfo_risk` | str | `Moderately High` |
| **Fund manager** | `fund_manager` | str | `Vinay Kulkarni` |
| **Benchmark** | `benchmark` / `benchmark_name` | str | `NIFTY 500 TRI` |
| RTA | `rta_details.rta_name` | str | `Cams` |
| ISIN | `isin` | str | `INF179K01YS4` |
| Launch date | `launch_date` | str | `01-Jan-2013` |
| Objective | `description` | str | prose |
| Registrar | `rta_details.address` / `.email` | str | |

### Verified per-scheme values (as of 27 Sep 2026 — do not hard-code, re-read at ingest)

| Slug | Sub-cat | Exp. ratio | Min SIP | Lock-in | Riskometer | Manager | Benchmark |
|---|---|---|---|---|---|---|---|
| `hdfc-large-cap` | Large Cap | 1.03 | 100 | none | Moderately High | Prashant Jain | NIFTY 100 TRI |
| `hdfc-flexi-cap` | Flexi Cap | 0.77 | 100 | none | Moderately High | Prashant Jain | NIFTY 500 TRI |
| `hdfc-elss-tax-saver` | ELSS | 1.21 | 500 | **3 years** | Moderately High | Vinay Kulkarni | NIFTY 500 TRI |
| `hdfc-small-cap` | Small Cap | 0.78 | 100 | none | Moderately High | Chirag Setalvad | BSE 250 SmallCap TRI |
| `hdfc-balanced-advantage` | Dynamic Asset Allocation | 0.78 | 100 | none | Moderately High | Srinivas Rao Ravuri | NIFTY 50 Hybrid Composite Debt 50:50 Index |

### EXCLUDE list — never extract, never index (enforces FR-6 / non-goals)

`stats`, `return_stats`, `simple_return`, `sip_return`, `analysis`, `holdings`, `nav`,
`nav_date`, `aum`, `portfolio_turnover`, `groww_rating`, `crisil_rating`,
`historic_fund_expense`, `historic_exit_loads`, `fund_manager_details[*].funds_managed`
(42 unrelated schemes per manager — pure retrieval poison),
**`fund_manager_details[*].person_name`** (see the data-quality trap below).

Enforce with an **allow-list** (`ALLOWED_FIELDS`), not a deny-list. New fields default to
excluded, so upstream page changes cannot silently leak returns into the corpus.

### Data-quality traps found during Phase 1–2 implementation

Both of these were caught by inspecting real output, and both would have quietly broken the
≥90% accuracy target:

1. **`fund_manager_details` is cross-scheme noise — never index it.** [verified] "Dhruv Muchhal"
   appears in the manager list of *all five* schemes, and the page's own stated `fund_manager`
   value (Prashant Jain, Vinay Kulkarni, Srinivas Rao Ravuri) is **missing from four of the
   five** lists. Balanced Advantage lists six names, none of them its stated manager. This list
   is clearly a related-funds / AMC-wide widget, not this scheme's management. Only the singular
   `fund_manager` field is authoritative. `tests/check_phase2.py` asserts no chunk names a
   cross-scheme manager.

2. **`nfo_risk` is inconsistent across schemes.** [verified] Large Cap and Small Cap return
   `"Moderately High Riskometer"`; Flexi Cap and ELSS return `"Moderately High"`. Always
   `normalise_risk()` before it reaches a chunk.

3. **Exit-load strings need two fixes.** [verified] they contain `\r\n`, and the
   balanced-advantage value has a missing space (`investment,1%`). One scheme states a
   conditions *clause* rather than a bare value, so the card phrasing must branch — see
   `_exit_load_card`.

4. **Python's `urllib.robotparser` cannot be trusted for wildcard rules.** It compares rule
   paths with a plain `str.startswith`, so it ignores `*` and `$`. Against Groww's robots.txt
   that makes `Disallow: /mutual-funds/compare/*` never match, and the fetch is wrongly
   permitted. `load.py` ships a small RFC 9309 matcher (`RobotsRules`) that compiles rules to
   regexes and resolves longest-match-wins.

### Normalisation required

1. `nfo_risk` — values are inconsistent: `"Moderately High Riskometer"` vs `"Moderately High"`.
   Strip a trailing `" Riskometer"`, then title-case to one of the six SEBI levels.
2. `exit_load` — contains `\r\n`; collapse all whitespace.
3. Display name — do **not** use `super_category` (inconsistent: `"HDFC Flexi Cap Direct Plan-Growth"`).
   Use a curated `DISPLAY_NAMES` map keyed by slug, with `scheme_name` as fallback.
4. `lock_in` — render only when `years+months+days > 0`.

---

## 4. Data flow

```
INGESTION (offline, `python ingest.py`)
  sources.csv (allow-list) → robots check → rate-limited fetch → data/raw/*.html
      ↓
  load.py      parse __NEXT_DATA__ → mfServerSideData → ALLOWED_FIELDS only
      ↓
  clean.py     normalise (risk levels, whitespace, display names)
      ↓
  chunk.py     fact_card (1 attr × 1 scheme) + curated manual steps chunks
      ↓  context header prepended: "[Scheme] | [Section] | [Source type]"
  embed_store.py  MiniLM 384-dim, batch 32 → ChromaDB hdfc_mf_faq (cosine), upsert by id
      ↓
  sources.csv refreshed with fetched_at

QUERY (per message, app.py)
  message
      ↓
  guard.py   PII ──hit──▶ refuse, never log, never embed
             advice ──▶ refusal template + AMFI link
             out-of-scope AMC/scheme ──▶ scope statement
             ambiguous "HDFC fund" ──▶ disambiguation question
      ↓ (survives)
  retrieve.py  alias→scheme, where={scheme in [s, general]}, top-5, cosine dist
             dist > threshold (start 0.6) ──▶ "not in my sources" + closest official link
      ↓
  generate.py  LLM adapter, temp 0, JSON {answer, source_chunk_id}
      ↓
  format.py   look up source_url + fetched_at from CHUNK METADATA (never from model text)
              strip model-written URLs, strip % figures, hard-trim to 3 sentences
              append "Source: <url>" + "Last updated from sources: <date>"
```

---

## 5. Chunk schema

`data/processed/chunks.jsonl`, one JSON object per line. `id` = `sha1(source_url + "#" + attribute + "#" + scheme)[:12] + "_" + f"{n:03d}"` → stable across re-runs, so upsert is idempotent (FR: refreshed pages replace old chunks).

| Field | Type | Example |
|---|---|---|
| `id` | str | `a1f3c9d2e107_004` |
| `document` | str | `"HDFC Small Cap Fund – Direct Growth \| Exit load \| Groww scheme page. HDFC Small Cap Fund – Direct Growth: exit load is 1% if redeemed within 1 year."` |
| `scheme` | str | `hdfc-small-cap` or `general` |
| `attribute` | str | `exit_load`, `expense_ratio`, `lock_in`, `riskometer`, `benchmark`, `fund_manager`, `category`, `min_sip`, `statement_download`, `other` |
| `source_url` | str | page URL |
| `source_type` | str | `groww`, `amc_sid`, `amc_kim`, `amc_factsheet`, `amfi`, `sebi`, `rta` |
| `fetched_at` | str | `2026-09-27` (ISO date) |
| `chunk_type` | str | `fact_card`, `prose`, `table`, `steps` |

**Chroma constraint:** metadata values must be `str | int | float | bool`. `None`, lists and
dicts raise. Coerce every value with a `clean_meta()` helper before upsert.

### Fact cards (one attribute per scheme, 30–80 tokens)

Exactly the shape the PRD asks for, e.g.
`"HDFC ELSS Tax Saver – Direct Plan Growth: lock-in period is 3 years from each investment date."`

One attribute per chunk is what makes retrieval sharp, and keeps every chunk far under the
256 word-piece limit where MiniLM silently truncates.

### Curated manual chunks (statement / tax documents) — REQUIRED

**[verified gap]** `mfServerSideData` contains **no** statement-download procedure. It has only
`rta_details.{rta_name,address,email}`. The 3 statement questions in the golden set therefore
**cannot** be answered from the scheme JSON.

Ship `data/manual/statement_steps.json` — hand-written, each entry
`{title, steps[], source_url, source_type: "rta", tier}`. Verified-reachable citation targets:

- `https://mfs.kfintech.com/investor/` → 200
- `https://investor.kfintech.com/` → 200
- `https://www.camsonline.com/` → 200 (JS-rendered; link only)
- `https://www.amfiindia.com/investor` → 200

These are chunked as `chunk_type: "steps"`, `scheme: "general"`, `attribute: "statement_download"`.
Every step must stay a whole procedure; split only between steps if a chunk exceeds 200 tokens.

---

## 6. Retrieval

1. Alias map → scheme slug (`"flexi cap"`, `"equity fund"`, `"hdfc equity"` → `hdfc-flexi-cap`).
   `scheme: general` chunks always included in the filter.
2. `col.query(query_embeddings=[q], n_results=5, where={"scheme": {"$in": [...]}})`.
3. Collection created with `metadata={"hnsw:space": "cosine"}` → returned `distances` are cosine
   distances in `[0, 2]`.
4. `distances[0][0] > 0.6` → FR-8 "I don't have that in my sources" + nearest official link.
   Tune on the golden set (Phase 7).
5. Because the five pages use near-identical wording, the scheme-name header inside every chunk
   plus the metadata filter are both mandatory — not optional belt-and-braces.

## 7. Generation

Adapter interface (`src/query/llm.py`):

```python
class LLMClient(Protocol):
    def complete_json(self, system: str, user: str) -> dict: ...
```

- `StubLLM` — deterministic, offline, extractive. Picks the top chunk and composes an answer from
  its own sentences. **Default when no API key is set.** Lets the full test suite, the demo and
  `sample_qa.md` be produced with zero credentials.
- `GroqClient` / `GeminiClient` / `OpenAIClient` — thin wrappers, temperature 0, JSON mode.
- Selected by `LLM_PROVIDER` env var; unknown value falls back to `StubLLM` with a warning.

Model returns `{answer, source_chunk_id}`. **Code** resolves `source_chunk_id` → `source_url` +
`fetched_at`. The model can never invent a link, because it never supplies one.

### Post-checks (in order)

1. Resolve `source_chunk_id` in a code-side chunk map; reject IDs not in the retrieved set.
2. Strip any `http(s)://` token the model emitted.
3. Strip `%`/percent return figures (FR-6).
4. Hard-trim to 3 sentences.
5. Append `Source: <url>` and `Last updated from sources: <date>`.

## 8. Guardrails (all in code, before and after the LLM)

| Gate | Position | Behaviour |
|---|---|---|
| PII | before retrieval | Regex; fixed reply; **message never logged or stored** |
| Advice | before retrieval | Keyword rules → refusal template + AMFI link |
| Performance | before retrieval | Return/NAV phrasing → factsheet pointer, no numbers |
| Out-of-scope | before retrieval | Names the five covered schemes only |
| Disambiguation | before retrieval | Bare "HDFC fund" → one question listing the five |
| Citation | after LLM | URL only from chunk metadata; exactly one |
| Length | after LLM | Hard trim to 3 sentences |
| No-% | after LLM | Percent figures stripped |

**PII patterns (FR-7):** PAN `[A-Z]{5}[0-9]{4}[A-Z]`; Aadhaar 12 digits with optional spaces;
mobile `(\+91)?[6-9]\d{9}`; email; OTP 4–8 digits near "OTP"; folio/account long digit runs near
"folio"/"account".

**Refusal template (verbatim):**
> I can only share factual information about these schemes, not investment advice. For help
> deciding what suits you, please see this investor education resource or consult a SEBI-registered
> adviser. Source: <AMFI/SEBI link>

**FR-8 unknown:** `I don't have that in my sources.` + closest relevant official link.

**FR-10 out-of-scope:** states the assistant covers only the five listed HDFC schemes.

## 9. Repo layout

```
mf-faq-rag/
  data/raw/                  # cached HTML (gitignored)
  data/manual/               # curated statement_steps.json
  data/processed/chunks.jsonl
  chroma_db/                 # persistent store
  src/ingest/{load,clean,chunk,embed_store}.py
  src/query/{guard,retrieve,generate,format,llm,config}.py
  tests/{golden.json,test_*.py}
  app.py  ingest.py  sources.csv  requirements.txt
  README.md  sample_qa.md  DISCLAIMER.md  architecture.md  implementation.md
  .env.example               # committed; .env never committed
```

## 10. Known limits (for README)

Covers 5 HDFC Direct-Growth schemes only; English only; facts reflect the **fetch date**, not
live data; no account-specific help; no returns, comparisons or rankings; scheme facts are
sourced from a distributor platform (Groww), not the AMC — users should verify against the
official HDFC SID/KIM/factsheet; Playwright fallback is not installed, so a JS-only field
regression would show up as a missing fact rather than a crash.
