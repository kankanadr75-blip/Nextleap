# Disclaimer

## The snippet shown in the UI

Reproduced verbatim from the PRD ("UI and disclaimer" → *Disclaimer snippet
(deliverable)*):

> Facts-only. No investment advice. This assistant shares publicly available
> information about selected HDFC Mutual Fund schemes and cites a source for
> every answer. It does not recommend buying, selling or holding any scheme.
> Mutual fund investments are subject to market risks; read all scheme-related
> documents carefully. For personal advice, consult a SEBI-registered investment
> adviser.

In the running app this is rendered as the persistent note beneath the input,
via `config.FACTS_ONLY_NOTE` — asserted by `tests/test_ui.py`.

**Provenance note.** The `PRD` file in this repository is 0 bytes, so the snippet
could not be read from the tree. It was taken from
`Downloads/PRD_Mutual_Fund_FAQ_Assistant.docx`, which is the source document for
this project. If that file is not the PRD you meant, replace this section.

The PRD also specifies this input hint, which the UI renders as the chat input's
placeholder area:

> Please don't share PAN, Aadhaar, account numbers, OTPs, email or phone.

## What this project is, and is not

This is a **facts-only retrieval demo** over five HDFC Mutual Fund schemes. It is
not a distribution platform, not a registered investment adviser, and not a
recommendation of any scheme.

Concretely, the assistant will not:

- recommend a scheme, or say which is better, safest or most suitable;
- state or imply returns, NAV, AUM, CAGR or any performance figure;
- rank, compare or score schemes against each other;
- give account-specific guidance, or handle personal data of any kind.

When asked any of the above it refuses and points at
[AMFI's Investor Corner](https://www.amfiindia.com/investor) or a
SEBI-registered adviser.

## About the facts it does give

Every answer cites exactly one source, and the source is a **distributor
platform (Groww), not the fund house**. Values reflect the **fetch date**, not
live data — the ingest date appears in the footer of every reply as
*"Last updated from sources: <date>"*.

Verify anything that matters against the official
[HDFC Mutual Fund documents](https://www.hdfcfund.com/) — the Scheme Information
Document (SID), Key Information Memorandum (KIM) and the monthly factsheet.
Those are the authoritative versions; this demo is not.

## No fiduciary relationship

Nothing here creates an adviser-client or fiduciary relationship. The assistant
is software that quotes a small, fixed set of indexed documents, and it is wrong
in ways a human adviser would not be — it cannot know your circumstances, and it
will confidently refuse a question that a human could answer.

## Open items

- **Statement/tax-document menu labels are unverified.** The step-by-step
  download procedures in `data/manual/statement_steps.json` carry
  `"ui_verified": false`; the exact menu labels have not been confirmed against
  a live CAMS or KFintech login. Treat those answers as indicative.
- **Source policy needs sign-off.** hdfcfund.com returns HTTP 403 to automated
  requests, so official SID/KIM/factsheet documents could not be ingested
  automatically and are cited but never fetched. See `sources.csv`.
