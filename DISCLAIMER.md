# Disclaimer

> Mutual fund investments are subject to market risks. Read all scheme related
> documents carefully before investing. Investment in mutual funds is never
> guaranteed and investors are advised to make their own due diligence before
> investing. The value of investments can go down as well as up.

## ⚠ The snippet above is a reconstruction, not a verified copy

`implementation.md` Phase 7 task 7 requires the snippet "verbatim", but the
`PRD` file in this repository is **0 bytes** - it is empty, and the snippet
appears nowhere else in the tree. So the wording above is standard SEBI/AMFI
boilerplate written to fit the brief, **not** text recovered from your PRD.

**Replace it with the real text before this ships.** Drop the snippet in from the
PRD and delete this section. The `Sources checked` note at the bottom lists what
was searched, so you can see the gap rather than take my word for it.

What *is* verified is task 7's sibling requirement, at `implementation.md` line
325: the UI carries a persistent *"Facts-only. No investment advice."* note
beneath the input, asserted by `tests/test_ui.py`.

## What this project is, and is not

This is a **facts-only retrieval demo** over five HDFC Mutual Fund schemes. It
is not a distribution platform, not a registered investment adviser, and not a
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
live data - the ingest date appears in the footer of every reply as
*"Last updated from sources: <date>"*.

Verify anything that matters against the official
[HDFC Mutual Fund documents](https://www.hdfcfund.com/) - the Scheme Information
Document (SID), Key Information Memorandum (KIM) and the monthly factsheet.
Those are the authoritative versions; this demo is not.

## No fiduciary relationship

Nothing here creates an adviser-client or fiduciary relationship. The assistant
is software that quotes a small, fixed set of indexed documents, and it is wrong
in ways a human adviser would not be - it cannot know your circumstances, and it
will confidently refuse a question that a human could answer.

## Open items

- **Statement/tax-document menu labels are unverified.** The step-by-step
  download procedures in `data/manual/statement_steps.json` carry
  `"ui_verified": false`; the exact menu labels have not been confirmed against
  a live CAMS or KFintech login. Treat those answers as indicative.
- **Source policy needs sign-off.** hdfcfund.com returns HTTP 403 to
  automated requests, so official SID/KIM/factsheet documents could not be
  ingested automatically and are cited but never fetched. See `sources.csv`.

## Sources checked for the snippet

Searched, all empty or absent:

| Source | Result |
|---|---|
| `PRD` (repo root) | 0 bytes |
| `architecture.md` | no disclaimer snippet |
| `implementation.md` | describes the task, does not contain it |
| `src/query/config.py` | only the refusal and facts-only copy |
| `data/manual/statement_steps.json` | no disclaimer snippet |

Note: `implementation.md` line 347 attributes `DISCLAIMER.md` to Phase 7 task 7
while the summary at the top of this file lists it under Phase 6 - the task
numbering in the plan is inconsistent, but the requirement itself is
unambiguous.
