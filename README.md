# Ask the 10-K

A local RAG assistant that answers questions about Apple's three most recent
10-K filings with **exact citations**, and **refuses to guess**. One company,
three documents, no frameworks — hand-written retrieval so the raw mechanics are
visible (PRD §1).

> **Status:** under construction, milestone by milestone (M0 → M6). This README
> is the single local-execution manual — it is updated in every PR that changes a
> command or adds a pending item (FR-10). When M6 merges, this file alone takes
> you from a fresh clone to a working system.

---

## Architecture

Two paths plus a harness (PRD §6):

- **Ingestion** (run once; re-run when parsing improves):
  `fetch_filings.py` → `ingest/parse.py` → `ingest/chunk.py` → `ingest/index.py`
  (EDGAR download → sections + tables→markdown → chunks + FY metadata →
  `bge-small` embeddings into persistent Chroma).
- **Query loop** (interactive): question → fiscal-year detection → Chroma top-k
  retrieval → strict grounded prompt → `call_llm()` → streamed, cited answer.
  Used identically by the CLI (`app/ask.py`) and the Streamlit UI (`app/ui.py`).
- **Eval harness** (`evals/run_evals.py`): replays `evals/questions.csv` through
  the *same* query loop and records results for human grading.

Swap seams (ring two): all LLM calls go through `call_llm()`, all vector-store
access through `retrieve()` — both in `app/core.py`.

---

## Prerequisites

- **Python 3.11+**
- A virtual environment (`venv`)
- Internet access to: SEC EDGAR (filings), Hugging Face (one-time ~130 MB
  embedding-model download), and your chosen LLM provider.
- An LLM API key — Google AI Studio (primary) or Groq (optional). See `.env`.

## Install

```bash
git clone https://github.com/Luqman-Ud-Din/filings-iq.git
cd filings-iq
python3 -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## `.env` setup

```bash
cp .env.example .env
```

Then edit `.env`:

| Key | Required | Notes |
|-----|----------|-------|
| `PROVIDER` | yes | `gemini` (primary) or `groq` |
| `GOOGLE_API_KEY` | if `PROVIDER=gemini` | from https://aistudio.google.com/apikey |
| `GROQ_API_KEY` | if `PROVIDER=groq` | from https://console.groq.com/keys |
| `MODEL_NAME` | yes | `gemini-2.5-flash` or `llama-3.3-70b-versatile` |
| `SEC_USER_AGENT` | yes (for fetch) | `Your Name your@email.com` — SEC 403s anonymous clients |
| `TOP_K` | no (default 6) | retrieval top-k |
| `EMBED_MODEL` | no | `BAAI/bge-small-en-v1.5` |

---

## Command sequence (run order)

> Filled in as each milestone lands. Full sequence: fetch → parse → chunk →
> index → ask → evals → UI (PRD §10.1).

```bash
# M0 — download filings + metadata (needs SEC_USER_AGENT in .env)
python fetch_filings.py

# Inspect which filings would be fetched, without downloading:
python fetch_filings.py --dry-run

# M1 — parse filings into sections + markdown tables
python -m ingest.parse                    # summarize every filing in data/raw
python -m ingest.parse --item 8           # print Item 8 (income statement) of the most recent FY
python -m ingest.parse --fy 2024 --item 7 # print Item 7 (MD&A) of FY2024

# M2 — chunk sections and build the vector index
python -m ingest.chunk                    # build all chunks, print stats
python -m ingest.chunk --sample 5         # QA: 5 random chunks + metadata
python -m ingest.index --dry-run          # chunk only (no model download / no Chroma)
python -m ingest.index --rebuild          # embed (bge-small, CPU) into persistent Chroma

# M3 — ask grounded, cited questions (streams the answer)
python -m app.ask "What was services revenue in FY2024?"
python -m app.ask --k 8 "How did risk factors change between FY2023 and FY2024?"

# M4 — replay the eval set through the SAME path, then score after grading
python -m evals.run_evals                       # writes evals/results_<ts>.csv (empty grade column)
python -m evals.run_evals --score evals/results_<ts>.csv   # §11.3 metric from your grades

# (UI command added in the last milestone)
```

Writes `data/raw/AAPL_10-K_FY<year>.html` (three files) and
`data/raw/filings_meta.json`. Fiscal year comes from each filing's
`report_date` (§9.1), never its filing date.

---

## Pending human verification

This checklist aggregates every item from merged PRs that needs *your* machine
(network access this build environment lacks, API keys, or human grading). Work
through it top to bottom after a fresh clone.

- [ ] **(M0)** Run `python fetch_filings.py` with a real `SEC_USER_AGENT` and
      confirm three `data/raw/AAPL_10-K_FY<year>.html` files (a few MB each) plus
      `data/raw/filings_meta.json` appear. *This build environment's egress
      policy blocks `data.sec.gov`, so the live download could not be run here.*
- [ ] **(M1)** After the fetch, run `python -m ingest.parse --item 8` and confirm
      the most-recent FY's **consolidated income statement** prints as a clean
      markdown table with figures under the correct fiscal-year columns (FR-1 AC).
      *Verified here only on a representative fixture — the real filings were not
      downloadable in this environment.*
- [ ] **(M2)** Run `python -m ingest.index --rebuild` on the real corpus and
      confirm it prints the final chunk count and completes on CPU in ≤ ~10 min
      (FR-3 AC). The first run downloads the `bge-small` model (~130 MB) from
      Hugging Face. *Both the model download and Chroma indexing were unrunnable
      here (Hugging Face blocked, heavy deps deferred); chunking itself is
      verified — `python -m ingest.chunk --sample 5` shows full §9.2 metadata.*
- [ ] **(M2 — decision point)** `bge-small-en-v1.5` accepts **512 model tokens**;
      chunks near the PRD's 800-token ceiling are silently truncated at embed
      time (see "Known deviations" below). Decide during M4/M5 tuning whether to
      lower the chunk ceiling toward ~512.
- [ ] **(M3)** With the index built and an API key in `.env`, run the three demo
      questions and confirm each answer carries ≥1 citation, plus one
      deliberately-unanswerable question that refuses (FR-5 AC):
      `python -m app.ask "What was services revenue in FY2024?"`. *Needs the live
      index + LLM — unrunnable here (no key, index not built).*
- [ ] **(M3)** Confirm FR-4's AC: ask an FY2024 question and check
      `data/traces.jsonl` shows **only** `FY2024-*` chunk ids for that query.
      *The year-filter logic is unit-verified; the live trace needs the real
      index.*
- [ ] **(M4 — do this BEFORE any tuning, §11.2)** Verify each `golden_answer` in
      `evals/questions.csv` against the real filings and fix any that are wrong;
      the drafted answers are marked `UNVERIFIED` and are **excluded from the
      metric** until you confirm them. Adjust the years if your ingested corpus
      differs from FY2022–FY2024 (see `filings_meta.json`; do not hardcode years).
- [ ] **(M4)** Produce the baseline: `python -m evals.run_evals` (writes
      `evals/results_<ts>.csv`), hand-grade the `grade` column per §11.3, then
      `python -m evals.run_evals --score <file>`. Record the baseline % in the
      eval table below. *Expect ~50–60% once graded — that is normal (§12).
      Unrunnable here: needs the live index + `GOOGLE_API_KEY`.*

---

## Known deviations from the PRD

- **Chunk size vs. embedding window (flagged, not worked around).** FR-2 targets
  ~500–800 token chunks; the pinned embedder `bge-small-en-v1.5` (§7) accepts
  only **512 model tokens** and silently truncates longer inputs. The chunker
  keeps the PRD's 800-token ceiling as specified rather than quietly overriding
  it; the practical effect is that the largest chunks embed on their first ~512
  tokens. This is a tuning knob for M4/M5 (chunk size is exactly the lever §14
  points at for retrieval misses) and a decision for you — not a silent change.

## Troubleshooting

- **SEC HTTP 403 on fetch** — your `SEC_USER_AGENT` is missing or still the
  placeholder. Set it to `Your Name your@email.com` in `.env`. SEC rejects
  anonymous clients.
- **First-run embedding download** — the first `index` run downloads
  `bge-small-en-v1.5` (~130 MB) from Hugging Face; this is a one-time cost and
  needs network access.
- **Free-tier 429s** — the Gemini/Groq free tiers rate-limit; `call_llm()`
  retries once with backoff (honoring `retry-after`), and eval runs are paced
  (~7 s/call on Gemini free tier). If you still hit limits, wait and re-run.

---

## Eval results

The project's source of truth is the human-graded eval score over 25 questions
(PRD §11). Metric = `(grounded_correct + refused_correctly) / 25`.

| Step | What changed | Score |
|------|--------------|-------|
| _baseline_ | initial pipeline | _pending human grading_ |

_(Baseline → final changelog is filled in as M4/M5 PRs merge and you grade the
results; scores are never auto-filled.)_
