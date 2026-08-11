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

# (parse / chunk / index / ask / evals / UI commands added in later milestones)
```

---

## Pending human verification

This checklist aggregates every item from merged PRs that needs *your* machine
(network access this build environment lacks, API keys, or human grading). Work
through it top to bottom after a fresh clone.

- [ ] **(M0)** Run `python fetch_filings.py` with a real `SEC_USER_AGENT` and
      confirm three `data/raw/AAPL_10-K_FY<year>.html` files (a few MB each) plus
      `data/raw/filings_meta.json` appear. *This build environment's egress
      policy blocks `data.sec.gov`, so the live download could not be run here.*

---

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
