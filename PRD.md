# PRD — Ask the 10-K

| | |
|---|---|
| **Version** | 1.0 (2026-08-11) |
| **Owner** | You (solo developer) |
| **Status** | Ready to build — greenfield, nothing implemented yet |
| **Timebox** | ~2 weeks of evenings |
| **Ring** | 1 of a multi-ring project ("FilingsIQ") — see §15 |

**One-liner:** A local RAG assistant that answers questions about Apple's three most recent 10-K filings with exact citations, and refuses to guess.

---

## 0. How to use this document (note to Claude Code)

- This PRD is the canonical spec. If a request conflicts with it, flag the conflict before coding.
- **Assume an empty repository.** No script, file, or data exists until its milestone builds it — including `fetch_filings.py`. Do not reference or assume prior artifacts.
- Work **one milestone at a time** (§12). Each milestone ends as a pull request carrying its checkpoint evidence; merge and continue per the §13 workflow — there is no human merge gate, but never skip, reorder, or blend milestones.
- Do not add dependencies beyond §7 without asking. Do not implement anything listed in §3 (non-goals).
- The human grades evals (§11). Never fill in eval scores yourself.

---

## 1. Overview

Analysts (and retail investors) answer questions like "what was services revenue in FY2024?" or "how did risk factors change year over year?" by manually searching 300-page SEC filings. This project builds a local assistant over Apple's three most recent 10-K filings that:

1. Answers questions **grounded only in the ingested filings**,
2. **Cites** the filing and section for every claim,
3. **Refuses** when the answer is not in the corpus.

This is a learning project with production habits: eval-gated iteration, tracing, and strict grounding. It is deliberately small (one company, three documents, no frameworks) so the developer learns the raw RAG mechanics that libraries abstract away.

## 2. Goals and definition of done

The project is **done** when all of the following hold:

- **G1.** ≥ 80% of the 25-question eval set is graded `grounded_correct` or `refused_correctly` (§11).
- **G2.** All 5 trap questions produce clean refusals (5/5).
- **G3.** Every non-refusal answer contains at least one citation in the format of §10.3.
- **G4.** A fresh `git clone` + README quickstart reaches a working CLI answer in ≤ 10 minutes (excluding model downloads).
- **G5.** README contains the eval table showing baseline score → final score with a short changelog of what moved it.

## 3. Non-goals (ring one exclusions — do not build)

- No LangChain / LangGraph / LlamaIndex — the pipeline is hand-written.
- No Docker, no Qdrant/Pinecone/Weaviate/pgvector — Chroma embedded mode only.
- No reranker, no hybrid/BM25 search — plain vector similarity + metadata filter.
- No FastAPI / REST layer — CLI and Streamlit only.
- No agents, tools, or live market data.
- No Langfuse or external observability — JSONL trace log only.
- No CI pipeline, no multi-company support, no multi-tenancy, no fine-tuning.
- No LLM-as-judge for evals — human grading at this scale.

Each exclusion is a planned ring-two upgrade; interfaces in §13 exist so those swaps stay cheap.

## 4. Users and example queries

Single user: the developer, acting as an equity analyst.

Representative queries the system must handle:

| Type | Example |
|---|---|
| Lookup | "What was Apple's services revenue in FY2024?" |
| Table | "What was total net sales in FY2023 vs FY2024?" |
| Comparison | "How did the risk factors change between FY2023 and FY2024?" |
| Synthesis | "Summarize what management said about supply chain in the most recent filing." |
| Trap (must refuse) | "What was Apple's revenue in FY2019?" / "What is Microsoft's cloud revenue?" |

## 5. Functional requirements

Each FR has acceptance criteria (AC). "Filing" = one 10-K HTML file produced by FR-0. Nothing is pre-built: every script below, including the fetch script, is written from scratch in its milestone.

**FR-0 — Data acquisition (`fetch_filings.py`).** Download Apple's three most recent 10-K filings from SEC EDGAR as HTML, plus a metadata file.
- Company CIK: `0000320193` (CIKs are zero-padded to 10 digits in the API path).
- Filing list from `https://data.sec.gov/submissions/CIK0000320193.json`. The `filings.recent` object holds **parallel arrays** — index *i* across `form`, `accessionNumber`, `filingDate`, `reportDate`, `primaryDocument` describes one filing.
- Keep entries where `form == "10-K"` exactly (exact match also skips `10-K/A` amendments); take the 3 most recent.
- Document URL pattern: `https://www.sec.gov/Archives/edgar/data/<cik-without-leading-zeros>/<accession-without-dashes>/<primaryDocument>`.
- Every request must send a `User-Agent` header of the form `Name email@example.com` — SEC returns 403 to anonymous clients. The script reads `SEC_USER_AGENT` from `.env` (§10.2) and must exit with an instructive error when it is missing or still a placeholder. Sleep between downloads to stay far below SEC's 10 requests/second limit.
- Save files as `data/raw/AAPL_10-K_FY<year>.html`, where `<year>` = first 4 characters of `reportDate`, and write `data/raw/filings_meta.json` per §9.1.
*AC: three HTML files (a few MB each) plus `filings_meta.json` exist on disk; running with the placeholder User-Agent refuses with a clear message.*

**FR-1 — Parsing.** Parse each filing into sections keyed to the real 10-K structure (at minimum: Item 1A Risk Factors, Item 7 MD&A, Item 8 Financial Statements; other items may be grouped). Strip inline-XBRL noise. Convert HTML tables to markdown, preserving row/column alignment.
*AC: printing the parsed Item 8 for the most recent FY shows the consolidated income statement as a readable markdown table with figures in the correct columns.*

**FR-2 — Chunking.** Split sections into chunks of ~500–800 tokens. Every chunk carries the metadata schema in §9.2. Tables must not be split mid-table; lists should not be split mid-item where feasible.
*AC: a `--sample 5` flag prints 5 random chunks with metadata; all schema fields present; text is human-readable.*

**FR-3 — Indexing.** Embed chunks with `BAAI/bge-small-en-v1.5` (sentence-transformers, CPU) into a **persistent** Chroma collection. Re-running the index command rebuilds from scratch (idempotent).
*AC: index command prints final chunk count; full rebuild completes on CPU in ≤ ~10 min.*

**FR-4 — Retrieval.** Top-k similarity search (k configurable, default 6). When the question names fiscal year(s), apply a Chroma `where` filter on `fiscal_year` (single year or `$in` list). Year detection may be a simple regex over `FY?20\d\d` patterns.
*AC: for a question naming FY2024, the trace log (FR-7) shows only FY2024 chunk IDs retrieved.*

**FR-5 — Grounded answering.** Build a prompt containing only the retrieved chunks + the question. System instructions: answer **only** from provided context; cite per §10.3; if the context is insufficient, output the refusal per §10.4. Stream tokens to the terminal. Provider is called only through `call_llm()` (§13).
*AC: 3 demo questions answered with ≥1 citation each; a deliberately unanswerable question refuses.*

**FR-6 — Refusal behavior.** Out-of-corpus questions (wrong company, year outside ingested range, topic absent) must produce the §10.4 refusal with **no** guessed content.
*AC: all 5 trap questions in the eval set refuse cleanly.*

**FR-7 — Tracing.** Every query appends one JSONL line to `data/traces.jsonl`: `{ts, question, fy_filter, chunk_ids, similarity_scores, model, answer, latency_ms}`.
*AC: after any eval run, `wc -l` on the trace file equals questions asked; any answer can be manually replayed by inspecting its retrieved chunk IDs.*

**FR-8 — Eval harness.** `run_evals.py` replays every row of `evals/questions.csv` **through the same code path** `ask.py` uses (shared function, not copied logic). Outputs `evals/results_<timestamp>.csv` with model answers alongside golden answers and an empty `grade` column for the human. A `--score` mode reads a graded results file and prints the metric in §11.3. Calls are spaced to respect the active provider's requests-per-minute limit (~7 s apart on the Gemini free tier; §7.1).
*AC: results file produced; score mode computes the correct percentage from a hand-graded file.*

**FR-9 — Streamlit UI.** A single-page chat (`app/ui.py`, ~30–50 lines) with streaming answers and citations rendered beneath each answer. Reuses the same core functions as the CLI.
*AC: `streamlit run app/ui.py` answers a question with visible citations.*

**FR-10 — README.** The README is the human's local-execution manual — everything runs on their machine from it. Required sections: prerequisites (Python version, venv), install, `.env` setup (which keys, where to get them), the full command sequence from §10.1 in run order (fetch → parse → chunk → index → ask → evals → UI), a **"Pending human verification"** checklist aggregating every pending item from merged PRs (§13), troubleshooting (SEC 403s from a bad User-Agent, first-run embedding model download, free-tier 429s), the eval table (baseline → final, with what changed at each step), and a short architecture description. The README is updated in **any** PR that changes a command or adds a pending item — never written once at the end.
*AC: G4 and G5 satisfied; a fresh clone can be run start-to-finish using only the README.*

## 6. Architecture

Two paths plus a harness:

- **Ingestion (run once, re-run when parsing improves):** `fetch_filings.py` (EDGAR download) → `parse.py` (sections, tables→markdown) → `chunk.py` (chunks + FY metadata) → `index.py` (bge-small → persistent Chroma).
- **Query loop (interactive):** question → year-filter detection → Chroma top-k retrieval → strict grounded prompt → `call_llm()` → streamed, cited answer. Used identically by CLI and Streamlit.
- **Eval harness:** replays `questions.csv` through the query loop and records results for human grading. Its score is the project's source of truth.

## 7. Tech stack (pinned)

| Layer | Choice | Notes |
|---|---|---|
| Language | Python 3.11+ | `venv`; no framework |
| HTTP | `requests` | already used by fetch script |
| Parsing | `beautifulsoup4` + `lxml` | table→markdown logic is hand-written |
| Chunking | hand-rolled splitter | token counting via `tiktoken` (optional) or word-count approximation |
| Embeddings | `sentence-transformers` / `BAAI/bge-small-en-v1.5` | CPU; ~130 MB first-run download |
| Vector store | `chromadb` (PersistentClient, local dir) | metadata filtering via `where` |
| LLM | Google AI Studio (primary) + Groq (optional) | locked decision in §7.1; accessed only through `call_llm()` |
| Config | `.env` + `python-dotenv` | keys never hardcoded |
| UI | `streamlit` | last milestone |
| Evals | `pandas` + CSV | human-graded |

No other dependencies without explicit approval.

### 7.1 LLM provider decision (locked)

- **Primary — Gemini 2.5 Flash** on the Google AI Studio free tier: `PROVIDER=gemini`, `MODEL_NAME=gemini-2.5-flash`. Rationale: the free tier's daily request budget covers several 25-question eval runs per day, its requests-per-minute allowance finishes one run in a few minutes, and its high token-per-minute ceiling means RAG prompts (~4–5K tokens each) never throttle.
- **Optional secondary — Groq** (`PROVIDER=groq`; e.g. `MODEL_NAME=llama-3.3-70b-versatile` or a hosted GPT-OSS model) for fast interactive questions during development. Caveat: Groq's free tier caps tokens-per-minute low enough (~6K on many models) that full eval runs crawl — use it for single questions, never for `run_evals.py`.
- Free-tier limits change without notice; the human verifies current numbers in each provider console. Model IDs live **only** in `.env`.
- **`call_llm()` implementation:** default is raw REST via `requests` with hand-parsed SSE streaming (ring-one ethos). The official thin SDKs (`google-genai`, `groq`) are **pre-approved** alternates if SSE parsing stalls progress — the only pre-approved dependency exception. No LangChain-style wrappers under any circumstances.
- **Rate-limit etiquette:** on HTTP 429, retry once with backoff, honoring a `retry-after` header when present.

## 8. Repository layout

```
filings-iq/
├── fetch_filings.py        # FR-0 / M0
├── ingest/
│   ├── parse.py            # FR-1
│   ├── chunk.py            # FR-2
│   └── index.py            # FR-3
├── app/
│   ├── core.py             # retrieve(), call_llm(), answer() — shared path
│   ├── ask.py              # CLI entry (FR-4/5/6/7)
│   └── ui.py               # Streamlit (FR-9)
├── evals/
│   ├── questions.csv       # §11.1
│   └── run_evals.py        # FR-8
├── data/
│   ├── raw/                # gitignored — filings + filings_meta.json
│   ├── chroma/             # gitignored — persistent index
│   └── traces.jsonl        # gitignored
├── .claude/
│   └── settings.json       # disables git attribution (§13)
├── .env.example
├── requirements.txt
└── README.md
```

## 9. Data spec

### 9.1 Raw data (produced by FR-0 / M0)

- `data/raw/AAPL_10-K_FY<year>.html` — three filings, most recent first.
- `data/raw/filings_meta.json` — list of `{accession, filing_date, report_date, primary_doc}`.
- **Fiscal-year source of truth = `report_date`** (Apple's FY ends late September). Never derive FY from `filing_date`.

### 9.2 Chunk metadata schema (every chunk, no exceptions)

```json
{
  "chunk_id": "FY2024-item7-014",
  "fiscal_year": "2024",
  "item": "7",
  "section_title": "Management's Discussion and Analysis",
  "source_file": "AAPL_10-K_FY2024.html",
  "contains_table": true
}
```

`fiscal_year` is a string (Chroma filters on exact match). The ingested FY range is whatever `filings_meta.json` says — do not hardcode years anywhere else.

## 10. Interface specs

### 10.1 CLI

```
python fetch_filings.py                # FR-0 — download filings + metadata
python -m app.ask "What was services revenue in FY2024?"
python -m app.ask --k 8 "..."          # override top-k
python -m ingest.index --rebuild       # wipe and re-index
python -m ingest.chunk --sample 5      # chunk QA
```

### 10.2 Environment

`.env.example` documents: `PROVIDER=gemini` (`gemini` | `groq`), `GOOGLE_API_KEY`, `GROQ_API_KEY` (optional), `SEC_USER_AGENT` ("Name email@example.com", FR-0), `MODEL_NAME=gemini-2.5-flash`, `TOP_K=6`, `EMBED_MODEL=BAAI/bge-small-en-v1.5`. `call_llm()` reads only these — switching providers must never require a code edit (§7.1).

### 10.3 Citation format (exact)

Inline or end-of-answer, one or more of:

```
[FY2024 10-K · Item 7 (MD&A)]
```

### 10.4 Refusal format (exact template)

```
I can't find this in the filings I have (Apple 10-Ks, FY<min>–FY<max>).
```

No additional speculation may follow the refusal sentence. `<min>`/`<max>` come from the indexed corpus at runtime.

## 11. Eval spec

### 11.1 `questions.csv` schema

`id, type, fiscal_years, question, golden_answer, golden_citation, notes`

`type ∈ {lookup, table, comparison, synthesis, trap}`.

### 11.2 Distribution (25 rows)

8 lookup · 5 table · 5 comparison · 2 synthesis · 5 trap. Claude Code may draft candidate questions and answers, but a row becomes canonical only after the human verifies its golden answer against the filings; unverified rows are excluded from the metric. Verification happens before any tuning begins. Traps target: years outside the ingested range, other companies, and topics genuinely absent from 10-Ks.

### 11.3 Grading protocol (human)

Each answer gets exactly one grade:

| Grade | Meaning |
|---|---|
| `grounded_correct` | Correct **and** supported by the retrieved chunks in the trace |
| `correct_ungrounded` | Correct but the trace shows retrieval didn't support it (model memory) — **counts as a failure** |
| `wrong` | Incorrect content |
| `refused_correctly` | Trap refused with §10.4 template |
| `refused_wrongly` | Refused an answerable question |

**Metric = (grounded_correct + refused_correctly) / 25.** The `correct_ungrounded` grade exists because the LLM knows Apple's public numbers from training; groundedness, not correctness, is what this project measures.

## 12. Milestones and checkpoints

| # | Deliverable | Checkpoint evidence (stop here for review) |
|---|---|---|
| M0 | `fetch_filings.py` | 3 filings + `filings_meta.json` on disk; placeholder User-Agent guard works (FR-0 AC) |
| M1 | `parse.py` | Item 8 income statement printed as clean markdown (FR-1 AC) |
| M2 | `chunk.py` + `index.py` | 5-chunk sample + final collection count (FR-2/3 AC) |
| M3 | `core.py` + `ask.py` | 3 cited demo answers + matching trace lines (FR-4/5/7 AC) |
| M4 | Eval harness + baseline run | Results CSV generated; baseline % pending human grading (expect ~50–60% once graded — that is normal) |
| M5 | Iteration toward ≥ 80% | One PR per fix; traps and citation checks verified mechanically; correctness deltas pending human grading |
| M6 | `ui.py` + README | Streamlit demo + G1–G5 checklist; every item needing local execution listed as pending in the README |

## 13. Working agreement (AI-assisted development)

- **One milestone per session.** Stop at each checkpoint; wait for human review.
- **Swap seams are sacred:** all LLM calls go through `call_llm()`; all vector-store access through `retrieve()`. Ring two swaps providers/stores by editing only `core.py`.
- **Evals share the production code path** (FR-8). Never a parallel implementation.
- **No silent scope growth:** anything in §3 requires an explicit human "yes" first.
- **Secrets:** `.env` only; `.env` gitignored; `.env.example` maintained.
- **Git workflow — branch per milestone, PR per checkpoint, self-merge:** work never lands on `main` directly. Each milestone gets a branch (`m0-fetch`, `m1-parse`, `m2-chunk-index`, `m3-core-ask`, `m4-evals`, `m5-iteration`, `m6-ui-readme`) with small, focused commits, and ends with a pull request whose description carries: FRs covered, the §12 checkpoint evidence (real output, not claims), how to verify locally, and any PRD deviations. Claude Code merges its own PR **only when every acceptance criterion it can verify passes**; anything requiring the human's machine (network fetches, API keys, eval grading) is listed under a **"Pending human verification"** heading in the PR and mirrored into the README checklist before merging. M5 splits into several small PRs — one per improvement, each showing the eval score before → after (or marked pending until graded results exist). The merged PR history becomes the README changelog (G5).
- **No AI attribution in git:** commit messages and PR descriptions must not contain `Co-Authored-By` trailers or "Generated with Claude Code" lines. Enforced by a checked-in `.claude/settings.json` at the repo root containing `{"attribution": {"commit": "", "pr": ""}}` — create it in the scaffold PR, never remove or override it, and verify the first commit's message is clean before proceeding.
- **Style:** plain functions over classes; readable > clever; every script runnable as `python -m <module>` with `--help`.
- **Honesty:** if a checkpoint AC fails, report the failure — do not adjust the AC.

## 14. Known risks and gotchas

- **Inline XBRL bloat:** filings are several MB of tag noise; parse must strip it or chunks become garbage.
- **Table flattening:** the #1 cause of wrong numbers; a misaligned column silently shifts figures across fiscal years.
- **FY vs filing date:** the 10-K *filed* in Nov 2024 is the *FY2024* report — conflating these is a bug the traps should catch.
- **Paraphrase misses:** "how much did Apple make" embeds far from "net sales"; expect retrieval whiffs and fix via chunking/section titles in chunk text, not by adding a reranker (ring two).
- **Mid-list splits:** Item 1A risk factors are long lists; splitting mid-item breaks comparison questions.
- **Free-tier 429s:** `call_llm()` retries once with backoff, honoring `retry-after` when present; eval runs are paced per §7.1.
- **Model memory contamination:** see `correct_ungrounded` in §11.3 — the eval design exists precisely because the model already knows Apple's numbers.

## 15. Ring two preview (out of scope, informs interfaces)

Planned upgrades, each measured as an eval delta on the same 25 questions: more companies → Qdrant in Docker (swap inside `retrieve()`), reranker stage, hybrid search, LangGraph agent + live price tool, Langfuse tracing, CI eval gate, Next.js UI, multi-tenant mode. None of these may leak into ring one.

---

*End of PRD. Build order: M0 first — the repository starts empty.*
