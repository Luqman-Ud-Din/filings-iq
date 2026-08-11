"""Shared query loop: retrieve(), call_llm(), answer().

These three functions are the swap seams (§13): every vector-store access goes
through ``retrieve()``; every LLM call goes through ``call_llm()``. The CLI
(app/ask.py), the Streamlit UI (app/ui.py) and the eval harness
(evals/run_evals.py) all call ``answer()`` — never a parallel implementation.

Covers FR-4 (retrieval + fiscal-year filter), FR-5 (grounded answering),
FR-6 (refusal), FR-7 (JSONL tracing).
"""

import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv()

CHROMA_DIR = os.environ.get("CHROMA_DIR", "data/chroma")
COLLECTION = os.environ.get("CHROMA_COLLECTION", "filings")
TRACE_PATH = Path(os.environ.get("TRACE_PATH", "data/traces.jsonl"))
EMBED_MODEL = os.environ.get("EMBED_MODEL", "BAAI/bge-small-en-v1.5")
DEFAULT_TOP_K = int(os.environ.get("TOP_K", "6") or "6")

# bge-small retrieval instruction (applied to the QUERY only; documents were
# embedded without it in ingest/index.py).
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

# Short parenthetical labels used in the §10.3 citation format.
SHORT_TITLES = {
    "1": "Business",
    "1A": "Risk Factors",
    "1B": "Unresolved Staff Comments",
    "1C": "Cybersecurity",
    "2": "Properties",
    "3": "Legal Proceedings",
    "7": "MD&A",
    "7A": "Market Risk",
    "8": "Financial Statements",
    "9A": "Controls and Procedures",
}

_STATE = {"embedder": None, "collection": None, "fy_range": None}


# --------------------------------------------------------------------------- #
# Fiscal-year detection and Chroma where-filter (FR-4)
# --------------------------------------------------------------------------- #
_FY_RE = re.compile(r"(?:FY\s*)?(20\d\d)", re.IGNORECASE)


def detect_fiscal_years(question):
    """Return distinct fiscal years named in the question, e.g. ['2023','2024'].

    Matches ``FY2024``, ``FY 2024`` and bare ``2024`` (§FR-4: "a simple regex
    over FY?20\\d\\d patterns").
    """
    years = []
    for match in _FY_RE.findall(question):
        if match not in years:
            years.append(match)
    return sorted(years)


def build_where(years):
    """Build a Chroma ``where`` filter for one or more fiscal years."""
    if not years:
        return None
    if len(years) == 1:
        return {"fiscal_year": years[0]}
    return {"fiscal_year": {"$in": years}}


# --------------------------------------------------------------------------- #
# Citations (§10.3) and refusal (§10.4)
# --------------------------------------------------------------------------- #
def citation_label(meta):
    """Render a chunk's §10.3 citation label, e.g. 'FY2024 10-K · Item 7 (MD&A)'."""
    short = SHORT_TITLES.get(meta["item"], meta.get("section_title", ""))
    return f"FY{meta['fiscal_year']} 10-K · Item {meta['item']} ({short})"


def refusal_text(fy_min, fy_max):
    """The exact §10.4 refusal template (no speculation may follow)."""
    return (
        f"I can't find this in the filings I have (Apple 10-Ks, "
        f"FY{fy_min}–FY{fy_max})."
    )


# --------------------------------------------------------------------------- #
# Vector store (FR-4) — all access goes through retrieve()
# --------------------------------------------------------------------------- #
def _get_embedder():
    if _STATE["embedder"] is None:
        from sentence_transformers import SentenceTransformer

        _STATE["embedder"] = SentenceTransformer(EMBED_MODEL, device="cpu")
    return _STATE["embedder"]


def _get_collection():
    if _STATE["collection"] is None:
        import chromadb

        client = chromadb.PersistentClient(path=CHROMA_DIR)
        try:
            _STATE["collection"] = client.get_collection(COLLECTION)
        except Exception as exc:
            raise RuntimeError(
                f"Chroma collection '{COLLECTION}' not found at {CHROMA_DIR}. "
                "Build it first: `python -m ingest.index --rebuild` (see README)."
            ) from exc
    return _STATE["collection"]


def embed_query(text):
    """Embed a query with the bge retrieval instruction; unit-normalized."""
    model = _get_embedder()
    return model.encode(
        BGE_QUERY_PREFIX + text, normalize_embeddings=True
    ).tolist()


def corpus_fy_range():
    """Return (min, max) fiscal_year strings present in the index."""
    if _STATE["fy_range"] is None:
        collection = _get_collection()
        metas = collection.get(include=["metadatas"])["metadatas"]
        years = sorted({m["fiscal_year"] for m in metas})
        if not years:
            raise RuntimeError("The index is empty; run `python -m ingest.index`.")
        _STATE["fy_range"] = (years[0], years[-1])
    return _STATE["fy_range"]


def retrieval_plan(question):
    """Decide which ``where`` filters to run for a question.

    §14 (comparison questions): when the question names two or more fiscal
    years, run one top-k query PER year and merge, so both years are represented
    instead of top-k skewing toward whichever year embeds closer. A single named
    year (or none) runs one query.
    """
    years = detect_fiscal_years(question)
    if len(years) >= 2:
        return [{"fiscal_year": y} for y in years]
    return [build_where(years)]


def _hits_from_result(result):
    hits = []
    for cid, doc, meta, dist in zip(
        result["ids"][0],
        result["documents"][0],
        result["metadatas"][0],
        result["distances"][0],
    ):
        hits.append(
            {
                "chunk_id": cid,
                "text": doc,
                "metadata": meta,
                "score": round(1.0 - dist, 4),  # cosine distance -> similarity
            }
        )
    return hits


def retrieve(question, k=None):
    """Similarity search with a fiscal-year ``where`` filter (FR-4).

    The sole vector-store access point (§13). Returns a list of hits
    ``{chunk_id, text, metadata, score}`` sorted best-first. For multi-year
    comparison questions, returns k hits per named year (balanced coverage).
    """
    k = k or DEFAULT_TOP_K
    collection = _get_collection()
    query_embedding = embed_query(question)  # embed once, reuse per year
    plans = retrieval_plan(question)
    hits = []
    for where in plans:
        result = collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            where=where,
            include=["documents", "metadatas", "distances"],
        )
        hits.extend(_hits_from_result(result))
    hits.sort(key=lambda h: h["score"], reverse=True)
    return hits if len(plans) > 1 else hits[:k]


# --------------------------------------------------------------------------- #
# Prompt construction (FR-5)
# --------------------------------------------------------------------------- #
def build_prompt(question, hits, fy_min, fy_max):
    """Build the (system, user) grounded prompt from retrieved chunks only."""
    system = (
        "You are a financial-filings assistant. Answer the user's question using "
        "ONLY the numbered context passages from Apple's 10-K filings provided "
        "below.\n"
        "Rules:\n"
        "- Use only facts stated in the context. Never use outside or prior "
        "knowledge about Apple or any other company.\n"
        "- You already know Apple's public figures from training; ignore that "
        "memory. If a specific number or fact is not stated verbatim in a passage "
        "below, you do not have it — do not supply, estimate, or round it.\n"
        "- If the question asks about a company other than Apple, or a fiscal year "
        "that does not appear in the passages below, you cannot answer it from "
        "these filings: output the refusal line.\n"
        "- Every answer must include at least one citation copied EXACTLY from the "
        "label shown above the passage you used, in this format: "
        "[FY2024 10-K · Item 7 (MD&A)].\n"
        "- If the context does not contain enough information to answer, reply with "
        "EXACTLY this line and nothing else:\n"
        f"{refusal_text(fy_min, fy_max)}\n"
        "- Never add any text, speculation, or citation after a refusal line."
    )
    passages = []
    for hit in hits:
        label = citation_label(hit["metadata"])
        passages.append(f"[{label}]\n{hit['text']}")
    user = (
        "Context passages:\n\n"
        + "\n\n".join(passages)
        + f"\n\nQuestion: {question}"
    )
    return system, user


# --------------------------------------------------------------------------- #
# LLM provider (FR-5) — all LLM access goes through call_llm()
# --------------------------------------------------------------------------- #
def _require_env(name):
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(
            f"{name} is not set. Add it to .env (see .env.example / README)."
        )
    return value


def _post_sse(url, headers, payload):
    """POST and stream an SSE response, retrying once on 429 (Retry-After)."""
    for attempt in range(2):
        resp = requests.post(
            url, headers=headers, json=payload, stream=True, timeout=120
        )
        if resp.status_code == 429 and attempt == 0:
            wait = resp.headers.get("Retry-After")
            time.sleep(float(wait) if wait and wait.isdigit() else 2.0)
            continue
        if resp.status_code != 200:
            raise RuntimeError(
                f"LLM provider returned HTTP {resp.status_code}: "
                f"{resp.text[:300]}"
            )
        return resp
    return resp


def _stream_gemini(system, user, model):
    key = _require_env("GOOGLE_API_KEY")
    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{model}:streamGenerateContent?alt=sse&key={key}"
    )
    payload = {
        "systemInstruction": {"parts": [{"text": system}]},
        "contents": [{"role": "user", "parts": [{"text": user}]}],
        "generationConfig": {"temperature": 0.0},
    }
    resp = _post_sse(url, {"Content-Type": "application/json"}, payload)
    for raw in resp.iter_lines(decode_unicode=True):
        if not raw or not raw.startswith("data:"):
            continue
        data = raw[5:].strip()
        if not data or data == "[DONE]":
            continue
        obj = json.loads(data)
        for cand in obj.get("candidates", []):
            for part in cand.get("content", {}).get("parts", []):
                if "text" in part:
                    yield part["text"]


def _stream_groq(system, user, model):
    key = _require_env("GROQ_API_KEY")
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": 0.0,
        "stream": True,
    }
    resp = _post_sse(url, headers, payload)
    for raw in resp.iter_lines(decode_unicode=True):
        if not raw or not raw.startswith("data:"):
            continue
        data = raw[5:].strip()
        if data == "[DONE]":
            break
        obj = json.loads(data)
        delta = obj["choices"][0].get("delta", {})
        if delta.get("content"):
            yield delta["content"]


def call_llm(system, user):
    """Stream a completion from the configured provider. Yields text pieces.

    The single LLM access point (§13). Provider/model come only from env
    (PROVIDER, MODEL_NAME) — switching providers never requires a code edit.
    """
    provider = os.environ.get("PROVIDER", "gemini").lower()
    model = _require_env("MODEL_NAME")
    if provider == "gemini":
        yield from _stream_gemini(system, user, model)
    elif provider == "groq":
        yield from _stream_groq(system, user, model)
    else:
        raise RuntimeError(
            f"Unknown PROVIDER '{provider}'. Use 'gemini' or 'groq' (§7.1)."
        )


# --------------------------------------------------------------------------- #
# Tracing (FR-7)
# --------------------------------------------------------------------------- #
def write_trace(record):
    """Append one JSONL line to data/traces.jsonl."""
    TRACE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with TRACE_PATH.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")


# --------------------------------------------------------------------------- #
# Orchestration (FR-5/6/7) — the shared entry point
# --------------------------------------------------------------------------- #
def answer(question, k=None, stream=True, trace=True, on_token=None):
    """Retrieve, ground, and answer one question (or refuse). Returns a dict.

    ``stream=True`` prints tokens to stdout as they arrive (FR-5). ``on_token``,
    if given, is called with each text piece (used by the Streamlit UI to stream
    into the page — same single path, no parallel implementation). Always
    appends a trace line (FR-7) unless ``trace=False``.
    """
    start = time.time()
    fy_filter = detect_fiscal_years(question)
    hits = retrieve(question, k)
    fy_min, fy_max = corpus_fy_range()

    if not hits:
        # Empty retrieval (e.g. a fiscal year outside the corpus) -> deterministic
        # refusal, no LLM call, no guessed content (FR-6).
        text = refusal_text(fy_min, fy_max)
        if on_token:
            on_token(text)
        if stream:
            print(text)
        refused = True
    else:
        system, user = build_prompt(question, hits, fy_min, fy_max)
        pieces = []
        for piece in call_llm(system, user):
            pieces.append(piece)
            if on_token:
                on_token(piece)
            if stream:
                print(piece, end="", flush=True)
        if stream:
            print()
        text = "".join(pieces).strip()
        refused = text.startswith("I can't find this in the filings I have")

    latency_ms = int((time.time() - start) * 1000)
    record = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "question": question,
        "fy_filter": fy_filter,
        "chunk_ids": [h["chunk_id"] for h in hits],
        "similarity_scores": [h["score"] for h in hits],
        "model": os.environ.get("MODEL_NAME", ""),
        "answer": text,
        "latency_ms": latency_ms,
    }
    if trace:
        write_trace(record)

    return {
        "answer": text,
        "hits": hits,
        "fy_filter": fy_filter,
        "refused": refused,
        "latency_ms": latency_ms,
        "citations": [citation_label(h["metadata"]) for h in hits],
    }
