"""FR-2 — Chunking.

Split parsed sections into ~500-800 token chunks, each carrying the §9.2
metadata schema. Tables are never split mid-table; list items arrive from the
parser as their own blocks, so they are never split mid-item.

Run from the repo root:

    python -m ingest.chunk                 # build all chunks, print stats
    python -m ingest.chunk --sample 5      # print 5 random chunks + metadata
    python -m ingest.chunk --file path.html
    python -m ingest.chunk --help

`chunk_all()` / `chunk_document()` are the importable entry points used by the
indexer (FR-3). Metadata (§9.2):

    {chunk_id, fiscal_year, item, section_title, source_file, contains_table}
"""

import argparse
import random
import re
import sys
from pathlib import Path

from ingest.parse import RAW_DIR, parse_filing

MAX_TOKENS = 800
MIN_TOKENS = 400  # soft floor; section tails may be shorter
SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z(])")

_ENCODER = None
_ENCODER_READY = False


def count_tokens(text):
    """Token count via tiktoken when available, else a word-count estimate."""
    global _ENCODER, _ENCODER_READY
    if not _ENCODER_READY:
        try:
            import tiktoken

            _ENCODER = tiktoken.get_encoding("cl100k_base")
        except Exception:  # tiktoken optional (§7) — fall back to word count
            _ENCODER = None
        _ENCODER_READY = True
    if _ENCODER is not None:
        return len(_ENCODER.encode(text))
    return int(len(text.split()) * 1.3) + 1


def split_oversized_text(text, max_tokens):
    """Split a too-long text block on sentence boundaries into balanced units.

    Splits toward an even target size (total / ceil(total/max)) rather than
    greedily to max, so the last piece is never a tiny remainder.
    """
    import math

    total = count_tokens(text)
    if total <= max_tokens:
        return [text]
    n_parts = math.ceil(total / max_tokens)
    target = math.ceil(total / n_parts)
    sentences = SENTENCE_RE.split(text)
    units, buf = [], ""
    for sentence in sentences:
        candidate = (buf + " " + sentence).strip() if buf else sentence
        if buf and count_tokens(candidate) > target:
            units.append(buf)
            buf = sentence
        else:
            buf = candidate
    if buf:
        units.append(buf)
    return units


def _make_chunk(fy, item, title, source, seq, text, contains_table):
    return {
        "text": text,
        "metadata": {
            "chunk_id": f"FY{fy}-item{item}-{seq:03d}",
            "fiscal_year": fy,
            "item": item,
            "section_title": title,
            "source_file": source,
            "contains_table": contains_table,
        },
    }


def chunk_document(doc, max_tokens=MAX_TOKENS):
    """Chunk one parsed document (from parse_filing) into §9.2 chunks."""
    fy = doc["fiscal_year"]
    source = doc["source_file"]
    chunks = []

    for section in doc["sections"]:
        item = section["item"]
        title = section["title"]
        seq = 0
        buf = []          # list of (text, is_table)
        buf_tokens = 0

        def flush():
            nonlocal seq, buf, buf_tokens
            if not buf:
                return
            text = "\n\n".join(t for t, _ in buf)
            contains_table = any(is_t for _, is_t in buf)
            chunks.append(
                _make_chunk(fy, item, title, source, seq, text, contains_table)
            )
            seq += 1
            buf = []
            buf_tokens = 0

        for block in section["blocks"]:
            if block["kind"] == "table":
                # Tables are atomic: flush first if they won't fit, then keep
                # the whole table together (even if it alone exceeds max).
                ttok = count_tokens(block["md"])
                if buf and buf_tokens + ttok > max_tokens:
                    flush()
                buf.append((block["md"], True))
                buf_tokens += ttok
                if buf_tokens >= max_tokens:
                    flush()
                continue

            for unit in split_oversized_text(block["text"], max_tokens):
                utok = count_tokens(unit)
                if buf and buf_tokens + utok > max_tokens:
                    flush()
                buf.append((unit, False))
                buf_tokens += utok

        flush()

    return _add_section_context(_merge_small_adjacent(chunks, max_tokens))


def _add_section_context(chunks):
    """Prefix each chunk's text with a natural-language section header.

    §14 (paraphrase misses): a bare figure like "391,035" embeds far from a
    query like "how much did Apple make". Prepending the filing/section context
    ("Apple 10-K — FY2024, Item 7: Management's Discussion and Analysis") pulls
    the chunk's embedding toward those section words, without a reranker. The
    header sits first so it survives the embedder's input-window truncation.
    """
    for chunk in chunks:
        meta = chunk["metadata"]
        header = (
            f"Apple 10-K — FY{meta['fiscal_year']}, "
            f"Item {meta['item']}: {meta['section_title']}"
        )
        chunk["text"] = f"{header}\n\n{chunk['text']}"
    return chunks


def _merge_small_adjacent(chunks, max_tokens):
    """Merge consecutive same-section chunks when the union still fits.

    Keeps tables whole (merging only concatenates text/markdown, never splits a
    table) and re-numbers chunk_id sequences per (fiscal_year, item) afterward.
    """
    merged = []
    for chunk in chunks:
        if merged:
            prev = merged[-1]
            same_section = (
                prev["metadata"]["item"] == chunk["metadata"]["item"]
                and prev["metadata"]["fiscal_year"] == chunk["metadata"]["fiscal_year"]
            )
            combined = prev["text"] + "\n\n" + chunk["text"]
            if same_section and count_tokens(combined) <= max_tokens:
                prev["text"] = combined
                prev["metadata"]["contains_table"] = (
                    prev["metadata"]["contains_table"]
                    or chunk["metadata"]["contains_table"]
                )
                continue
        merged.append(chunk)

    seqs = {}
    for chunk in merged:
        meta = chunk["metadata"]
        key = (meta["fiscal_year"], meta["item"])
        seq = seqs.get(key, 0)
        meta["chunk_id"] = f"FY{meta['fiscal_year']}-item{meta['item']}-{seq:03d}"
        seqs[key] = seq + 1
    return merged


def chunk_all(raw_dir=RAW_DIR, max_tokens=MAX_TOKENS):
    """Chunk every filing in ``raw_dir`` (parses each via parse_filing)."""
    files = sorted(Path(raw_dir).glob("AAPL_10-K_FY*.html"))
    chunks = []
    for path in files:
        chunks.extend(chunk_document(parse_filing(path), max_tokens=max_tokens))
    return chunks


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _print_stats(chunks):
    if not chunks:
        print("No chunks produced.")
        return
    token_counts = [count_tokens(c["text"]) for c in chunks]
    in_band = sum(1 for t in token_counts if MIN_TOKENS <= t <= MAX_TOKENS)
    with_tables = sum(1 for c in chunks if c["metadata"]["contains_table"])
    print(f"Total chunks: {len(chunks)}")
    print(
        f"Tokens/chunk: min={min(token_counts)} "
        f"mean={sum(token_counts) // len(token_counts)} "
        f"max={max(token_counts)}  "
        f"({in_band}/{len(chunks)} within {MIN_TOKENS}-{MAX_TOKENS})"
    )
    print(f"Chunks containing a table: {with_tables}")
    print("\nBy filing / item:")
    counts = {}
    for c in chunks:
        key = (c["metadata"]["source_file"], c["metadata"]["item"])
        counts[key] = counts.get(key, 0) + 1
    for (src, item), n in sorted(counts.items()):
        print(f"  {src}  Item {item:<3} {n} chunk(s)")


def _print_chunk(chunk):
    meta = chunk["metadata"]
    print("-" * 70)
    print(f"chunk_id      : {meta['chunk_id']}")
    print(f"fiscal_year   : {meta['fiscal_year']}")
    print(f"item          : {meta['item']}")
    print(f"section_title : {meta['section_title']}")
    print(f"source_file   : {meta['source_file']}")
    print(f"contains_table: {meta['contains_table']}")
    print(f"tokens        : {count_tokens(chunk['text'])}")
    print("text:")
    print(chunk["text"])


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Chunk parsed 10-K sections into ~500-800 token chunks (FR-2).",
    )
    parser.add_argument(
        "--sample", type=int, metavar="N",
        help="print N random chunks with full metadata (QA)",
    )
    parser.add_argument("--file", help="chunk a single HTML file instead of data/raw")
    parser.add_argument("--seed", type=int, help="random seed for --sample")
    parser.add_argument(
        "--max-tokens", type=int, default=MAX_TOKENS,
        help=f"max tokens per chunk (default {MAX_TOKENS})",
    )
    args = parser.parse_args(argv)

    if args.file:
        path = Path(args.file)
        if not path.exists():
            print(f"error: file not found: {path}", file=sys.stderr)
            return 1
        chunks = chunk_document(parse_filing(path), max_tokens=args.max_tokens)
    else:
        chunks = chunk_all(max_tokens=args.max_tokens)
        if not chunks:
            print(
                "No chunks: no filings in data/raw/. Run `python fetch_filings.py` "
                "then `python -m ingest.parse` (see README).",
                file=sys.stderr,
            )
            return 1

    if args.sample:
        if args.seed is not None:
            random.seed(args.seed)
        picked = random.sample(chunks, min(args.sample, len(chunks)))
        print(f"{len(picked)} random chunk(s) of {len(chunks)} total:\n")
        for chunk in picked:
            _print_chunk(chunk)
        return 0

    _print_stats(chunks)
    return 0


if __name__ == "__main__":
    sys.exit(main())
