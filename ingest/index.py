"""FR-3 — Indexing.

Embed chunks with BAAI/bge-small-en-v1.5 (sentence-transformers, CPU) into a
persistent Chroma collection. Re-running rebuilds from scratch (idempotent).

Run from the repo root:

    python -m ingest.index --rebuild       # wipe and re-index
    python -m ingest.index --dry-run       # chunk only; no model / no Chroma
    python -m ingest.index --help

Heavy dependencies (sentence-transformers, chromadb) are imported lazily so
that --help and --dry-run work before the first model download.
"""

import argparse
import os
import sys

from dotenv import load_dotenv

from ingest.chunk import MAX_TOKENS, chunk_all, count_tokens

DEFAULT_PERSIST_DIR = "data/chroma"
DEFAULT_COLLECTION = "filings"
DEFAULT_EMBED_MODEL = "BAAI/bge-small-en-v1.5"


def load_embedder(model_name):
    """Load the sentence-transformers model on CPU (lazy import)."""
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name, device="cpu")


def build_index(persist_dir, collection_name, model_name, batch_size=32,
                max_tokens=MAX_TOKENS):
    """Chunk, embed, and (re)build the Chroma collection from scratch."""
    import chromadb

    chunks = chunk_all(max_tokens=max_tokens)
    if not chunks:
        print(
            "No chunks to index: no filings in data/raw/. Run the fetch/parse "
            "steps first (see README).",
            file=sys.stderr,
        )
        return 1

    print(f"Embedding {len(chunks)} chunks with {model_name} on CPU ...")
    model = load_embedder(model_name)
    texts = [c["text"] for c in chunks]
    ids = [c["metadata"]["chunk_id"] for c in chunks]
    metadatas = [c["metadata"] for c in chunks]
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,  # cosine-ready unit vectors
    ).tolist()

    client = chromadb.PersistentClient(path=persist_dir)
    # Idempotent rebuild: drop any existing collection, then recreate.
    try:
        client.delete_collection(collection_name)
    except Exception:
        pass
    collection = client.create_collection(
        collection_name, metadata={"hnsw:space": "cosine"}
    )

    # Add in batches to keep memory flat on large corpora.
    for start in range(0, len(ids), 256):
        end = start + 256
        collection.add(
            ids=ids[start:end],
            documents=texts[start:end],
            metadatas=metadatas[start:end],
            embeddings=embeddings[start:end],
        )

    print(
        f"Indexed {collection.count()} chunks into collection "
        f"'{collection_name}' at {persist_dir}/"
    )
    return 0


def dry_run(max_tokens=MAX_TOKENS):
    """Chunk only — report what would be indexed, without model or Chroma."""
    chunks = chunk_all(max_tokens=max_tokens)
    if not chunks:
        print(
            "No chunks: no filings in data/raw/. Run fetch/parse first.",
            file=sys.stderr,
        )
        return 1
    tokens = [count_tokens(c["text"]) for c in chunks]
    print(f"Would index {len(chunks)} chunks.")
    print(
        f"Tokens/chunk: min={min(tokens)} "
        f"mean={sum(tokens) // len(tokens)} max={max(tokens)}"
    )
    print("(dry-run: no embeddings computed, no Chroma collection written)")
    return 0


def main(argv=None):
    load_dotenv()
    parser = argparse.ArgumentParser(
        description="Embed chunks into a persistent Chroma collection (FR-3).",
    )
    parser.add_argument(
        "--rebuild", action="store_true",
        help="wipe and re-index (the index always rebuilds from scratch)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="chunk only and print counts; no model download, no Chroma write",
    )
    parser.add_argument("--persist-dir", default=DEFAULT_PERSIST_DIR)
    parser.add_argument("--collection", default=DEFAULT_COLLECTION)
    parser.add_argument(
        "--embed-model",
        default=os.environ.get("EMBED_MODEL", DEFAULT_EMBED_MODEL),
        help="sentence-transformers model id (default from EMBED_MODEL env)",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--max-tokens", type=int, default=MAX_TOKENS,
        help=f"max tokens per chunk (default {MAX_TOKENS}, aligned to the "
        "bge-small 512-token window; raise to A/B against the ~800 baseline)",
    )
    args = parser.parse_args(argv)

    if args.dry_run:
        return dry_run(args.max_tokens)

    return build_index(
        args.persist_dir, args.collection, args.embed_model, args.batch_size,
        args.max_tokens,
    )


if __name__ == "__main__":
    sys.exit(main())
