"""FR-4/5/6/7 — CLI entry point.

    python -m app.ask "What was services revenue in FY2024?"
    python -m app.ask --k 8 "..."          # override top-k
    python -m app.ask --help

Thin wrapper over app.core.answer() — the same shared path the UI and eval
harness use. Streams the grounded, cited answer, then lists the retrieved
sources beneath it.
"""

import argparse
import sys

from app.core import DEFAULT_TOP_K, answer


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Ask a grounded, cited question about Apple's 10-K filings.",
    )
    parser.add_argument("question", help="the question to ask")
    parser.add_argument(
        "--k", type=int, default=None,
        help=f"top-k chunks to retrieve (default {DEFAULT_TOP_K} / TOP_K env)",
    )
    parser.add_argument(
        "--no-stream", action="store_true",
        help="print the full answer at once instead of streaming tokens",
    )
    args = parser.parse_args(argv)

    try:
        result = answer(args.question, k=args.k, stream=not args.no_stream)
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if args.no_stream:
        print(result["answer"])

    if not result["refused"] and result["citations"]:
        print("\nRetrieved sources:")
        for label in result["citations"]:
            print(f"  - [{label}]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
