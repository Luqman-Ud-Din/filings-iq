"""FR-8 — Eval harness.

Replay every row of evals/questions.csv through the SAME code path the CLI uses
(app.core.answer — never a copied implementation), writing a timestamped
results file with an empty ``grade`` column for the human. A ``--score`` mode
reads a hand-graded results file and prints the §11.3 metric.

    python -m evals.run_evals                     # run; writes evals/results_<ts>.csv
    python -m evals.run_evals --sleep 7           # pace calls for the free tier
    python -m evals.run_evals --score evals/results_<ts>.csv
    python -m evals.run_evals --help

Grades are the human's job (§11) — this tool never fills them in.
"""

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

QUESTIONS = Path("evals/questions.csv")
RESULTS_DIR = Path("evals")

# §11.3 grading vocabulary. Metric = (grounded_correct + refused_correctly) / N.
VALID_GRADES = {
    "grounded_correct",
    "correct_ungrounded",
    "wrong",
    "refused_correctly",
    "refused_wrongly",
}
METRIC_NUMERATOR = {"grounded_correct", "refused_correctly"}

RESULT_COLUMNS = [
    "id", "type", "fiscal_years", "question",
    "golden_answer", "golden_citation",
    "model_answer", "retrieved_citations", "chunk_ids",
    "refused", "latency_ms", "grade", "notes",
]


def run(questions_path, out_dir, sleep_seconds, k):
    """Replay every question through app.core.answer and write a results CSV."""
    from app.core import answer  # shared path (lazy import; needs index + key)

    df = pd.read_csv(questions_path, dtype=str).fillna("")
    rows = []
    total = len(df)
    print(f"Running {total} questions through app.core.answer() ...")
    for i, q in df.iterrows():
        print(f"[{i + 1}/{total}] {q['id']}: {q['question'][:70]}")
        result = answer(q["question"], k=k, stream=False, trace=True)
        rows.append(
            {
                "id": q["id"],
                "type": q["type"],
                "fiscal_years": q["fiscal_years"],
                "question": q["question"],
                "golden_answer": q["golden_answer"],
                "golden_citation": q["golden_citation"],
                "model_answer": result["answer"],
                "retrieved_citations": " | ".join(result["citations"]),
                "chunk_ids": " | ".join(h["chunk_id"] for h in result["hits"]),
                "refused": result["refused"],
                "latency_ms": result["latency_ms"],
                "grade": "",  # human fills this (§11)
                "notes": q.get("notes", ""),
            }
        )
        if i < total - 1:
            time.sleep(sleep_seconds)  # respect provider RPM (§7.1)

    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"results_{stamp}.csv"
    pd.DataFrame(rows, columns=RESULT_COLUMNS).to_csv(out_path, index=False)
    print(f"\nWrote {out_path} ({len(rows)} rows). Grade the 'grade' column, then:")
    print(f"  python -m evals.run_evals --score {out_path}")
    return 0


def score(results_path):
    """Compute the §11.3 metric from a hand-graded results file."""
    df = pd.read_csv(results_path, dtype=str).fillna("")
    if "grade" not in df.columns:
        print("error: no 'grade' column in results file.", file=sys.stderr)
        return 1

    grades = [g.strip() for g in df["grade"].tolist() if g.strip()]
    unknown = sorted({g for g in grades if g not in VALID_GRADES})
    if unknown:
        print(f"warning: ignoring unknown grade value(s): {unknown}", file=sys.stderr)
    graded = [g for g in grades if g in VALID_GRADES]

    ungraded = len(df) - len(grades)
    if ungraded:
        print(
            f"warning: {ungraded} of {len(df)} rows are ungraded and excluded "
            f"(§11.2).",
            file=sys.stderr,
        )

    counts = {g: graded.count(g) for g in sorted(VALID_GRADES)}
    numerator = sum(counts[g] for g in METRIC_NUMERATOR)
    denominator = len(graded)

    print("Grade breakdown:")
    for grade, n in counts.items():
        mark = "  *" if grade in METRIC_NUMERATOR else "   "
        print(f"{mark} {grade:<20} {n}")
    print(f"\nPass = grounded_correct + refused_correctly = {numerator}")
    print(f"Graded rows (denominator) = {denominator}")
    if denominator:
        pct = 100.0 * numerator / denominator
        print(f"Metric = {numerator}/{denominator} = {pct:.1f}%")
    else:
        print("Metric = n/a (no graded rows)")
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Replay questions.csv through the shared answer path and "
        "score hand-graded results (FR-8).",
    )
    parser.add_argument(
        "--score", metavar="RESULTS_CSV",
        help="compute the §11.3 metric from a hand-graded results file",
    )
    parser.add_argument("--questions", default=str(QUESTIONS))
    parser.add_argument("--out-dir", default=str(RESULTS_DIR))
    parser.add_argument(
        "--sleep", type=float, default=7.0,
        help="seconds between calls (default 7 for the Gemini free tier; §7.1)",
    )
    parser.add_argument("--k", type=int, default=None, help="retrieval top-k")
    args = parser.parse_args(argv)

    if args.score:
        return score(args.score)

    questions_path = Path(args.questions)
    if not questions_path.exists():
        print(f"error: questions file not found: {questions_path}", file=sys.stderr)
        return 1
    try:
        return run(questions_path, Path(args.out_dir), args.sleep, args.k)
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
