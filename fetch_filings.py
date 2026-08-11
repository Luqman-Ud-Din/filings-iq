"""FR-0 — Data acquisition.

Download Apple's three most recent 10-K filings from SEC EDGAR as HTML, plus a
metadata file. Run from the repo root:

    python fetch_filings.py                 # download 3 filings + metadata
    python fetch_filings.py --dry-run       # show what would be downloaded
    python fetch_filings.py --help

Every SEC request carries a descriptive ``User-Agent`` read from ``SEC_USER_AGENT``
in ``.env`` (SEC returns HTTP 403 to anonymous clients). The script refuses to
run when that value is missing or still the ``.env.example`` placeholder.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import requests
from dotenv import load_dotenv

# Apple Inc. — CIKs are zero-padded to 10 digits in the submissions API path.
CIK = "0000320193"
CIK_NO_ZEROS = str(int(CIK))  # "320193" — used in the Archives document URL
SUBMISSIONS_URL = f"https://data.sec.gov/submissions/CIK{CIK}.json"
ARCHIVES_BASE = "https://www.sec.gov/Archives/edgar/data"

PLACEHOLDER_UA = "Name email@example.com"
DEFAULT_RAW_DIR = Path("data/raw")
META_FILENAME = "filings_meta.json"

# SEC allows up to 10 requests/second; we make only ~4 requests total and sleep
# a full second between them to stay far below the limit and be polite.
DEFAULT_SLEEP = 1.0


def die(message):
    """Print an instructive error to stderr and exit non-zero."""
    print(f"error: {message}", file=sys.stderr)
    sys.exit(1)


def resolve_user_agent():
    """Return a valid SEC User-Agent or exit with an instructive message.

    Rejects a missing value and the ``.env.example`` placeholder so the fetch
    never silently hits SEC anonymously (which returns 403).
    """
    load_dotenv()
    ua = (os.environ.get("SEC_USER_AGENT") or "").strip()
    hint = (
        'Set SEC_USER_AGENT in .env to "Your Name your@email.com". '
        "SEC EDGAR returns HTTP 403 to anonymous clients "
        "(see https://www.sec.gov/os/webmaster-faq#developers)."
    )
    if not ua:
        die("SEC_USER_AGENT is not set. " + hint)
    if ua == PLACEHOLDER_UA or "example.com" in ua.lower():
        die(
            f"SEC_USER_AGENT is still the placeholder ({ua!r}). " + hint
        )
    if "@" not in ua:
        die(
            f"SEC_USER_AGENT ({ua!r}) does not contain an email address. " + hint
        )
    return ua


def http_get(url, user_agent, *, accept, max_retries=3):
    """GET ``url`` with the SEC User-Agent, retrying transient failures.

    Retries on connection errors and on HTTP 429/5xx with exponential backoff,
    honoring a ``Retry-After`` header when present.
    """
    headers = {
        "User-Agent": user_agent,
        "Accept": accept,
        "Accept-Encoding": "gzip, deflate",
        "Host": requests.utils.urlparse(url).netloc,
    }
    backoff = 1.0
    last_error = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.get(url, headers=headers, timeout=30)
        except requests.RequestException as exc:  # network hiccup
            last_error = str(exc)
        else:
            if resp.status_code == 200:
                return resp
            if resp.status_code == 403:
                die(
                    f"SEC returned 403 for {url}. Your SEC_USER_AGENT is likely "
                    "not accepted; use a real name and email."
                )
            if resp.status_code in (429, 500, 502, 503, 504):
                last_error = f"HTTP {resp.status_code}"
                retry_after = resp.headers.get("Retry-After")
                if retry_after and retry_after.isdigit():
                    backoff = float(retry_after)
            else:
                die(f"Unexpected HTTP {resp.status_code} fetching {url}")
        if attempt < max_retries:
            time.sleep(backoff)
            backoff *= 2
    die(f"Failed to fetch {url} after {max_retries} attempts ({last_error}).")


def select_recent_10k(submissions, count):
    """Pick the ``count`` most recent 10-K filings from a submissions object.

    ``filings.recent`` holds parallel arrays: index i across ``form``,
    ``accessionNumber``, ``filingDate``, ``reportDate`` and ``primaryDocument``
    describes one filing. Only exact ``"10-K"`` forms are kept (this also skips
    ``10-K/A`` amendments). Results are sorted most-recent-first by filing date.
    """
    recent = submissions["filings"]["recent"]
    forms = recent["form"]
    accessions = recent["accessionNumber"]
    filing_dates = recent["filingDate"]
    report_dates = recent["reportDate"]
    primary_docs = recent["primaryDocument"]

    rows = []
    for i in range(len(forms)):
        if forms[i] == "10-K":  # exact match — excludes "10-K/A"
            rows.append(
                {
                    "accession": accessions[i],
                    "filing_date": filing_dates[i],
                    "report_date": report_dates[i],
                    "primary_doc": primary_docs[i],
                }
            )

    # filingDate is ISO (YYYY-MM-DD), so lexical sort == chronological sort.
    rows.sort(key=lambda r: r["filing_date"], reverse=True)
    return rows[:count]


def document_url(accession, primary_doc):
    """Build the Archives URL for a filing's primary document.

    Pattern: /Archives/edgar/data/<cik-without-leading-zeros>/<accession-without-dashes>/<primaryDocument>
    """
    accession_no_dashes = accession.replace("-", "")
    return f"{ARCHIVES_BASE}/{CIK_NO_ZEROS}/{accession_no_dashes}/{primary_doc}"


def output_filename(report_date):
    """``AAPL_10-K_FY<year>.html`` where <year> = first 4 chars of report_date.

    Fiscal year comes from report_date (Apple's FY ends late September), never
    from filing_date (§9.1).
    """
    fiscal_year = report_date[:4]
    return f"AAPL_10-K_FY{fiscal_year}.html"


def load_submissions(user_agent, submissions_file):
    """Read submissions JSON from a local file (offline) or fetch from SEC."""
    if submissions_file:
        return json.loads(Path(submissions_file).read_text(encoding="utf-8"))
    resp = http_get(SUBMISSIONS_URL, user_agent, accept="application/json")
    return resp.json()


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Download Apple's 3 most recent 10-K filings from SEC EDGAR "
        "(FR-0). Reads SEC_USER_AGENT from .env.",
    )
    parser.add_argument(
        "--out-dir",
        default=str(DEFAULT_RAW_DIR),
        help="directory for downloaded filings + metadata (default: data/raw)",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=3,
        help="number of most-recent 10-K filings to download (default: 3)",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=DEFAULT_SLEEP,
        help="seconds to sleep between SEC requests (default: 1.0)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="list the selected filings and URLs without downloading anything",
    )
    parser.add_argument(
        "--submissions",
        metavar="PATH",
        default=None,
        help="read the submissions JSON from a local file instead of SEC "
        "(offline; skips the User-Agent guard when combined with --dry-run)",
    )
    args = parser.parse_args(argv)

    # A network request is needed unless we are reading submissions from disk.
    # The User-Agent guard protects every network request.
    offline = args.submissions is not None
    user_agent = None
    if not offline:
        user_agent = resolve_user_agent()

    submissions = load_submissions(user_agent, args.submissions)
    selected = select_recent_10k(submissions, args.count)

    if not selected:
        die("No 10-K filings found in the submissions feed.")
    if len(selected) < args.count:
        print(
            f"warning: only {len(selected)} 10-K filing(s) found "
            f"(requested {args.count}).",
            file=sys.stderr,
        )

    print(f"Selected {len(selected)} most-recent 10-K filing(s):")
    for row in selected:
        url = document_url(row["accession"], row["primary_doc"])
        print(
            f"  FY{row['report_date'][:4]}  filed {row['filing_date']}  "
            f"report {row['report_date']}  {url}"
        )

    if args.dry_run:
        print("\n--dry-run: nothing downloaded.")
        return 0

    # Downloading requires a User-Agent even if submissions came from a file.
    if user_agent is None:
        user_agent = resolve_user_agent()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    meta = []
    for i, row in enumerate(selected):
        url = document_url(row["accession"], row["primary_doc"])
        filename = output_filename(row["report_date"])
        dest = out_dir / filename

        if i > 0:
            time.sleep(args.sleep)  # pace requests below SEC's 10 req/s limit
        print(f"Downloading {filename} <- {url}")
        resp = http_get(url, user_agent, accept="text/html")
        dest.write_bytes(resp.content)
        size_mb = len(resp.content) / (1024 * 1024)
        print(f"  saved {dest} ({size_mb:.1f} MB)")

        meta.append(
            {
                "accession": row["accession"],
                "filing_date": row["filing_date"],
                "report_date": row["report_date"],
                "primary_doc": row["primary_doc"],
            }
        )

    meta_path = out_dir / META_FILENAME
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {meta_path} ({len(meta)} entries).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
