"""FR-1 — Parsing.

Parse a 10-K HTML filing into sections keyed to the real 10-K structure
(Item 1A Risk Factors, Item 7 MD&A, Item 8 Financial Statements, ...), strip
inline-XBRL noise, and convert HTML tables to markdown that preserves
row/column alignment.

Run from the repo root:

    python -m ingest.parse                      # summarize every filing in data/raw
    python -m ingest.parse --item 8             # print Item 8 of the most recent FY
    python -m ingest.parse --fy 2024 --item 7   # print Item 7 of FY2024
    python -m ingest.parse --file path.html --item 1A
    python -m ingest.parse --help

`parse_filing(path)` is the importable entry point used by the chunker (FR-2);
the query loop never re-implements parsing.
"""

import argparse
import json
import re
import sys
from pathlib import Path

from bs4 import BeautifulSoup, Comment, NavigableString

RAW_DIR = Path("data/raw")
META_FILENAME = "filings_meta.json"

# Canonical 10-K item titles (§9.2 uses clean titles, not raw heading text).
ITEM_TITLES = {
    "1": "Business",
    "1A": "Risk Factors",
    "1B": "Unresolved Staff Comments",
    "1C": "Cybersecurity",
    "2": "Properties",
    "3": "Legal Proceedings",
    "4": "Mine Safety Disclosures",
    "5": "Market for Registrant's Common Equity",
    "6": "Selected Financial Data",
    "7": "Management's Discussion and Analysis",
    "7A": "Quantitative and Qualitative Disclosures About Market Risk",
    "8": "Financial Statements and Supplementary Data",
    "9": "Changes in and Disagreements with Accountants",
    "9A": "Controls and Procedures",
    "9B": "Other Information",
    "9C": "Disclosure Regarding Foreign Jurisdictions that Prevent Inspections",
    "10": "Directors, Executive Officers and Corporate Governance",
    "11": "Executive Compensation",
    "12": "Security Ownership of Certain Beneficial Owners and Management",
    "13": "Certain Relationships and Related Transactions",
    "14": "Principal Accountant Fees and Services",
    "15": "Exhibits, Financial Statement Schedules",
    "16": "Form 10-K Summary",
}

# A leaf "block" is one of these tags with no block-level descendant.
BLOCKISH = {
    "p", "div", "li", "tr", "th", "td", "caption", "section", "article",
    "header", "footer", "h1", "h2", "h3", "h4", "h5", "h6", "dd", "dt",
}
BLOCK_DESCENDANT = {
    "p", "div", "li", "ul", "ol", "table", "tr", "h1", "h2", "h3", "h4",
    "h5", "h6", "section", "article",
}

# Item heading like "Item 7." / "Item 1A." / "ITEM 7A —"
ITEM_RE = re.compile(r"^item\s+(\d{1,2})\s*([A-Ca-c])?\s*[.\-—:)]", re.IGNORECASE)
NUMERIC_RE = re.compile(r"^[\(\$\-]?[\d,]+(\.\d+)?[\)%]?$")
CURRENCY_SYMBOLS = {"$", "%"}


def normspace(text):
    """Collapse all runs of whitespace (incl. non-breaking spaces) to one space."""
    return re.sub(r"\s+", " ", text.replace("\xa0", " ")).strip()


# --------------------------------------------------------------------------- #
# Inline-XBRL / boilerplate stripping
# --------------------------------------------------------------------------- #
def strip_ixbrl(soup):
    """Remove inline-XBRL scaffolding and non-content noise, in place.

    - drop <script>, <style>, HTML comments, and hidden iXBRL facts;
    - drop the <ix:header> block entirely;
    - unwrap the remaining ix:* wrappers so their visible text survives.
    """
    for element in soup.find_all(string=lambda s: isinstance(s, Comment)):
        element.extract()

    for tag in soup.find_all(["script", "style"]):
        tag.decompose()

    # Hidden iXBRL facts and header metadata carry no reader-visible content.
    for tag in soup.find_all(True):
        name = (tag.name or "").lower()
        if name in ("ix:header", "ix:hidden"):
            tag.decompose()

    # Elements explicitly hidden via CSS are layout/iXBRL noise.
    for tag in soup.find_all(style=re.compile(r"display\s*:\s*none", re.I)):
        tag.decompose()

    # Unwrap any surviving ix:* wrappers (keep their text in place).
    for tag in soup.find_all(True):
        if (tag.name or "").lower().startswith("ix:"):
            tag.unwrap()

    return soup


# --------------------------------------------------------------------------- #
# HTML table -> markdown (hand-written; §14 "#1 cause of wrong numbers")
# --------------------------------------------------------------------------- #
def _direct_cells(tr):
    return [c for c in tr.find_all(["td", "th"], recursive=False)]


def table_to_grid(table):
    """Expand an HTML table into a rectangular grid, honoring colspan/rowspan."""
    rows = [tr for tr in table.find_all("tr") if tr.find_parent("table") is table]
    grid = []
    carry = {}  # column index -> (text, remaining_rows) for active rowspans
    for tr in rows:
        row = []
        col = 0

        def drain_carry():
            nonlocal col
            while col in carry:
                text, remaining = carry[col]
                row.append(text)
                if remaining - 1 > 0:
                    carry[col] = (text, remaining - 1)
                else:
                    del carry[col]
                col += 1

        for cell in _direct_cells(tr):
            drain_carry()
            text = normspace(cell.get_text(" "))
            colspan = int(cell.get("colspan", 1) or 1)
            rowspan = int(cell.get("rowspan", 1) or 1)
            for k in range(colspan):
                value = text if k == 0 else ""
                row.append(value)
                if rowspan > 1:
                    carry[col] = (value, rowspan - 1)
                col += 1
        drain_carry()
        grid.append(row)

    width = max((len(r) for r in grid), default=0)
    for r in grid:
        r.extend([""] * (width - len(r)))
    return grid


def _drop_empty_columns(grid):
    if not grid:
        return grid
    ncols = len(grid[0])
    keep = [c for c in range(ncols) if any(row[c] for row in grid)]
    return [[row[c] for c in keep] for row in grid]


def _drop_empty_rows(grid):
    return [row for row in grid if any(cell for cell in row)]


def _merge_cell(prefix, value):
    if not prefix:
        return value
    if not value:
        return prefix
    if prefix == "$":
        return "$" + value
    return prefix + " " + value


def _merge_symbol_columns(grid):
    """Fold spacer/currency-only columns (e.g. a lone "$") into their neighbor.

    Apple's statements put "$" and the figure in separate <td>s with layout
    spacer columns between years; without folding, a year header sitting above
    the "$" drifts one column off its number. Data rows (row 0 is the header)
    decide whether a column is symbol-only; the header text rides along in the
    merge, landing directly above its figure.
    """
    if not grid:
        return grid
    ncols = len(grid[0])
    data_rows = grid[1:] if len(grid) > 1 else grid

    def is_symbol_col(c):
        non_empty = [row[c] for row in data_rows if row[c]]
        return bool(non_empty) and all(cell in CURRENCY_SYMBOLS for cell in non_empty)

    columns = [[row[c] for row in grid] for c in range(ncols)]
    result = []
    pending = None  # a symbol column waiting to merge into the next real column
    for c in range(ncols):
        if is_symbol_col(c):
            pending = columns[c]
            continue
        column = columns[c]
        if pending is not None:
            column = [_merge_cell(pending[r], column[r]) for r in range(len(column))]
            pending = None
        result.append(column)
    if pending is not None:  # trailing symbol column
        if result:
            result[-1] = [_merge_cell(result[-1][r], pending[r]) for r in range(len(pending))]
        else:
            result.append(pending)

    nrows = len(grid)
    return [[col[r] for col in result] for r in range(nrows)]


def _looks_numeric(cell):
    return bool(NUMERIC_RE.match(cell.replace(" ", "")))


def grid_to_markdown(grid):
    """Render a cleaned grid as a markdown table (or plain text if 1 column)."""
    grid = _drop_empty_rows(_merge_symbol_columns(_drop_empty_columns(grid)))
    if not grid or not grid[0]:
        return ""

    def esc(cell):
        return cell.replace("|", "\\|")

    ncols = len(grid[0])
    if ncols == 1:
        # A single-column "table" is a layout wrapper — render as lines of text.
        return "\n".join(esc(row[0]) for row in grid if row[0])

    header, body = grid[0], grid[1:]
    aligns = []
    for c in range(ncols):
        data = [row[c] for row in body if row[c]]
        numeric = sum(1 for cell in data if _looks_numeric(cell))
        aligns.append("---:" if data and numeric >= len(data) / 2 else ":---")

    lines = ["| " + " | ".join(esc(h) for h in header) + " |"]
    lines.append("| " + " | ".join(aligns) + " |")
    for row in body:
        lines.append("| " + " | ".join(esc(cell) for cell in row) + " |")
    return "\n".join(lines)


def table_to_markdown(table):
    return grid_to_markdown(table_to_grid(table))


# --------------------------------------------------------------------------- #
# Linearize the document into ordered typed blocks
# --------------------------------------------------------------------------- #
def _has_block_descendant(tag):
    return tag.find(lambda t: t.name in BLOCK_DESCENDANT) is not None


def _is_toc_block(tag):
    """A block whose content is a hyperlink (href) is a table-of-contents entry."""
    return tag.find(lambda t: t.name == "a" and t.get("href")) is not None


def iter_blocks(node, in_link=False):
    """Yield ordered blocks: {'kind': 'text'|'table', 'text'/'md', 'in_link'}."""
    for child in node.children:
        if isinstance(child, NavigableString):
            continue
        name = (child.name or "").lower()
        if name == "table":
            md = table_to_markdown(child)
            if md:
                yield {"kind": "table", "md": md, "in_link": in_link}
            continue
        child_in_link = in_link or name == "a"
        if name in BLOCKISH and not _has_block_descendant(child):
            text = normspace(child.get_text(" "))
            if text:
                yield {
                    "kind": "text",
                    "text": text,
                    "in_link": child_in_link or _is_toc_block(child),
                }
        else:
            yield from iter_blocks(child, child_in_link)


# --------------------------------------------------------------------------- #
# Section assembly
# --------------------------------------------------------------------------- #
def _match_item(text):
    """Return a canonical item key (e.g. "7", "1A") if text is an item heading."""
    if len(text) > 160:
        return None
    m = ITEM_RE.match(text)
    if not m:
        return None
    number, letter = m.group(1), (m.group(2) or "").upper()
    return number + letter


def assemble_sections(blocks):
    """Split the block stream into item sections (front matter dropped)."""
    sections = []
    current = None
    seen = set()
    for block in blocks:
        item = None
        if block["kind"] == "text" and not block["in_link"]:
            item = _match_item(block["text"])
        if item and item not in seen:
            seen.add(item)
            title = ITEM_TITLES.get(item, block["text"])
            current = {"item": item, "title": title, "blocks": []}
            sections.append(current)
            continue
        if current is not None:
            current["blocks"].append(block)
    return sections


def fiscal_year_from_name(path):
    m = re.search(r"FY(\d{4})", Path(path).name)
    return m.group(1) if m else None


def parse_filing(path):
    """Parse one 10-K HTML file into a structured document.

    Returns ``{fiscal_year, source_file, sections}`` where each section is
    ``{item, title, blocks}`` and each block is a typed text/table dict.
    """
    path = Path(path)
    html = path.read_text(encoding="utf-8", errors="replace")
    soup = BeautifulSoup(html, "lxml")
    strip_ixbrl(soup)
    body = soup.body or soup
    blocks = list(iter_blocks(body))
    sections = assemble_sections(blocks)
    return {
        "fiscal_year": fiscal_year_from_name(path),
        "source_file": path.name,
        "sections": sections,
    }


def section_to_text(section):
    """Render a section's blocks back to readable text (tables as markdown)."""
    parts = []
    for block in section["blocks"]:
        parts.append(block["md"] if block["kind"] == "table" else block["text"])
    return "\n\n".join(parts)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _load_meta():
    meta_path = RAW_DIR / META_FILENAME
    if meta_path.exists():
        return json.loads(meta_path.read_text(encoding="utf-8"))
    return []


def _most_recent_file():
    """Resolve the most-recent-FY filing via metadata, else by filename."""
    meta = _load_meta()
    if meta:
        newest = max(meta, key=lambda m: m["report_date"])
        return RAW_DIR / f"AAPL_10-K_FY{newest['report_date'][:4]}.html"
    candidates = sorted(RAW_DIR.glob("AAPL_10-K_FY*.html"))
    return candidates[-1] if candidates else None


def _resolve_file(args):
    if args.file:
        return Path(args.file)
    if args.fy:
        return RAW_DIR / f"AAPL_10-K_FY{args.fy}.html"
    return _most_recent_file()


def _print_summary(doc):
    print(f"{doc['source_file']}  (FY{doc['fiscal_year']})")
    for section in doc["sections"]:
        n_tables = sum(1 for b in section["blocks"] if b["kind"] == "table")
        words = sum(
            len(b["text"].split()) for b in section["blocks"] if b["kind"] == "text"
        )
        print(
            f"  Item {section['item']:<3} {section['title'][:48]:<50} "
            f"blocks={len(section['blocks']):<4} tables={n_tables:<3} words={words}"
        )


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Parse a 10-K into sections and markdown tables (FR-1).",
    )
    parser.add_argument("--file", help="path to a specific 10-K HTML file")
    parser.add_argument("--fy", help="fiscal year, e.g. 2024 (resolves data/raw)")
    parser.add_argument(
        "--item",
        help="print the parsed markdown for one item (e.g. 8, 7, 1A)",
    )
    args = parser.parse_args(argv)

    if not args.file and not args.fy and not args.item:
        # Default: summarize every filing in data/raw.
        files = sorted(p for p in RAW_DIR.glob("AAPL_10-K_FY*.html"))
        if not files:
            print(
                "No filings found in data/raw/. Run `python fetch_filings.py` "
                "first (see README).",
                file=sys.stderr,
            )
            return 1
        for path in files:
            _print_summary(parse_filing(path))
            print()
        return 0

    path = _resolve_file(args)
    if not path or not Path(path).exists():
        print(f"error: filing not found: {path}", file=sys.stderr)
        return 1

    doc = parse_filing(path)
    if not args.item:
        _print_summary(doc)
        return 0

    want = args.item.upper()
    for section in doc["sections"]:
        if section["item"] == want:
            print(f"# Item {section['item']} — {section['title']}")
            print(f"# ({doc['source_file']}, FY{doc['fiscal_year']})\n")
            print(section_to_text(section))
            return 0
    print(
        f"error: Item {want} not found in {path.name}. "
        f"Available: {', '.join(s['item'] for s in doc['sections'])}",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
