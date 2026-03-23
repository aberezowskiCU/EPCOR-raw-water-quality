# -*- coding: utf-8 -*-
"""
Extract "Raw Water Quality – North Saskatchewan River" table(s) from monthly PDF reports
and write a tidy CSV for analysis.

Output (raw_water_quality.csv):
    date, year, month, day, plant, parameter, stat, value

Highlights:
- pdfplumber-only (no Camelot); tries multiple table strategies per page.
- Finds pages whose text includes BOTH "Raw Water Quality" and "North Saskatchewan River".
- Infers month/year from page text (e.g., "December 2025"); falls back to filename.
- Removes footer rows like "Monthly Min/Max/Avg".
- Accepts both .pdf and .PDF.
- CLI usage examples:
    python scripts/extract_raw_water_quality.py --in-dir data/pdf --verbose
    python scripts/extract_raw_water_quality.py --file data/pdf/2025-01_edmonton_water-quality_monthly-report.pdf --verbose
"""

import argparse
import glob
import os
import re
import sys
import warnings
from typing import List, Optional, Tuple

import pandas as pd
from dateutil.parser import parse as parse_date
import pdfplumber


# ----------------------- Patterns & Constants -----------------------

# Flexible match: tolerate different dashes, line breaks, spacing
TITLE_RE = re.compile(
    r'Raw\s+Water\s+Quality.*North\s+Saskatchewan\s+River',
    re.IGNORECASE | re.DOTALL,
)

MONTH_RE = re.compile(
    r'\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}\b',
    re.IGNORECASE,
)

PLANTS  = ["Rosedale", "E.L. Smith"]
PARAMS  = ["Turbidity (NTU)", "pH", "Colour (TCU)"]
STATS   = ["Min", "Max", "Avg"]


# ----------------------- Helper utilities -----------------------

def log(msg: str, verbose: bool = True) -> None:
    if verbose:
        print(msg, flush=True)


def find_title_text(page) -> Tuple[bool, Optional[str]]:
    """Return (is_target_page, month_label) by scanning page text."""
    text = page.extract_text() or ""
    if TITLE_RE.search(text):
        m = MONTH_RE.search(text)
        month_label = m.group(0) if m else None
        return True, month_label
    return False, None


def parse_month_year(label: Optional[str]) -> Tuple[Optional[pd.Timestamp], Optional[str]]:
    """Parse 'December 2025' → (Timestamp(2025-12-01), '2025-12')."""
    if not label:
        return None, None
    dt = parse_date(label, default=pd.Timestamp(2000, 1, 1))
    ym = pd.Timestamp(year=dt.year, month=dt.month, day=1)
    return ym, f"{ym:%Y-%m}"


def clean_df(df: pd.DataFrame) -> pd.DataFrame:
    """Basic cleaning on extracted table frames."""
    df = df.dropna(axis=1, how='all')
    df.columns = [str(c).strip() for c in df.columns]
    # Drop header rows mistakenly captured as data
    try:
        first_col = df.columns[0]
        df = df[~df[first_col].astype(str).str.contains(r"^\s*Day\s*$", case=False, na=False)]
    except Exception:
        pass
    return df


def reshape_table(df: pd.DataFrame, month_ym: pd.Timestamp) -> pd.DataFrame:
    """
    Convert wide table with merged headers to tidy rows:
    date, year, month, day, plant, parameter, stat, value
    """
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    # Identify 'Day' column (sometimes unlabeled)
    day_col = next((c for c in df.columns if c.strip().lower() == "day"), df.columns[0])

    # Remove footer summary rows like "Monthly Min/Max/Avg"
    df = df[~df[day_col].astype(str).str.contains("Monthly", case=False, na=False)]

    # Keep rows with a 1–31 day number
    day_num = df[day_col].astype(str).str.extract(r'(\d{1,2})')[0]
    df = df[day_num.notna()].copy()

    # Build a column map to (plant, parameter, stat)
    colmap = {}
    for c in df.columns:
        if c == day_col:
            continue
        hdr = str(c)
        plant = next((p for p in PLANTS if p.lower() in hdr.lower()), None)
        param = next((p for p in PARAMS if p.lower().replace(" ", "") in hdr.lower().replace(" ", "")), None)
        stat  = next((s for s in STATS  if re.search(rf'\b{s}\b', hdr, re.I)), None)

        # If headers are generic, infer by position (assumes Day then 9 cols per plant)
        if not plant or not param or not stat:
            idx = df.columns.get_loc(c) - 1  # after Day
            plant_idx = idx // 9
            within    = idx % 9
            param_idx = within // 3
            stat_idx  = within % 3
            plant = PLANTS[plant_idx] if plant_idx < len(PLANTS) else "Unknown"
            param = PARAMS[param_idx] if param_idx < len(PARAMS) else "Unknown"
            stat  = STATS[stat_idx]  if stat_idx  < len(STATS)  else "Value"

        colmap[c] = (plant, param, stat)

    # Melt to long
    value_cols = [c for c in df.columns if c != day_col]
    long_df = df.melt(id_vars=[day_col], value_vars=value_cols, var_name="column", value_name="value")
    long_df["plant"]     = long_df["column"].map(lambda c: colmap[c][0])
    long_df["parameter"] = long_df["column"].map(lambda c: colmap[c][1])
    long_df["stat"]      = long_df["column"].map(lambda c: colmap[c][2])
    long_df = long_df.drop(columns=["column"])

    # Build date
    long_df["day"] = long_df[day_col].astype(str).str.extract(r'(\d{1,2})')[0].astype(int)
    long_df = long_df.drop(columns=[day_col])
    long_df["year"]  = month_ym.year
    long_df["month"] = month_ym.month
    long_df["date"]  = pd.to_datetime(
        dict(year=long_df["year"], month=long_df["month"], day=long_df["day"]),
        errors="coerce"
    )

    # Numeric coercion
    long_df["value"] = pd.to_numeric(long_df["value"], errors="coerce")

    # Order
    cols = ["date", "year", "month", "day", "plant", "parameter", "stat", "value"]
    long_df = long_df[cols].sort_values(["date", "plant", "parameter", "stat"], ignore_index=True)
    return long_df


# ----------------------- pdfplumber extraction -----------------------

def extract_tables_from_page(page) -> List[pd.DataFrame]:
    """
    Try multiple pdfplumber strategies to extract tables from a single page.
    Returns a list of DataFrames (may include non-target tables; we filter later).
    """
    frames: List[pd.DataFrame] = []

    # Strategy A: grid (lines) – best if borders are present
    settings_lines = {
        "vertical_strategy": "lines",
        "horizontal_strategy": "lines",
        "intersection_y_tolerance": 8,
        "intersection_x_tolerance": 8,
        "snap_tolerance": 6,
        "join_tolerance": 6,
        "edge_min_length": 40,
    }
    tbl = page.extract_table(table_settings=settings_lines)
    if tbl:
        frames.append(pd.DataFrame(tbl))
    for tb in page.extract_tables(table_settings=settings_lines) or []:
        frames.append(pd.DataFrame(tb))

    # Strategy B: mix – lines for one direction, text for the other
    settings_mixed = {
        "vertical_strategy": "lines",
        "horizontal_strategy": "text",
        "intersection_y_tolerance": 8,
        "snap_tolerance": 6,
        "join_tolerance": 6,
        "edge_min_length": 40,
    }
    tbl = page.extract_table(table_settings=settings_mixed)
    if tbl:
        frames.append(pd.DataFrame(tbl))
    for tb in page.extract_tables(table_settings=settings_mixed) or []:
        frames.append(pd.DataFrame(tb))

    # Strategy C: text only – when borders are very faint or broken
    settings_text = {
        "vertical_strategy": "text",
        "horizontal_strategy": "text",
        "snap_tolerance": 6,
        "join_tolerance": 6,
    }
    tbl = page.extract_table(table_settings=settings_text)
    if tbl:
        frames.append(pd.DataFrame(tbl))
    for tb in page.extract_tables(table_settings=settings_text) or []:
        frames.append(pd.DataFrame(tb))

    return frames


def pages_with_title(pdf) -> List[Tuple[int, Optional[str]]]:
    """Return list of (1-based page_index, month_label) for pages containing the title."""
    hits: List[Tuple[int, Optional[str]]] = []
    for idx, page in enumerate(pdf.pages, start=1):
        ok, label = find_title_text(page)
        if ok:
            hits.append((idx, label))
    return hits


def extract_with_pdfplumber(pdf_path: str, verbose: bool = True) -> List[Tuple[pd.DataFrame, Optional[str]]]:
    """Open a PDF, pick target pages, and return candidate tables with month labels."""
    frames: List[Tuple[pd.DataFrame, Optional[str]]] = []
    with pdfplumber.open(pdf_path) as pdf:
        title_hits = pages_with_title(pdf)
        # If no explicit title page found, scan all pages (lenient mode)
        plan = title_hits if title_hits else [(i, None) for i in range(1, len(pdf.pages) + 1)]
        log(f"  pdfplumber: scanning page(s) {[p for p, _ in plan]}", verbose)

        for page_idx, month_label in plan:
            page = pdf.pages[page_idx - 1]
            text = page.extract_text() or ""
            if month_label is None:
                m = MONTH_RE.search(text)
                month_label = m.group(0) if m else None

            for df in extract_tables_from_page(page):
                frames.append((df, month_label))
    return frames


# ----------------------- Orchestration -----------------------

def gather_pdfs(in_dir: str) -> List[str]:
    """Case-insensitive glob for PDFs in a directory."""
    return sorted(
        glob.glob(os.path.join(in_dir, "*.pdf")) +
        glob.glob(os.path.join(in_dir, "*.PDF"))
    )


def main(
    in_dir: str = "data/pdf",
    out_csv: str = "data/output/raw_water_quality.csv",
    out_month_csv: Optional[str] = "data/output/monthly_summary.csv",
    file_path: Optional[str] = None,
    verbose: bool = True,
) -> None:

    # Choose files
    if file_path:
        if not os.path.isfile(file_path):
            print(f"[!] File not found: {file_path}", flush=True)
            sys.exit(1)
        pdfs = [file_path]
    else:
        pdfs = gather_pdfs(in_dir)

    if not pdfs:
        print(f"[!] No PDFs found. Check --in-dir or --file.", flush=True)
        sys.exit(1)

    # Ensure output folders exist
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    if out_month_csv:
        os.makedirs(os.path.dirname(out_month_csv), exist_ok=True)

    all_rows: List[pd.DataFrame] = []

    for pdf_path in pdfs:
        log(f"\n--- Processing: {os.path.basename(pdf_path)} ---", verbose)
        month_ym: Optional[pd.Timestamp] = None
        frames: List[Tuple[pd.DataFrame, Optional[str]]] = []

        # pdfplumber candidates (also helps identify month)
        frames_pp = extract_with_pdfplumber(pdf_path, verbose=verbose)
        if frames_pp:
            frames.extend(frames_pp)
            # Capture month_ym from any label found
            for _, label in frames_pp:
                if label:
                    month_ym, _ = parse_month_year(label)
                    break

        # Fallback: infer month from filename if still missing
        if month_ym is None:
            m = MONTH_RE.search(os.path.basename(pdf_path))
            if m:
                month_ym, _ = parse_month_year(m.group(0))

        # Reshape & collect
        got_any = False
        for df, _label in frames:
            df = clean_df(pd.DataFrame(df))
            # Skip very small/narrow frames
            if df.shape[1] < 10 or df.shape[0] < 5:
                continue
            try:
                if month_ym is None:
                    # If month is truly unknown, skip this frame
                    continue
                long_df = reshape_table(df, month_ym)
                if not long_df.empty:
                    all_rows.append(long_df)
                    got_any = True
            except Exception as e:
                warnings.warn(f"Reshape failed for {os.path.basename(pdf_path)}: {e}")

        if not got_any:
            warnings.warn(
                f"No valid tables found in {os.path.basename(pdf_path)}. "
                f"If this is a scanned PDF or the section title is different, we may need a custom rule."
            )

    if not all_rows:
        print("[!] No data extracted from any PDF.", flush=True)
        sys.exit(2)

    final = pd.concat(all_rows, ignore_index=True).drop_duplicates()
    final.to_csv(out_csv, index=False)
    log(f"\n✅ Wrote {out_csv} with {len(final):,} rows.", verbose)

    if out_month_csv:
        monthly = (
            final.groupby(["year", "month", "plant", "parameter", "stat"], dropna=False)
                 .agg(value_mean=("value", "mean"))
                 .reset_index()
                 .sort_values(["year", "month", "plant", "parameter", "stat"], ignore_index=True)
        )
        monthly.to_csv(out_month_csv, index=False)
        log(f"✅ Wrote {out_month_csv}.", verbose)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract Raw Water Quality – North Saskatchewan River tables from PDFs.")
    parser.add_argument("--in-dir", default="data/pdf", help="Folder containing PDF files (default: data/pdf)")
    parser.add_argument("--file",   default=None,      help="Process a single PDF path (overrides --in-dir)")
    parser.add_argument("--out-csv", default="data/output/raw_water_quality.csv", help="Output CSV path")
    parser.add_argument("--out-month-csv", default="data/output/monthly_summary.csv", help="Monthly summary CSV path (or empty to skip)")
    parser.add_argument("--verbose", action="store_true", help="Print detailed progress")
    args = parser.parse_args()

    out_month = args.out_month_csv if (args.out_month_csv and args.out_month_csv.strip()) else None
    main(in_dir=args.in_dir, out_csv=args.out_csv, out_month_csv=out_month, file_path=args.file, verbose=args.verbose)


