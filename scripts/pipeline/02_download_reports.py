"""
Download annual report / 10-K PDFs from candidate URLs.

Input: a CSV produced by the entry-page stage, expected to contain:
- idrssd, username, best_pdf_url, (optional) score, year, pdf_type

Output:
- PDFs saved under data/raw/pdf_downloads/<pdf_type>/<year>/<bank_id>/
- A download result CSV with status and local_path
"""
import os
import re
import math
import time
import traceback
from pathlib import Path
from typing import Optional, Tuple
import sys

import pandas as pd
import requests

def find_repo_root(start: Path) -> Path:
    p = start.resolve()
    for _ in range(10):  # Search up to 10 parent directories for the project root.
        if (p / ".git").exists() or (p / "README.md").exists() or (p / "data").exists():
            return p
        p = p.parent
    raise RuntimeError(f"Cannot locate repo root from: {start}")

ROOT = find_repo_root(Path(__file__))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ========= Configuration =========
# Input CSV produced by the entry-page stage (expects best_pdf_url and basic metadata columns).
INPUT_CSV = ROOT/"data"/"interim"/"output"/"log"/"bank_candidate_entry_pages_for_pdf_2rd.csv" 

# Output CSV (overwritten on each run).
OUTPUT_CSV = ROOT/"data"/"interim"/"output"/"log"/"download_results.csv"

# Root directory for downloaded PDFs.
OUTPUT_ROOT = ROOT/"data"/"raw"/"pdf_downloads"

# Minimum score threshold for downloads (0.0 disables score-based filtering).
MIN_SCORE_FOR_DOWNLOAD = 0.0

# HTTP request configuration
MAX_RETRIES = 3
RETRY_BACKOFF = 2.0  # Exponential backoff multiplier per retry.
TIMEOUT = 40

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/121.0.0.0 Safari/537.36"
    ),
    "Accept": "application/pdf,application/octet-stream;q=0.9,*/*;q=0.8",
}


# ======== Utility functions ========

def ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def slugify(text: str, max_len: int = 60) -> str:
    """
    Convert an identifier (e.g., bank name / ID) into a filesystem-safe slug.
    """
    if not isinstance(text, str):
        text = str(text)

    text = text.strip()
    # Normalize whitespace into underscores.
    text = re.sub(r"[ \t\n\r]+", "_", text)
    # Retain only alphanumeric characters and a small set of safe symbols.
    text = re.sub(r"[^0-9A-Za-z_\-]+", "", text)
    if not text:
        text = "unknown"
    return text[:max_len]


def is_pdf_url(url: str) -> bool:
    """
    Heuristic check for PDF URLs based on the filename suffix.
    """
    if not isinstance(url, str):
        return False
    u = url.lower().split("?", 1)[0].split("#", 1)[0]
    return u.endswith(".pdf")


def parse_year_from_url_or_text(url: str, extra: str = "") -> Optional[int]:
    """
    Extract a plausible year from URL or auxiliary text (2000 .. current_year+1).
    """
    text = f"{url} {extra}"
    years = re.findall(r"(20\d{2})", text)
    if not years:
        return None

    years_int = []
    current = time.localtime().tm_year
    for y in years:
        yi = int(y)
        if 2000 <= yi <= current + 1:
            years_int.append(yi)

    if not years_int:
        return None

    # Prefer the most recent year found.
    return max(years_int)


def classify_pdf_type_from_row(row: pd.Series) -> str:
    """
    Infer a coarse PDF type label from the row content.
    If pdf_type is already provided in the input CSV, reuse it.
    """
    if "pdf_type" in row and isinstance(row["pdf_type"], str) and row["pdf_type"].strip():
        return row["pdf_type"].strip()

    url = str(row.get("url", "")).lower()
    name = str(row.get("username", "")).lower()

    text = f"{url} {name}"

    if "10-k" in text or "form10k" in text.replace(" ", ""):
        return "10k"
    if "annual-report" in text or "annualreport" in text.replace(" ", "") or "annual review" in text:
        return "annual_report"
    if "proxy" in text and "statement" in text:
        return "proxy"
    if "presentation" in text or "slides" in text or "deck" in text:
        return "presentation"
    if "quarterly" in text or re.search(r"q[1-4]\s*20\d{2}", text):
        return "quarterly"
    if "8-k" in text or "6-k" in text:
        return "sec_other"

    return "other"


def download_pdf(
    session: requests.Session,
    url: str,
    dest_path: str,
    referer: Optional[str] = None,
) -> Tuple[bool, str]:
    """
    Download a PDF to dest_path with retries.

    Returns:
        (success, reason): reason is either "ok"/"exists" or an error/warning tag
        such as http_403, empty_file, non_pdf_content_type:<ct>, exception:<name>.
    """
    if not isinstance(url, str) or not url.strip():
        return False, "empty_url"

    url = url.strip()

    headers = HEADERS.copy()
    if referer:
        headers["Referer"] = referer

    last_err = ""
    delay = 1.0
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = session.get(url, headers=headers, timeout=TIMEOUT, stream=True)
            if resp.status_code >= 400:
                last_err = f"http_{resp.status_code}"
            else:
                # Validate Content-Type (best-effort).
                ct = resp.headers.get("Content-Type", "").lower()
                if "pdf" not in ct and not is_pdf_url(url):
                    # Still save the response, but record a warning.    
                    last_err = f"non_pdf_content_type:{ct}"

                ensure_dir(os.path.dirname(dest_path))
                with open(dest_path, "wb") as f:
                    for chunk in resp.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)

                # Verify file size after writing.
                if os.path.getsize(dest_path) == 0:
                    last_err = "empty_file"
                    # Remove empty file before retrying.
                    try:
                        os.remove(dest_path)
                    except OSError:
                        pass
                    # Retry on empty output.
                else:
                    # Success (may include a non-fatal warning).
                    if last_err and "non_pdf_content_type" in last_err:
                        return True, last_err
                    return True, "ok"

        except Exception as e:
            last_err = f"exception:{e.__class__.__name__}"

        # Backoff before next retry.
        if attempt < MAX_RETRIES:
            time.sleep(delay)
            delay *= RETRY_BACKOFF

    return False, last_err or "unknown_error"


# ======== Main pipeline ========

def main():
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    Path(OUTPUT_CSV).parent.mkdir(parents=True, exist_ok=True)

    if not os.path.exists(INPUT_CSV):
        raise FileNotFoundError(f"Input CSV not found: {INPUT_CSV}")

    print(f"Reading input CSV: {INPUT_CSV}")
    df = pd.read_csv(INPUT_CSV)

    # 1) Validate required columns (best_pdf_url is the canonical URL field).
    for col in ["idrssd", "username", "best_pdf_url"]:
        if col not in df.columns:
            raise ValueError(f"Input CSV is missing required column: {col}")

    # 2) Backward compatibility: add optional output columns if missing.
    for col in ["status", "reason", "local_path"]:
        if col not in df.columns:
            df[col] = ""

    # 3) Filter rows with non-empty best_pdf_url.
    df["best_pdf_url"] = df["best_pdf_url"].fillna("").astype(str)
    df = df[df["best_pdf_url"].str.strip() != ""].copy()
    df.reset_index(drop=True, inplace=True)

    # Create a unified "url" column to keep downstream logic stable.
    df["url"] = df["best_pdf_url"]

    total = len(df)
    print(f"Valid rows: {total}")

    session = requests.Session()

    # Accumulate per-row results.
    results = []

    for idx, row in df.iterrows():
        idrssd = row.get("idrssd", "")
        username = str(row.get("username", "")).strip()
        url = str(row.get("url", "")).strip()
        score = row.get("score", math.nan)

        # Use pdf_type/year from input when available; otherwise infer from URL/text.
        pdf_type = classify_pdf_type_from_row(row)
        year = row.get("year", None)

        # Normalize year types (may be NaN/float/str).
        if isinstance(year, float) and math.isnan(year):
            year = None
        if isinstance(year, str) and not year.strip():
            year = None

        if year is None:
            year = parse_year_from_url_or_text(url) or "unknown"

        # Score-based filtering (disabled by default when threshold is 0.0)
        status = ""
        reason = ""
        local_path = ""

        if not url.lower().startswith("http"):
            status = "skip_invalid_url"
            reason = "not_http_url"
        else:
            strong_hint = ("annual" in url.lower()
                           or "10-k" in url.lower()
                           or "annual-report" in url.lower())

            if (
                isinstance(score, (int, float))
                and not math.isnan(score)
                and score < MIN_SCORE_FOR_DOWNLOAD
                and not strong_hint
            ):
                status = "skip_low_score"
                reason = f"score<{MIN_SCORE_FOR_DOWNLOAD}"
            else:
                # Perform download.
                dir_name = os.path.join(
                    OUTPUT_ROOT,
                    slugify(str(pdf_type)),
                    str(year),
                    # Add a per-bank subfolder to avoid too many files in a single directory.
                    f"{slugify(str(username))}_{slugify(str(idrssd))}",
                )
                ensure_dir(dir_name)

                # File name derived from the URL path (best-effort).
                url_last = url.split("?", 1)[0].split("#", 1)[0].rstrip("/")
                fname_part = url_last.rsplit("/", 1)[-1]
                fname_base = slugify(fname_part, max_len=80)
                if not fname_base.lower().endswith(".pdf"):
                    fname_base = fname_base + ".pdf"

                dest_path = os.path.join(dir_name, fname_base)

                # Skip download if the target file already exists and is non-empty.
                if os.path.exists(dest_path) and os.path.getsize(dest_path) > 0:
                    status = "success"
                    reason = "exists"
                    local_path = dest_path
                else:
                    print(f"[{idx+1}/{total}] Downloading {username} | {pdf_type} | {year}")
                    ok, msg = download_pdf(session, url, dest_path)
                    if ok:
                        status = "success"
                        reason = msg
                        local_path = dest_path
                    else:
                        status = "fail"
                        reason = msg
                        local_path = ""

        results.append(
            {
                "idx": idx,
                "idrssd": idrssd,
                "username": username,
                "url": url,
                "score": score,
                "pdf_type": pdf_type,
                "year": year,
                "status": status,
                "reason": reason,
                "local_path": local_path,
            }
        )

    out_df = pd.DataFrame(results)
    out_df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print(f"\nDone. Results written to: {OUTPUT_CSV}")
    print(out_df["status"].value_counts())


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print("Unhandled exception:")
        traceback.print_exc()
