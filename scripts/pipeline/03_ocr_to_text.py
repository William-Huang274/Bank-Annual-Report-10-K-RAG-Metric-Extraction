"""
OCR and text extraction for downloaded annual report PDFs.

This script:
1. Runs OCR on PDFs using ocrmypdf when needed
2. Normalizes outputs into a unified OCR PDF directory
3. Extracts text from OCR-processed PDFs into plain text files
4. Records per-file status in a structured CSV log

Designed for batch processing with bounded parallelism.
"""
import subprocess
from pathlib import Path
import csv
from typing import Optional
import os         
import fitz  # PyMuPDF
import shutil  
import sys

def find_repo_root(start: Path) -> Path:
    p = start.resolve()
    for _ in range(10):  # Search up to 10 parent directories for the project root
        if (p / ".git").exists() or (p / "README.md").exists() or (p / "data").exists():
            return p
        p = p.parent
    raise RuntimeError(f"Cannot locate repo root from: {start}")

ROOT = find_repo_root(Path(__file__))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Configure a project-local temporary directory for OCR-related intermediate files
CUSTOM_TEMP = ROOT/"_tmp"/"ocr_temp"
os.makedirs(CUSTOM_TEMP, exist_ok=True)

os.environ["TMP"] = CUSTOM_TEMP
os.environ["TEMP"] = CUSTOM_TEMP
os.environ["TMPDIR"] = CUSTOM_TEMP

# ========= Configuration =========

YEAR = "2024"

PDF_RAW_DIR = ROOT/"data"/"raw"/"pdf_downloads"/"annual_report"/ YEAR
PDF_OCR_DIR = ROOT/"data"/"interim"/"output"/ YEAR
TXT_OUT_DIR = ROOT/"data"/"interim"/"txt"/ YEAR

# CSV log recording OCR and text extraction status
LOG_CSV = ROOT/"data"/"interim"/"output"/"log"/f"ocr_extract_{YEAR}.csv"

# Determine a reasonable level of parallelism based on available CPU cores
CPU_COUNT = os.cpu_count() or 4
# Reserve 1–2 cores for the system to avoid resource contention
NUM_JOBS = max(CPU_COUNT - 2, 1)

PDF_OCR_DIR.mkdir(parents=True, exist_ok=True)
TXT_OUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_CSV.parent.mkdir(parents=True, exist_ok=True)

print(f"[INFO] Detected {CPU_COUNT} CPU cores; using --jobs {NUM_JOBS}")

# Base ocrmypdf command; additional flags can be added if needed.
OCR_CMD_BASE = [
    "ocrmypdf",
    "--force-ocr",          # Force OCR even if a text layer exists
    "--deskew",             # Deskew pages
    "--optimize", "1",      # Light PDF optimization
    "--skip-big", "200",    # Skip very large PDFs (MB)
    "--jobs", str(NUM_JOBS) # Enable parallel page-level OCR
]
# ========================


def pdf_has_text(pdf_path: Path, max_pages: int = 5) -> bool:
    """
    Heuristically determine whether a PDF already contains extractable text.

    If any of the first `max_pages` pages contains non-empty text, the PDF
    is considered text-searchable.
    """
    try:
        doc = fitz.open(pdf_path)
    except Exception:
        return False

    pages_to_check = min(len(doc), max_pages)
    for i in range(pages_to_check):
        text = doc.load_page(i).get_text("text").strip()
        if text:
            return True
    return False


def ocr_pdf_if_needed(src: Path, dst: Path) -> str:
    """
    Run OCR on a single PDF if required.

    Behavior:
    - If the destination OCR PDF already exists, reuse it
    - If the source PDF contains a text layer, skip OCR and copy the file
    - Otherwise, invoke ocrmypdf to generate an OCR-processed PDF

    Returns:
        "ok"   : OCR completed or reused successfully
        "fail" : OCR failed with a non-recoverable error
    """
    # Ensure output directory exists
    dst.parent.mkdir(parents=True, exist_ok=True)

    # Reuse existing OCR-processed PDF if it already exists.
    if dst.exists():
        print(f"[OCR SKIP] Reusing existing OCR PDF: {dst}")
        return "ok"

    cmd = OCR_CMD_BASE + [str(src), str(dst)]
    print(f"[OCR] Running OCR for: {src.name}")
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
    )

    rc = result.returncode

    if rc == 0:
        print(f"[OCR OK] {src.name}")
        return "ok"

    if rc == 3:
        # PDF already contains a text layer; OCR is not required.
        print(f"[OCR SKIP] {src.name}: existing text layer detected")
        try:
            shutil.copy2(src, dst)  # Copy to the OCR output location so downstream always reads from `dst`.
        except Exception as e:
            print(f"[OCR SKIP WARN] Failed to copy source PDF to {dst}: {e}")
        return "ok"

    # All other return codes are treated as unrecoverable OCR failures
    print(f"[OCR FAIL] {src.name} (code={rc})")
    print("stdout:", result.stdout[:300])
    print("stderr:", result.stderr[:300])
    return "fail"


def extract_text(pdf_path: Path, txt_path: Path) -> bool:
    """
    Extract plain text from a PDF using PyMuPDF and write it to a UTF-8 text file.
    """
    txt_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        doc = fitz.open(pdf_path)
        parts = []
        for page in doc:
            parts.append(page.get_text("text"))
        txt = "\n".join(parts)
    except Exception as e:
        print(f"[TXT FAIL] {pdf_path}: {e}")
        return False

    try:
        txt_path.write_text(txt, encoding="utf-8", errors="ignore")
        print(f"[TXT OK] {txt_path}")
        return True
    except Exception as e:
        print(f"[TXT WRITE FAIL] {txt_path}: {e}")
        return False


def iter_all_pdfs(root: Path):
    """
    Iterate over all PDF files under per-bank subdirectories.
    """
    for bank_dir in sorted(root.glob("*")):
        if not bank_dir.is_dir():
            continue
        for pdf in sorted(bank_dir.glob("*.pdf")):
            yield bank_dir.name, pdf


def infer_bank_and_year(bank_folder: str, pdf_name: str) -> tuple[str, Optional[int]]:
    """
    Infer bank identifier and report year.

    - The bank identifier is derived from the parent folder name
    - The report year is extracted from the filename if present,
    otherwise defaults to the configured YEAR
    """
    import re

    bank = bank_folder
    years = re.findall(r"(20\d{2})", pdf_name)
    yr = int(years[0]) if years else int(YEAR)
    # Filenames like '2025-SAR-Annual-Report' may require custom year mapping in a later stage.
    return bank, yr


def main():
    PDF_OCR_DIR.mkdir(parents=True, exist_ok=True)
    TXT_OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Collect all PDFs upfront to report overall progress.
    all_pdfs = list(iter_all_pdfs(PDF_RAW_DIR))
    total = len(all_pdfs)
    print(f"Found {total} PDF files under {PDF_RAW_DIR}")
    if total == 0:
        return

    # Open the log file; all writer.writerow calls must remain within this context.
    with LOG_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "bank_folder",
            "pdf_file",
            "bank_name",
            "report_year",
            "raw_pdf_path",
            "ocr_pdf_path",
            "txt_path",
            "ocr_status",
            "txt_status",
        ])

        for idx, (bank_folder, pdf_path) in enumerate(all_pdfs, start=1):
            rel_bank_dir = bank_folder  # Example: DeutscheBank_214807
            bank_name, year = infer_bank_and_year(rel_bank_dir, pdf_path.name)

            print(f"\n[{idx}/{total}] Processing {rel_bank_dir} / {pdf_path.name}")

            ocr_pdf_path = PDF_OCR_DIR / rel_bank_dir / pdf_path.name
            txt_path = TXT_OUT_DIR / rel_bank_dir / (pdf_path.stem + ".txt")

            # Skip processing if the text output already exists
            if txt_path.exists():
                print(f"    [SKIP] Text output already exists: {txt_path}")
                writer.writerow([
                    rel_bank_dir,
                    pdf_path.name,
                    bank_name,
                    year,
                    str(pdf_path),
                    str(ocr_pdf_path if ocr_pdf_path.exists() else ""),
                    str(txt_path),
                    "skip_exist",
                    "ok",
                ])
                continue

            # Run OCR first
            print(f"    [OCR] Starting OCR: {pdf_path.name}")
            ocr_status = ocr_pdf_if_needed(pdf_path, ocr_pdf_path)
            if ocr_status != "ok":
                print(f"    [OCR FAIL] OCR failed: {pdf_path.name}")
                writer.writerow([
                    rel_bank_dir,
                    pdf_path.name,
                    bank_name,
                    year,
                    str(pdf_path),
                    str(ocr_pdf_path),
                    str(txt_path),
                    ocr_status,
                    "fail",
                ])
                continue
            print(f"    [OCR OK] OCR PDF written: {ocr_pdf_path}")

            # Extract text
            print(f"    Extracting text: {pdf_path.name}")
            txt_ok = extract_text(ocr_pdf_path, txt_path)  
            txt_status = "ok" if txt_ok else "fail"

            if txt_ok:
                print(f"    [TXT OK] Text file written: {txt_path}")
            else:
                print(f"    [TXT FAIL] Text extraction failed: {txt_path}")

            # Write one structured log row per processed PDF (success or failure).
            writer.writerow([
                rel_bank_dir,
                pdf_path.name,
                bank_name,
                year,
                str(pdf_path),
                str(ocr_pdf_path),
                str(txt_path),
                ocr_status,
                txt_status,
            ])


if __name__ == "__main__":
    main()
