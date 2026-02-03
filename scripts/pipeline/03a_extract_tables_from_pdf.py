# -*- coding: utf-8 -*-
"""
Extract table-like blocks directly from PDF (preferred over OCR txt).

Output schema matches 03b_extract_tables.py:
TableBlock: {doc_id, source_path, block_id, start_line, end_line, header_lines, rows, raw_text}

Usage:
  python scripts/pipeline/03a_extract_tables_from_pdf.py ^
    --inputs data/interim/pdf/2024/SavingsFirst_613679 ^
    --out data/interim/tables/table_sidecar_savingsfirst_pdf.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional, Iterable

import pdfplumber

# reuse-ish anchors (keep minimal & robust)
ANCHOR_PATTERNS = [
    re.compile(r"\bselected\s+financial\s+data\b", re.I),
    re.compile(r"\bfive\s+year\s+summary\b", re.I),
    re.compile(r"\bperformance\s+ratios\b", re.I),
    re.compile(r"\bselected\s+balance\s+sheet\s+data\b", re.I),
    re.compile(r"\bselected\s+income\s+statement\s+data\b", re.I),
    re.compile(r"\bconsolidated\s+statements?\b", re.I),
    re.compile(r"\bstatement\s+of\s+(condition|operations|income|earnings|cash\s+flows)\b", re.I),
    re.compile(r"\bnet\s+interest\s+(income|margin)\b", re.I),
    re.compile(r"\breturn\s+on\b", re.I),
    re.compile(r"\bprovision\s+for\s+(credit\s+losses|loan\s+losses)\b", re.I),
]

NUM_LIKE_RE = re.compile(r"(\$?\(?\d{1,3}(?:,\d{3})+(?:\.\d+)?\)?|\$?\(?\d+(?:\.\d+)?\)?|\d+(?:\.\d+)?\s*%)")

@dataclass
class TableBlock:
    doc_id: str
    source_path: str
    block_id: str
    start_line: int
    end_line: int
    header_lines: List[str]
    rows: List[str]
    raw_text: str

def expand_inputs(inputs: List[str]) -> List[Path]:
    out: List[Path] = []
    for x in inputs:
        p = Path(x)
        if p.is_dir():
            out.extend(sorted(p.rglob("*.pdf")))
        elif p.is_file() and p.suffix.lower() == ".pdf":
            out.append(p)
    # de-dup
    seen = set()
    uniq = []
    for p in out:
        if str(p) not in seen:
            uniq.append(p); seen.add(str(p))
    return uniq

def normalize_cell(s: Optional[str]) -> str:
    if s is None:
        return ""
    s = s.replace("\u00a0", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s

def table_quality(table: List[List[Optional[str]]]) -> int:
    # heuristic score: more numeric cells + more non-empty cells => higher
    sc = 0
    for row in table:
        for c in row:
            t = normalize_cell(c)
            if not t:
                continue
            sc += 1
            if NUM_LIKE_RE.search(t):
                sc += 2
    return sc

def table_to_rows(table: List[List[Optional[str]]]) -> List[str]:
    rows: List[str] = []
    for row in table:
        cells = [normalize_cell(c) for c in row]
        # drop fully empty rows
        if not any(cells):
            continue
        # join with two spaces (keeps readability, downstream regex still works)
        rows.append("  ".join([c for c in cells if c != ""]))
    return rows

def page_has_anchor(text: str) -> bool:
    if not text:
        return False
    return any(p.search(text) for p in ANCHOR_PATTERNS)

def pick_header_lines(text: str) -> List[str]:
    if not text:
        return []
    lines = [re.sub(r"\s+", " ", l).strip() for l in text.splitlines() if l.strip()]
    # pick top few lines that look like headings (all caps or contains keywords)
    heads = []
    for l in lines[:30]:
        if l.isupper() or any(p.search(l) for p in ANCHOR_PATTERNS):
            heads.append(l)
        if len(heads) >= 6:
            break
    if not heads:
        heads = lines[:4]
    return heads[:6]

def extract_tables_from_page(page) -> List[List[List[Optional[str]]]]:
    # try "lines" first, then "text" fallback
    settings_list = [
        dict(vertical_strategy="lines", horizontal_strategy="lines", snap_tolerance=3, join_tolerance=3, intersection_tolerance=3),
        dict(vertical_strategy="text", horizontal_strategy="text", snap_tolerance=3, join_tolerance=3, intersection_tolerance=3, min_words_vertical=2, min_words_horizontal=1),
    ]
    all_tables = []
    for st in settings_list:
        try:
            ts = page.extract_tables(table_settings=st) or []
            if ts:
                all_tables.extend(ts)
        except Exception:
            continue
    return all_tables

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--min_rows", type=int, default=4)
    ap.add_argument("--min_score", type=int, default=40)  # quality threshold
    ap.add_argument("--max_pages", type=int, default=0, help="0 = no limit")
    args = ap.parse_args()

    pdf_paths = expand_inputs(args.inputs)
    if not pdf_paths:
        print("[FATAL] No pdf files found under given inputs.")
        return

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    total_blocks = 0
    with out_path.open("w", encoding="utf-8") as f:
        for pdf in pdf_paths:
            doc_id = pdf.stem
            blocks: List[TableBlock] = []

            with pdfplumber.open(str(pdf)) as pdfdoc:
                n_pages = len(pdfdoc.pages)
                limit = n_pages if args.max_pages <= 0 else min(n_pages, args.max_pages)

                for pi in range(limit):
                    page = pdfdoc.pages[pi]
                    text = page.extract_text() or ""
                    if not page_has_anchor(text) and (text.count("%") + len(NUM_LIKE_RE.findall(text)) < 30):
                        continue

                    tables = extract_tables_from_page(page)
                    if not tables:
                        continue

                    # pick good tables only
                    good = []
                    for t in tables:
                        if not t or len(t) < args.min_rows:
                            continue
                        sc = table_quality(t)
                        if sc >= args.min_score:
                            good.append((sc, t))
                    if not good:
                        continue

                    good.sort(key=lambda x: x[0], reverse=True)
                    header_lines = pick_header_lines(text)
                    raw_text = "\n".join([l for l in (text.splitlines() if text else [])[:60]])

                    # emit top-N tables on this page (usually 1-2)
                    for ti, (sc, t) in enumerate(good[:2]):
                        rows = table_to_rows(t)
                        if len(rows) < args.min_rows:
                            continue
                        block_id = f"pdf_p{pi+1:03d}_t{ti:02d}"
                        blocks.append(TableBlock(
                            doc_id=doc_id,
                            source_path=str(pdf),
                            block_id=block_id,
                            start_line=0,
                            end_line=0,
                            header_lines=header_lines,
                            rows=rows,
                            raw_text=raw_text,
                        ))

            for b in blocks:
                f.write(json.dumps(asdict(b), ensure_ascii=False) + "\n")

            total_blocks += len(blocks)
            print(f"[OK] {pdf.name}: blocks={len(blocks)}")

    print(f"[DONE] wrote {total_blocks} blocks -> {out_path}")

if __name__ == "__main__":
    main()
