# -*- coding: utf-8 -*-
"""
Extract table-like blocks from OCR TXT outputs and write a sidecar JSONL.

Design goals:
- Work well on messy OCR text (bank annual reports, 10-K).
- Prefer deterministic extraction for numeric rows; keep LLM optional for header interpretation later.
- Output table blocks with header_lines + rows + raw_text to support rule-first extraction downstream.

Usage:
  python scripts/pipeline/03b_extract_tables.py ^
    --inputs data/raw/_GermanAmerican_37640/2025-SAR-Annual-Reportpdf.txt data/raw/Ally_3284070/2024-10kpdf.txt ^
    --out data/interim/tables/table_sidecar_2024.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict, Tuple, Optional

DEBUG_UNSTACK = True

# -----------------------------
# Heuristics tuned for 4 sample formats
# -----------------------------

ANCHOR_PATTERNS = [
    # GermanAmerican: "Five Year Summary", "Selected Performance Ratios", "Summary of Operations", etc.
    re.compile(r"\bfive\s+year\s+summary\b", re.I),
    re.compile(r"\bselected\s+financial\s+data\b", re.I),
    re.compile(r"\bselected\s+performance\s+ratios\b", re.I),
    re.compile(r"\bsummary\s+of\s+operations\b", re.I),

    # 10-K / annual report: "Statement of ...", "Consolidated statements ..."
    re.compile(r"\bstatement\s+of\s+(condition|operations|income|earnings|cash\s+flows)\b", re.I),
    re.compile(r"\bconsolidated\s+statements?\b", re.I),

    # Common table section labels
    re.compile(r"\bassets\b", re.I),
    re.compile(r"\bliabilities\b", re.I),
    re.compile(r"\bcapital\b", re.I),
    re.compile(r"\bperformance\s+ratios\b", re.I),
    re.compile(r"\basset\s+quality\b", re.I),
    re.compile(r"\bnet\s+interest\s+income\b", re.I),
    re.compile(r"\bnet\s+interest\s+margin\b", re.I),
    re.compile(r"\breturn\s+on\b", re.I),
    re.compile(r"\bprovision\s+for\s+(credit\s+losses|loan\s+losses)\b", re.I),
]

PHONE_RE = re.compile(r"\(\d{3}\)\s*\d{3}[-.\s]\d{4}")
ZIP_RE = re.compile(r"\b[A-Z]{2}\s+\d{5}(?:-\d{4})?\b")

# Lines likely to be chart axis labels / noise; we downweight them.
NOISE_PATTERNS = [
    re.compile(r"\b(total\s+assets|total\s+deposits|total\s+loans)\b.*\(\$\s*in\s*(millions|billions)\)", re.I),
    re.compile(r"\bfigure\s+\d+\b", re.I),
    re.compile(r"\bsource:\b", re.I),
    re.compile(r"\bpage\s+\d+\b", re.I),
]

# Detect numeric tokens including commas, decimals, percentages, and parentheses negatives
NUM_TOKEN_RE = re.compile(r"""
(?:
    \$?\(?\d{1,3}(?:,\d{3})+(?:\.\d+)?\)?     # 190,591 or (1,234) or $1,234.56
  | \$?\(?\d+(?:\.\d+)?\)?                    # 1234 or (1234) or 12.34
  | \d+(?:\.\d+)?\s*%                         # 3.27% or 3.27 %
)
""", re.VERBOSE)

YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")
DATE_COL_RE = re.compile(r"\b(?:december|june|march|september)\s+\d{1,2},\s+(19|20)\d{2}\b", re.I)

# Common unit indicators in headers
UNIT_HINT_RE = re.compile(r"\b(dollars?\s+in\s+(thousands|millions|billions)|\$\s*in\s+(thousands|millions|billions)|in\s+(thousands|millions|billions))\b", re.I)


VAL_TOKEN_RE = re.compile(r"""
    \(\s*\$?\d{1,3}(?:,\d{3})*(?:\.\d+)?\s*\)     |
    \d+(?:\.\d+)?\s*%                             |
    \$?\d{1,3}(?:,\d{3})*(?:\.\d+)?               |
    \$?\d+(?:\.\d+)?
""", re.VERBOSE)


def extract_value_tokens(line: str) -> list[str]:
    toks = [t.strip() for t in VAL_TOKEN_RE.findall(line or "")]
    # Normalize spacing quirks like "0.24 %" -> "0.24%".
    toks = [t.replace(" %", "%").replace("% ", "%") for t in toks]
    return toks

def normalize_line(s: str) -> str:
    s = s.replace("\u00a0", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def count_num_tokens(line: str) -> int:
    return len(NUM_TOKEN_RE.findall(line))


def has_noise(line: str) -> bool:
    if PHONE_RE.search(line) or ZIP_RE.search(line):
        return True
    return any(p.search(line) for p in NOISE_PATTERNS)


def looks_like_header(line: str) -> bool:
    # Header-ish if contains "Dollars in thousands", dates/years, or all-caps section label
    if UNIT_HINT_RE.search(line):
        return True
    if DATE_COL_RE.search(line):
        return True
    if len(YEAR_RE.findall(line)) >= 2:
        return True
    # a short all-caps label can be a section header
    if line.isupper() and 3 <= len(line) <= 60:
        return True
    return False


ALPHA_RE = re.compile(r"[A-Za-z]")
def alpha_count(s: str) -> int:
    return len(ALPHA_RE.findall(s))

def looks_like_table_row(line: str) -> bool:
    """
    Stricter table-row heuristic to suppress:
      - chart axis / year ticks (mostly digits, few letters)
      - long prose lines with many numbers/percents
    """

    if PHONE_RE.search(line) or ZIP_RE.search(line):
        return False

    if not line:
        return False
    if has_noise(line):
        return False

    # Prose lines are usually long. Tables rows are relatively short.
    if len(line) > 160:
        return False

    nnums = count_num_tokens(line)
    a = alpha_count(line)

    # Year-tick lines like "2015 2018 2021 2024" should NOT be treated as a row.
    if a < 4 and len(YEAR_RE.findall(line)) >= 2:
        return False

    # Numeric-heavy with almost no letters is likely chart ticks, not table rows.
    if nnums >= 2 and a < 6:
        return False

    # Typical multi-column row: >=2 numeric tokens and has some label text.
    if nnums >= 2 and a >= 6:
        return True

    # Single-number row allowed ONLY if it contains strong metric cues.
    if nnums == 1:
        low = line.lower()
        cues = [
            "net interest income", "net interest margin", "nim",
            "provision for credit losses", "provision for loan losses",
            "return on", "roa", "roe"
        ]
        if any(c in low for c in cues) and a >= 6:
            return True

    return False


def anchor_strength(line: str) -> int:
    """
    Higher means more likely table-related context.
    """
    score = 0
    for p in ANCHOR_PATTERNS:
        if p.search(line):
            score += 2
    if looks_like_header(line):
        score += 1
    return score


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

def is_numeric_only_line(line: str) -> bool:
    # numeric-only here means: has number token(s) and almost no letters
    if not line:
        return False
    if has_noise(line):
        return False
    return (count_num_tokens(line) >= 1) and (alpha_count(line) < 3)

def infer_ncols_from_nearby(lines: List[str], start: int, span: int = 20) -> int:
    """
    Infer number of value columns from nearby year/date header lines.
    Returns 1 if not confident.
    """
    seg = [lines[i] for i in range(start, min(len(lines), start + span)) if lines[i]]
    # Count distinct year tokens
    years = []
    for s in seg:
        years += YEAR_RE.findall(s)
    # YEAR_RE returns tuples because of (19|20) group; just count occurrences via a simpler pattern
    years2 = re.findall(r"\b(19|20)\d{2}\b", "\n".join(seg))
    # Dates like "June 30, 2025" lines often appear one per column
    date_lines = sum(1 for s in seg if DATE_COL_RE.search(s))

    # Prefer date lines if present (e.g., 3 lines: June 30, 2025/2024/2023)
    if date_lines >= 2 and date_lines <= 6:
        return date_lines

    # Else fallback: if we see many year tokens, guess ncols by unique years count (cap to 5)
    # (Not perfect but good enough for 2~3 cols)
    uniq_years = set(re.findall(r"\b(19|20)\d{2}\b", "\n".join(seg)))
    # uniq_years is just '19'/'20' because regex group, so use full-year regex:
    uniq_years_full = set(re.findall(r"\b(?:19|20)\d{2}\b", "\n".join(seg)))
    if 2 <= len(uniq_years_full) <= 5:
        return len(uniq_years_full)

    return 1

def try_extract_performance_ratios_block(doc_id, source_path, clean, anchor_i, ncols, dbg, block_id: str):
    # 1) Locate the PERFORMANCE RATIOS anchor line.
    def is_allcaps_header(s: str) -> bool:
        ss = (s or "").strip()
        return ss and ss == ss.upper() and sum(ch.isalpha() for ch in ss) >= 6

    # locate header line inside clean window
    pr_i = None
    for k, line in enumerate(clean):
        if (line or "").strip().upper() == "PERFORMANCE RATIOS":
            pr_i = k
            break
    if pr_i is None:
        return None

    # 2) Collect labels starting after PR until the next ALLCAPS header.
    labels = []
    for k in range(pr_i + 1, len(clean)):
        s = (clean[k] or "").strip()
        if not s:
            continue
        if is_allcaps_header(s):  # ASSET QUALITY RATIOS / CAPITAL RATIOS / SHAREHOLDER DATA...
            break
        if count_num_tokens(s) > 0:  # Stop label capture when a numeric line appears.
            break
        # Drop long narrative lines to avoid treating prose as labels.
        if len(s) > 55:
            continue
        labels.append(s)

    # Lightly merge wrapped lines (lowercase starts are treated as continuations).
    merged = []
    for s in labels:
        if merged and (s[:1].islower() or merged[-1].endswith("average") or merged[-1].endswith("to")):
            merged[-1] = (merged[-1] + " " + s).strip()
        else:
            merged.append(s)
    labels = merged

    # 3) Collect numeric tokens in the window after PR (scan only after pr_i).
    value_tokens = []
    for k in range(pr_i + 1, len(clean)):
        s = (clean[k] or "").strip()
        if s and is_allcaps_header(s):   # ASSET QUALITY RATIOS / CAPITAL RATIOS / SHAREHOLDER DATA ...
            break
        value_tokens.extend(extract_value_tokens(clean[k]))

    # merge cases like ["0.19", "%"] -> ["0.19%"]
    merged = []
    i = 0
    while i < len(value_tokens):
        if i + 1 < len(value_tokens) and value_tokens[i + 1] == "%":
            merged.append(value_tokens[i] + "%")
            i += 2
        else:
            merged.append(value_tokens[i])
            i += 1
    value_tokens = merged

    L = len(labels)
    if L < 3:
        return None
    need = L * ncols
    if len(value_tokens) < need:
        dbg(f"[DBG][PR] too few tokens: L={L} ncols={ncols} need={need} got={len(value_tokens)}")
        return None

    # 4) Auto-search offset so return/margin rows line up with percent columns.
    # layout assumed column-major: col0 all rows, then col1, then col2...
    def score_offset(off: int) -> int:
        # slice in col-major
        s = 0
        for r, lab in enumerate(labels):
            lab_l = lab.lower()
            # Apply strong constraints only to the key three rows.
            if ("return on average assets" in lab_l) or ("return on average equity" in lab_l) or ("net interest margin" in lab_l):
                # Expect three values (2025/2024/2023).
                vals = []
                for c in range(ncols):
                    idx = off + c * L + r
                    if idx >= len(value_tokens):
                        return -10**9
                    vals.append(value_tokens[idx])
                # Require at least two entries with a percent sign.
                pct = sum(1 for v in vals if "%" in v)
                s += pct * 10
                # Penalize obviously non-ratio big numbers (e.g., 106.10 without a percent).
                for v in vals:
                    vv = v.replace("$", "").replace(",", "").replace("(", "-").replace(")", "").replace("%", "").strip()
                    try:
                        x = float(vv)
                        if ("%" not in v) and x > 50:
                            s -= 30
                    except:
                        pass
        return s

    best_off, best_sc = 0, -10**9
    # Limit offset search to the first 400 tokens (tables here are short).
    max_off = min(400, len(value_tokens) - need)
    for off in range(0, max_off + 1):
        sc = score_offset(off)
        if sc > best_sc:
            best_sc, best_off = sc, off

        if best_sc <= 0:
            dbg("[DBG][PR] score too weak, skip PR block")
            return None

    dbg(f"[DBG][PR] best_off={best_off} best_sc={best_sc} L={L} ncols={ncols} tokens={len(value_tokens)}")
    
    def _score_perf_window(win_tokens, labels, ncols):
        """
        win_tokens length == len(labels)*ncols
        Simple heuristic scoring for PERFORMANCE RATIOS block.
        """
        sc = 0
        for r, lab in enumerate(labels):
            vals = win_tokens[r*ncols:(r+1)*ncols]
            low = lab.lower()

            # helpers
            has_pct = sum(('%' in v) for v in vals)
            has_dollar = sum(v.strip().startswith('$') for v in vals)
            has_comma = sum((',' in v) for v in vals)

            # ROA/ROE/NIM should often be percent-like (may appear as "0.24 %" etc.)
            if ("return on average" in low) or ("margin" in low):
                sc += has_pct * 3
                # also reward small-ish decimals even if OCR dropped '%'
                for v in vals:
                    vv = v.replace(' ', '').replace('%', '').replace('(', '').replace(')', '').replace('$', '')
                    try:
                        x = float(vv)
                        if 0 <= x <= 30:
                            sc += 1
                    except:
                        pass

            # EPS often $ or small decimals
            if "earnings per share" in low:
                sc += has_dollar * 3
                for v in vals:
                    vv = v.replace(' ', '').replace('(', '').replace(')', '').replace('$', '')
                    try:
                        x = float(vv)
                        if 0 <= x <= 20:
                            sc += 1
                    except:
                        pass

            # "Average interest-earning assets ..." tends to be big numbers with commas
            if "average interest-earning" in low:
                sc += has_comma * 2

        return sc

    # 5) pack rows using best_off (column-major)
    need = L * ncols
    picked = value_tokens[best_off:best_off + need]

    rows = []
    for r, lab in enumerate(labels):
        vals = []
        for c in range(ncols):
            idx = best_off + c * L + r
            if idx < len(value_tokens):
                vals.append(value_tokens[idx])
        rows.append(lab + " " + " ".join(vals))

    # extra debug: show picked window head
    dbg(f"[DBG][PR_PICK] off={best_off} need={need} head={picked[:min(30,len(picked))]}")
    return TableBlock(
        doc_id=doc_id,
        source_path=source_path,
        block_id=block_id,
        start_line=anchor_i + pr_i,
        end_line=min(len(clean) - 1, anchor_i + pr_i + 120),
        header_lines=["PERFORMANCE RATIOS"],
        rows=rows,
        raw_text="\n".join(rows),
    )

def extract_unstacked_table_blocks(
    clean: List[str],
    doc_id: str,
    source_path: str,
    header_lookback: int = 10,
    min_pairs: int = 8,
    scan_window: int = 260,
    ) -> List[TableBlock]:
    """
    Handle OCR where table is split into:
    [many label lines] ... then [many numeric-only lines]
    We reconstruct rows by position.
    """
    blocks = []
    n = len(clean)
    block_idx = 100000  # avoid colliding with existing b0000 numbering
    def dbg(*a):
        if DEBUG_UNSTACK:
            print(*a)


    # anchor triggers: strong table headers
    strong_anchors = [
        re.compile(r"\bSTATEMENT OF OPERATIONS\b", re.I),
        re.compile(r"\bSTATEMENT OF CONDITION\b", re.I),
        re.compile(r"\bSELECTED FINANCIAL DATA\b", re.I),

        # Allow matching both ratio/ratios and across wrapped lines.
        re.compile(r"\bPERFORMANCE\s+RATIO(?:S|\s+S)?\b", re.I),

        # In the SavingFirst sample, the vertical numbers sit under SHAREHOLDER DATA.
        re.compile(r"\bSHAREHOLDER\s+DATA\b", re.I),
    ]


    def is_anchor_at(idx: int) -> bool:
        if idx < 0 or idx >= n:
            return False
        s = (clean[idx] or "").strip()
        if not s:
            return False

        # Try stitching the next line (e.g., "PERFORMANCE RATIO" + "S").
        s2 = s
        if idx + 1 < n:
            t = (clean[idx + 1] or "").strip()
            if t and t.isupper() and len(t) <= 12:  # Very short suffix from a wrapped line.
                s2 = f"{s} {t}"

        # Allow s2 to be non-uppercase because OCR may mix characters; don't force isupper.
        if len(s2) > 90:
            return False

        return any(p.search(s2) or p.search(s) for p in strong_anchors)


    i = 0
    while i < n:
        if DEBUG_UNSTACK:
            s0 = (clean[i] or "")
            if "PERFORMANCE" in s0.upper() or "SHAREHOLDER" in s0.upper():
                dbg(f"[DBG][ANCH_CAND] i={i} line={repr(s0)}")
        if not clean[i] or not is_anchor_at(i):
            i += 1
            continue


        anchor_i = i
        dbg(f"[DBG][ANCH_HIT] anchor_i={anchor_i} line={repr(clean[anchor_i])}")

        end = min(n, anchor_i + scan_window)
        seg = clean[anchor_i:end]

        # Print the first 40 lines to see whether numeric-only lines were caught.
        for jj, line in enumerate(seg[:40]):
            dbg(f"   [SEG {jj:02d}] numTok={count_num_tokens(line)} alpha={alpha_count(line)} numericOnly={is_numeric_only_line(line)} | {repr(line)}")

        # Look ahead a window to find a split: label list then numeric list
        end = min(n, anchor_i + scan_window)
        seg = clean[anchor_i:end]

        # Find first long run of numeric-only lines (>= min_pairs)
                # --- find first numeric-only line as run_start
        run_start = None
        for j, line in enumerate(seg):
            if is_numeric_only_line(line):
                run_start = j
                break
        if run_start is None:
            dbg("[DBG][RUN_START] not found")
            i += 1
            continue

        # --- collect numeric-only lines with small gap tolerance
        value_tokens = []
        gap = 0
        GAP_LIMIT = 6  # allow a few junk lines like "@)" between year columns
        while j < len(seg) and gap <= GAP_LIMIT:
            line = seg[j]

            # For obvious non-numeric lines: allow a gap but do not break immediately.
            toks = extract_value_tokens(line)

            if toks:
                value_tokens.extend(toks)
                gap = 0
            else:
                gap += 1

            j += 1
            
            run_end = j  # j is the line index within seg (open interval).

        run_len = len(value_tokens)
        dbg(f"[DBG][RUN_LEN] run_len={run_len} first10={value_tokens[:10]}")
        if run_len < min_pairs:
            i += 1
            continue

        # Labels are non-empty lines before run_start that contain letters and no numbers
        label_lines = []
        for line in seg[:run_start]:
            if not line:
                continue
            if has_noise(line):
                continue
            low = line.lower()

            # skip section titles / headings
            if line.isupper():
                continue
            if low.endswith("data") or "selected financial data" in low:
                continue
            if low.startswith("(") and "thousand" in low:
                continue
            if re.match(r"^[A-Z\s&,'().-]{5,}$", line) and alpha_count(line) < 25:
                # short all-caps-ish header
                continue

            # drop labels that contain dates like 12/31/24
            if re.search(r"\b\d{1,2}/\d{1,2}/\d{2,4}\b", line):
                continue

            if count_num_tokens(line) == 0:
                s = line.strip()

                # Filter out section headers first (all-caps titles).
                # e.g. "PERFORMANCE RATIOS", "ASSET QUALITY RATIOS", "CAPITAL RATIOS", "SHAREHOLDER DATA"
                if re.fullmatch(r"[A-Z][A-Z\s]{6,}", s):
                    continue
                a = alpha_count(line)
                # Normal label: has many letters.
                if a >= 6:
                    label_lines.append(line)
                # Truncated word tail: very short but has letters (e.g., "me").
                elif 2 <= a <= 5 and len(line.strip()) <= 12:
                    label_lines.append(line)


        # Merge wrapped label lines (continuations)
        merged = []
        for s in label_lines:
            if merged and s and count_num_tokens(s) == 0:
                prev = merged[-1].rstrip()
                if prev.endswith(("to", "of", "and", "-", ",")) or prev.endswith(("average", "total")):
                    merged[-1] = prev + " " + s.strip()
                    continue
            # Keep the line even if not merged.
            merged.append(s)

        label_lines = merged

        dbg(f"[DBG][LABELS] labels={len(label_lines)} first20={label_lines[:20]}")

        if len(label_lines) < min_pairs:
            i += 1
            continue

        # Infer number of columns (2 for 12/31/24 & 12/31/23, or 3 for June 30 2025/2024/2023)
        ncols = infer_ncols_from_nearby(clean, anchor_i, span=30)
        dbg(f"[DBG][NCOLS] ncols={ncols}")

        if ncols < 1:
            ncols = 1

        # If values look like multi-col: allow label_count * ncols values
        max_rows = min(len(label_lines), len(value_tokens) // ncols if ncols > 1 else len(value_tokens))
        if max_rows < min_pairs:
            # fallback: treat as 1-col
            ncols = 1
            max_rows = min(len(label_lines), len(value_tokens))

        if max_rows < min_pairs:
            i += 1
            continue
        
        # keep only "table-like" labels to avoid long narrative lines shifting alignment
        filtered = []
        for s in label_lines:
            ss = s.strip()
            if len(ss) > 45:   # Threshold can be tuned (40-55).
                continue
            if ss.count(" ") >= 8:
                continue
            filtered.append(s)
        label_lines = filtered


        def is_section_header(s: str) -> bool:
            ss = (s or "").strip()
            if not ss:
                return True
            # All-caps titles: PERFORMANCE RATIOS / SELECTED FINANCIAL DATA / SHAREHOLDER DATA ...
            if ss == ss.upper() and sum(ch.isalpha() for ch in ss) >= 6:
                return True
            return False

        def is_trivial_label(s: str) -> bool:
            ss = (s or "").strip().lower()
            # These are usually not metric rows in this table type.
            bad = {
                "basic", "diluted",
                "at june 30",
                "(s thousands)", "(s thousand)", "(s)", "(000s)", "(in thousands)",
            }
            if ss in bad:
                return True
            # Too short or overly generic single-word lines.
            if len(ss) <= 2:
                return True
            return False
        
        def merge_wrapped_labels(lines: list[str]) -> list[str]:
            out = []
            for s in lines:
                ss = (s or "").strip()
                if not ss:
                    continue
                # If the previous line exists and this one looks like a continuation (starts lowercase, etc.), merge it.
                if out:
                    prev = out[-1]
                    if ss[:1].islower() or prev.endswith("to") or prev.endswith("average"):
                        out[-1] = (prev + " " + ss).strip()
                        continue
                out.append(ss)
            return out

        label_lines = merge_wrapped_labels(label_lines)

        # --- apply filter ---
        filtered = []
        for s in label_lines:
            if is_section_header(s):
                continue
            if is_trivial_label(s):
                continue
            filtered.append(s)
        label_lines = filtered

        # ---- NEW: emit PR-only subtable before packing the big table ----
        pr_block = try_extract_performance_ratios_block(
            doc_id=doc_id,
            source_path=source_path,
            clean=clean,
            anchor_i=anchor_i,
            ncols=ncols,
            dbg=dbg,
            block_id=f"u{block_idx}_pr",
        )
        if pr_block:
            blocks.append(pr_block)
        # ---- END NEW ----

        need = len(label_lines) * ncols
        if len(value_tokens) < need:
            dbg(f"[DBG][ALIGN_FAIL] need={need} got={len(value_tokens)} -> skip")
            i += 1
            continue

        # If there are extra rows, truncate to just enough to avoid pulling numbers from later sections.
        if len(value_tokens) > need:
            value_tokens = value_tokens[:need]

        # Number of values per column (true stride)
        Lcol = len(value_tokens) // ncols
        L = len(label_lines)  # Already ensured the list length is just enough.
        
        if Lcol < L:
            dbg(f"[DBG][STRIDE_FAIL] L={L} Lcol={Lcol} -> trim labels")
            label_lines = label_lines[:Lcol]
            L = len(label_lines)

        rows = []
        for r in range(L):
            if ncols > 1:
                vals = []
                for c in range(ncols):
                    idx = r + c * Lcol
                    if idx < len(value_tokens):
                        vals.append(value_tokens[idx])
                rows.append(f"{label_lines[r]} " + " ".join(vals))
            else:
                rows.append(f"{label_lines[r]} {value_tokens[r]}")


        
        # header lines: lookback around anchor
        hlines = []
        for k in range(max(0, anchor_i - header_lookback), anchor_i + 1):
            if clean[k]:
                hlines.append(clean[k])
        header_lines = [h for h in hlines if looks_like_header(h) or anchor_strength(h) > 0]

        raw_text = "\n".join(seg[:run_end])

        blocks.append(TableBlock(
            doc_id=doc_id,
            source_path=source_path,
            block_id=f"u{block_idx}",
            start_line=anchor_i,
            end_line=anchor_i + run_end,
            header_lines=header_lines,
            rows=rows,
            raw_text=raw_text,
        ))
        block_idx += 1

        # jump forward to avoid duplicate captures
        i = anchor_i + run_start + run_len
    return blocks

def extract_table_blocks_from_lines(
    lines: List[str],
    doc_id: str,
    source_path: str,
    min_rows: int = 4,
    header_lookback: int = 6,
    gap_break: int = 2,
) -> List[TableBlock]:
    """
    Scan lines and form blocks of consecutive table-like rows.
    Additionally, include a small header window above the block.
    """
    clean = [normalize_line(x) for x in lines]
    blocks: List[TableBlock] = []

    i = 0
    n = len(clean)
    block_idx = 0

    while i < n:
        if looks_like_table_row(clean[i]):
            # start block
            start = i
            last_row = i
            gaps = 0
            i += 1
            while i < n:
                if looks_like_table_row(clean[i]):
                    last_row = i
                    gaps = 0
                else:
                    # allow small gaps (blank lines, one-off captions) within a block
                    if clean[i] == "" or anchor_strength(clean[i]) > 0:
                        gaps += 1
                    else:
                        gaps += 1
                    if gaps > gap_break:
                        break
                i += 1

            end = last_row

            # Collect rows within [start, end], skipping empty/noise-only
            row_lines = []
            for j in range(start, end + 1):
                if clean[j] and not has_noise(clean[j]) and looks_like_table_row(clean[j]):
                    # push year-only header lines into header_lines instead of rows
                    if alpha_count(clean[j]) < 4 and len(YEAR_RE.findall(clean[j])) >= 2:
                        continue
                    row_lines.append(clean[j])
            if len(row_lines) >= min_rows:
                # header lookback: take previous non-empty lines, prioritizing strong anchors/headers
                hlines = []
                for j in range(max(0, start - header_lookback), start):
                    if clean[j]:
                        hlines.append(clean[j])

                # prune header lines: keep those that look header-ish or anchor-strong
                header_lines = [h for h in hlines if looks_like_header(h) or anchor_strength(h) > 0]
                # if nothing picked, keep last 2 non-empty as fallback
                if not header_lines:
                    header_lines = hlines[-2:] if len(hlines) >= 2 else hlines

                raw_text = "\n".join(x for x in clean[start:end+1] if x)

                block = TableBlock(
                    doc_id=doc_id,
                    source_path=source_path,
                    block_id=f"b{block_idx:04d}",
                    start_line=start,
                    end_line=end,
                    header_lines=header_lines,
                    rows=row_lines,
                    raw_text=raw_text,
                )
                blocks.append(block)
                block_idx += 1

            # continue from end + 1
            i = end + 1
        else:
            i += 1

    # Post-filter: drop blocks that look like pure chart axis (few anchors and many numeric-only lines)
    filtered = []
    for b in blocks:
        a = sum(anchor_strength(h) for h in b.header_lines) + sum(anchor_strength(r) for r in b.rows[:3])
        # require some anchoring or at least some non-numeric label content
        label_chars = sum(len(re.sub(r"[\d\W_]+", "", r)) for r in b.rows)
        if a >= 2 or label_chars >= 20:
            filtered.append(b)

    # Add reconstructed blocks for "unstacked" OCR tables (labels then numeric list).
    filtered += extract_unstacked_table_blocks(
        clean=clean,
        doc_id=doc_id,
        source_path=source_path,
        header_lookback=10,
        min_pairs=8,
        scan_window=600,
    )

    return filtered


def infer_doc_id_from_path(p: Path) -> str:
    # try to use parent folder name if it looks like bank_id, else stem
    parent = p.parent.name
    if parent and any(ch.isdigit() for ch in parent) and "_" in parent:
        return parent
    return p.stem

def expand_inputs(inputs):
    """
    Expand file/dir inputs into a list of txt file paths.
    - If input is a file: keep it
    - If input is a directory: recursively collect *.txt
    """
    paths = []
    for x in inputs:
        p = Path(x)
        if p.is_file():
            paths.append(p)
        elif p.is_dir():
            paths.extend(p.rglob("*.txt"))
        else:
            print(f"[WARN] input not found: {p}")
    return paths

def read_txt(path: Path) -> List[str]:
    # Try utf-8, fallback to latin-1 (some OCR outputs are weird)
    try:
        return path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except Exception:
        return path.read_text(encoding="latin-1", errors="ignore").splitlines()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True, help="One or more OCR txt files.")
    ap.add_argument("--out", required=True, help="Output JSONL path.")
    ap.add_argument("--min-rows", type=int, default=4)
    ap.add_argument("--header-lookback", type=int, default=6)
    ap.add_argument("--gap-break", type=int, default=2)
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    total_blocks = 0
    with out_path.open("w", encoding="utf-8") as f:
        input_paths = expand_inputs(args.inputs)

        if not input_paths:
            print("[FATAL] No txt files found under given inputs.")
            return

        for p in input_paths:
            lines = read_txt(p)
            doc_id = infer_doc_id_from_path(p)

            blocks = extract_table_blocks_from_lines(
                lines,
                doc_id=doc_id,
                source_path=str(p),
                min_rows=args.min_rows,
                header_lookback=args.header_lookback,
                gap_break=args.gap_break,
            )

            for b in blocks:
                f.write(json.dumps(asdict(b), ensure_ascii=False) + "\n")

            total_blocks += len(blocks)
            print(f"[OK] {p.name}: blocks={len(blocks)}")

    print(f"[DONE] wrote {total_blocks} blocks -> {out_path}")


if __name__ == "__main__":
    main()
