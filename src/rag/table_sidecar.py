"""
Shared table-sidecar helpers reused by table_patch and pipeline prefills.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

_RE_PCT_SPLIT = re.compile(r"(\d)\s+%")
_RE_WS = re.compile(r"\s+")
_RE_WORD_BREAK = re.compile(r"([A-Za-z]{3,})\s{2,}([A-Za-z]{1,3})\b")


def _norm_row(s: str) -> str:
    s = (s or "").replace("\u00a0", " ")
    s = _RE_WS.sub(" ", s).strip()
    s = _RE_PCT_SPLIT.sub(r"\1%", s)
    for _ in range(3):
        s2 = _RE_WORD_BREAK.sub(r"\1\2", s)
        if s2 == s:
            break
        s = s2
    return s


def _block_text_lower(b: dict) -> str:
    return " ".join((b.get("header_lines") or []) + (b.get("rows") or []) + [(b.get("raw_text") or "")]).lower()


def _is_noise_block(b: dict) -> bool:
    t = _block_text_lower(b)
    bad = [
        "board of directors",
        "executive management",
        "officers of",
        "transfer agent",
        "independent auditors",
    ]
    return any(k in t for k in bad)


def _is_fin_table_block(b: dict) -> bool:
    t = _block_text_lower(b)
    if "selected financial data" in t:
        return True
    return ("total assets" in t) and ("net interest" in t)


def _sanitize_block(b: dict) -> dict:
    rows = b.get("rows") or []
    b["rows"] = [_norm_row(r) for r in rows]
    hs = b.get("header_lines") or []
    b["header_lines"] = [_norm_row(h) for h in hs]
    return b


def _pick_best_blocks(blocks: List[dict]) -> List[dict]:
    blocks = [b for b in blocks if not _is_noise_block(b)]
    good = [b for b in blocks if _is_fin_table_block(b)]
    return good if good else blocks


def norm_bank(s: str) -> str:
    s = (s or "").strip().lstrip("_")
    return s.lower()


def extract_bank_from_source_path(p: str) -> str:
    if not p:
        return ""
    parts = re.split(r"[\\/]+", p)
    for i, seg in enumerate(parts):
        if re.fullmatch(r"(?:19|20)\d{2}", seg) and i + 1 < len(parts):
            return parts[i + 1]
    for seg in parts:
        if "_" in seg and re.search(r"\d", seg):
            return seg
    return ""


METRIC_ROW_PATTERNS = {
    "ROA": [
        r"return\s+on\s+(average\s+)?(total\s+)?assets",
        r"\broa\b",
        r"\broaa\b",
    ],
    "ROE": [
        r"return\s+on\s+(average\s+)?(common\s+)?(shareholders'?\s+)?equity",
        r"\broe\b",
        r"\broae\b",
    ],
    "NIM": [
        r"net\s+interest\s+margin",
        r"\bnim\b",
        r"interest\s+margin",
    ],
    "NII": [
        r"net\s+interest\s+income",
        r"net\s+financing\s+revenue",
        r"financing\s+revenue(\s+and\s+other\s+interest\s+income)?",
        r"\bnii\b",
    ],
    "Provision for Credit Losses": [
        r"provision\s+for\s+credit\s+losses",
        r"provision\s+for\s+loan\s+losses",
        r"\bpcl\b",
    ],
}

PATCH_ALLOW = {"NII", "NIM", "ROA", "ROE", "Provision for Credit Losses"}

PCT_RE = re.compile(r"%|\bpercent(age)?\b", re.I)
UNIT_HINT_RE = re.compile(
    r"\b(dollars?\s+in\s+(thousands|millions|billions)|\$\s*in\s+(thousands|millions|billions)|in\s+(thousands|millions|billions))\b",
    re.I
)
NUM_TOKEN_RE = re.compile(
    r"""(?:
        \$?\(?\d{1,3}(?:,\d{3})+(?:\.\d+)?\)?   # 190,591 or (1,234) or $1,234.56
      | \$?\(?\d+(?:\.\d+)?\)?                  # 1234 or (1234) or 12.34
      | \d+(?:\.\d+)?%                          # 3.27%
    )""",
    re.VERBOSE
)


def _join_block_text(b: dict) -> str:
    header = "\n".join(b.get("header_lines") or [])
    rows = "\n".join(b.get("rows") or [])
    raw = b.get("raw_text") or ""
    return (header + "\n" + rows + "\n" + raw).strip()


def _header_years(b: dict) -> List[str]:
    header_lines = b.get("header_lines") or []
    rows = b.get("rows") or []
    header_candidates = list(header_lines) + rows[:15]
    text = "\n".join(header_candidates)
    years = re.findall(r"\b(?:19|20)\d{2}\b", text)
    seen = set()
    out = []
    for y in years:
        if y not in seen:
            out.append(y)
            seen.add(y)
    return out


def _unit_hint(b: dict) -> Optional[str]:
    header_lines = b.get("header_lines") or []
    rows = b.get("rows") or []
    blob = "\n".join(header_lines + rows[:10])

    m = UNIT_HINT_RE.search(blob)
    if m:
        g = m.group(2) or m.group(3) or m.group(4) or ""
        return g.lower() if g else "dollars"
    return None


def _row_matches_metric(row: str, metric_name: str) -> bool:
    pats = METRIC_ROW_PATTERNS.get(metric_name, [])
    if not pats:
        return False
    for p in pats:
        if re.search(p, row, re.I):
            return True
    return False


def _extract_numbers(row: str) -> List[str]:
    return NUM_TOKEN_RE.findall(row)


def _pick_year_from_row(row: str, header_years: List[str], target_year: str) -> Optional[str]:
    nums = _extract_numbers(row)
    if not nums:
        return None

    tokens = []
    for x in nums:
        if isinstance(x, tuple):
            tokens.append(next((t for t in x if t), ""))
        else:
            tokens.append(x)
    tokens = [t for t in tokens if t]

    if header_years:
        year_set = set(header_years)
        no_year = [t for t in tokens if t not in year_set]
        if len(no_year) >= len(header_years):
            tokens = no_year

    if not tokens:
        return None

    if header_years and (2 <= len(header_years) <= 5) and (target_year in header_years):
        ncols = len(header_years)
        if len(tokens) < ncols:
            return None
        if len(tokens) > ncols:
            tokens = tokens[-ncols:]
        idx = header_years.index(target_year)
        if idx < len(tokens):
            return tokens[idx]
        return None

    return None


def infer_unit_from_value_token(tok, block_unit_hint, metric_name=None):
    if "%" in tok or PCT_RE.search(tok):
        return "%"
    if block_unit_hint:
        return block_unit_hint
    if metric_name in {"ROA", "ROE", "NIM"}:
        return "%"


def load_sidecar_index(sidecar_path: Path) -> Dict[str, List[dict]]:
    idx: Dict[str, List[dict]] = defaultdict(list)
    with sidecar_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue

            doc_id = obj.get("doc_id") or ""
            src = obj.get("source_path") or ""
            bank2 = extract_bank_from_source_path(src)

            k1 = norm_bank(doc_id)
            k2 = norm_bank(bank2)

            if k1:
                idx[k1].append(obj)
            if k2 and k2 != k1:
                idx[k2].append(obj)
    return idx


def load_sidecar_index_multi(
    base_sidecar: Optional[Path] = None,
    pdf_sidecar: Optional[Path] = None,
    pdf_sidecar_dir: Optional[Path] = None,
) -> Dict[str, List[dict]]:
    idx: Dict[str, List[dict]] = defaultdict(list)
    if base_sidecar and base_sidecar.exists():
        idx = load_sidecar_index(base_sidecar)

    def _merge_one(p: Path):
        if not p or (not p.exists()):
            return
        idx_pdf = load_sidecar_index(p)
        for k, bs in idx_pdf.items():
            bs2 = [_sanitize_block(b) for b in bs]
            bs2 = _pick_best_blocks(bs2)
            if not bs2:
                continue
            idx[k] = bs2 + idx.get(k, [])

    if pdf_sidecar:
        _merge_one(pdf_sidecar)

    if pdf_sidecar_dir and pdf_sidecar_dir.exists():
        for p in sorted(pdf_sidecar_dir.rglob("*.jsonl")):
            _merge_one(p)

    return idx


def extract_metric_from_bank_tables(
    blocks: List[dict],
    metric_name: str,
    target_year: str,
) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    best = None  # (score, val, unit, block_id)

    for b in blocks:
        rows = b.get("rows") or []
        if not rows:
            continue

        header_years = _header_years(b)
        if target_year not in header_years:
            continue

        bu = _unit_hint(b)
        block_id = b.get("block_id") or ""

        text = _join_block_text(b).lower()
        score_base = 0
        if "selected performance ratios" in text or "performance ratios" in text:
            score_base += 10
        if metric_name.lower() in text:
            score_base += 3

        for r in rows:
            r2 = _norm_row(r)
            if not _row_matches_metric(r2, metric_name):
                continue
            val = _pick_year_from_row(r2, header_years, target_year)
            if not val:
                continue

            looks_money = ("$" in val) or ("," in val) or (len(val.replace(".", "").replace("-", "")) >= 4)
            vv = None
            try:
                vv = float(val.replace(",", "").replace("$", "").replace("(", "-").replace(")", "").strip().replace("%", ""))
            except Exception:
                vv = None
            if metric_name in ("NII", "Provision for Credit Losses"):
                if (bu in (None, "", "NOT FOUND")):
                    if (vv is not None and vv < 100) or (not looks_money):
                        continue
                if (bu in (None, "", "NOT FOUND")):
                    if (vv is not None and vv < 100) or (not looks_money):
                        continue
            unit = infer_unit_from_value_token(val, bu, metric_name=metric_name)

            score = score_base + 20
            if unit == "%":
                score += 2

            cand = (score, val, unit, block_id)
            if best is None or cand[0] > best[0]:
                best = cand

    if not best:
        return None, None, None
    _, v, u, bid = best
    evidence = f"table:{bid}" if bid else "NOT FOUND"
    return v, (u or "NOT FOUND"), evidence
