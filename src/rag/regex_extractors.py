import re
from typing import Optional, Tuple, Dict, Any
import os
import logging
LOGGER = logging.getLogger("extract.regex")


# allow matching "3.25" as percent when context implies percent
ENABLE_BARE_PERCENT = os.getenv("ENABLE_BARE_PERCENT", "1") == "1"


def _guess_unit(text: str) -> str:
    """
    Infer a unit scale token from nearby text.
    Returns one of: thousand/million/billion/dollars/NOT FOUND based on common table header conventions.
    """
    t = (text or "").lower()

    # --- NEW: $ heuristic (many tables omit wording but show $) ---
    # Heuristic: if '$' appears without an explicit scale (thousand/million), default to 'dollars' and let neighbor/header scans refine it later.
    if "$" in (text or ""):
        # Do not default to 'thousand' to avoid false positives; mark as 'dollars' first.
        # If an explicit scale (e.g., 'in thousands/millions') appears, it will override this below.
        unit_dollar = "dollars"
    else:
        unit_dollar = None

    # Common patterns: "(dollars in thousands)" / "($ in thousands)" / "(in thousands)"
    if re.search(r"\(\s*(?:\$|dollars)?\s*in\s+thousands\s*\)", t) or "in thousands" in t:
        return "thousand"
    if re.search(r"\(\s*(?:\$|dollars)?\s*in\s+millions\s*\)", t) or "in millions" in t:
        return "million"
    if re.search(r"\(\s*(?:\$|dollars)?\s*in\s+billions\s*\)", t) or "in billions" in t:
        return "billion"

    # amounts in thousands / amounts (in thousands)
    if "amounts in thousands" in t or "amounts (in thousands)" in t:
        return "thousand"
    if "amounts in millions" in t or "amounts (in millions)" in t:
        return "million"

    # Broader header patterns for unit inference
    if re.search(r"\b(thousands)\b", t) and ("dollar" in t or "$" in t or "usd" in t):
        return "thousand"
    if re.search(r"\b(millions)\b", t) and ("dollar" in t or "$" in t or "usd" in t):
        return "million"
    if re.search(r"\b(billions)\b", t) and ("dollar" in t or "$" in t or "usd" in t):
        return "billion"

    return unit_dollar or "NOT FOUND"

def try_regex_extract_pcl_from_context(context: str, year: int = 2024):
    """
    Extract Provision for Credit Losses (PCL) using a tiered strategy:
    1) Parse a vertical year table from the "Summary of Operations" section when available.
    2) Fallback: scan blocks for PCL-like labels and pick the value aligned to the target year.
    Notes:
    - Unit is inferred from local/neighbor windows (headers may appear in adjacent blocks).
    - Parentheses are treated as negative only for benefit/recovery wording to avoid sign artifacts.
    Returns: (value, unit, source_chunk_id) or (None, None, None).
    """
    if not context:
        return None, None, None


    def _pcl_header_from_block(b: str) -> str:
        # Prefer returning the full citation header line; fall back to a numeric chunk id if needed.
        for ln in (b or "").splitlines():
            t = ln.strip()
            if t.startswith("[k=") and "|chunk=" in t:
                return t
        mm = re.search(r"\bchunk=(\d+)\b", b or "")
        return mm.group(1) if mm else "NOT FOUND"

    def _pcl_detect_unit_and_scale(t: str):
        low = (t or "").lower()
        # common table headers
        if re.search(r"(dollars?\s+in\s+thousands|in\s+thousands)", low):
            return ("thousand", 1.0)
        if re.search(r"(dollars?\s+in\s+millions|in\s+millions)", low):
            return ("million", 1.0)
        return ("NOT FOUND", 1.0)

    def _pcl_parse_money(s: str) -> float:
        ss = (s or "").strip()
        neg = False
        if ss.startswith("(") and ss.endswith(")"):
            neg = True
            ss = ss[1:-1].strip()
        ss = ss.replace("$", "").replace(",", "").replace(" ", "")
        if ss.startswith("-"):
            neg = True
            ss = ss[1:]
        v = float(ss)
        return -v if neg else v

    def _pick_by_year(near_text: str, nums: list[str], year: int = 2024) -> str:
        # try infer year column order from nearby header like "2024 2023 2022 ..."
        yrs = re.findall(r"\b(20\d{2})\b", near_text)
        # keep first occurrence order + dedup
        seen, order = set(), []
        for y in yrs:
            if y not in seen:
                seen.add(y); order.append(int(y))
        if year in order:
            idx = order.index(year)
            if 0 <= idx < len(nums):
                return nums[idx]
        return nums[0]

    def _pcl_fmt(v: float) -> str:
        if abs(v - round(v)) < 1e-9:
            return f"{int(round(v)):,}"
        s = f"{v:.3f}".rstrip("0").rstrip(".")
        return s

    # High-precision path: parse a vertical year table from the "Selected Performance Ratios" section when available.
    m = _parse_vertical_year_table(
        context,
        section_title="Summary of Operations",
        year=int(year),
        stop_titles=["Year-end Balances", "Average Balances", "Per Share Data", "Selected Performance Ratios"],
    )
    if not m:
        # ===== PCL fallback: scan any block, accept "credit losses" OR "loan losses" =====
        blocks = re.split(r"\n---\n", context)

        pat = re.compile(
            r"provision\s*(?:\(|\[)?(?:recovery|benefit)?(?:\)|\])?\s*for\s+"
            r"(?:credit\s+loss(?:es)?|loan\s+loss(?:es)?)"
            r"(?:\s+expense)?",
            re.I
        )

        # money-like number: supports commas, decimals, optional $ and parentheses for negatives
        num_pat = re.compile(
            r"""
            (?<![A-Za-z0-9])                 # left boundary
            \(?                              # optional '('
            -?                               # optional '-'
            \$?                              # optional '$'
            (?:\d{1,3}(?:,\d{3})+|\d+)        # 1,234 or 1234
            (?:\.\d+)?                       # optional decimals
            \)?                              # optional ')'
            (?![A-Za-z0-9])                  # right boundary
            """,
            re.VERBOSE,
        )
        best = None  # (score, val, unit, cid)

        for blk in blocks:
            # Use the citation from the current block to avoid mis-attributing evidence.
            cid = _pcl_header_from_block(blk)

            lines = blk.splitlines()
            for i, ln in enumerate(lines):
                picked_raw = None
                if not pat.search(ln):
                    continue

                nums = num_pat.findall(ln)
                if not nums:
                    continue

                # 1) remove obvious years like 2024/2023/2022
                nums2 = [x for x in nums if not (len(x) == 4 and (x.startswith("19") or x.startswith("20")))]
                if not nums2:
                    continue

                unit_window = "\n".join(lines[max(0, i-40): i+1])
                money_hint = (unit_window or "").lower()

                def _is_amount_token(s: str, money_hint: str) -> bool:
                    d = re.sub(r"\D", "", s)
                    if not d:
                        return False

                    # Allow small decimal amounts only when clear monetary signals exist (e.g., $ / thousand / million) to reduce false positives.
                    if "." in s:
                        if ("million" in money_hint) or ("thousand" in money_hint) or ("in millions" in money_hint) or ("in thousands" in money_hint) or ("$" in money_hint):
                            return True
                        return False

                    if "," in s:
                        return True
                    return len(d) >= 3

                nums2 = [n for n in nums2 if _is_amount_token(n, money_hint)]
                if not nums2:
                    continue
                # Local window used to infer the year-column order near the matched line.
                near_for_pick = "\n".join(lines[max(0, i-12): i+1]).lower()

                raw = _pick_by_year(near_for_pick, nums2, year=int(year))
                picked_raw = raw
                LOGGER.debug("[DBG][PCL_PICK] cid=%s year=%s nums2=%s picked_raw=%s", cid, year, nums2, picked_raw)
                if not raw:
                    continue

                # hard reject day-of-month like "December 31" -> 31
                if raw in {"31", "30", "29", "28"}:
                    if re.search(r"december\s+\d{1,2}", near_for_pick, re.I):
                        continue

                # Parentheses often indicate negative values; for PCL we only treat it as negative when the text explicitly signals benefit/recovery.
                # This block only adjusts sign; it does not apply unit scaling.
                is_paren = "(" in ln and ")" in ln and raw in ln
                val = raw

                if is_paren:
                    low_ln = ln.lower()
                    # If parentheses are merely a presentation style (expense shown in brackets), keep the magnitude positive.
                    if re.search(r"\bbenefit\b|\brecovery\b|negative\s+provision", low_ln):
                        if not val.startswith("-"):
                            val = "-" + val
                    else:
                        # In many reports, parentheses are formatting for expenses; keep magnitude positive unless wording implies recovery/benefit.
                        val = raw

                # Infer unit from a wider local window near the matched row (headers may appear above).
                unit_window = "\n".join(lines[max(0, i-80): i+2])
                money_hint = unit_window.lower()
                if not (("$" in ln) or ("million" in money_hint) or ("thousand" in money_hint) or ("in millions" in money_hint) or ("in thousands" in money_hint)):
                    continue
                unit, _ = _pcl_detect_unit_and_scale(unit_window)
                # hard override if explicit hint exists
                if "million" in money_hint or "in millions" in money_hint:
                    unit = "million"
                elif "thousand" in money_hint or "in thousands" in money_hint:
                    unit = "thousand"

                # Scoring: prefer rows whose nearby window contains the target fiscal year.
                s = 0
                near = "\n".join(lines[max(0, i-8): i+1]).lower()
                if str(year) in near:
                    s += 5
                if "year ended" in near or "december 31" in near:
                    s += 2
                # Slightly prefer "credit losses" wording (closer to the target metric label) over "loan losses".
                if "credit losses" in ln.lower():
                    s += 1

                if "statements of income" in near or "summary of operations" in near:
                    s += 60
                if "allowance for credit losses" in near or "allowance for loan losses" in near or "rollforward" in near or "activity" in near:
                    s -= 120

                near = "\n".join(lines[max(0, i-12): i+1]).lower()
                # HARD reject quarterly tables / quarter-ended blocks
                if ("quarter" in near) or ("for the quarter ended" in near) or ("summary of quarterly statements of income" in near):
                    continue 

                if "consolidated statements of income" in near or "statements of income" in near:
                    s += 200

                if re.search(r"year ended\s+december\s+31,\s+2024", near):
                    s += 120

                if "segment income statement data" in near or "consumer commercial wealth elimination" in near:
                    s -= 250

                # segment income statement totals table (consumer/commercial/wealth/elimination/totals)
                if ("consumer" in near and "commercial" in near and "wealth" in near and "elimination" in near and "totals" in near):
                    s -= 250

                # Rollforward / allowance table (not the consolidated PCL line)
                if "allowance for credit losses" in near or "liability for unfunded" in near:
                    s -= 250

                # extra boost: classic 3-year row for PCL
                if ln.strip().lower().startswith("provision for credit losses") and len(nums2) == 3:
                    s += 120

                cand = (s, val, unit, cid)
                if (best is None) or (cand[0] > best[0]):
                    best = cand

                LOGGER.debug("[DBG][PCL_CAND] cid=%s score=%s line=%s", cid, s, lines[i])
                LOGGER.debug("[DBG][PCL_CAND] near=\n%s", near)
                LOGGER.debug("[DBG][PCL_CAND] nums=%s picked_raw=%s", nums2, raw)

        # force absolute magnitude + normalize formatting
        if best:
            _, val, unit, cid = best
            # hard reject day-of-month artifacts like "December 31" when unit/cid missing
            if isinstance(val, str) and val.strip() in {"28","29","30","31"}:
                if unit in (None, "", "NOT FOUND") and cid in (None, "", "NOT FOUND"):
                    return None  # Reject day-of-month artifacts (e.g., "December 31" -> "31").
            try:
                val_num = abs(_pcl_parse_money(str(val)))
                val = _pcl_fmt(val_num)
            except Exception:
                # If parsing fails, strip a leading '-' to avoid propagating a sign artifact.
                if isinstance(val, str) and val.startswith("-"):
                    val = val[1:]

            if unit == "NOT FOUND":
                if re.search(r"in\s+millions", context, re.I):
                    unit = "million"
                elif re.search(r"in\s+thousands", context, re.I):
                    unit = "thousand"
            LOGGER.debug("[DBG][PCL_FALLBACK] year=%s val=%s unit=%s cid=%s", year, val, unit, cid)
            return val, unit, cid

        return None


    v = m.get("Provision for Credit Losses")
    if v is None:
        return None

    # unit is "Dollars in million" in this table
    unit = "million"
    # best-effort: find the chunk number containing the phrase
    cid = "NOT FOUND"
    for blk in re.split(r"\n---\n", context):
        if "provision for credit losses" in (blk or "").lower():
            cid = _pcl_header_from_block(blk)
            break

    # Table prefill returns a raw value string; parse it directly and keep the table-implied unit (no scaling here).
    # Try to infer unit near "Summary of Operations" header; if not found, keep the existing default.
    try:
        val_num = _pcl_parse_money(str(v))
    except Exception:
        return None

    # Try to infer unit near the "Summary of Operations" header; fallback keeps the table default (million).
    pos = (context or "").lower().find("summary of operations")
    unit_window = context[max(0, pos - 300): pos + 1200] if pos != -1 else context[:2000]
    unit_guess, _ = _pcl_detect_unit_and_scale(unit_window)
    if unit_guess != "NOT FOUND":
        unit = unit_guess  # Override unit when explicitly detected; do not rescale values here (table already implies the scale).

    val = _pcl_fmt(abs(val_num))
    return val, unit, cid

def try_regex_extract_nii_from_context(context: str, year: int | None = None, neighbor_k: int = 3, head_scan_chars: int = 2200):
    """
    Improvements:
    1) Collect multiple candidates and select the best-scoring match instead of returning the first hit
    2) Stronger unit inference: current chunk + neighbor chunks + context header + local window near the matched span
    3) Filter obvious false positives: pure years, unrealistically small values without nearby scale indicators
    """
    if year is None:
        year = 2024  # backward default (or set to current pipeline year if you prefer)
    if not context:
        return None

    # Context is built as concatenated blocks separated by "\n\n---\n\n"; each block starts with a citation header line.
    blocks = re.split(r"\n\n---\n\n", context)

    # Capture full cite keys like "[k=...|stem=...|chunk=...]" so downstream can keep a traceable source_chunk_id.
    header_pat = re.compile(r"^\[(?P<cid>k=.*?\|stem=.*?\|chunk=\d+)\]", flags=re.M)

    # numeric token (allow $, commas, parentheses negatives)
    num_pat = r"(?P<val>\(?\$?\s*\d[\d,]*(?:\.\d+)?\)?)"
    num_tok = r"\(?\$?\s*\d[\d,]*(?:\.\d+)?\)?"

    # NII label patterns; keep this forgiving because some banks use "net financing revenue"
    nii_label_pat = r"(net\s+interest\s+income|\bNII\b|net\s+financing\s+revenue|financing\s+revenue(?:\s+and\s+other\s+interest\s+income)?)"

    # label + nearby number (often split across lines / OCR)
    val_pat = re.compile(
        rf"(?is){nii_label_pat}.{{0,180}}?{num_pat}"
    )

    # Fallback for "highlights" style tables where labels and numbers are far apart: capture 3 adjacent year columns.
    # Enabled only for highlight/ratio sections to avoid matching unrelated large numeric blocks.
    fallback_pat3 = re.compile(
        rf"(?is){nii_label_pat}.{{0,6000}}?(?P<v1>{num_tok})\s+(?P<v2>{num_tok})\s+(?P<v3>{num_tok})"
    )


    def _num(s: str):
        # Normalize to float (strip currency symbols, commas, and parentheses)
        if not s:
            return None
        t = s.strip()
        neg = False
        if t.startswith("(") and t.endswith(")"):
            neg = True
            t = t[1:-1]
        t = t.replace("$", "").replace(",", "").strip()
        try:
            x = float(t)
            return -x if neg else x
        except:
            return None

    def _looks_like_highlights(text: str) -> bool:
        t = (text or "").lower()
        return (
            "selected financial data" in t
            or "performance ratios" in t
            or "financial highlights" in t
            or "selected income statement data" in t
        )

    def _pick_by_year(v1, v2, v3, scan_blk: str):
            # Infer year-column order from nearby text (e.g., "2025 2024 2023" or date headers like "June 30, 2025").
            yrs = re.findall(r"\b20\d{2}\b", (scan_blk or "")[:2000])
            # Deduplicate while preserving first-seen order.
            seen = set()
            yrs2 = []
            for y in yrs:
                if y not in seen:
                    seen.add(y)
                    yrs2.append(y)

            if str(year) in yrs2 and len(yrs2) >= 3:
                idx = yrs2.index(str(year))
                return [v1, v2, v3][idx]

            # Common layout: 2025 / 2024 / 2023 -> pick the middle column for 2024.
            if year == 2024:
                return v2
            return v1  # Default fallback: take the first column.

    def _scale_from_near(text: str) -> str:
        """Infer scale keywords from a local text window near the matched span (more precise than _guess_unit)."""
        t = (text or "").lower()
        # Common implicit scale patterns (e.g., '$ in thousands', '($000s)').
        if re.search(r"\$\s*in\s*thousands|\$\s*\(?\s*000s?\)?|\bin\s*\$0{3}s\b|\(\s*\$0{3}s?\s*\)", t):
            return "thousand"
        if re.search(r"\$\s*in\s*millions|\bin\s*\$0{6}s\b|\(\s*\$0{6}s?\s*\)", t):
            return "million"
        if "thousand" in t:
            return "thousand"
        if "million" in t:
            return "million"
        if "billion" in t:
            return "billion"
        return "NOT FOUND"

    def _year_score(local_text: str, target_year: int) -> int:
        low = (local_text or "").lower()
        y = str(int(target_year))
        score = 0

        if re.search(rf"year\s+ended\s+december\s+31,\s*{y}", low):
            score += 140
        if re.search(rf"\bfor\s+{y}\b", low):
            score += 80

        years = re.findall(r"\b20\d{2}\b", low)
        if y in years:
            score += 30
        if years and (y not in years):
            score -= 60

        prev_y = str(int(target_year) - 1)
        if re.search(rf"year\s+ended\s+december\s+31,\s*{prev_y}", low):
            score -= 140
        return score

    candidates = []

    head = context[:head_scan_chars]

    for i, blk in enumerate(blocks):
        m_header = header_pat.search(blk)
        if not m_header:
            continue
        chunk_id = m_header.group("cid")

        next_blk = blocks[i + 1] if (i + 1 < len(blocks)) else ""
        next_chunk_id = chunk_id
        if next_blk:
            m_next = header_pat.search(next_blk)
            if m_next:
                next_chunk_id = m_next.group("cid")

        # Lookahead: append next 1 block to capture split keyword/value across chunks
        scan_blk = blk
        if next_blk:
            scan_blk = blk + "\n" + next_blk
        boundary = len(blk) + 1
        before_n = len(candidates)

        for m in val_pat.finditer(scan_blk):
            val = (m.group(m.lastindex) or "").strip()
            # print(f"[DBG][NII] match groups last={m.lastindex} val={val!r}", flush=True)
            if not val:
                continue
            x = _num(val)

            # Filter year-like false positives (e.g., 2024/2025)
            if 1900 <= x <= 2100 and abs(x - int(x)) < 1e-9:
                continue

            # Anchor the source cid by match position in lookahead text.
            in_next = m.start() >= boundary
            src_chunk_id = next_chunk_id if in_next else chunk_id
            src_blk = next_blk if (in_next and next_blk) else blk

            # 1) Unit signals in the source chunk (fallback to scan block)
            unit = _guess_unit(src_blk) or _guess_unit(scan_blk)

            # 2) Local context window around the matched value to infer scale indicators
            # (e.g., thousand / million / billion)
            span_l = max(0, m.start() - 180)
            span_r = min(len(scan_blk), m.end() + 220)
            near = scan_blk[span_l:span_r]
            unit_near = _scale_from_near(near)
            if unit_near != "NOT FOUND":
                unit = unit_near

            # 3) Neighbor chunks (table headers often appear earlier)
            if unit in ("NOT FOUND", "dollars"):
                neigh = "\n\n".join(blocks[max(0, i-neighbor_k): min(len(blocks), i+neighbor_k+1)])
                unit2 = _guess_unit(neigh)
                if unit2 != "NOT FOUND":
                    unit = unit2
                else:
                    unit2b = _scale_from_near(neigh)
                    if unit2b != "NOT FOUND":
                        unit = unit2b

            # 4) Scan context header (table titles frequently appear near the beginning)
            if unit in ("NOT FOUND", "dollars"):
                unit3 = _guess_unit(head)
                if unit3 != "NOT FOUND":
                    unit = unit3
                else:
                    unit3b = _scale_from_near(head)
                    if unit3b != "NOT FOUND":
                        unit = unit3b

            # Filter low-confidence matches:
            # values without scale indicators and below a reasonable monetary threshold
            if unit == "dollars":
                has_comma = ("," in val)
                if (not has_comma) and abs(x) < 1000:
                    # Keep as a low-confidence candidate, but do not return immediately.
                    penalty_small = 1
                else:
                    penalty_small = 0
            else:
                penalty_small = 0

            # Scoring: prefer explicit scale over plain dollars; prefer large/comma-formatted values; penalize small likely false positives.
            score = 0
            if unit in ("thousand", "million", "billion"):
                score += 100
            elif unit == "dollars":
                score += 30
            if "," in val:
                score += 20
            if abs(x) >= 1000:
                score += 10
            score -= 50 * penalty_small
            lab = m.group(0).lower()
            if "net interest income" in lab:
                score += 80
            elif "interest income" in lab and "net" not in lab:
                score -= 40

            yscore = _year_score(near, int(year))
            score += yscore
            if yscore <= -140 and (str(int(year)) not in near):
                continue

            candidates.append((score, val, unit, src_chunk_id))

        # Highlights-table fallback: only run when this block produced no direct label+value candidates.
        if (len(candidates) == before_n) and _looks_like_highlights(scan_blk):
            m3 = fallback_pat3.search(scan_blk)
            if m3:
                v1, v2, v3 = m3.group("v1"), m3.group("v2"), m3.group("v3")
                picked = _pick_by_year(v1, v2, v3, scan_blk)
                x = _num(picked)
                if x is not None:
                    unit = _guess_unit(scan_blk)
                    candidates.append((300, picked.strip(), unit, chunk_id))
                    print(f"[DBG][NII_FALLBACK] cid={chunk_id} v1={v1} v2={v2} v3={v3} picked={picked}", flush=True)

    if not candidates:
        return None

    candidates.sort(key=lambda z: z[0], reverse=True)
    best = candidates[0]
    return (best[1], best[2], best[3])


def try_extract_nim_from_yield_row(block_text: str) -> Optional[str]:
    """
    Extract NIM value from a single table-like row such as:
    "Net yield on interest-earning assets (j) 3.27 % 3.33 % 3.85 %"
    Returns the first percentage (assumed to be 2024) as a string like "3.27", or None.
    """

    if not block_text:
        return None

    # Normalize: collapse multiple spaces; keep newlines for line-by-line scanning
    lines = (block_text or "").splitlines()

    # Row label patterns (aliases for NIM in many annual reports)
    row_label_pat = re.compile(
        r"(?i)\bnet\s+yield\s+on\s+interest[-\s]*[a-z]{0,2}arn\w*\s+assets\b"
    )

    # Percent numbers: accept "3.27%", "3.27 %", "3.27"
    pct_pat = re.compile(r"(?P<val>\d{1,2}\.\d{1,3})\s*%?")

    for ln in lines:
        if not row_label_pat.search(ln):
            continue

        # Collect percentages on the same line
        vals = [m.group("val") for m in pct_pat.finditer(ln)]
        # We expect something like [3.27, 3.33, 3.85]
        if len(vals) >= 3:
            return vals[0]  # assume 2024 is first column

        # Some OCR formats split the numbers to the next line(s)
        # If this happens, caller can pass "lookahead" joined text; keep this function simple.

    return None


def try_regex_extract_for_metric(
    *,
    metric_name: str,
    bank: str,
    year: int,
    context_text: str,
    context_blocks: Optional[list] = None,
    logger=None,
) -> Optional[dict]:
    """
    Return normalized item dict or None if regex path did not produce a value.
    """
    m = metric_name
    if m == "NII":
        got = try_regex_extract_nii_from_context(context_text)
        if got:
            val, unit, cid = got
            return {
                "metric_name": "NII",
                "value": val,
                "unit": unit,
                "fiscal_year": int(year),
                "source_chunk_id": cid,
            }
        return None

    if m == "NIM":
        got_nim = try_regex_extract_nim_from_context(context_text)
        if got_nim:
            val, unit, cid = got_nim
            if logger:
                LOGGER.debug("[DEBUG][NIM_REGEX] bank=%s metric=%s val=%s unit=%s cid=%s", bank, m, val, unit, cid)
            return {
                "metric_name": "NIM",
                "value": val,
                "unit": unit,
                "fiscal_year": int(year),
                "source_chunk_id": cid,
            }
        return None

    if m in ("ROA", "ROE"):
        LOGGER.debug("[DBG][ROA_ROE_CALL] bank=%s metric=%s len=%d head=%s", bank, m, len(context_text or ""), (context_text or "")[:200].replace("\n"," "))
        got_rr = try_regex_extract_roa_roe_from_context(context_text, year=int(year)) or {}
        if got_rr.get(m):
            val, unit, cid = got_rr[m]
            return {
                "metric_name": m,
                "value": val,
                "unit": unit,
                "fiscal_year": int(year),
                "source_chunk_id": cid,
            }
        return None

    if m == "Provision for Credit Losses":
        got_pcl = try_regex_extract_pcl_from_context(context_text, year=int(year))
        if got_pcl:
            val, unit, cid = got_pcl
            return {
                "metric_name": m,
                "value": val,
                "unit": unit,
                "fiscal_year": int(year),
                "source_chunk_id": cid,
            }
        return None

    return None


def regex_prefill_from_contexts(bank_id: str, year: int, contexts_by_metric: Dict[str, str], dprint=None):
    if dprint is None:
        dprint = lambda *a, **k: None
    prefill = {}
    stats = {}
    for metric, ctx in (contexts_by_metric or {}).items():
        got = try_regex_extract_for_metric(
            metric_name=metric,
            bank=bank_id,
            year=year,
            context_text=ctx or "",
            context_blocks=None,
            logger=None,
        )
        if got:
            prefill[metric] = got
            stats[metric] = "hit"
            dprint(f"[REGEX_PREFILL] bank={bank_id} metric={metric} val={got.get('value')} unit={got.get('unit')} cid={got.get('source_chunk_id')}")
        else:
            stats[metric] = "miss"
    return prefill, stats

def try_regex_extract_nim_from_context(context: str, head_scan_chars: int = 2400):
    """
    Extract NIM from context using regex. Returns (value, unit, source_chunk_id) or None.

    Handles:
    - prose: "net interest margin ... 3.63%"
    - table-ish: "Net interest margin" in one line, value in next 1-3 lines
    - cases where '%' is missing but 'percent' is present nearby (rare OCR/table formats)

    Notes:
    - NIM is constrained to explicit percent forms by default to avoid money/ratio false positives.
    - Returns (value, unit, source_chunk_id_headerline) for downstream traceability.
    ENABLE_BARE_PERCENT = False  # Require explicit '%' for NIM to avoid numeric false positives.
    """
    if not context:
        return None

    header_pat = re.compile(r"^\[k=.*?\|stem=.*?\|chunk=(\d+)\]", flags=re.M)

    
    def _header_line(s: str) -> str:
        for ln in (s or "").splitlines():
            ln = ln.strip()
            if ln.startswith("[k=") and "|chunk=" in ln:
                return ln
        m = header_pat.search(s or "")
        return m.group(0).strip() if m else "NOT FOUND"

    # NIM sanity range: typical bank NIM is low single digits; cap at 15 to reject obvious artifacts.
    def _valid_nim(v: str) -> bool:
        try:
            x = float(v)
            return (x > 0) and (x <= 15)
        except Exception:
            return False
    blocks = re.split(r"\n\n---\n\n", context)

    # --- table header driven extraction: NIM table spans multiple blocks ---
    hdr_idx = None
    for i, blk in enumerate(blocks):
        if "net interest margin table" in (blk or "").lower():
            hdr_idx = i
            break

    if hdr_idx is not None:
        # Limit scan window to bound runtime and avoid drifting into unrelated tables.
        # scan following blocks for the "net yield ..." row
        row_pat = re.compile(r"net\s+yield\s+on\s+interest", re.I)
        pct_pat_row = re.compile(r"(?P<val>\d{1,2}(?:\.\d{1,3})?)\s*%")
        for j in range(hdr_idx, min(hdr_idx + 12, len(blocks))):
            b = blocks[j] or ""
            if not row_pat.search(b):
                continue

            vals = [m.group("val") for m in pct_pat_row.finditer(b)]
            # expect 3 values for 2024/2023/2022
            if len(vals) >= 3:
                cid = _header_line(b)  # Attribute evidence to the block containing the matched row.
                v = vals[0]            # Convention: first column corresponds to the target fiscal year in this table section.
                if _valid_nim(v):
                    return (v, "%", cid)

    # percent like 3.63% or 3.63 %
    pct_pat_val = re.compile(r'(?<![\d.])(?P<val>\d{1,2}(?:\.\d{1,4})?)(?=\s*%)(?!\d)')
    pct_pat_any = re.compile(r"(?P<val>\d{1,2}(?:\.\d{1,3})?)\s*%")

    # bare number like 3.63 (used only if 'percent' nearby)
    bare_pat = re.compile(r"(?P<val>\d{1,2}(?:\.\d{1,4})?)\b")

    nim_pat = re.compile(r"(net\s+interest\s+margin|interest\s+margin|\bNIM\b)", flags=re.I)

    header_pat = re.compile(r"^\[k=.*?\|stem=.*?\|chunk=(\d+)\]", flags=re.M)

    candidates = []

    # Keywords for NIM (net interest margin) evidence gating
    keywords = [
        "net interest margin",
        "net yield",
        "net yield on",
        "interest margin",
        "earning assets",
        "nim",
        "tax-equivalent",
        "tax equivalent",
        "interest-earning assets",
    ]

    for bi, blk in enumerate(blocks):
        cid_blk = _header_line(blk)
        cid_next = _header_line(blocks[bi + 1]) if bi + 1 < len(blocks) else "NOT FOUND"
        # Lookahead: include the next block to capture cases where label/value split across chunk boundaries.
        scan_blk = blk
        if bi + 1 < len(blocks):
            scan_blk = blk + "\n" + blocks[bi + 1]
        # Fast path: detect "net yield ..." row and return immediately with correct citation attribution.

        # 1) try current block only
        v = try_extract_nim_from_yield_row(blk)
        if v is not None and _valid_nim(v):
            cid_blk = _header_line(blk)
            print(f"[DEBUG][NIM_YIELD_ROW] hit(cur) block={bi} val={v} cid={cid_blk}", flush=True)
            return (v, "%", cid_blk)

        # 2) try next block only (lookahead), attribute cid to next block
        if bi + 1 < len(blocks):
            v = try_extract_nim_from_yield_row(blocks[bi + 1])
            if v is not None and _valid_nim(v):
                cid_next = _header_line(blocks[bi + 1])
                print(f"[DEBUG][NIM_YIELD_ROW] hit(next) block={bi+1} val={v} cid={cid_next}", flush=True)
                return (v, "%", cid_next)

        low = scan_blk.lower()
        # allow keyword-only blocks; % may appear in following lines
        if not any(kw in low for kw in keywords):
            continue
        print(f"[DEBUG][NIM_REGEX] keyword_block_idx={bi}", flush=True)

        lines = scan_blk.splitlines()
        focus_lines = []
        for i, ln in enumerate(lines):
            low = ln.lower()
            if ("net interest margin" in low) or ("net yield on" in low) or ("net interest spread" in low) or re.search(r"\bnim\b", low):
                focus_lines.append(ln)
                if i + 1 < len(lines): focus_lines.append(lines[i + 1])
                if i + 2 < len(lines): focus_lines.append(lines[i + 2])
        focus_text = "\n".join(focus_lines) if focus_lines else ""

        # If keyword exists, search for percent values around it
        for m in pct_pat_val.finditer(focus_text):
            val = m.group("val")
            if not _valid_nim(val):
                continue
            # reject year-like
            if re.fullmatch(r"(19|20)\d{2}", val):
                continue
            # reject absurd
            if float(val) >= 100:
                continue

            # decide which block header to attribute to
            cid = cid_blk

            print(
                    f"[DEBUG][NIM_REGEX] pct_match block={bi} val={val} cid={cid}",
                    flush=True
                )

            bonus = 10 if ("2024" in scan_blk) else 0
            # Guard: reject percent tokens that are part of money amounts (e.g., "$12.2") to reduce false positives.
            start = m.start()
            end = m.end()

            # Reject if the matched token is immediately preceded by '$' (money amount mis-match).
            if start > 0 and focus_text[start - 1] == "$":
                continue

            # Reject if '$' appears in a tight neighborhood around the token.
            near = focus_text[max(0, start - 4): min(len(focus_text), end + 4)]
            if "$" in near:
                continue
            # ----------------------------------------------------------
            candidates.append((100 + bonus, val, "%", cid))

            # Fallback: if percent values exist near NIM keywords, score candidates and select the best.
            cid = cid_blk

        # Label/value split: find NIM line then scan the next few lines for a percent value.
        for i, ln in enumerate(lines):
            if not nim_pat.search(ln):
                continue

            # SAME-LINE label + % (e.g., "FTE NIM to 3.00%")
            if "%" in ln:
                m_label = re.search(r"(net\s+interest\s+margin|interest\s+margin|nim|net\s+interest\s+spread|net\s+yield)", ln, flags=re.I)
                if m_label is None:
                    # no label -> skip same-line path
                    pass
                else:
                    label_pos = m_label.start()
                    best = None
                    for mline in pct_pat_any.finditer(ln):
                        val = mline.group("val")
                        if not _valid_nim(val):
                            continue
                        pos = mline.start()
                        near = ln[max(0, pos - 15): min(len(ln), pos + len(mline.group(0)) + 15)]
                        if "$" in near:
                            continue
                        if re.search(r"\b(increase|decrease|up|down)\b", ln, re.I) and abs(pos - label_pos) > 12:
                            continue
                        if abs(pos - label_pos) > 30:
                            continue
                        score = 180 - min(abs(pos - label_pos), 20)
                        if (best is None) or (score > best[0]):
                            best = (score, val, "%", cid_blk, ln.strip()[:160])
                    if best:
                        candidates.append((best[0], best[1], best[2], best[3]))
                        LOGGER.debug("[DBG][NIM_SAMELINE] cid=%s picked=%s line=%s", best[3], best[1], best[4])

            if "unfunded" in ln or "lending commitment" in ln or "commitments" in ln:
                continue

            # scan next 3 lines (table values often immediately follow)
            look = " ".join(lines[i:i+8])
            if re.search(r"december\s+31", look, re.I):
                continue

            # (a) percent form
            m2 = pct_pat_any.search(look)
            if m2:
                val = m2.group("val")
                if _valid_nim(val):
                    bonus = 20 if ("2024" in look) else 0
                    print(
                        f"[DEBUG][NIM_REGEX] line_match block={bi} line={i} val={val} cid={cid_blk}",
                        flush=True
                    )
                    # reject money-adjacent number for this candidate (do NOT reject the whole line)
                    # if the matched percent is immediately preceded by '$', it's not NIM
                    pos = look.find(m2.group(0))
                    if pos != -1:
                        if pos > 0 and look[pos - 1] == "$":
                            continue
                        near2 = look[max(0, pos - 6): min(len(look), pos + len(m2.group(0)) + 6)]
                        if "$" in near2:
                            continue
                    candidates.append((120 + bonus, val, "%", cid_blk))
                    continue

            # (b) bare number but 'percent' keyword nearby (rare)
            if ENABLE_BARE_PERCENT and re.search(r"(?i)\bpercent\b", look):
                m3 = bare_pat.search(look)
                if m3:
                    val = m3.group("val")
                    if _valid_nim(val):
                        near_low = look.lower()   # Score using the local lookahead window (not the full context).
                        score = 100
                        if "$" in near_low or "billion" in near_low or "million" in near_low or "thousand" in near_low:
                            continue
                        if "selected performance ratios" in near_low: score += 80
                        if "year ended" in near_low or "twelve months" in near_low or "december" in near_low: score += 40
                        if "quarter" in near_low or "three months" in near_low or "unaudited" in near_low: score -= 120
                        if "2024" in near_low: score += 20
                        # reject money-adjacent number for this candidate (do NOT reject the whole line)
                        # if the matched percent is immediately preceded by '$', it's not NIM
                        pos = look.find(m3.group("val"))
                        if pos != -1:
                            if pos > 0 and look[pos - 1] == "$":
                                continue
                            near2 = look[max(0, pos - 6): min(len(look), pos + len(m3.group(0)) + 6)]
                            if "$" in near2:
                                continue
                        candidates.append((score, val, "%", cid))

    if not candidates:
        return None

    # tie-break: if we have plausible NIM (<6%), drop very large ones (>=8%)
    _vals = []
    for s, v, u, cid in candidates:
        try:
            _vals.append(float(str(v)))
        except Exception:
            pass

    if any(x < 6.0 for x in _vals):
        filtered = []
        for s, v, u, cid in candidates:
            try:
                if float(str(v)) < 8.0:
                    filtered.append((s, v, u, cid))
            except Exception:
                filtered.append((s, v, u, cid))
        if filtered:
            candidates = filtered

    # pick best score
    candidates.sort(key=lambda x: x[0], reverse=True)
    for score, v, u, cid in candidates[:5]:
        print(
            f"[DEBUG][NIM_REGEX] cand score={score} val={v} cid={cid}",
            flush=True
        )
    _, val, unit, cid = candidates[0]
    print(
        f"[DEBUG][NIM_REGEX] PICK val={val} unit={unit} cid={cid}",
        flush=True
    )
    return (val, unit, cid)

def try_regex_extract_roa_roe_from_context(context: str, year: int, head_scan_chars: int = 2600):
    """
    Try to extract ROA/ROE (usually percentages) directly from context.

    Handles prose like:
    - "Return on average assets and equity were 1.99% and 8.69%, respectively, for the year ended December 31, 2024."
    And table-ish blocks containing "Return on Assets"/"Return on Equity" near percentage values.

    Return: dict with optional keys 'ROA'/'ROE' each as (value, unit, source_chunk_id_headerline)
    """
    if not context:
        return {}
    LOGGER.debug("[DBG][ROA_ROE_ENTER] year=%s head=%s", year, (context or "")[:120].replace("\n"," "))

    m = _parse_vertical_year_table(
    context,
    section_title="Selected Performance Ratios",
    year=int(year),
    stop_titles=["Summary of Operations", "Average Balances", "Per Share Data"]
    )
    # Some reports use variant section titles (e.g., different casing); try a small set of common aliases.
    if not m:
        m = _parse_vertical_year_table(
            context,
            section_title="Selected performance ratios",
            year=int(year),
            stop_titles=["Summary of Operations", "Average Balances", "Per Share Data"],
        )

    if m:
        print(f"[DBG][VERT_RATIO] keys={list(m.keys())[:10]}", flush=True)
        out = {}
        # normalize keys
        for k, v in m.items():
            kl = (k or "").lower()
            if "return on average assets" in kl:
                out["ROA"] = (str(v), "%", _first_block_cid(context)) # ROA/ROE are ratios; standardize unit as percent for downstream CSV normalization.
            # ROE: only accept "return on average equity" (exclude tangible/ROTCE variants) to avoid non-target ratios.
            if "return on average equity" in kl and ("tangible" not in kl) and ("rotce" not in kl):
                out["ROE"] = (str(v), "%", _first_block_cid(context))
        if out:
            return out

    blocks = re.split(r"\n\s*---\s*\n", context)
    header_pat = re.compile(r"^\[k=.*?\|stem=.*?\|chunk=\d+\]", flags=re.M)

    pct_pat = re.compile(r"(?P<val>\d{1,2}(?:\.\d{1,4})?)\s*%")

    # Keyword patterns: allow variants like "Return (net income) on ..."
    RET_ON = r"return(?:\s*\([^)]*\))?\s+on"  
    roa_kw = re.compile(rf"({RET_ON}\s+(?:average\s+)?(?:total\s+)?assets|\bROA\b|\bROAA\b)", flags=re.I)
    roe_kw = re.compile(rf"({RET_ON}\s+(?:average\s+)?(?:common\s+)?(?:shareholders[’']?\s+)?equity|\bROE\b|\bROAE\b)", flags=re.I)

    # joint sentence pattern: assets ... equity ... X% and Y% ... respectively ... 2024
    joint_pat = re.compile(
        r"return\s+on\s+average\s+assets\s+and\s+equity\s+were\s+(?P<a>\d{1,2}(?:\.\d{1,4})?)\s*%\s+and\s+(?P<e>\d{1,2}(?:\.\d{1,4})?)\s*%.*?respectively",
        flags=re.I | re.S,
    )

    def _header_line(blk: str) -> str:
        # take first header line in block
        m = header_pat.search(blk)
        if not m:
            return "NOT FOUND"
        # usually the first line is the header already
        for ln in blk.splitlines():
            ln = ln.strip()
            if ln.startswith("[k=") and "|chunk=" in ln:
                return ln
        return m.group(0).strip()

    def _near_money(text: str, start: int, end: int, window: int = 15) -> bool:
        seg = text[max(0, start - window): min(len(text), end + window)]
        return "$" in seg

    def _kw_before(kw_pat, text: str, pos: int, max_back: int) -> bool:
        """True if kw_pat occurs shortly BEFORE position pos."""
        last_end = None
        for m in kw_pat.finditer(text):
            if m.end() <= pos:
                last_end = m.end()
            else:
                break
        if last_end is None:
            return False
        return (pos - last_end) <= max_back

    def _score_candidate(val: str, blk: str, metric: str) -> int:
        # score heuristic: prefer mentions of 2024 / Dec 31 2024 and plausible ranges
        t = (blk or "").lower()
        score = 0
        if "2024" in t or "december" in t and "2024" in t:
            score += 30
        try:
            fv = float(val)
        except Exception:
            return -999
        if metric == "ROA":
            if 0 < fv <= 10:
                score += 30
            if 0 < fv <= 5:
                score += 10
        if metric == "ROE":
            if 0 < fv <= 40:
                score += 30
            if 0 < fv <= 25:
                score += 10
        # shorter distance to keyword helps; approximate by first occurrence
        if metric == "ROA" and roa_kw.search(blk):
            score += 10
        if metric == "ROE" and roe_kw.search(blk):
            score += 10
        return score

    best = {}  # metric -> (score, val, unit, cid)

    head = context[:head_scan_chars].lower()

    num_pat = re.compile(r"(?<!\d)\d{1,2}(?:\.\d{1,4})?(?!\d)")

    # same-line / label-down ROA/ROE candidates per block
    blocks = re.split(r"\n\s*---\s*\n", context)
    for blk in blocks:
        cid = _header_line(blk)
        lines_blk = blk.splitlines()
        for idx, ln in enumerate(lines_blk):
            lnl = ln.lower()
            if any(x in lnl for x in ["cet1", "tier", "total capital", "capital ratio", "leverage"]):
                continue

            # ROA label + nearby
            if ("return on average assets" in lnl) or ("return on average total assets" in lnl) or re.search(r"\broa\b|\broaa\b", lnl):
                val = None
                val_line = ln
                found_pct = False
                mp_self = pct_pat.search(ln)
                if mp_self and not _near_money(ln, mp_self.start(), mp_self.end(), 15):
                    val = mp_self.group("val")
                    found_pct = True
                if val is None:
                    window = lines_blk[idx + 1: idx + 7]
                    for w in window:
                        wl = w.lower()
                        if any(x in wl for x in ["cet1", "tier", "total capital", "capital ratio", "leverage"]):
                            continue
                        mp = pct_pat.search(w)
                        if mp and not _near_money(w, mp.start(), mp.end(), 15):
                            val = mp.group("val")
                            val_line = w
                            found_pct = True
                            break
                if val is None:
                    window = lines_blk[idx + 1: idx + 7]
                    for w in window:
                        wl = w.lower()
                        if any(x in wl for x in ["cet1", "tier", "total capital", "capital ratio", "leverage"]):
                            continue
                        if re.search(r"\b(19|20)\d{2}\b", w):
                            continue
                        if re.search(r"\(\s*\d+\s*\)", w):
                            continue
                        mn = num_pat.search(w)
                        if mn:
                            try:
                                fv = float(mn.group())
                            except Exception:
                                continue
                            if 0.1 <= fv <= 5:
                                val = mn.group()
                                val_line = w
                                break
                if val:
                    sc = 170 if found_pct else 150
                    prev = best.get("ROA")
                    if (prev is None) or (sc > prev[0]):
                        best["ROA"] = (sc, val, "%", cid)
                    LOGGER.debug("[DBG][ROA_LABEL_DOWN] cid=%s picked=%s label_line=%s val_line=%s",
                                 cid, val, ln.strip()[:160], val_line.strip()[:160])

            # ROE label + nearby (exclude tangible/ROTCE)
            if ("return on average equity" in lnl) or ("return on average common shareholders' equity" in lnl) or re.search(r"return on average common shareholders[’'] equity", lnl) or re.search(r"\broe\b|\broae\b", lnl):
                if ("tangible" in lnl) or ("rotce" in lnl):
                    continue
                val = None
                val_line = ln
                found_pct = False
                mp_self = pct_pat.search(ln)
                if mp_self and not _near_money(ln, mp_self.start(), mp_self.end(), 15):
                    val = mp_self.group("val")
                    found_pct = True
                if val is None:
                    window = lines_blk[idx + 1: idx + 7]
                    for w in window:
                        wl = w.lower()
                        if any(x in wl for x in ["cet1", "tier", "total capital", "capital ratio", "leverage"]):
                            continue
                        mp = pct_pat.search(w)
                        if mp and not _near_money(w, mp.start(), mp.end(), 15):
                            val = mp.group("val")
                            val_line = w
                            found_pct = True
                            break
                if val is None:
                    window = lines_blk[idx + 1: idx + 7]
                    for w in window:
                        wl = w.lower()
                        if any(x in wl for x in ["cet1", "tier", "total capital", "capital ratio", "leverage"]):
                            continue
                        if re.search(r"\b(19|20)\d{2}\b", w):
                            continue
                        if re.search(r"\(\s*\d+\s*\)", w):
                            continue
                        mn = num_pat.search(w)
                        if mn:
                            try:
                                fv = float(mn.group())
                            except Exception:
                                continue
                            if 2 <= fv <= 30:
                                val = mn.group()
                                val_line = w
                                break
                if val:
                    sc = 170 if found_pct else 150
                    prev = best.get("ROE")
                    if (prev is None) or (sc > prev[0]):
                        best["ROE"] = (sc, val, "%", cid)
                    LOGGER.debug("[DBG][ROE_LABEL_DOWN] cid=%s picked=%s label_line=%s val_line=%s",
                                 cid, val, ln.strip()[:160], val_line.strip()[:160])

    for blk in blocks:
        cid = _header_line(blk)

        # 0) Ratio-table style (e.g., "Financial ratios: Return on average assets 0.35 % 0.49 % 0.93 %")
        # Prefer GAAP / "Return on average ..." rows; avoid non-GAAP like "Core ROTCE".
        lines = [ln.strip() for ln in blk.splitlines() if ln.strip()]
        for ln in lines:
            ln_norm = re.sub(r"\s+", " ", ln.strip())
            lnl = ln_norm.lower()

            # Skip obvious non-target ROE variants
            if ("return on average equity" in lnl or re.search(r"\broe\b", lnl)) and ("rotce" in lnl or "tangible" in lnl):
                continue

            # ROA row: "return on average assets"
            if ("return on average assets" in lnl) \
                or ("return on average total assets" in lnl) \
                or (re.search(r"\broa\b", lnl) and "return" in lnl):
                pos = lnl.find("return on average assets")
                tail = ln_norm[pos:] if pos >= 0 else ln

                m = re.search(r"(?P<v>\d{1,2}(?:\.\d{1,4})?)\s*%", tail)
                if m:
                    v = m.group("v")
                    cid = _header_line(blk)
                    print(f"[DBG][ROA_ROW] cid={cid} ln={ln_norm}")
                    print(f"[DBG][ROA_ROW] tail={tail}")
                    print(f"[DBG][ROA_ROW] picked={v}")
                    prev = best.get("ROA")
                    sc = 1000  # Row-level hit: assign a high priority to prevent being overridden by unrelated percent matches.
                    if (prev is None) or (sc > prev[0]):
                        best["ROA"] = (sc, v, "%", cid)


            # ROE row: prefer "return on average equity" / "gaap return on equity"
            if ("return on average equity" in lnl) \
                or ("return on average common equity" in lnl) \
                or ("return on average shareholders' equity" in lnl) \
                or ("return on average stockholders' equity" in lnl) \
                or (re.search(r"\broe\b", lnl) and "return" in lnl):
                pos = lnl.find("return on average equity")
                tail = ln_norm[pos:] if pos >= 0 else ln_norm

                m = re.search(r"(?P<v>\d{1,2}(?:\.\d{1,4})?)\s*%", tail)
                if m:
                    v = m.group("v")
                    cid = _header_line(blk)

                    prev = best.get("ROE")
                    sc = 1000
                    if (prev is None) or (sc > prev[0]):
                        best["ROE"] = (sc, v, "%", cid)


        # 1) joint sentence (grab both at once)
        m_joint = joint_pat.search(blk)
        if m_joint:
            a = m_joint.group("a")
            e = m_joint.group("e")
            for metric, val in [("ROA", a), ("ROE", e)]:
                sc = _score_candidate(val, blk, metric) + 50  # bonus for joint pattern
                prev = best.get(metric)
                if (prev is None) or (sc > prev[0]):
                    best[metric] = (sc, val, "%", cid)

        # 2) ROA: percent should be CLOSE to ROA keyword (distance-based)
        if roa_kw.search(blk):
            kw_spans = [(m0.start(), m0.end()) for m0 in roa_kw.finditer(blk)]
            for m in pct_pat.finditer(blk):
                val = m.group("val")

                # distance from this % to nearest ROA keyword span
                p = m.start()
                dist = min(min(abs(p - ks), abs(p - ke)) for ks, ke in kw_spans)

                # Distance gate: require the percent token to be close to the ROA/ROE keyword to avoid capturing unrelated growth-rate percentages.
                # hard gate: too far => likely unrelated % (e.g., growth rate)
                if dist > 140:
                    continue

                # keep a window only for scoring "year" etc.
                s = max(0, m.start() - 220)
                e = min(len(blk), m.end() + 220)
                window = blk[s:e]

                sc = _score_candidate(val, window, "ROA") + int(max(0, 140 - dist))  # closer => higher
                prev = best.get("ROA")
                if (prev is None) or (sc > prev[0]):
                    best["ROA"] = (sc, val, "%", cid)

        # 3) ROE: percent should be CLOSE to ROE keyword (distance-based)
        if roe_kw.search(blk):
            kw_spans = [(m0.start(), m0.end()) for m0 in roe_kw.finditer(blk)]
            for m in pct_pat.finditer(blk):
                val = m.group("val")

                # distance from this % to nearest ROE keyword span
                p = m.start()
                dist = min(min(abs(p - ks), abs(p - ke)) for ks, ke in kw_spans)

                # hard gate: too far => likely unrelated % (e.g., growth rate)
                if dist > 140:
                    continue

                # keep a window only for scoring "year" etc.
                s = max(0, m.start() - 220)
                e = min(len(blk), m.end() + 220)
                window = blk[s:e]

                sc = _score_candidate(val, window, "ROE") + int(max(0, 140 - dist))  # closer => higher
                prev = best.get("ROE")
                if (prev is None) or (sc > prev[0]):
                    best["ROE"] = (sc, val, "%", cid)


    out = {}
    for metric in ("ROA", "ROE"):
        if metric in best:
            _, val, unit, cid = best[metric]
            out[metric] = (val, unit, cid)

    # Fallback for highlight-style tables where labels appear earlier and value-only rows appear later in the same block.
    # --- FALLBACK: "Selected Performance Ratios" table style (labels separated from values) ---
    # Many reports list:
    #   Selected Performance Ratios:
    #   Return on Assets
    #   Return on Equity
    #   ...
    # and later (still in the same chunk) show only value rows like:
    #   1.42 % 1.26% 1.57% 1.32%
    #   14.70 % 13.41 % ...
    # In this case, keyword-near-% search fails, so we map the first two %-rows to ROA/ROE.
    if ("ROA" not in out) or ("ROE" not in out):
        for blk in blocks:
            if "selected performance ratios" not in (blk or "").lower():
                continue

            cid = _header_line(blk)

            # pick lines that look like ratio rows: multiple % values on the same line
            value_lines = []

            for ln in (blk or "").splitlines():
                ln_norm = re.sub(r"\s+", " ", ln.strip())
                lnl = ln_norm.lower()

                vals = pct_pat.findall(ln_norm)
                if len(vals) >= 2:  # table row typically has multiple years' % values
                    value_lines.append((ln_norm, vals))

            # We expect at least 2 rows: ROA row then ROE row
            if len(value_lines) >= 2:
                roa_vals = value_lines[0][1]
                roe_vals = value_lines[1][1]

                # --- determine which column corresponds to target fiscal year ---
                year_idx = 0  # default fallback
                years_pat = re.compile(r"\b20\d{2}\b")

                # try to find a year header line inside the same block (often above the % rows)
                hdr_years = None
                for ln in (blk or "").splitlines():
                    ln_norm = re.sub(r"\s+", " ", ln.strip())
                    yrs = years_pat.findall(ln)
                    if len(yrs) >= 2 and str(year) in yrs:
                        hdr_years = yrs
                        break

                if hdr_years:
                    try:
                        year_idx = hdr_years.index(str(year))
                    except Exception:
                        year_idx = 0

                # clamp index to available values length (sometimes values count != years count)
                def _pick(vals):
                    if not vals:
                        return None
                    i = max(0, min(year_idx, len(vals) - 1))
                    return vals[i]

                roa_pick = _pick(roa_vals)
                roe_pick = _pick(roe_vals)

                if ("ROA" not in out) and roa_pick:
                    out["ROA"] = (roa_pick, "%", cid)
                if ("ROE" not in out) and roe_pick:
                    out["ROE"] = (roe_pick, "%", cid)

                # once filled, stop scanning
                if ("ROA" in out) and ("ROE" in out):
                    break
    # --- END FALLBACK ---

    return out

def _to_intish(s: str):
    if s is None:
        return None
    x = str(s).strip()
    if not x:
        return None

    # handle parentheses negative
    neg = False
    if x.startswith("(") and x.endswith(")"):
        neg = True
        x = x[1:-1].strip()

    # OCR fix: "2,/75" is often "2,775"
    x = x.replace(",/", ",7")

    # remove common symbols then strip again
    x = x.replace("$", "").replace(",", "").strip()

    # remove stray non-numeric characters (keeps digits, dot, minus)
    x = re.sub(r"[^0-9.\-]", "", x).strip()

    if not x or not re.fullmatch(r"-?\d+(\.\d+)?", x):
        return None

    v = float(x)
    return -v if neg else v

def _parse_label_value_block(text: str, section_title: str, stop_titles: list[str]) -> dict:
    """
    Parse a simple vertical label/value block that does NOT repeat the year marker.

    Example pattern:
        Average Balances:
        Total Assets
        Total Loans
        ...
        $ 6,233,753
        4,035,670
        ...

    Returns: {label -> numeric_value}
    """
    if not text:
        return {}

    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    st = (section_title or "").lower()

    def _is_stop(ln: str) -> bool:
        ll = (ln or "").lower()
        if any(t.lower() in ll for t in (stop_titles or [])):
            return True
        # next section header like "Per Share Data:" / "Selected Performance Ratios:"
        if ll.endswith(":") and (st not in ll):
            return True
        return False

    def _looks_numeric(ln: str) -> bool:
        # allow "$ 6,233,753" / "685,862" / "(1,234)" etc.
        return bool(re.search(r"[\d,]", ln)) and not bool(re.fullmatch(r"\d{4}", ln))

    # locate section
    start = -1
    for i, ln in enumerate(lines):
        if st in ln.lower():
            start = i
            break
    if start < 0:
        return {}

    labels: list[str] = []
    vals: list[str] = []
    reading_vals = False

    for ln in lines[start + 1:]:
        if _is_stop(ln):
            break

        ll = ln.lower()
        if "dollars" in ll and "thousand" in ll:
            continue

        if not reading_vals:
            if _looks_numeric(ln):
                reading_vals = True
                vals.append(ln)
            else:
                labels.append(ln)
        else:
            # values area
            if not _is_stop(ln):
                vals.append(ln)

    out = {}
    n = min(len(labels), len(vals))
    for i in range(n):
        out[labels[i]] = _to_intish(vals[i])
    return out

def _parse_vertical_year_table(text: str, section_title: str, year: int, stop_titles: list):
    """
    Parse the 'vertical' table layout like:
    Summary of Operations:
    Interest Income
    Interest Expense
    ...
    Net Income
    2024
    291,043
    100,452
    ...
    83,811
    Year-end Balances:
    ...

    Return: dict[label -> number(float)]
    """
    lines = [ln.strip() for ln in (text or "").splitlines()]
    lines = [ln for ln in lines if ln]  # drop empty

    def _find_idx(substr):
        sub = substr.lower()
        for i, ln in enumerate(lines):
            if sub in ln.lower():
                return i
        return -1

    start = _find_idx(section_title)
    if start < 0:
        return {}

    # find year marker after section title
    ystr = str(int(year))
    year_idx = -1
    for i in range(start, len(lines)):
        ln = lines[i]
        # Accept both a standalone "2024" and a header like "2024 2023 2022 2021 2020"
        if re.search(rf"(?<!\d){ystr}(?!\d)", ln):
            year_idx = i
            break
        if any(t.lower() in ln.lower() for t in stop_titles):
            return {}
    if year_idx < 0:
        return {}

    # labels are between (start+1 .. year_idx-1)
    labels = []
    for ln in lines[start+1:year_idx]:
        # ignore obvious headers
        if ln.endswith(":"):
            continue
        # ignore currency/unit headers
        if "dollars" in ln.lower():
            continue
        labels.append(ln)

    # values follow year_idx+1 in the same order until we hit a stop title
    vals = []
    for i in range(year_idx+1, len(lines)):
        if any(t.lower() in lines[i].lower() for t in stop_titles):
            break
        vals.append(lines[i])

    # map by position (label_i -> vals_i)
    out = {}
    for i, lab in enumerate(labels):
        if i >= len(vals):
            break
        v = _to_intish(vals[i])
        out[lab] = v
    return out

def _find_first_money_after_label(text: str, labels: list):
    """
    Convert a numeric-looking string to an int-like value when safe.
    Used for cleaning values extracted from text (e.g., removing commas or currency symbols).
    Returns None if conversion is not reliable.
    """
    if not text:
        return None
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]

    # Stricter pass: prefer comma-formatted large numbers (e.g., 1,234,567).
    money_commas = re.compile(r"(?<!\d)(\(?\$?\d{1,3}(?:,\d{3})+(?:\.\d+)?\)?)(?!\d)")
    # Fallback pass: accept plain numbers, but filter out years and percentages.
    num_plain = re.compile(r"(?<!\d)(\(?\$?\d+(?:\.\d+)?\)?)(?!\d)")

    def _clean_to_float(s: str):
        s = s.strip()
        if "%" in s:
            return None
        # Normalize by removing '$', commas, and handling parenthesized negatives.
        neg = s.startswith("(") and s.endswith(")")
        s2 = s.replace("$", "").replace(",", "").strip("()")
        try:
            v = float(s2)
        except:
            return None
        if neg:
            v = -v
        # Filter out year-like values (e.g., 2023/2024).
        if 1900 <= v <= 2100 and abs(v - int(v)) < 1e-9:
            return None
        return v

    labels_l = [x.lower() for x in labels if x]
    for i, ln in enumerate(lines):
        lnl = ln.lower()
        if not any(lb in lnl for lb in labels_l):
            continue

        # Search this line and a few subsequent lines to avoid capturing only a nearby year token.
        window = " ".join(lines[i:i+6])

        # (1) Try comma-formatted large numbers first.
        for m in money_commas.finditer(window):
            v = _clean_to_float(m.group(1))
            if v is not None:
                return v

        # (2) Then fallback to plain numbers with stricter filtering.
        for m in num_plain.finditer(window):
            v = _clean_to_float(m.group(1))
            if v is not None and abs(v) >= 1000:  # Extra constraint: average assets/equity should be sufficiently large (sanity check).
                return v

    return None

def mine_ratio_candidates_from_hits(hits: list, metric: str, year: int) -> list[dict]:
    """
    Mine ROA/ROE ratio candidates from *raw retrieval hits* (pre-context-join, pre-truncation).

    Return candidates:
      {
        "label": "...",
        "value": "0.99",
        "unit": "%",
        "fiscal_year": 2024,
        "source_chunk_id": "[k=...|stem=...|chunk=269]"  (or "269")
        "snippet": "...",
        "score": 0.71,   # optional: from retrieval score
      }
    """
    metric = (metric or "").strip().upper()
    if metric not in ("ROA", "ROE"):
        return []

    def _norm_txt(s: str) -> str:
        # Normalize OCR apostrophe variants so regex matching is stable.
        return (s or "").replace("’", "'").replace("`", "'").replace("鈥?", "'")

    # Label patterns (conservative; you can extend later)
    if metric == "ROA":
        label_pat = re.compile(r"(return\s+on\s+(average\s+)?(total\s+)?assets|\broa\b|\broaa\b)", re.I)
        reject_pat = re.compile(r"(rotce|tangible|core\s+rotce|non-?gaap)", re.I)  # avoid wrong “equity-ish” rows
    else:
        label_pat = re.compile(
            r"(return\s+on\s+(average\s+)?(?:(?:common\s+)?(?:shareholders|stockholders)['’]?\s+)?equity|\broe\b|\broae\b)",
            re.I,
        )
        reject_pat = re.compile(r"(rotce|tangible|core\s+rotce|non-?gaap)", re.I)  # avoid ROTCE proxy unless you later allow

    pct_pat = re.compile(r"(?<!\d)(\d{1,2}(?:\.\d{1,4})?)\s*%")
    num_pat = re.compile(r"(?<!\d)(\d{1,2}(?:\.\d{1,4})?)(?!\d)")
    year_pat = re.compile(rf"(?<!\d){int(year)}(?!\d)")

    cands = []
    for h in (hits or []):
        text = (h.get("text") or "")
        if not text:
            continue

        # Build a stable citation header similar to your context blocks
        bank = h.get("bank") or h.get("bank_folder") or ""
        stem = h.get("stem") or ""
        chunk = h.get("chunk_id") if h.get("chunk_id") is not None else h.get("chunk")
        cid = f"[k={bank}|stem={stem}|chunk={chunk}]" if (bank or stem or chunk is not None) else "NOT FOUND"

        lines = text.splitlines()
        for i, ln in enumerate(lines):
            ln_n = _norm_txt(ln)
            if not ln_n or not label_pat.search(ln_n):
                continue

            # Reject by label line only; table windows can contain mixed ROE/ROTCE rows.
            win_lines = lines[i : min(i + 9, len(lines))]
            win = "\n".join(win_lines)
            if reject_pat.search(ln_n):
                continue

            def _ok(val: str) -> bool:
                try:
                    fv = float(val)
                except Exception:
                    return False
                if metric == "ROA" and not (0.0 < fv <= 10.0):
                    return False
                if metric == "ROE" and not (0.0 < fv <= 40.0):
                    return False
                if 1900 <= int(fv) <= 2100:
                    return False
                return True

            # Prefer values on the label line itself.
            same_line_picked = False
            for m in pct_pat.finditer(ln_n):
                val = m.group(1)
                if not _ok(val):
                    continue
                cands.append({
                    "label": ln.strip()[:120],
                    "value": val,
                    "unit": "%",
                    "fiscal_year": int(year),
                    "source_chunk_id": cid,
                    "snippet": ln_n[:400],
                    "score": h.get("score"),
                })
                same_line_picked = True
                break
            if same_line_picked:
                continue

            # Table style fallback: some rows have values without '%' (e.g., "10.4 11.2 13.2").
            for m in num_pat.finditer(ln_n):
                val = m.group(1)
                if not _ok(val):
                    continue
                cands.append({
                    "label": ln.strip()[:120],
                    "value": val,
                    "unit": "%",
                    "fiscal_year": int(year),
                    "source_chunk_id": cid,
                    "snippet": ln_n[:400],
                    "score": h.get("score"),
                })
                same_line_picked = True
                break
            if same_line_picked:
                continue

            # Prefer year-local window if present (helps on “Selected ratios” tables)
            win2 = win
            if year_pat.search(text):
                # if the whole block contains year, we keep as-is; otherwise still ok

                pass

            for m in pct_pat.finditer(win2):
                val = m.group(1)
                if not _ok(val):
                    continue

                snippet = win2[:800]  # keep short; judge will see a few
                cands.append({
                    "label": ln.strip()[:120],
                    "value": val,
                    "unit": "%",
                    "fiscal_year": int(year),
                    "source_chunk_id": cid,
                    "snippet": snippet,
                    "score": h.get("score"),
                })

    # Light de-dup: same cid+value
    seen = set()
    out = []
    for c in sorted(cands, key=lambda x: (-(x.get("score") or 0), x.get("source_chunk_id",""), x.get("value",""))):
        k = (c.get("source_chunk_id"), c.get("value"))
        if k in seen:
            continue
        seen.add(k)
        out.append(c)

    return out
