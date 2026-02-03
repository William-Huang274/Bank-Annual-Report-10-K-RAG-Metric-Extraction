import re
from typing import Callable

from src.rag.regex_extractors import (
    _parse_vertical_year_table,
    _parse_label_value_block,
    _to_intish,
)
from src.rag.retrieval import search_faiss
from src.rag.packing import build_context


def maybe_compute_roa_roe_from_context(final_obj: dict, context: str, year: int, dprint: Callable = None):
    """
    Compute ROA/ROE from (Net Income, Avg Assets/Avg Equity) when explicit ROA/ROE are missing.

    Notes:
    - This writes computed ratios back into final_obj["results"] with unit="%".
    - It is intentionally conservative and only computes when inputs are confidently extracted.
    """
    if dprint is None:
        dprint = lambda *a, **k: None

    if not isinstance(final_obj, dict) or "results" not in final_obj:
        return

    def _get_row(name: str):
        for r in final_obj.get("results", []):
            if r.get("metric_name") == name:
                return r
        return None

    roa_row = _get_row("ROA")
    roe_row = _get_row("ROE")

    need_roa = (
        roa_row is not None
        and str(roa_row.get("value", "")).strip().upper() in ("NOT FOUND", "NOT_FOUND", "")
    )
    need_roe = (
        roe_row is not None
        and str(roe_row.get("value", "")).strip().upper() in ("NOT FOUND", "NOT_FOUND", "")
    )
    if not (need_roa or need_roe):
        return

    def _slice_section(text: str, start_title: str, stop_titles: list) -> str:
        lines = text.splitlines()
        out = []
        in_sec = False
        for ln in lines:
            lnl = (ln or "").lower()
            if start_title.lower() in lnl:
                in_sec = True
            if in_sec:
                if any(t.lower() in lnl for t in stop_titles):
                    break
                out.append(ln)
        return "\n".join(out).strip()

    # 1) Slice sections (prevents year-end numbers polluting average balance extraction)
    summary_block = _slice_section(
        context,
        start_title="Summary of Operations",
        stop_titles=["Year-end Balances", "Average Balances", "Per Share Data", "Selected Performance Ratios"],
    )
    avg_block = _slice_section(
        context,
        start_title="Average Balances",
        stop_titles=["Per Share Data", "Selected Performance Ratios", "Other Data at Year-end", "Year-end Balances"],
    )

    # 2) Extract inputs (reuse existing helpers)
    # Prefer the vertical-year table parser (correct for layouts like: labels... then "2024" then values column)
    summary_map = _parse_vertical_year_table(
        summary_block or context,
        section_title="Summary of Operations",
        year=int(year),
        stop_titles=["Year-end Balances", "Average Balances", "Per Share Data", "Selected Performance Ratios"],
    )

    def _parse_vertical_no_year(section_text: str, section_title: str, stop_titles: list[str]) -> dict:
        lines = [ln.strip() for ln in (section_text or "").splitlines() if ln.strip()]
        if not lines:
            return {}

        # locate section start
        start = -1
        st = section_title.lower()
        for i, ln in enumerate(lines):
            if st in ln.lower():
                start = i
                break
        if start < 0:
            return {}

        # collect until stop title
        block = []
        for ln in lines[start+1:]:
            if any(t.lower() in ln.lower() for t in stop_titles):
                break
            block.append(ln)

        # split into labels then numeric values column (first numeric marks the start of values)
        def _is_numlike(s: str) -> bool:
            x = s.replace("$", "").replace(",", "").strip()
            return bool(re.fullmatch(r"-?\d+(\.\d+)?", x))

        labels = []
        vals = []
        in_vals = False
        for ln in block:
            if not in_vals and _is_numlike(ln):
                in_vals = True
            if in_vals:
                v = _to_intish(ln)
                if v is not None:
                    vals.append(float(v))
            else:
                # drop unit headers etc.
                if "dollars" in ln.lower():
                    continue
                if ln.endswith(":"):
                    continue
                labels.append(ln)

        out = {}
        for i, name in enumerate(labels):
            if i < len(vals):
                out[name] = vals[i]
        return out

    avg_map = _parse_vertical_no_year(
        avg_block or context,
        section_title="Average Balances",
        stop_titles=["Per Share Data", "Selected Performance Ratios", "Other Data at Year-end", "Year-end Balances"],
    )

    def _get_first(d: dict, keys: list[str]):
        for k in keys:
            if k in d and d[k] is not None:
                return float(d[k])
        return None

    def _pick_contains(d: dict, must: list[str], must_not: list[str]):
        for k, v in (d or {}).items():
            kl = (k or "").lower()
            if any(x in kl for x in must) and not any(x in kl for x in must_not):
                if v is not None:
                    return float(v)
        return None

    net_income = _get_first(summary_map, ["Net Income", "Net income", "Net income (loss)", "Net loss"])
    if net_income is None:
        # avoid grabbing "Net Interest Income" by excluding interest/margin
        net_income = _pick_contains(summary_map, ["net", "income"], ["interest", "margin"])

    avg_assets = _get_first(avg_map, ["Total Assets", "Total assets", "Average Total Assets", "Average total assets"])
    dprint(f"[ENH] avg_map keys sample: {list(avg_map.keys())[:20]}")
    dprint("[ENH] avg_block first 40 lines:\n" + "\n".join((avg_block or "").splitlines()[:40]))
    dprint(f"[ENH] avg_block contains 'Average Balances'? {('average balances' in (avg_block or '').lower())}")
    if avg_assets is None:
        dprint(f"[ENH] avg_map has Total Assets? {'Total Assets' in avg_map}")
        avg_assets = _pick_contains(avg_map, ["assets"], ["equity", "shareholders", "stockholders"])

    avg_equity = _get_first(avg_map, [
        "Total Shareholders鈥?Equity", "Total Shareholders' Equity",
        "Total Stockholders鈥?Equity", "Total Stockholders' Equity",
        "Average Shareholders鈥?Equity", "Average Shareholders' Equity",
        "Average Stockholders鈥?Equity", "Average Stockholders' Equity",
        "Total equity", "Average equity",
    ])
    if avg_equity is None:
        avg_equity = _pick_contains(avg_map, ["equity"], ["assets"])

    # Fallback for reports that do NOT include a standalone year marker in "Average Balances"
    if avg_assets is None or avg_equity is None:
        avg_simple = _parse_label_value_block(
            avg_block or context,
            section_title="Average Balances",
            stop_titles=["Per Share Data", "Selected Performance Ratios", "Other Data at Year-end", "Year-end Balances"],
        )
        if avg_assets is None:
            avg_assets = _get_first(avg_simple, ["Total Assets", "Total assets", "Average Total Assets", "Average total assets"])
        if avg_equity is None:
            avg_equity = _get_first(avg_simple, [
                "Total Shareholders鈥?Equity", "Total Shareholders' Equity",
                "Total Stockholders鈥?Equity", "Total Stockholders' Equity",
                "Average Shareholders鈥?Equity", "Average Shareholders' Equity",
                "Average Stockholders鈥?Equity", "Average Stockholders' Equity",
                "Total equity", "Average equity",
            ])

    dprint(f"[ENH] inputs: net_income={net_income} avg_assets={avg_assets} avg_equity={avg_equity}")

    # 3) Compute independently (do NOT require all 3 for both ratios)
    def _fmt_pct(x: float) -> str:
        return f"{x:.2f}"

    m = re.search(r"\[k=.*?\|stem=.*?\|chunk=\d+\]", context)
    cite = m.group(0) if m else "NOT FOUND"

    if need_roa and net_income is not None and avg_assets is not None and avg_assets > 0:
        # sanity: avg assets should be larger than net income (common scale: "in thousands")
        if avg_assets > max(1.0, net_income):
            roa = (net_income / avg_assets) * 100.0
            roa_row["value"] = _fmt_pct(roa)
            roa_row["unit"] = "%"
            roa_row["fiscal_year"] = int(year)
            roa_row["source_chunk_id"] = f"COMPUTED_FROM {cite} (Net Income / Avg Assets)"

    if need_roe and net_income is not None and avg_equity is not None and avg_equity > 0:
        if avg_equity > max(1.0, net_income):
            roe = (net_income / avg_equity) * 100.0
            roe_row["value"] = _fmt_pct(roe)
            roe_row["unit"] = "%"
            roe_row["fiscal_year"] = int(year)
            roe_row["source_chunk_id"] = f"COMPUTED_FROM {cite} (Net Income / Avg Equity)"


def augment_context_with_avg_balances(index, meta, emb, bank_id: str, year: int, base_context: str) -> str:
    """
    Ensure the evidence context contains Average Balances / Avg Assets / Avg Equity blocks,
    so that maybe_compute_roa_roe_from_context() can parse avg_assets/avg_equity.
    """
    if not bank_id:
        return base_context

    tb = (bank_id or "").strip().lower().lstrip("_")

    def _bank_match(hit_bank: str) -> bool:
        hb = (hit_bank or "").strip().lower().lstrip("_")
        if not hb or not tb:
            return False
        return hb == tb or hb.startswith(tb) or tb.startswith(hb) or (len(tb) >= 8 and tb in hb) or (len(hb) >= 8 and hb in tb)

    queries = [
        f"{bank_id} Average Balances {year}",
        f"{bank_id} average total assets {year}",
        f"{bank_id} average assets {year}",
        f"{bank_id} average total equity {year}",
        f"{bank_id} average shareholders' equity {year}",
        f"{bank_id} average stockholders' equity {year}",
        "Average Balances total assets",
        "Average Total Assets",
        "Average shareholders' equity",
        "Average stockholders' equity",
    ]

    pooled = []
    for q in queries:
        pooled.extend(search_faiss(index, meta, emb, q, topk=60))

    # filter + dedup + add "k" header for build_context()
    filtered = []
    seen = set()
    for h in sorted(pooled, key=lambda x: x.get("score", 0.0), reverse=True):
        if h.get("score", 0.0) < 0.45:
            continue
        if not _bank_match(h.get("bank")):
            continue
        key = (h.get("bank"), h.get("stem"), str(h.get("chunk_id")))
        if key in seen:
            continue
        seen.add(key)

        if "k" not in h:
            h["k"] = f'k={h.get("bank")}|stem={h.get("stem")}|chunk={h.get("chunk_id")}'
        filtered.append(h)
        if len(filtered) >= 12:
            break

    if not filtered:
        return base_context

    extra = build_context(filtered)
    return (base_context or "") + "\n\n---\n\n" + extra
