from __future__ import annotations

from typing import Callable, Dict, Any

from src.rag.table_sidecar import PATCH_ALLOW, norm_bank, extract_metric_from_bank_tables


def apply_table_sidecar_prefill(
    final: dict,
    target_bank: str,
    year: int,
    side_idx: Dict[str, Any] | None,
    table_prefill_allow,
    dprint: Callable,
    _get_final_row: Callable,
    _is_nf: Callable,
):
    if dprint is None:
        dprint = lambda *a, **k: None

    if side_idx is None:
        return

    allow = set(table_prefill_allow or PATCH_ALLOW) & set(PATCH_ALLOW)

    bkey = norm_bank(target_bank)
    blocks = side_idx.get(bkey, [])

    dprint(f"[TBL_PREFILL] bank={target_bank} blocks={len(blocks)} allow={sorted(list(allow))}")

    for m in allow:
        row = _get_final_row(final, m)
        cur_val = row.get("value") if row else None
        cur_unit = row.get("unit") if row else None
        cand_val, cand_unit, cand_src = extract_metric_from_bank_tables(blocks, m, str(year))
        allow_write = bool(row and _is_nf(cur_val) and cand_val and (not _is_nf(cand_val)))
        reason = []
        reason.append("metric_allowed")
        if not row:
            reason.append("no_row")
        if row and _is_nf(cur_val):
            reason.append("cur_val_is_nf")
        else:
            reason.append("cur_val_not_nf")
        if _is_nf(cur_unit):
            reason.append("cur_unit_is_nf")
        if _is_nf(cand_val):
            reason.append("cand_val_is_nf")
        if cand_unit and (not _is_nf(cand_unit)):
            reason.append("cand_unit_ok")
        dprint(f"[TBL_PREFILL_TRY] bank={target_bank} metric={m} cur_val={cur_val} cur_unit={cur_unit} -> cand_val={cand_val} cand_unit={cand_unit} src={cand_src} allow_write={allow_write} reason={','.join(reason)}")

        if not row or not _is_nf(row.get("value")):
            continue

        val, unit, src = cand_val, cand_unit, cand_src
        if m in ("ROA", "ROE", "NIM"):
            reject = False
            try:
                s = str(val).strip().replace(",", "").replace("%", "").replace("$", "")
                num_val = float(s)
            except Exception:
                num_val = None
            if num_val is not None:
                if 1900 <= int(num_val) <= 2099:
                    reject = True
                    dprint(f"[TBL_PREFILL_SKIP] bank={target_bank} metric={m} reason=year_like_val val={val}")
                elif not (0 < num_val < 50):
                    reject = True
                    dprint(f"[TBL_PREFILL_SKIP] bank={target_bank} metric={m} reason=out_of_range val={val}")
            if isinstance(val, str):
                lv = val.lower()
                if ("million" in lv) or ("thousand" in lv) or ("billion" in lv):
                    reject = True
                    dprint(f"[TBL_PREFILL_SKIP] bank={target_bank} metric={m} reason=scale_word val={val}")
            if not reject:
                if (unit is None) or _is_nf(unit):
                    unit = "%"
                elif "%" not in str(unit):
                    reject = True
                    dprint(f"[TBL_PREFILL_SKIP] bank={target_bank} metric={m} reason=unit_missing_percent val={val} unit={unit}")
                elif isinstance(unit, str) and (("million" in unit.lower()) or ("thousand" in unit.lower()) or ("billion" in unit.lower())):
                    reject = True
                    dprint(f"[TBL_PREFILL_SKIP] bank={target_bank} metric={m} reason=unit_scale_word val={val} unit={unit}")
            if reject:
                continue
        if val and (not _is_nf(val)):
            if m in ("ROA", "ROE", "NIM"):
                try:
                    x = float(str(val).replace(",", "").strip())
                except Exception:
                    x = None
                reject_extra = False
                if m == "ROA" and not (0 <= (x if x is not None else 0) <= 3.0):
                    reject_extra = True
                if m == "NIM" and not (0 <= (x if x is not None else 0) <= 6.0):
                    reject_extra = True
                if m == "ROE" and not (0 <= (x if x is not None else 0) <= 40.0):
                    reject_extra = True
                if reject_extra:
                    dprint(f"[TBL_PREFILL_SKIP] bank={target_bank} metric={m} reason=sanity_range val={val} unit={unit}")
                    continue
            row["value"] = val
            if _is_nf(row.get("unit")) and unit and (not _is_nf(unit)):
                row["unit"] = unit
            if _is_nf(row.get("source_chunk_id")) and src:
                if not str(src).startswith("table:"):
                    src = f"table:{src}"
                row["source_chunk_id"] = src

            dprint(f"[TBL_PREFILL] bank={target_bank} metric={m} val={val} unit={row.get('unit')} cid={row.get('source_chunk_id')}")
            print(f"[TBL_PREFILL] bank={target_bank} metric={m} val={val} unit={row.get('unit')} cid={row.get('source_chunk_id')}", flush=True)
