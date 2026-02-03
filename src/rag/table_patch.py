"""
Table-sidecar backfill utilities.
"""

from __future__ import annotations

import csv
import json
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from src.rag.metrics_io import bucket_from_row
from src.rag.table_sidecar import (
    PATCH_ALLOW,
    norm_bank,
    load_sidecar_index_multi,
    extract_metric_from_bank_tables,
)


def _norm_metric(m: str) -> str:
    m0 = (m or "").strip()
    ml = m0.lower()
    if ml in ("net interest income", "nii"):
        return "NII"
    if ml in ("net interest margin", "nim"):
        return "NIM"
    if ml in ("roa", "return on assets", "return on average assets", "roaa"):
        return "ROA"
    if ml in ("roe", "return on equity", "return on average equity", "roae"):
        return "ROE"
    if ml in ("provision for credit losses", "provision for loan losses", "pcl"):
        return "Provision for Credit Losses"

    if ("net" in ml and "interest" in ml and "income" in ml) or ("interest income" in ml and "net" in ml):
        return "NII"
    if "net interest margin" in ml:
        return "NIM"
    if ("return on" in ml and "asset" in ml) or ("return on average assets" in ml):
        return "ROA"
    if ("return on" in ml and "equity" in ml):
        return "ROE"
    if ("provision" in ml) and (("credit loss" in ml) or ("loan loss" in ml)):
        return "Provision for Credit Losses"

    return m0


def fill_values_from_tables(
    metrics_path: Path,
    out_path: Path,
    sidecar_idx: Dict[str, List[dict]],
    year: str = "2024",
) -> Tuple[Counter, Counter]:
    patched = Counter()
    stats = Counter()
    out_rows = []

    with metrics_path.open("r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        fieldnames = r.fieldnames or []
        needed = {"bank", "year", "metric_name", "val", "unit", "source_chunk_id"}
        missing = needed - set(fieldnames)
        if missing:
            raise RuntimeError(f"metrics CSV missing columns: {sorted(missing)}")

        for row in r:
            stats["rows_total"] += 1

            if (row.get("source_chunk_id") in (None, "", "NOT FOUND")) and ("source_chunk" in row):
                row["source_chunk_id"] = row.get("source_chunk")

            if str(row.get("year", "")).strip() != str(year):
                out_rows.append(row)
                stats["rows_skip_year_mismatch"] += 1
                continue

            metric_raw = row.get("metric_name") or ""
            metric_norm = _norm_metric(metric_raw)
            bank = (row.get("bank") or "").strip()
            val = (row.get("val") or "").strip()
            unit = (row.get("unit") or "").strip()

            if metric_norm not in PATCH_ALLOW:
                out_rows.append(row)
                stats["rows_skip_metric_not_allowed"] += 1
                continue

            if val.upper() != "NOT FOUND":
                out_rows.append(row)
                stats["rows_value_already_present"] += 1
                continue

            bkey = norm_bank(bank)
            blocks = sidecar_idx.get(bkey, [])

            if not blocks:
                out_rows.append(row)
                stats["rows_no_sidecar_blocks"] += 1
                continue

            new_val, new_unit, bid = extract_metric_from_bank_tables(blocks, metric_norm, str(year))
            if new_val:
                if metric_norm in ("ROA", "ROE", "NIM"):
                    try:
                        s = str(new_val).strip().replace(",", "").replace("%", "").replace("$", "")
                        num_val = float(s)
                    except Exception:
                        num_val = None
                    reject = False
                    if num_val is not None and 1900 <= int(num_val) <= 2099:
                        reject = True
                    if num_val is not None and not (0 < num_val < 50):
                        reject = True
                    if isinstance(new_val, str):
                        lv = new_val.lower()
                        if ("million" in lv) or ("thousand" in lv) or ("billion" in lv):
                            reject = True
                    if new_unit is None or str(new_unit).strip().upper() in ("", "NOT FOUND", "NOT_FOUND"):
                        new_unit = "%"
                    elif "%" not in str(new_unit):
                        reject = True
                    elif isinstance(new_unit, str) and (("million" in new_unit.lower()) or ("thousand" in new_unit.lower()) or ("billion" in new_unit.lower())):
                        reject = True
                    if reject:
                        out_rows.append(row)
                        stats["rows_patch_miss"] += 1
                        continue
                row["metric_name"] = metric_norm
                row["val"] = new_val
                if unit.upper() == "NOT FOUND" and new_unit:
                    row["unit"] = new_unit
                src0 = (row.get("source_chunk_id") or "").strip()
                if (not src0) or src0.upper() == "NOT FOUND":
                    row["source_chunk_id"] = bid if (bid and str(bid).startswith("table:")) else ("table:" + str(bid) if bid else "NOT FOUND")
                row["bucket"] = bucket_from_row(row)
                patched[metric_norm] += 1
                stats["rows_patched"] += 1
            else:
                stats["rows_patch_miss"] += 1

            out_rows.append(row)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as fw:
        if "bucket" not in fieldnames:
            fieldnames.append("bucket")
        w = csv.DictWriter(fw, fieldnames=fieldnames)
        w.writeheader()
        for row in out_rows:
            w.writerow(row)

    return patched, stats


def patch_metrics_csv_from_table_sidecar(
    metrics_csv: Path,
    sidecar_jsonl: Path | None,
    year: int,
    pdf_sidecar: Path | None = None,
    pdf_sidecar_dir: Path | None = None,
) -> dict:
    base_ok = sidecar_jsonl is not None and sidecar_jsonl.exists()
    pdf_ok  = pdf_sidecar is not None and pdf_sidecar.exists()
    dir_ok  = pdf_sidecar_dir is not None and pdf_sidecar_dir.exists()

    if not (base_ok or pdf_ok or dir_ok):
        return {}

    side_idx = load_sidecar_index_multi(
        base_sidecar=sidecar_jsonl if base_ok else None,
        pdf_sidecar=pdf_sidecar if pdf_ok else None,
        pdf_sidecar_dir=pdf_sidecar_dir if dir_ok else None,
    )

    if not side_idx:
        return {}

    tmp_csv = metrics_csv.with_suffix(".tmp.csv")
    patched, stats = fill_values_from_tables(
        metrics_path=metrics_csv,
        out_path=tmp_csv,
        sidecar_idx=side_idx,
        year=str(year),
    )
    tmp_csv.replace(metrics_csv)
    return {
        "patched_total": sum(patched.values()),
        "patched_by_metric": dict(patched),
        "stats": dict(stats),
        "rows_total": stats.get("rows_total", 0),
        "rows_patched": stats.get("rows_patched", 0),
        "rows_patch_miss": stats.get("rows_patch_miss", 0),
    }
