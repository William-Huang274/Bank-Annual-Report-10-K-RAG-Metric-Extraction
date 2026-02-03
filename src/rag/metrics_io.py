"""
Metrics I/O helpers (CSV/JSONL) with stable schema and merge rules.
Behavior matches prior pipeline; changes focus on compatibility and safety.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, List


def _is_not_found(x) -> bool:
    if x is None:
        return True
    s = str(x).strip().upper()
    return s in ("NOT FOUND", "NOT_FOUND", "")


def _is_nf(x) -> bool:
    """Missing check used by merge_keep_existing (kept for compatibility)."""
    if x is None:
        return True
    s = str(x).strip().upper()
    return s in ("", "NOT FOUND", "NOT_FOUND", "N/A", "NA", "VALUE", "UNIT", "SOURCE_CHUNK_ID")


def normalize_value_unit(val, unit):
    if val is None:
        return "NOT FOUND", unit or "NOT FOUND"

    s = str(val).strip()
    u = (unit or "").strip()

    if s.endswith("%"):
        s = s[:-1].strip()
        u = "%"

    if s.strip().lower() in ("value", "val"):
        s = "NOT FOUND"
    if u.strip().lower() in ("unit",):
        u = "NOT FOUND"

    return s, u


def bucket_from_row(row: Dict) -> str:
    cid = (
        row.get("source_chunk_id")
        or row.get("source_chunk")
        or row.get("source_chunkid")
        or row.get("chunk_id")
        or ""
    )
    cid = str(cid).strip()
    v = row.get("val")
    u = row.get("unit")

    if row.get("error"):
        return "pipeline_error"
    if cid.startswith("table:"):
        return "table_prefill"
    if cid.startswith("regex:"):
        return "regex_prefill"
    if cid.startswith("llm:"):
        return "llm_filled"
    if _is_nf(v):
        return "value_missing"
    if _is_nf(u):
        return "unit_missing"
    return "ok"


def flatten_metrics(records: List[dict], audit_buffer: List[dict] | None = None) -> List[dict]:
    rows: List[dict] = []
    audit = audit_buffer

    for rec in records:
        meta0 = rec.get("_meta", {})
        bank = meta0.get("bank")
        year = meta0.get("year")
        has_err = bool(rec.get("error"))

        for item in rec.get("results", []):
            metric = item.get("metric_name", "UNKNOWN")
            val = item.get("value")
            unit = item.get("unit")
            val, unit = normalize_value_unit(val, unit)

            if audit is not None and not _is_not_found(val):
                audit.append({
                    "bank": bank,
                    "year": year,
                    "metric": metric,
                    "val": val,
                    "unit": unit,
                    "chunk": item.get("source_chunk_id"),
                })

            cid = (
                item.get("source_chunk_id")
                or item.get("source_chunk")
                or item.get("source_chunkid")
                or item.get("chunk_id")
            )

            row = {
                "bank": bank,
                "year": year,
                "metric_name": metric,
                "val": val,
                "unit": unit,
                "source_chunk_id": cid,
            }
            if has_err:
                row["error"] = rec.get("error") or True
            row["bucket"] = bucket_from_row(row)
            row.pop("error", None)  # keep output schema stable
            rows.append(row)
    return rows


def merge_keep_existing(old_csv_path: Path, new_rows: List[dict]) -> List[dict]:
    """
    Keep existing non-missing values from old CSV; otherwise insert/overwrite with new rows.
    Preserve rows for banks not present in the current run.
    """
    old_map: Dict[tuple, dict] = {}
    if old_csv_path.exists():
        with old_csv_path.open("r", encoding="utf-8") as f:
            r = csv.DictReader(f)
            for row in r:
                # compatibility: map legacy source_chunk -> source_chunk_id
                if (row.get("source_chunk_id") in (None, "")) and ("source_chunk" in row):
                    row["source_chunk_id"] = row.get("source_chunk")
                k = (row.get("bank"), str(row.get("year")), row.get("metric_name"))
                old_map[k] = row

    merged_map: Dict[tuple, dict] = dict(old_map)

    for nr in new_rows:
        k = (nr.get("bank"), str(nr.get("year")), nr.get("metric_name"))
        orow = old_map.get(k)

        if orow and (not _is_nf(orow.get("val"))):
            kept = {
                "bank": orow.get("bank"),
                "year": orow.get("year"),
                "metric_name": orow.get("metric_name"),
                "val": orow.get("val"),
                "unit": orow.get("unit"),
                "source_chunk_id": orow.get("source_chunk_id"),
            }
            kept["bucket"] = bucket_from_row(kept)
            merged_map[k] = kept
            continue

        nr = dict(nr)
        nr["bucket"] = bucket_from_row(nr)
        merged_map[k] = nr

    # Stable order: bank, year, metric_name
    def _sort_key(item):
        bank, year, metric = item[0]
        try:
            y = int(year)
        except Exception:
            y = 0
        return (str(bank), y, str(metric))

    return [v for _, v in sorted(merged_map.items(), key=_sort_key)]


def write_metrics_csv(path: Path, rows: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = ["bank", "year", "metric_name", "val", "unit", "source_chunk_id", "bucket"]
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            if not r.get("bucket"):
                r["bucket"] = bucket_from_row(r)
            w.writerow(r)


def write_jsonl(path: Path, records: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
