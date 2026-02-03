import json
import re
from typing import Optional, Tuple

from src.rag.prompts import METRICS


def make_template(year: int):
    return {
        "results": [
            {"metric_name": m, "value": "NOT FOUND", "unit": "NOT FOUND", "fiscal_year": int(year), "source_chunk_id": "NOT FOUND"}
            for m in METRICS
        ]
    }


def parse_json_loose(s: str):
    """
    Parse a JSON-like string with best-effort tolerance.
    Used to handle common model output issues (extra text, trailing commas, or wrapped JSON).
    Returns a Python object or None on failure.
    """
    s = (s or "").strip()
    # 1) Direct json.loads
    try:
        return json.loads(s)
    except Exception:
        pass

    # 2) Use JSONDecoder.raw_decode to locate the first valid JSON object in a mixed string
    dec = json.JSONDecoder()
    for start in range(len(s)):
        if s[start] not in "{[":
            continue
        try:
            obj, _ = dec.raw_decode(s[start:])
            return obj
        except Exception:
            continue

    raise ValueError("Model output is not valid JSON.")


def parse_or_fallback(raw_text: str, year: int = 2024):
    """
    Try `parse_json_loose` first; if it fails, fall back to regex extraction from non-JSON explanatory outputs.
    Returns:
    - A compliant object (e.g., {"results":[...]})
    - Or a flat dict: {"value": "...", "unit": "...", "source_chunk_id": "...", "fiscal_year": 2024}
    """
    # This is a heuristic preference, not a hard filter (never return empty hits).
    s = (raw_text or "").strip()

    # 1) Parse as JSON directly
    try:
        return parse_json_loose(s)
    except Exception:
        pass

    # 2) Extract a JSON substring from mixed text (some models produce prose + a JSON object)
    #    Use the first "{" and the last "}" as a coarse boundary
    l = s.find("{")
    r = s.rfind("}")
    if l != -1 and r != -1 and r > l:
        candidate = s[l:r+1]
        try:
            return parse_json_loose(candidate)
        except Exception:
            pass

    # 3) Fallback: extract a best-effort (value/unit/source) triple from non-JSON model outputs using regex.
    # value: prefer numeric patterns like 190,591 / 190591 / $190,591 / 190.591
    # value: prefer numeric patterns like 190,591 / 190591 / $190,591 / 190.591 (no unit scaling applied here)
    m_val = re.search(r"(?i)\bvalue\b[^0-9$]*\$?\s*`?\s*([0-9][0-9,\.]*)", s)
    value = m_val.group(1) if m_val else "NOT FOUND"
    value = value.replace(",", "") if value != "NOT FOUND" else value

    # unit: infer scale from phrases like "(in thousands)" / "thousand dollars" / "million"; otherwise NOT FOUND
    m_unit = re.search(r"(?i)\bunit\b[^A-Za-z]*`?\s*([A-Za-z][A-Za-z \-\(\)%]+)", s)
    unit = m_unit.group(1).strip() if m_unit else "NOT FOUND"

    # source: prefer structured cite keys like [k=...|stem=...|chunk=12]; fallback to a numeric chunk id when present
    m_cite = re.search(r"(\[k=[^\]]*?\|chunk=\d+\])", s)
    if m_cite:
        source_chunk_id = m_cite.group(1)
    else:
        m_num = re.search(r"(?i)\bchunk\b[^0-9]{0,10}(\d+)", s)
        source_chunk_id = m_num.group(1) if m_num else "NOT FOUND"

    return {
        "value": value,
        "unit": unit,
        "source_chunk_id": source_chunk_id,
        "fiscal_year": int(year),
    }


def _valid_cite(cid: str) -> bool:  # Relaxed validation: accept either full header-style citations or a plain numeric chunk id.
    """
    Validate a source_chunk_id value.
    Accepts either a full header-style citation (k=...|stem=...|chunk=...) or a plain numeric chunk id.
    """
    if not isinstance(cid, str):
        return False
    cid = cid.strip().strip("[]")
    return (
        "chunk=" in cid
        or cid.isdigit()
    )


def is_results_schema(obj) -> bool:
    """
    Return True if obj matches the expected {'results': [...]} schema.
    This is a minimal structural check used to decide whether schema repair is needed.
    """
    if not isinstance(obj, dict):
        return False
    rs = obj.get("results")
    if not isinstance(rs, list):
        return False
    # Minimal schema check: at least one dict item with metric_name and value.
    for it in rs:
        if isinstance(it, dict) and ("metric_name" in it) and ("value" in it):
            return True
    return False


def normalize_extraction(obj, year: int):
    """
    Normalize model outputs into a standard schema:
    {"results": [...]}
    """
    out = make_template(year)
    
    # Backward-compatibility: accept legacy single-metric outputs (value/unit/source_chunk_id) and map into the standard results schema.
    if isinstance(obj, dict) and ("value" in obj) and ("source_chunk_id" in obj) and ("results" not in obj):
        out = make_template(int(year))
        # Legacy path: earlier pipeline versions returned a single-metric extraction (NII only).
        metric = "NII"

        val = str(obj.get("value", "")).strip()
        cid = str(obj.get("source_chunk_id", "")).strip()
        fy  = obj.get("fiscal_year", year)

        # Align the extracted fields to the template row via metric_name.
        for row in out["results"]:
            if row.get("metric_name") == metric:
                row["value"] = val if val else "NOT FOUND"
                row["unit"] = obj.get("unit", "NOT FOUND")
                row["fiscal_year"] = int(fy) if str(fy).isdigit() else int(year)
                row["source_chunk_id"] = cid if cid else "NOT FOUND"
                break
        return out

    # Case A: Already in the expected {'results': [...]} schema.
    if isinstance(obj, dict) and isinstance(obj.get("results"), list) and len(obj["results"]) > 0:
        # Fill into the template when possible by aligning on metric_name.
        # Ignore unknown metric_name entries to keep the output schema stable.
        by_name = {}
        for it in obj["results"]:
            if not isinstance(it, dict):
                continue
            name = it.get("metric_name")
            if name in METRICS:
                by_name[name] = it
        for row in out["results"]:
            name = row["metric_name"]
            if name in by_name:
                row.update(by_name[name])
        return out

    # Case B: Flat dict output without 'results' (e.g., {metric_alias: value}).
    if isinstance(obj, dict):
        keymap = {
            # Common synonym mappings and non-standard key normalization
            "Return on Average Assets": "ROA",
            "Return on Assets": "ROA",
            "ROAA": "ROA",
            "Return on Average Equity": "ROE",
            "Return on Equity": "ROE",
            "ROAE": "ROE",
            "Net Interest Margin": "NIM",
            "NIM": "NIM",
            "Net Interest Income": "NII",
            "NII": "NII",
            "Provision for Credit Losses": "Provision for Credit Losses",
            "Provision for credit losses": "Provision for Credit Losses",
        }

        # Fill the template rows with values from a flat dict output (unit/source_chunk_id may remain NOT FOUND).
        for k, v in obj.items():
            if k in keymap:
                metric = keymap[k]
                for row in out["results"]:
                    if row["metric_name"] == metric:
                        row["value"] = str(v)
                        # If there is no supporting evidence for unit/source_chunk_id, keep them as 'NOT FOUND'.
                        break
        return out

    # Case C: Unsupported output shape -> return the empty template (with post-normalization applied).
    
    # --- value/unit cleanup: e.g. "1.99%" + unit="%" -> value="1.99" ---
    for row in out.get("results", []):
        if not isinstance(row, dict):
            continue
        v = row.get("value")
        u = row.get("unit")
        if isinstance(v, str):
            vs = v.strip()
            if vs.endswith("%"):
                v2 = vs[:-1].strip()
                if re.fullmatch(r"[-+]?\d+(\.\d+)?", v2):
                    row["value"] = v2
                    if (u is None) or (str(u).strip().upper() in ("", "NOT FOUND")):
                        row["unit"] = "%"
            # also normalize unit if model returns "not applicable"
            if isinstance(row.get("unit"), str) and row["unit"].strip().lower() in ("not applicable", "n/a"):
                row["unit"] = "NOT FOUND"

    # Normalize and validate source citations (accept bracket-wrapped cite keys; drop invalid ids to 'NOT FOUND').
    for row in out["results"]:
        cid = row.get("source_chunk_id", "")
        cid = (cid or "").strip()
        # Allow bracket-wrapped cite IDs and strip surrounding brackets
        # Accept cite keys like "[k=...|stem=...|chunk=...]" or numeric chunk ids; validate via _valid_cite().
        if cid.startswith("[") and cid.endswith("]"):
            cid = cid[1:-1].strip()

        if not _valid_cite(cid):
            row["source_chunk_id"] = "NOT FOUND"
        else:
            row["source_chunk_id"] = cid
    
    # Post-pass: normalize percent tokens/unit synonyms after schema mapping (kept for robustness against mixed model outputs).
    for row in out.get("results", []):
        v = str(row.get("value", "")).strip()
        u = str(row.get("unit", "")).strip()

        if v.endswith("%"):
            vv = v[:-1].strip()
            if vv:
                row["value"] = vv
            if (not u) or u == "NOT FOUND":
                row["unit"] = "%"
            # if unit already %, keep

        # common: unit contains 'percent'
        if u.lower() in ["percent", "percentage"]:
            row["unit"] = "%"

    # ---- end clean ----

    return out
