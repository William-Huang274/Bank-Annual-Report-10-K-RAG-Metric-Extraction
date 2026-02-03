from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional

import requests
from src.rag.schema import make_template, parse_json_loose, parse_or_fallback
from src.rag.prompts import METRICS, make_prompt_one_metric, make_repair_prompt_one_metric


def _valid_cite(cid: str) -> bool:
    if not isinstance(cid, str):
        return False
    cid = cid.strip().strip("[]")
    if not cid:
        return False
    if re.match(r"^k=.+\|stem=.+\|chunk=\d+$", cid):
        return True
    return ("chunk=" in cid) or cid.isdigit()


def _make_prompt_one_metric(metric: str, context: str, year: int = 2024) -> str:
    # Delegate to the original 06patch prompt (schema-enforcing, head/tail truncation inside prompts module)
    return make_prompt_one_metric(metric, context, year)


def _ollama_generate(url: str, model: str, prompt: str, temperature: float, seed: Optional[int], timeout_s: int) -> str:
    payload = {
        "model": model,
        "prompt": prompt,
        "system": "You are a financial information extraction assistant. no think.",
        "stream": False,
        "format": "json",
        "keep_alive": "30m",
        "options": {
            "temperature": temperature,
            "num_ctx": 8192,
            "num_predict": 512,
            "stop": ["\nQ (empty to exit):"],
        },
    }
    if seed is not None:
        payload["options"]["seed"] = seed

    r = requests.post(url, json=payload, timeout=timeout_s)
    debug_text = r.text or ""
    if r.status_code != 200:
        raise RuntimeError(f"Ollama HTTP {r.status_code}: {debug_text[:500]}")

    j = r.json()
    resp = (j.get("response") or "").strip()
    think = (j.get("thinking") or "").strip()
    msg = j.get("message") or {}
    content = (msg.get("content") or "").strip() if isinstance(msg, dict) else ""
    out = resp or think or content
    return out


def _parse_json_loose(s: str):
    # Reuse schema.parse_json_loose (06patch equivalent)
    return parse_json_loose(s)


def _parse_or_fallback(raw_text: str, year: int = 2024):
    # Reuse schema.parse_or_fallback (06patch equivalent)
    return parse_or_fallback(raw_text, year)


def _is_results_schema(obj) -> bool:
    if not isinstance(obj, dict):
        return False
    rs = obj.get("results")
    if not isinstance(rs, list):
        return False
    for it in rs:
        if isinstance(it, dict) and ("metric_name" in it) and ("value" in it):
            return True
    return False


def _normalize_extraction(obj, metric: str, year: int):
    """
    Restore the 06patch.py normalize_extraction logic (schema-level) and pick the requested metric row.
    """
    out = make_template(year)
    
    # Compatibility patch: accept flat extraction outputs without an explicit 'results' schema
    if isinstance(obj, dict) and ("value" in obj) and ("source_chunk_id" in obj) and ("results" not in obj):
        metric_flat = metric or "NII"

        val = str(obj.get("value", "")).strip()
        cid = str(obj.get("source_chunk_id", "")).strip()
        fy  = obj.get("fiscal_year", year)

        for row in out["results"]:
            if row.get("metric_name") == metric_flat:
                row["value"] = val if val else "NOT FOUND"
                row["unit"] = obj.get("unit", "NOT FOUND")
                row["fiscal_year"] = int(fy) if str(fy).isdigit() else int(year)
                row["source_chunk_id"] = cid if cid else "NOT FOUND"
                break
        # return single metric row below

    # Case A: Already in the expected {'results': [...]} schema.
    elif isinstance(obj, dict) and isinstance(obj.get("results"), list) and len(obj["results"]) > 0:
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

    # Case B: Model returned a dict without 'results'
    elif isinstance(obj, dict):
        keymap = {
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
        for k, v in obj.items():
            if k in keymap:
                m = keymap[k]
                for row in out["results"]:
                    if row["metric_name"] == m:
                        row["value"] = str(v)
                        break

    # Case C: unexpected / unsupported output shape -> keep template defaults

    # value/unit cleanup (percent)
    for row in out["results"]:
        v = str(row.get("value", "")).strip()
        u = str(row.get("unit", "")).strip()
        if v.endswith("%"):
            vv = v[:-1].strip()
            if vv:
                row["value"] = vv
            if (not u) or u == "NOT FOUND":
                row["unit"] = "%"
        if u.lower() in ["percent", "percentage"]:
            row["unit"] = "%"

    # return only the requested metric row
    for row in out["results"]:
        if row.get("metric_name") == metric:
            return row
    return {
        "metric_name": metric,
        "value": "NOT FOUND",
        "unit": "NOT FOUND",
        "fiscal_year": int(year),
        "source_chunk_id": "NOT FOUND",
    }


def _clean_citation(cid: str) -> str:
    cid = (cid or "").strip()
    if cid.startswith("[") and cid.endswith("]"):
        cid = cid[1:-1].strip()
    if not _valid_cite(cid):
        return "NOT FOUND"
    return cid


def call_llm_for_metric(
    *,
    metric_name: str,
    bank: str,
    year: int,
    context_text: str,
    model_name: str,
    ollama_base_url: str,
    temperature: float,
    seed: Optional[int],
    timeout_s: int,
    debug_dir: Optional[Path] = None,
    repair: bool = True,
    logger=None,
) -> Optional[dict]:
    prompt = _make_prompt_one_metric(metric_name, context_text, year=int(year))
    if logger:
        logger.debug("prompt(%s) length=%s", metric_name, len(prompt))
    try:
        raw = _ollama_generate(ollama_base_url, model_name, prompt, temperature, seed, timeout_s)
    except Exception as e:
        print(f"[WARN] ollama error bank={bank} metric={metric_name} -> {repr(e)}", flush=True)
        return None

    if debug_dir:
        (debug_dir / f"{bank}_{year}_raw_{metric_name}.txt").write_text(raw if raw is not None else "", encoding="utf-8", errors="ignore")

    if not (raw or "").strip():
        return None

    obj = _parse_or_fallback(raw, year=int(year))
    if isinstance(obj, dict) and ("results" not in obj) and ("value" in obj) and ("source_chunk_id" in obj):
        obj = {
            "results": [{
                "metric_name": metric_name,
                "value": obj.get("value", "NOT FOUND"),
                "unit": obj.get("unit", "NOT FOUND"),
                "fiscal_year": int(obj.get("fiscal_year", year)),
                "source_chunk_id": obj.get("source_chunk_id", "NOT FOUND"),
            }]
        }

    if repair and isinstance(obj, dict) and (not _is_results_schema(obj)):
        repair_prompt = make_repair_prompt_one_metric(metric_name, raw, year=int(year))
        try:
            raw2 = _ollama_generate(ollama_base_url, model_name, repair_prompt, temperature, seed, timeout_s)
            if debug_dir:
                (debug_dir / f"{bank}_{year}_raw_repair_{metric_name}.txt").write_text(raw2 if raw2 is not None else "", encoding="utf-8", errors="ignore")
            obj = _parse_or_fallback(raw2, year=int(year))
        except Exception as e:
            print(f"[WARN] repair failed bank={bank} metric={metric_name} -> {repr(e)}", flush=True)

    norm = _normalize_extraction(obj, metric_name, int(year))

    norm["source_chunk_id"] = _clean_citation(norm.get("source_chunk_id", ""))
    norm["fiscal_year"] = int(norm.get("fiscal_year", year))
    norm["metric_name"] = metric_name
    return norm
