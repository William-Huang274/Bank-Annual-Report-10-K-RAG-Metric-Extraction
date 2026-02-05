from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Optional

import requests
from src.rag.schema import make_template, parse_json_loose, parse_or_fallback
from src.rag.prompts import METRICS, make_prompt_one_metric, make_repair_prompt_one_metric


# def _valid_cite(cid: str) -> bool:
#     if not isinstance(cid, str):
#         return False
#     cid = cid.strip().strip("[]")
#     if not cid:
#         return False
#     if re.match(r"^k=.+\|stem=.+\|chunk=\d+$", cid):
#         return True
#     return ("chunk=" in cid) or cid.isdigit()


def _valid_cite(cid: str) -> bool:
    if not isinstance(cid, str):
        return False
    cid = cid.strip().strip("[]")
    if not cid:
        return False

    # allow optional "llm:" prefix for attribution
    if cid.startswith("llm:"):
        cid2 = cid[len("llm:"):].strip()
    else:
        cid2 = cid

    if re.match(r"^k=.+\|stem=.+\|chunk=\d+$", cid2):
        return True
    return ("chunk=" in cid2) or cid2.isdigit()

def _make_prompt_one_metric(metric: str, context: str, year: int = 2024) -> str:
    # Delegate to the original 06patch prompt (schema-enforcing, head/tail truncation inside prompts module)
    return make_prompt_one_metric(metric, context, year)


def _ollama_generate(
    url: str,
    model: str,
    prompt: str,
    temperature: float,
    seed: Optional[int],
    timeout_s: int,
    num_predict: int = 512,
) -> str:
    try:
        npred = int(num_predict)
    except Exception:
        npred = 512
    if npred <= 0:
        npred = 512

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
            "num_predict": npred,
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

    # clean citation first
    norm = _normalize_extraction(obj, metric_name, int(year))

    norm["source_chunk_id"] = _clean_citation(norm.get("source_chunk_id", ""))
    # tag attribution for participation analysis
    cid = str(norm.get("source_chunk_id", "")).strip()
    if cid and cid != "NOT FOUND" and not cid.startswith("llm:"):
        norm["source_chunk_id"] = "llm:" + cid

    norm["fiscal_year"] = int(norm.get("fiscal_year", year))
    norm["metric_name"] = metric_name
    return norm

def judge_select_candidate(
    *,
    metric_name: str,
    bank: str,
    year: int,
    candidates: list[dict],
    model_name: str,
    ollama_base_url: str,
    temperature: float = 0.0,
    seed: Optional[int] = None,
    timeout_s: int = 60,
    debug_dir: Optional[Path] = None,
    logger=None,
) -> int:
    """
    Return selected candidate index [0..n-1], or -1 if none should be accepted.
    IMPORTANT: judge only selects from candidates; it must not invent new values.
    """
    metric = (metric_name or "").strip()
    if metric not in ("ROA", "ROE"):
        return -1
    if not candidates:
        return -1

    # Keep prompt compact
    items = []
    for i, c in enumerate(candidates[:12]):  # cap
        cid = c.get("source_chunk_id", "NOT FOUND")
        val = c.get("value", "NOT FOUND")
        lab = (c.get("label") or "").strip().replace("\n", " ")
        snip = (c.get("snippet") or "").strip()
        snip = re.sub(r"\s+", " ", snip)[:320]
        items.append(f"{i}) value={val}% cid={cid} label={lab} snippet={snip}")

    prompt = (
        "You are a strict selection judge.\n"
        "Return ONLY JSON. No extra text.\n"
        f"Task: choose the best {metric} value for fiscal year {int(year)}.\n"
        "Rules:\n"
        "1) You MUST select from the provided candidates only.\n"
        "2) If none clearly matches the metric (e.g., wrong definition like ROTCE/tangible/core/non-GAAP), return -1.\n"
        "3) Prefer explicit 'Return on average assets/equity' wording.\n"
        "Output JSON (exact): {\"selected_index\": -1}\n\n"
        "Candidates:\n"
        + "\n".join(items)
        + "\n"
    )

    if logger:
        logger.debug("[JUDGE] bank=%s metric=%s n=%s prompt_len=%s", bank, metric, len(candidates), len(prompt))

    try:
        raw = _ollama_generate(ollama_base_url, model_name, prompt, temperature, seed, timeout_s)
    except Exception as e:
        if logger:
            logger.warning("[JUDGE] ollama error bank=%s metric=%s -> %r", bank, metric, e)
        return -1

    if debug_dir:
        (debug_dir / f"{bank}_{year}_judge_raw_{metric}.txt").write_text(raw or "", encoding="utf-8", errors="ignore")

    try:
        obj = _parse_json_loose(raw or "")
        idx = obj.get("selected_index", -1) if isinstance(obj, dict) else -1
        idx = int(idx)
    except Exception:
        return -1

    if idx < 0 or idx >= min(len(candidates), 12):
        return -1
    return idx


def _norm_cid_for_match(cid: str) -> str:
    c = str(cid or "").strip()
    if c.startswith("llm:"):
        c = c[len("llm:"):].strip()
    if c.startswith("[") and c.endswith("]"):
        c = c[1:-1].strip()
    return c


def _format_hit_cid(hit: dict, default_bank: str = "") -> str:
    bank = hit.get("bank") or hit.get("bank_folder") or default_bank or ""
    stem = hit.get("stem") or ""
    chunk = hit.get("chunk_id")
    if chunk is None:
        chunk = hit.get("chunk")
    if chunk is None:
        return "NOT FOUND"
    return f"[k={bank}|stem={stem}|chunk={chunk}]"


def _score_of_hit(hit: dict) -> float:
    try:
        return float(hit.get("score", 0.0))
    except Exception:
        return 0.0


def _collect_review_hits(metric: str, metric_contexts: dict) -> list[dict]:
    hits = list((((metric_contexts or {}).get(metric) or {}).get("hits", []) or []))
    if metric in ("ROA", "ROE"):
        peer = "ROE" if metric == "ROA" else "ROA"
        hits.extend((((metric_contexts or {}).get(peer) or {}).get("hits", []) or []))

    best = {}
    for h in hits:
        cidn = _norm_cid_for_match(_format_hit_cid(h))
        if not cidn:
            continue
        prev = best.get(cidn)
        if (prev is None) or (_score_of_hit(h) > _score_of_hit(prev)):
            best[cidn] = h

    out = sorted(best.values(), key=_score_of_hit, reverse=True)
    return out


def _cid_to_meta_key(cid: str):
    c = _norm_cid_for_match(cid)
    m = re.match(r"^k=([^|]+)\|stem=([^|]+)\|chunk=(\d+)$", c)
    if not m:
        return None
    return (m.group(1), m.group(2), int(m.group(3)))


def _render_cid(cid: str) -> str:
    c = _norm_cid_for_match(cid)
    if c.startswith("k="):
        return f"[{c}]"
    return c or "NOT FOUND"


def _metric_anchor_keywords(metric: str, year: int) -> list[str]:
    y = str(int(year))
    if metric == "ROA":
        return [
            "return on average total assets",
            "return on average assets",
            "return on assets",
            f"roa {y}",
        ]
    if metric == "ROE":
        return [
            "return on average common shareholders",
            "return on average shareholders",
            "return on average equity",
            "return on equity",
            f"roe {y}",
        ]
    if metric == "NII":
        return [
            f"net interest income for {y}",
            "net interest income",
            "nii",
        ]
    if metric == "NIM":
        return [
            "net interest margin",
            "nim",
        ]
    if metric == "Provision for Credit Losses":
        return [
            "provision for credit losses",
            "provision for loan losses",
        ]
    return [str(int(year))]


def _anchored_evidence_text(metric: str, text: str, year: int, max_chars: int = 420) -> str:
    """
    Build evidence text by anchoring around metric keywords, then expanding a local window.
    This avoids clipping away the metric label when it appears late in the chunk.
    """
    txt = re.sub(r"\s+", " ", str(text or "")).strip()
    if not txt:
        return ""

    low = txt.lower()
    anchors = _metric_anchor_keywords(metric, year)

    best_idx = -1
    for k in anchors:
        i = low.find(str(k).lower())
        if i >= 0 and (best_idx < 0 or i < best_idx):
            best_idx = i

    if best_idx < 0:
        y = str(int(year))
        i = low.find(y)
        if i >= 0:
            best_idx = i

    if best_idx < 0:
        return txt[:max_chars]

    left = max(0, best_idx - 120)
    right = min(len(txt), left + max_chars)
    return txt[left:right]


def review_all_metrics_after_extract(
    *,
    bank: str,
    year: int,
    final_obj: dict,
    metric_contexts: dict,
    meta_rows: Optional[list[dict]] = None,
    model_name: str,
    ollama_base_url: str,
    temperature: float = 0.0,
    seed: Optional[int] = None,
    timeout_s: int = 120,
    num_predict: int = 768,
    metrics_to_review: Optional[list[str]] = None,
    debug_dir: Optional[Path] = None,
    logger=None,
) -> dict:
    """
    One-shot LLM audit over all 5 metrics after extraction.
    Returns:
      {
        "ok": bool,
        "decisions": {metric_name -> {action, value, unit, source_chunk_id, reason}},
        "raw": str
      }
    """
    results = final_obj.get("results", []) if isinstance(final_obj, dict) else []
    row_by_metric = {}
    for r in results:
        if isinstance(r, dict) and r.get("metric_name") in METRICS:
            row_by_metric[r.get("metric_name")] = r

    metrics = METRICS
    if metrics_to_review:
        allow = {str(x).strip() for x in metrics_to_review if str(x).strip()}
        metrics = [m for m in METRICS if m in allow]
    if not metrics:
        return {"ok": True, "decisions": {}, "raw": ""}

    decisions = {}
    allowed_cids = {}
    prompt_parts = []
    meta_text = {}
    if isinstance(meta_rows, list):
        for mm in meta_rows:
            try:
                b = mm.get("bank_folder")
                s = mm.get("stem")
                c = int(mm.get("chunk_id"))
                if not b or not s:
                    continue
                # Keep only target bank to cap lookup size.
                if str(b).strip() != str(bank).strip():
                    continue
                meta_text[(str(b), str(s), int(c))] = str(mm.get("text") or "")
            except Exception:
                continue

    for metric in metrics:
        row = row_by_metric.get(metric, {})
        cur_val = str(row.get("value", "NOT FOUND"))
        cur_unit = str(row.get("unit", "NOT FOUND"))
        cur_cid = str(row.get("source_chunk_id", "NOT FOUND"))

        hits = _collect_review_hits(metric, metric_contexts)
        hit_by_cid = {}
        for h in hits:
            cidn_h = _norm_cid_for_match(_format_hit_cid(h, default_bank=bank))
            if cidn_h and (cidn_h not in hit_by_cid):
                hit_by_cid[cidn_h] = h

        ev_lines = []
        ev_seen = set()
        metric_allowed = set()

        # Keep the current citation as an allowed option so "keep" can be represented safely.
        if cur_cid and cur_cid != "NOT FOUND":
            cur_cidn = _norm_cid_for_match(cur_cid)
            metric_allowed.add(cur_cidn)

            # Force include current citation evidence, even if it is not in the metric top hits.
            cur_txt = ""
            cur_score = ""
            h0 = hit_by_cid.get(cur_cidn)
            if h0 is not None:
                cur_txt = str(h0.get("text") or "")
                cur_score = str(h0.get("score", ""))
            else:
                mk = _cid_to_meta_key(cur_cidn)
                if mk is not None:
                    cur_txt = meta_text.get(mk, "")
            cur_txt = _anchored_evidence_text(metric, cur_txt, int(year))
            ev_lines.append(f"- cid={_render_cid(cur_cidn)} score={cur_score} text={cur_txt}")
            ev_seen.add(cur_cidn)

        for h in hits[:10]:
            cid = _format_hit_cid(h, default_bank=bank)
            cidn = _norm_cid_for_match(cid)
            if not cidn:
                continue
            metric_allowed.add(cidn)
            if cidn in ev_seen:
                continue
            txt = _anchored_evidence_text(metric, str(h.get("text") or ""), int(year))
            score = h.get("score", "")
            ev_lines.append(f"- cid={cid} score={score} text={txt}")
            ev_seen.add(cidn)
            if len(ev_lines) >= 5:
                break

        allowed_cids[metric] = metric_allowed
        if not ev_lines:
            ev_lines = ["- cid=NOT FOUND score= text="]

        prompt_parts.append(
            f"Metric: {metric}\n"
            f"Current: value={cur_val} unit={cur_unit} source_chunk_id={cur_cid}\n"
            f"Evidence:\n" + "\n".join(ev_lines)
        )

        decisions[metric] = {
            "action": "keep",
            "value": cur_val,
            "unit": cur_unit,
            "source_chunk_id": cur_cid,
            "reason": "default_keep",
        }

    prompt = (
        "You are a strict reviewer for financial metric extraction.\n"
        "Review the listed metrics for fiscal year "
        f"{int(year)} and decide whether to keep, replace, or reject each value.\n"
        "Rules:\n"
        "1) Action must be one of: keep, replace, reject.\n"
        "2) For replace, choose a better value supported by evidence.\n"
        "3) For replace, source_chunk_id MUST be one of the evidence cids for that metric.\n"
        "4) For reject, set value/unit/source_chunk_id to NOT FOUND.\n"
        "5) Do not invent metrics or citations.\n"
        "Output JSON only in this exact shape:\n"
        "{\n"
        "  \"reviews\": [\n"
        "    {\"metric_name\":\"ROA\",\"action\":\"keep\",\"value\":\"...\",\"unit\":\"...\",\"source_chunk_id\":\"...\",\"reason\":\"...\"}\n"
        "  ]\n"
        "}\n\n"
        "Review payload:\n"
        + "\n\n".join(prompt_parts)
    )

    if logger:
        logger.debug("[REVIEW_ALL] bank=%s prompt_len=%s", bank, len(prompt))

    raw = ""
    review_num_predict = num_predict
    if review_num_predict is None:
        try:
            review_num_predict = int(os.getenv("LLM_REVIEW_NUM_PREDICT", "768"))
        except Exception:
            review_num_predict = 768
    try:
        review_num_predict = int(review_num_predict)
    except Exception:
        review_num_predict = 768
    if review_num_predict <= 0:
        review_num_predict = 768

    try:
        raw = _ollama_generate(
            ollama_base_url,
            model_name,
            prompt,
            temperature,
            seed,
            timeout_s,
            num_predict=review_num_predict,
        )
    except Exception as e:
        if logger:
            logger.warning("[REVIEW_ALL] ollama error bank=%s -> %r", bank, e)
        return {"ok": False, "decisions": decisions, "raw": raw}

    if debug_dir:
        (debug_dir / f"{bank}_{year}_review_raw_all_metrics.txt").write_text(raw or "", encoding="utf-8", errors="ignore")

    try:
        obj = _parse_json_loose(raw or "")
    except Exception:
        return {"ok": False, "decisions": decisions, "raw": raw}

    revs = obj.get("reviews") if isinstance(obj, dict) else None
    if not isinstance(revs, list):
        return {"ok": False, "decisions": decisions, "raw": raw}

    for it in revs:
        if not isinstance(it, dict):
            continue
        metric = str(it.get("metric_name", "")).strip()
        if metric not in decisions:
            continue
        action = str(it.get("action", "")).strip().lower()
        reason = str(it.get("reason", "")).strip()[:240]

        if action == "keep":
            decisions[metric]["action"] = "keep"
            decisions[metric]["reason"] = reason or "review_keep"
            continue

        if action == "reject":
            decisions[metric] = {
                "action": "reject",
                "value": "NOT FOUND",
                "unit": "NOT FOUND",
                "source_chunk_id": "NOT FOUND",
                "reason": reason or "review_reject",
            }
            continue

        if action == "replace":
            val = str(it.get("value", "")).strip()
            unit = str(it.get("unit", "")).strip() or "NOT FOUND"
            cid_raw = str(it.get("source_chunk_id", "")).strip()
            cid_clean = _clean_citation(cid_raw)
            cid_norm = _norm_cid_for_match(cid_clean)

            if (not val) or (val.upper() == "NOT FOUND"):
                continue
            if (not cid_norm) or (cid_norm not in allowed_cids.get(metric, set())):
                continue

            decisions[metric] = {
                "action": "replace",
                "value": val,
                "unit": unit,
                "source_chunk_id": cid_clean,
                "reason": reason or "review_replace",
            }

    return {"ok": True, "decisions": decisions, "raw": raw}
