"""
Extract financial metrics (NII, NIM, ROA, ROE, Provision for Credit Losses) from bank annual reports / 10-K.

Workflow:
- Load FAISS index + meta.jsonl built from OCR text chunks
- Retrieve evidence chunks (multi-query + optional neighbor expansion)
- Build an evidence context string for LLM extraction
- Parse/normalize model output into a stable tabular schema and write CSV

Notes:
- This script assumes the index and meta files already exist under data/interim/index/.
- Ollama must be running locally when LLM extraction is enabled.
"""
# Collect non-missing extractions for a lightweight QA/audit export (not a primary output artifact).
AUDIT_ROWS = []
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# Constrain CPU threads early (must be set before importing FAISS / BLAS) to reduce oversubscription and improve stability.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import csv
import json
import re
import logging
import traceback
from pathlib import Path
from collections import Counter
import shutil
import json, traceback, sys
import numpy as np
import faiss
try:
    faiss.omp_set_num_threads(1)
except Exception:
    pass
import requests
from sentence_transformers import SentenceTransformer
import sys
from src.rag.packing import build_context
from src.rag.regex_extractors import mine_ratio_candidates_from_hits
from src.rag.llm_extract import judge_select_candidate, review_all_metrics_after_extract


from src.rag.api import (
    load_meta,
    get_repo_root,
    default_paths,
    parse_batch_command,
    read_bank_list,
    resolve_batch_path,
    retrieve_and_build_context_for_bank,
    regex_prefill_from_contexts,
    apply_table_sidecar_prefill,
    call_llm_for_metric,
    flatten_metrics,
    merge_keep_existing,
    write_metrics_csv,
    write_jsonl,
    patch_metrics_csv_from_table_sidecar,
    make_template,
    augment_context_with_avg_balances,
)
from src.rag.table_sidecar import load_sidecar_index_multi
from src.rag.llm_extract import _valid_cite


# Repository root inferred from this script location (keeps paths portable across machines).
ROOT = get_repo_root(__file__)

# ----------------------------
# Logging (default: INFO; set LOG_LEVEL=DEBUG to enable verbose debug logs)
# ----------------------------
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger("extract")

try:
    dprint  # type: ignore
except NameError:
    def dprint(*args, **kwargs):
        return

def _setup_logger():
    level = os.getenv("LOG_LEVEL", "INFO").upper()  # DEBUG/INFO/WARNING/ERROR
    logger = logging.getLogger("extract")
    if logger.handlers:
        return logger
    logger.setLevel(level)
    h = logging.StreamHandler()
    h.setLevel(level)
    fmt = logging.Formatter("[%(levelname)s] %(message)s")
    h.setFormatter(fmt)
    logger.addHandler(h)
    logger.propagate = False
    return logger

LOGGER = _setup_logger()
DBG = LOGGER.isEnabledFor(logging.DEBUG)

def dprint(*args, **kwargs):
    if DBG:
        kwargs.setdefault("flush", True)
        print(*args, **kwargs)


# Target fiscal year (string for path building; cast to int only when writing schema fields).
YEAR = "2024"

# To use sample index, please change INDEX_DIR = ROOT / "data" / "sample" / "index" / f"faiss_{YEAR}_sample"
INDEX_DIR = ROOT / "data" / "interim" / "index" / f"faiss_{YEAR}_full"
INDEX_PATH = INDEX_DIR / "faiss.index"
META_PATH  = INDEX_DIR / "meta.jsonl"

EMB_MODEL = "BAAI/bge-m3"
EMB_DEVICE = "cuda"

# Retrieval budget: TOPK_SEARCH from the global index -> filter by bank -> keep TOPK_FINAL for evidence/context.
# Note: TOPK is legacy/interactive setting; keep only if used by a CLI path.
TOPK_SEARCH = 50   # Initial retrieval size from the full index (before bank filtering)
TOPK_FINAL  = 20   # Final number of chunks kept after filtering by bank

TOPK = 10

# Ollama endpoint + model. Keep model name aligned with `ollama list` on the target machine.
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "qwen3:8b"   # Model name must match a local Ollama model (see `ollama list`)
TEMPERATURE = 0.0

# Guardrails: large prompts increase schema drift on small local models; cap context to reduce variance.
MAX_PROMPT_CHARS = 20000
MAX_CONTEXT_CHARS_PER_METRIC = 8000



import re
from datetime import datetime

# Default multi-metric query used to build a shared evidence pool (improves recall when per-metric hits are sparse).
DEFAULT_RETRIEVAL_QUERY = "FY2024 ROA ROE NIM NII net interest income net interest margin return on assets return on equity provision for credit losses"

import re, json


METRICS = ["ROA", "ROE", "NIM", "NII", "Provision for Credit Losses"]

import re

def extract_for_bank(index, meta, emb, bank_id=None, year=YEAR, retrieval_query=DEFAULT_RETRIEVAL_QUERY, side_idx=None, table_prefill_allow=None):
    """
    Run metric extraction for a single bank and fiscal year.
    Pipeline: retrieve evidence -> build context -> regex prefill (where applicable) -> per-metric LLM extraction ->
    schema repair/normalization -> merge into final results dict.
    """
    print(f"[EXTRACT] start bank={bank_id} year={year}", flush=True)
    target_bank = bank_id  # The batch input bank_id is the target bank identifier.

    debug_dir = ROOT / "data" / "outputs" / "debug"

    retrieval = retrieve_and_build_context_for_bank(
        index=index,
        meta=meta,
        emb=emb,
        bank_id=target_bank,
        year=int(year),
        metrics=METRICS,
        topk_final=TOPK_FINAL,
        max_context_chars_per_metric=MAX_CONTEXT_CHARS_PER_METRIC,
        debug_dir=debug_dir,
        dprint=dprint,
        logger=logger,
    )

    hits = retrieval["hits"]
    context = retrieval["context"]
    nim_ctx = retrieval["nim_ctx"]
    pcl_ctx = retrieval["pcl_ctx"]
    nii_ctx = retrieval["nii_ctx"]
    hits_by_metric = retrieval["hits_by_metric"]
    metric_contexts = retrieval["metric_contexts"]

    print(f"[EXTRACT] hits={len(hits)}", flush=True)

    if not hits:
        obj = make_template(int(year))
        obj["_meta"] = {
            "bank": target_bank,
            "year": int(year),
            "retrieval_query": "MULTIQUERY(FY{year}: ROA/ROE/NIM/NII/PCL)",
            "topk": len(hits),
            "generated_at": datetime.now().isoformat(timespec="seconds"),
        }
        obj["error"] = "NO_HITS"
        return obj

    # Extraction order (cheap -> expensive):
    # 1) regex prefills (high precision), 2) table-sidecar backfill, 3) per-metric regex on capped context, 4) LLM as last resort + schema repair.
    prefill = {}  # metric_name -> {value/unit/source_chunk_id}
    try:
        from src.rag.regex_extractors import (
            try_regex_extract_nii_from_context,
            try_regex_extract_nim_from_context,
            try_regex_extract_roa_roe_from_context,
            try_regex_extract_pcl_from_context,
        )
    except Exception:
        try_regex_extract_nii_from_context = try_regex_extract_nim_from_context = None
        try_regex_extract_roa_roe_from_context = try_regex_extract_pcl_from_context = None

    # NII first-pass regex (global/per-metric context)
    if try_regex_extract_nii_from_context:
        _nii_ctx = nii_ctx if "nii_ctx" in locals() and nii_ctx else context
        got = try_regex_extract_nii_from_context(_nii_ctx)
        if got:
            val, unit, cid = got
            prefill["NII"] = {
                "metric_name": "NII",
                "value": val,
                "unit": unit,
                "fiscal_year": int(year),
                "source_chunk_id": cid,
            }
            print("[EXTRACT] regex prefill NII ok, continue to LLM for other metrics", flush=True)

    # ROA / ROE regex prefill (global context)
    if try_regex_extract_roa_roe_from_context:
        got_rr = try_regex_extract_roa_roe_from_context(context, year=int(year))
        if got_rr:
            for mn in ("ROA", "ROE"):
                if mn in got_rr and mn not in prefill:
                    v, u, cid = got_rr[mn]
                    prefill[mn] = {
                        "metric_name": mn,
                        "value": v,
                        "unit": u,
                        "fiscal_year": int(year),
                        "source_chunk_id": cid,
                    }
            if ("ROA" in prefill) or ("ROE" in prefill):
                print("[EXTRACT] regex prefill ROA/ROE ok, continue to LLM for other metrics", flush=True)

    # PCL regex prefill
    if try_regex_extract_pcl_from_context and ("Provision for Credit Losses" in METRICS):
        _pcl_ctx = pcl_ctx if "pcl_ctx" in locals() and pcl_ctx else context
        got = try_regex_extract_pcl_from_context(_pcl_ctx, year=int(year))
        if got:
            if isinstance(got, dict):
                val = got.get("value", "NOT FOUND")
                unit = got.get("unit", "NOT FOUND")
                cid = got.get("source_chunk_id", "NOT FOUND")
            else:
                val, unit, cid = got
            prefill["Provision for Credit Losses"] = {
                "metric_name": "Provision for Credit Losses",
                "value": val,
                "unit": unit,
                "fiscal_year": int(year),
                "source_chunk_id": cid,
            }
            print("[EXTRACT] regex prefill PCL ok, continue to LLM for other metrics", flush=True)

    # NIM regex prefill
    if try_regex_extract_nim_from_context and ("NIM" in METRICS):
        _nim_ctx = nim_ctx if "nim_ctx" in locals() and nim_ctx else context
        got = try_regex_extract_nim_from_context(_nim_ctx)
        if got:
            if isinstance(got, dict):
                got = (got.get("value"), got.get("unit"), got.get("source_chunk_id"))
            val, unit, cid = got
            prefill["NIM"] = {
                "metric_name": "NIM",
                "value": val,
                "unit": unit,
                "fiscal_year": int(year),
                "source_chunk_id": cid,
            }
            print("[EXTRACT] regex prefill NIM ok, continue to LLM for other metrics", flush=True)

    for metric in ["ROA", "ROE", "NII", "NIM", "Provision for Credit Losses"]:
        cur = prefill.get(metric)
        print(f"[PREFILL_TRY] bank={target_bank} metric={metric}", flush=True)
        if cur:
            print(f"[PREFILL_HIT] bank={target_bank} metric={metric} val={cur.get('value')} unit={cur.get('unit')} cid={cur.get('source_chunk_id')}", flush=True)
        else:
            print(f"[PREFILL_MISS] bank={target_bank} metric={metric}", flush=True)

    # 2) derived/compute (cheap) BEFORE table prefill/LLM.
    final = make_template(int(year))
    
    def _is_nf(v) -> bool:
        v = ("" if v is None else str(v)).strip().upper()
        return v in ("", "NOT FOUND", "NOT_FOUND", "N/A", "NA")

    def _get_final_row(final_obj: dict, metric_name: str):
        for r in final_obj.get("results", []):
            if r.get("metric_name") == metric_name:
                return r
        return None


    def _write_if_missing(final_obj: dict, metric_name: str, value: str, unit: str, source_chunk_id: str) -> bool:
        row = _get_final_row(final_obj, metric_name)
        if row is None:
            return False
        if not _is_nf(row.get("value")):
            return False  # already filled (by any earlier step), do not overwrite
        row["value"] = value
        row["unit"] = unit
        row["source_chunk_id"] = source_chunk_id
        return True 

    def sanity_check(metric: str, val, unit=None) -> bool:
        try:
            s = str(val).strip()
            s = s.replace(",", "").replace("%", "").replace("$", "")
            if s.startswith("(") and s.endswith(")"):
                s = "-" + s[1:-1]
            x = float(s)
        except Exception:
            return False   
        if metric == "ROA":
            return 0 <= x <= 3.0
        if metric == "NIM":
            return 0 <= x <= 6.0
        if metric == "ROE":
            return 0 <= x <= 40.0
        return True

    def _to_float_safe(v):
        try:
            s = str(v).strip()
            if not s:
                return None
            s = s.replace(",", "").replace("%", "").replace("$", "")
            if s.startswith("(") and s.endswith(")"):
                s = "-" + s[1:-1]
            return float(s)
        except Exception:
            return None

    def _norm_cid(c):
        c = str(c or "").strip()
        if c.startswith("llm:"):
            c = c[len("llm:"):].strip()
        if c.startswith("[") and c.endswith("]"):
            c = c[1:-1].strip()
        return c

    def _hit_cid_norm(h):
        bank_h = h.get("bank") or h.get("bank_folder") or target_bank or ""
        stem_h = h.get("stem") or ""
        chunk_h = h.get("chunk_id")
        if chunk_h is None:
            chunk_h = h.get("chunk")
        if chunk_h is None:
            return ""
        return f"k={bank_h}|stem={stem_h}|chunk={chunk_h}"

    def _score(h):
        try:
            return float(h.get("score"))
        except Exception:
            return None

    def _collect_hits_for_conf(metric_name: str, all_ctx: dict) -> list:
        hits_local = list((((all_ctx or {}).get(metric_name) or {}).get("hits", []) or []))
        if metric_name in ("ROA", "ROE"):
            peer = "ROE" if metric_name == "ROA" else "ROA"
            hits_local.extend((((all_ctx or {}).get(peer) or {}).get("hits", []) or []))

        best = {}
        for h in hits_local:
            cidn = _hit_cid_norm(h)
            if not cidn:
                continue
            prev = best.get(cidn)
            prev_s = _score(prev) if prev is not None else None
            cur_s = _score(h)
            if (prev is None) or ((cur_s is not None) and (prev_s is None or cur_s > prev_s)):
                best[cidn] = h
        out = list(best.values())
        out.sort(key=lambda x: (_score(x) if _score(x) is not None else -1.0), reverse=True)
        return out

    def _keyword_ok(metric_name: str, text: str) -> bool:
        t = str(text or "").lower()
        if metric_name == "ROA":
            return ("return on average" in t and "asset" in t) or ("roa" in t)
        if metric_name == "ROE":
            return ("return on average" in t and "equity" in t) or ("roe" in t)
        return True

    def _assess_confidence(metric_name: str, row: dict, all_ctx: dict) -> dict:
        hard = []
        soft = []

        val = row.get("value")
        unit = row.get("unit")
        cid = str(row.get("source_chunk_id", "NOT FOUND")).strip()

        if _is_nf(val):
            hard.append("value_missing")
        if _is_nf(unit):
            hard.append("unit_missing")
        if _is_nf(cid):
            hard.append("citation_missing")
        else:
            cid_chk = _norm_cid(cid)
            if (not cid_chk.startswith("table:")) and (not cid_chk.startswith("regex:")):
                if not _valid_cite(cid_chk):
                    hard.append("citation_invalid")

        fv = _to_float_safe(val)
        if metric_name == "ROA" and (fv is not None) and not (0.0 <= fv <= 3.0):
            hard.append("range_roa")
        if metric_name == "ROE" and (fv is not None) and not (0.0 <= fv <= 40.0):
            hard.append("range_roe")
        if metric_name == "NIM" and (fv is not None) and not (0.0 <= fv <= 6.0):
            hard.append("range_nim")

        hits_local = _collect_hits_for_conf(metric_name, all_ctx)
        s1 = _score(hits_local[0]) if len(hits_local) >= 1 else None
        s2 = _score(hits_local[1]) if len(hits_local) >= 2 else None

        matched_hit = None
        cidn = _norm_cid(cid)
        if cidn and hits_local:
            for h in hits_local:
                if _hit_cid_norm(h) == cidn:
                    matched_hit = h
                    break

        s_ref = _score(matched_hit) if matched_hit is not None else s1
        if s_ref is not None and s_ref < 0.60:
            soft.append(f"low_score:{s_ref:.3f}")
        if (s1 is not None) and (s2 is not None) and ((s1 - s2) < 0.015):
            soft.append(f"small_gap:{(s1 - s2):.3f}")
        if cid.startswith("llm:"):
            soft.append("llm_citation")
        if metric_name in ("ROA", "ROE"):
            txt = str((matched_hit or {}).get("text") or "")
            if (not txt) and hits_local:
                txt = str(hits_local[0].get("text") or "")
            if not _keyword_ok(metric_name, txt):
                soft.append("weak_metric_keyword")

        if hard:
            level = "low"
            needs_review = "1"
        elif len(soft) >= 2:
            level = "low"
            needs_review = "1"
        elif len(soft) == 1:
            level = "medium"
            needs_review = "0"
        else:
            level = "high"
            needs_review = "0"

        reason = ";".join(hard + soft)[:240]
        if not reason:
            reason = "rule_ok"
        return {
            "confidence_level": level,
            "needs_review": needs_review,
            "confidence_reason": reason,
        }

    # Apply regex prefills (e.g., NII/NIM) before merging LLM outputs.
    for mname, pobj in prefill.items():
        if mname in ("ROA", "ROE", "NIM") and not sanity_check(mname, pobj.get("value"), pobj.get("unit")):
            r = _get_final_row(final, mname)
            if r:
                r["bucket"] = "sanity_reject"
                r["failure_reason"] = f"sanity_check_failed val={pobj.get('value')} unit={pobj.get('unit')}"
                if pobj.get("source_chunk_id"):
                    r["source_chunk_id"] = pobj["source_chunk_id"]
            logger.warning("[SANITY_REJECT] metric=%s val=%s unit=%s cid=%s",
                    mname, pobj.get("value"), pobj.get("unit"), pobj.get("source_chunk_id"))
            continue
        _write_if_missing(
            final,
            mname,
            pobj.get("value", "NOT FOUND"),
            pobj.get("unit", "NOT FOUND"),
            pobj.get("source_chunk_id", "NOT FOUND"),
        )

    # DEBUG: snapshot after regex prefill applied into `final`
    for _m in ["NII", "NIM", "ROA", "ROE", "Provision for Credit Losses"]:
        _row = _get_final_row(final, _m)
        if _row:
            dprint(f"[DEBUG][PREFILL_AFTER_WRITE] bank={target_bank} {_m}: "
                f"val={_row.get('value')} unit={_row.get('unit')} cid={_row.get('source_chunk_id')}")

    def _is_nf_val(v) -> bool:
        return str(v).strip().upper() in ("NOT FOUND", "NOT_FOUND", "")

    _need_compute = False
    for _m in ("ROA", "ROE"):
        _r = _get_final_row(final, _m)
        if _r and _is_nf_val(_r.get("value", "")):
            _need_compute = True
            break

    if _need_compute:
        try:
            # Only expand context if we actually need compute (avoid extra cost)
            _ctx2 = augment_context_with_avg_balances(
                index, meta, emb,
                bank_id=target_bank,
                year=int(year),
                base_context=context,
            )
            from src.rag.derived_metrics import maybe_compute_roa_roe_from_context  # correct module path
            maybe_compute_roa_roe_from_context(final, _ctx2, int(year), dprint=dprint)
            print("[ENH] compute ROA/ROE tried", flush=True)
        except Exception as e:
            print(f"[ENH][ERROR] compute ROA/ROE failed: {e!r}", flush=True)

    # 3) table-sidecar prefill (before LLM)
    # ---- table-sidecar prefill (before LLM) ----
    apply_table_sidecar_prefill(
        final=final,
        target_bank=target_bank,
        year=int(year),
        side_idx=side_idx,
        table_prefill_allow=table_prefill_allow,
        dprint=dprint,
        _get_final_row=_get_final_row,
        _is_nf=_is_nf,
    )

        # 3.5) ROA/ROE judge hook (only when still missing; uses pre-truncation hits_by_metric)
    for _m in ("ROA", "ROE"):
        _r = _get_final_row(final, _m)
        if not _r or not _is_nf(_r.get("value")):
            continue

        _hits_m = (hits_by_metric or {}).get(_m) or []
        cands = mine_ratio_candidates_from_hits(_hits_m, _m, int(year))

        dprint(f"[DEBUG][JUDGE_MINE] bank={target_bank} metric={_m} cands={len(cands)}")
        if not cands:
            continue

        sel = judge_select_candidate(
            metric_name=_m,
            bank=target_bank,
            year=int(year),
            candidates=cands,
            model_name=OLLAMA_MODEL,
            ollama_base_url=OLLAMA_URL,
            temperature=0.0,
            seed=None,
            timeout_s=90,
            debug_dir=debug_dir,
            logger=logger,
        )
        dprint(f"[DEBUG][JUDGE_SEL] bank={target_bank} metric={_m} sel={sel}")

        if sel >= 0:
            picked = cands[sel]
            _write_if_missing(
                final,
                _m,
                picked.get("value", "NOT FOUND"),
                picked.get("unit", "NOT FOUND"),
                picked.get("source_chunk_id", "NOT FOUND"),
            )

    # ---- END TABLE PREFILL ----

    # Ask the LLM to fill remaining metrics (NIM/ROA/ROE/PCL); NII is considered stable via regex.
    llm_metrics = []
    for m in METRICS:
        row = _get_final_row(final, m)
        if row and _is_nf(row.get("value")):
            llm_metrics.append(m)
    # de-dup while preserving order
    llm_metrics = list(dict.fromkeys(llm_metrics))

    dprint(f"[DEBUG][LLM_GATE] bank={target_bank} llm_metrics={llm_metrics}")

    # Optional switch: skip per-metric LLM fallback and rely on regex/table + unified post-review.
    if os.getenv("ENABLE_PER_METRIC_LLM_FALLBACK", "0") != "1":
        print(
            f"[LLM] per-metric fallback skipped (ENABLE_PER_METRIC_LLM_FALLBACK=0) "
            f"bank={target_bank} would_call={llm_metrics}",
            flush=True,
        )
        llm_metrics = []

    # enable llm for testing
    if os.getenv("DISABLE_LLM", "0") == "1":
        print(f"[LLM] skipped (DISABLE_LLM=1) bank={target_bank} would_call={llm_metrics}", flush=True)
        llm_metrics = []

    for metric in llm_metrics:
        _row0 = _get_final_row(final, metric)
        dprint(
            f"[DEBUG][LOOP_ENTER] bank={target_bank} metric={metric} "
            f"current_val={(_row0 or {}).get('value')} "
            f"current_unit={(_row0 or {}).get('unit')} "
            f"current_cid={(_row0 or {}).get('source_chunk_id')}"
        )

        mhits = (metric_contexts.get(metric) or {}).get("hits", [])
        mctx = (metric_contexts.get(metric) or {}).get("context", "")

        # -------- per-metric regex passes (mirrors 06patch) --------
        try:
            from src.rag.regex_extractors import (
                try_regex_extract_nii_from_context,
                try_regex_extract_nim_from_context,
                try_regex_extract_roa_roe_from_context,
                try_regex_extract_pcl_from_context,
            )
        except Exception:
            try_regex_extract_nii_from_context = try_regex_extract_nim_from_context = None
            try_regex_extract_roa_roe_from_context = try_regex_extract_pcl_from_context = None

        if metric == "NII" and try_regex_extract_nii_from_context:
            row = _get_final_row(final, "NII")
            if row and _is_nf(row.get("value")):
                got = try_regex_extract_nii_from_context(mctx or context)
                if got:
                    val, unit, cid = got
                    row.update({"value": val, "unit": unit, "source_chunk_id": cid, "metric_name": "NII", "fiscal_year": int(year)})
                    print(f"[EXTRACT] regex prefill NII(per-metric) ok: {val} {unit} @ {cid}", flush=True)
                    continue

        if metric == "NIM" and try_regex_extract_nim_from_context:
            row = _get_final_row(final, "NIM")
            if row and _is_nf(row.get("value")):
                got_nim = try_regex_extract_nim_from_context(mctx) or try_regex_extract_nim_from_context(context)
                if got_nim:
                    val, unit, cid = got_nim
                    if sanity_check("NIM", val):
                        row.update({"value": val, "unit": unit, "source_chunk_id": cid, "metric_name": "NIM", "fiscal_year": int(year)})
                        print(f"[EXTRACT] regex prefill NIM ok (per-metric), skip LLM", flush=True)
                        continue
                    else:
                        logger.warning("[SANITY_REJECT] metric=NIM val=%s cid=%s", val, cid)

        if metric in ("ROA", "ROE") and try_regex_extract_roa_roe_from_context:
            need = _get_final_row(final, metric)
            if need and _is_nf(need.get("value")):
                got_rr = try_regex_extract_roa_roe_from_context(mctx, year=int(year)) or try_regex_extract_roa_roe_from_context(context, year=int(year))
                if got_rr:
                    filled_current = False
                    for mm in ("ROA", "ROE"):
                        if mm not in got_rr:
                            continue
                        val, unit, cid = got_rr[mm]
                        rowm = _get_final_row(final, mm)
                        if rowm and _is_nf(rowm.get("value")):
                            if sanity_check(mm, val):
                                rowm.update({"value": val, "unit": unit, "source_chunk_id": cid, "metric_name": mm, "fiscal_year": int(year)})
                                if mm == metric:
                                    filled_current = True
                            else:
                                logger.warning("[SANITY_REJECT] metric=%s val=%s cid=%s", mm, val, cid)
                    if filled_current:
                        print(f"[EXTRACT] regex prefill {metric} ok (per-metric), skip LLM", flush=True)
                        continue

        if metric == "Provision for Credit Losses" and try_regex_extract_pcl_from_context:
            row = _get_final_row(final, metric)
            if row and _is_nf(row.get("value")):
                got_pcl = try_regex_extract_pcl_from_context(mctx, year=int(year)) or try_regex_extract_pcl_from_context(context, year=int(year))
                if got_pcl:
                    val, unit, cid = got_pcl
                    row.update({"value": val, "unit": unit, "source_chunk_id": cid, "metric_name": metric, "fiscal_year": int(year)})
                    print("[EXTRACT] regex prefill PCL ok (per-metric), skip LLM", flush=True)
                    continue
        # -------- end per-metric regex passes --------

        print(f"[EXTRACT] calling ollama metric={metric} ...", flush=True)
        item = call_llm_for_metric(
            metric_name=metric,
            bank=bank_id,
            year=int(year),
            context_text=mctx,
            model_name=OLLAMA_MODEL,
            ollama_base_url=OLLAMA_URL,
            temperature=TEMPERATURE,
            seed=None,
            timeout_s=180,
            debug_dir=debug_dir,
            repair=True,
            logger=logger,
        )

        val = None
        unit = None
        cid = None
        fy = None
        try:
            val = item.get("value", item.get("val"))
            unit = item.get("unit")
            cid = item.get("source_chunk_id")
            fy = item.get("fiscal_year", item.get("year"))
        except Exception:
            pass

        print(
            f"[LLM_RET] bank={bank_id} metric={metric} "
            f"val={val} unit={unit} cid={cid} fy={fy} item_type={type(item).__name__}",
            flush=True,
        )

        if item:
            for row in final["results"]:
                if row["metric_name"] == metric:
                    row.update(item)
                    break

    # 4.5) confidence scoring (rule-based) to decide whether review is needed
    low_conf_metrics = []
    for row in final.get("results", []):
        metric = row.get("metric_name")
        conf = _assess_confidence(metric, row, metric_contexts)
        row.update(conf)
        if conf.get("needs_review") == "1":
            low_conf_metrics.append(metric)
    low_conf_metrics = list(dict.fromkeys(low_conf_metrics))
    dprint(f"[DEBUG][CONF] bank={target_bank} low_conf_metrics={low_conf_metrics}")

    # 5) unified post-extraction LLM review (all 5 metrics together)
    review_changed = 0
    if os.getenv("DISABLE_LLM_REVIEW", "0") != "1":
        review_only_low = os.getenv("REVIEW_ONLY_LOW_CONFIDENCE", "1") == "1"
        metrics_to_review = low_conf_metrics if review_only_low else list(METRICS)
        if review_only_low and not metrics_to_review:
            dprint(f"[DEBUG][REVIEW_ALL] skipped bank={target_bank} (no low-confidence metrics)")
            for row in final.get("results", []):
                if not row.get("review_action"):
                    row["review_action"] = "skip_by_confidence"
                if not row.get("review_note"):
                    row["review_note"] = "confidence_not_low"
                row["review_model"] = OLLAMA_MODEL
            review_out = None
        else:
            try:
                review_num_predict = int(os.getenv("LLM_REVIEW_NUM_PREDICT", "768"))
            except Exception:
                review_num_predict = 768
            if review_num_predict <= 0:
                review_num_predict = 768

            review_out = review_all_metrics_after_extract(
                bank=target_bank,
                year=int(year),
                final_obj=final,
                metric_contexts=metric_contexts,
                meta_rows=meta,
                model_name=OLLAMA_MODEL,
                ollama_base_url=OLLAMA_URL,
                temperature=0.0,
                seed=None,
                timeout_s=120,
                num_predict=review_num_predict,
                metrics_to_review=metrics_to_review,
                debug_dir=debug_dir,
                logger=logger,
            )

        if review_out:
            review_ok = bool((review_out or {}).get("ok"))
            decisions = (review_out or {}).get("decisions", {}) or {}
            for row in final.get("results", []):
                metric = row.get("metric_name")
                d = decisions.get(metric)
                if not d:
                    if not row.get("review_action"):
                        row["review_action"] = "skip_by_confidence"
                    if not row.get("review_note"):
                        row["review_note"] = "not_in_review_scope"
                    row["review_model"] = OLLAMA_MODEL
                    continue

                action = str(d.get("action", "keep")).strip().lower()
                note = str(d.get("reason", "")).strip()
                if not review_ok:
                    action = "review_fail"
                    note = note or "review_parse_or_call_failed"
                row["review_model"] = OLLAMA_MODEL
                row["review_note"] = note
                row["review_action"] = action if action in ("keep", "replace", "reject") else "review_fail"

                if action in ("replace", "reject"):
                    old_val = row.get("value", "NOT FOUND")
                    old_unit = row.get("unit", "NOT FOUND")
                    old_cid = row.get("source_chunk_id", "NOT FOUND")
                    row["orig_value"] = old_val
                    row["orig_unit"] = old_unit
                    row["orig_source_chunk_id"] = old_cid

                    row["value"] = d.get("value", "NOT FOUND")
                    row["unit"] = d.get("unit", "NOT FOUND")
                    row["source_chunk_id"] = d.get("source_chunk_id", "NOT FOUND")
                    review_changed += 1
    else:
        dprint(f"[DEBUG][REVIEW_ALL] skipped bank={target_bank} (DISABLE_LLM_REVIEW=1)")

    # ===== meta =====
    final["_meta"] = {
        "bank": target_bank,
        "year": int(year),
        "retrieval_query": "PER_METRIC_MULTIQUERY",
        "topk": sum(len(v or []) for v in hits_by_metric.values()),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "prefill": list(prefill.keys()) if prefill else [],
        "review_changed": review_changed,
        "low_conf_metrics": low_conf_metrics,
    }
    return final


import csv

def _is_not_found(x):
    """
    Return True if a field value represents a missing extraction.
    Treats empty strings and the sentinel 'NOT FOUND' (case-insensitive) as missing.
    """
    if x is None:
        return True
    s = str(x).strip().upper()
    return s in ("NOT FOUND", "NOT_FOUND", "")


def _extract_stem(cid: str) -> str:
    """
    Extract a document stem from a citation or hit payload when present.
    This is used in diagnostics to detect potential year mismatches and to summarize evidence provenance.
    """
    if not isinstance(cid, str):
        return ""
    cid = cid.strip()
    if cid.startswith("llm:"):
        cid = cid[len("llm:"):].strip()
    if cid.startswith("[") and cid.endswith("]"):
        cid = cid[1:-1].strip()
    m = re.search(r"stem=([^|]+)", cid)
    return m.group(1) if m else ""

def analyze_extractions(jsonl_path: Path, out_csv_path: Path, year: int):
    """
    Compute summary statistics over extraction outputs.
    Produces diagnostics such as hit rate, citation compliance, and year mismatch indicators for QA.
    """
    rows = []
    cnt = Counter()

    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            meta = obj.get("_meta", {}) if isinstance(obj, dict) else {}
            bank = meta.get("bank") or obj.get("bank") or "UNKNOWN"
            topk = meta.get("topk", "")
            err = obj.get("error", "")

            results = obj.get("results", None) if isinstance(obj, dict) else None

            # Aggregate counters by category.
            if err == "NO_HITS":
                cnt["NO_HITS"] += 1
            elif err:
                cnt["HAS_ERROR"] += 1

            if not isinstance(results, list):
                cnt["RESULTS_NOT_LIST"] += 1
                rows.append({
                    "bank": bank,
                    "year": year,
                    "topk": topk,
                    "status": "RESULTS_NOT_LIST",
                    "error": err,
                    "n_found": 0,
                    "found_metrics": "",
                    "n_bad_cite": 0,
                    "year_mismatch_metrics": "",
                })
                continue

            # Expected: results is a list of per-metric dict objects.
            if len(results) == 0:
                cnt["RESULTS_EMPTY"] += 1

            found = []
            bad_cite = 0
            year_mismatch = []

            for it in results:
                if not isinstance(it, dict):
                    continue
                # Backward-compatibility: some outputs may store the value field as "val" (flattened form) vs "value" (normalized form).
                m = it.get("metric_name", "")
                v = it.get("val", None)
                cid = it.get("source_chunk_id", "NOT FOUND")

                if (m in METRICS) and (not _is_not_found(v)):
                    found.append(m)

                # Citation compliance: if source_chunk_id is present (not NOT FOUND), it must match the accepted cite format.
                if not _is_not_found(cid) and (not _valid_cite(cid)):
                    bad_cite += 1

                # Year mismatch: document stem suggests a different fiscal year than the requested year.
                stem = _extract_stem(cid)
                if stem and (str(year) not in stem) and re.search(r"\b(2020|2021|2022|2023)\b", stem):
                    year_mismatch.append(m or "UNKNOWN")          # Heuristic check: stem string may contain other years; this is only a warning signal, not a definitive error.

            n_found = len(set(found))
            if n_found == 0:
                cnt["FOUND_0"] += 1
            else:
                cnt["FOUND_GT0"] += 1

            if bad_cite > 0:
                cnt["BAD_CITE"] += 1

            if year_mismatch:
                cnt["YEAR_MISMATCH"] += 1

            rows.append({
                "bank": bank,
                "year": year,
                "topk": topk,
                "status": "OK" if (not err) else "OK_WITH_ERROR",
                "error": err,
                "n_found": n_found,
                "found_metrics": ",".join(sorted(set(found))),
                "n_bad_cite": bad_cite,
                "year_mismatch_metrics": ",".join(sorted(set(year_mismatch))),
            })

    # Write diagnostics.csv for summary statistics and QA checks.
    out_csv_path.parent.mkdir(parents=True, exist_ok=True)
    cols = ["bank", "year", "topk", "status", "error", "n_found", "found_metrics", "n_bad_cite", "year_mismatch_metrics"]
    with out_csv_path.open("w", encoding="utf-8", newline="") as fw:
        w = csv.DictWriter(fw, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    # Print a terminal summary for quick review.    
    print("\n=== STATS SUMMARY ===", flush=True)
    print(f"[STATS] jsonl: {jsonl_path}", flush=True)
    print(f"[STATS] csv : {out_csv_path}", flush=True)
    for k, v in cnt.most_common():
        print(f"{k}: {v}", flush=True)
    print("=====================\n", flush=True)

def main():
    """
    CLI entry point for batch extraction and diagnostics commands.
    Supports interactive inputs and batch file mode; errors are handled to continue processing remaining items.
    """
    print(f"[INFO] loading faiss: {INDEX_PATH}", flush=True)
    index = faiss.read_index(str(INDEX_PATH))
    print(f"[INFO] ntotal={index.ntotal} dim={index.d}", flush=True)

    print(f"[INFO] loading meta : {META_PATH}", flush=True)
    meta = load_meta(META_PATH)
    if len(meta) != index.ntotal:
        raise RuntimeError(f"meta({len(meta)}) != index({index.ntotal})")

    print(f"[INFO] loading embed model: {EMB_MODEL} device={EMB_DEVICE}", flush=True)
    emb = SentenceTransformer(EMB_MODEL, device=EMB_DEVICE)

    print(f"[INFO] ollama model: {OLLAMA_MODEL} url={OLLAMA_URL}", flush=True)

    while True:
        try:
            q = input("\nQ (empty to exit): ").strip()
        except EOFError:
            break
        
        if q.startswith(":batch"):
            batch_path_str = parse_batch_command(q)
            if not batch_path_str:
                print("Usage: :batch <banks.txt>", flush=True)
                continue
            bank_file = resolve_batch_path(ROOT, batch_path_str)

            if not bank_file.exists():
                raise FileNotFoundError(f"banks file not found: {bank_file}")

            banks = read_bank_list(bank_file)

            paths = default_paths(ROOT, YEAR)
            out_jsonl = paths["out_jsonl"]
            out_csv   = paths["out_csv"]

            # Sidecar locations (base table sidecar + optional PDF-derived sidecar sources).
            sidecar_path      = paths["sidecar_path"]
            pdf_sidecar_path  = paths["pdf_sidecar_path"]
            pdf_sidecar_dir   = paths["pdf_sidecar_dir"]

            side_idx = None
            try:
                base_ok = sidecar_path.exists()
                pdf_ok  = pdf_sidecar_path.exists()
                dir_ok  = pdf_sidecar_dir.exists()
                if base_ok or pdf_ok or dir_ok:
                    print("[SIDECAR] loading sidecar index ...", flush=True)
                    side_idx = load_sidecar_index_multi(
                        base_sidecar=sidecar_path if base_ok else None,
                        pdf_sidecar=pdf_sidecar_path if pdf_ok else None,
                        pdf_sidecar_dir=pdf_sidecar_dir if dir_ok else None,
                    )
                    print(f"[SIDECAR] loaded={bool(side_idx)} banks={len(side_idx) if side_idx else 0}", flush=True)
            except Exception as e:
                print(f"[SIDECAR] load_failed err={e!r}", flush=True)
                side_idx = None

            records = []
            for i, b in enumerate(banks, 1):
                print(f"[BATCH] {i}/{len(banks)} bank={b}", flush=True)
                try:
                    rec = extract_for_bank(index, meta, emb, bank_id=b, year=YEAR, side_idx=side_idx)
                except Exception as e:
                    print("[BATCH][ERROR]", b, "->", repr(e), flush=True)
                    traceback.print_exc()
                    rec = make_template(int(YEAR))
                    rec["_meta"] = {"bank": b, "year": int(YEAR)}
                    rec["error"] = repr(e)
                records.append(rec)

            write_jsonl(out_jsonl, records)
            rows = flatten_metrics(records, audit_buffer=AUDIT_ROWS)
            rows = merge_keep_existing(out_csv, rows)  # Scheme-1: do not overwrite existing filled values
            for r in rows:
                if r.get("bank") == "SavingsFirst_613679" and r.get("metric_name") in ("ROA","ROE"):
                    print("[DBG][BEFORE_WRITE]",
                        r.get("metric_name"),
                        "val=", r.get("val"),
                        "cid=", r.get("source_chunk_id"),
                        "bucket=", r.get("bucket"))
            write_metrics_csv(out_csv, rows)
            # Optional: backfill NOT FOUND cells from sidecar tables.
            if os.getenv("DISABLE_TABLE_PATCH", "0") == "1":
                print("[PATCH][TABLE] skipped (DISABLE_TABLE_PATCH=1)", flush=True)
            else:
                try:
                    base_ok = sidecar_path.exists()
                    pdf_ok  = pdf_sidecar_path.exists()
                    dir_ok  = pdf_sidecar_dir.exists()

                    if base_ok or pdf_ok or dir_ok:
                        stats = patch_metrics_csv_from_table_sidecar(
                            metrics_csv=out_csv,
                            sidecar_jsonl=sidecar_path if base_ok else None,
                            year=int(YEAR),
                            pdf_sidecar=pdf_sidecar_path if pdf_ok else None,
                            pdf_sidecar_dir=pdf_sidecar_dir if dir_ok else None,
                        )
                        if stats:
                            print(f"[PATCH][TABLE] ok stats={stats}", flush=True)
                        else:
                            print("[PATCH][TABLE] skipped (no sidecar blocks)", flush=True)
                    else:
                        print("[PATCH][TABLE] skipped (no sidecar files)", flush=True)
                except Exception as e:
                    print(f"[PATCH][TABLE][ERROR] {type(e).__name__}: {e}", flush=True)
                    traceback.print_exc()   

            print(f"[DONE] wrote: {out_jsonl}", flush=True)
            print(f"[DONE] wrote: {out_csv}", flush=True)   
            # Write a lightweight audit file of non-missing extractions for manual QA (not a primary output artifact).
            audit_csv = paths["audit_csv"]
            with audit_csv.open("w", encoding="utf-8", newline="") as f:
                w = csv.DictWriter(
                    f,
                    fieldnames=["bank", "year", "metric", "val", "unit", "chunk"]
                )
                w.writeheader()
                for r in AUDIT_ROWS:
                    w.writerow(r)
            print(f"[DONE] wrote audit: {audit_csv}", flush=True)
            continue

        if q.startswith(":stats"):
            # Usage: :stats [optional_jsonl_path]
            parts = q.split(maxsplit=1)
            if len(parts) == 2:
                jsonl_path = Path(parts[1].strip())
            else:
                jsonl_path = ROOT / "data" / "outputs" / "logs" / f"extractions_{YEAR}.jsonl"

            out_csv = ROOT / "data" / "outputs" / "logs" / f"diagnostics_{YEAR}.csv"
            analyze_extractions(jsonl_path, out_csv, int(YEAR))
            continue

        try:
            dprint("[DEBUG] encoding query ...")
            qvec = emb.encode([q], batch_size=1, normalize_embeddings=True, show_progress_bar=False,
            convert_to_numpy=True).astype(np.float32)

            dprint("[DEBUG] searching faiss ...")
            D, I = index.search(qvec, TOPK_SEARCH)


            hits = []
            for rnk, (score, idx) in enumerate(zip(D[0], I[0]), 1):
                hits.append((rnk, float(score), meta[int(idx)]))

            # --- choose target bank (default: bank of the top-1 hit) ---
            target_bank = hits[0][2].get("bank_folder")

            # --- keep only hits from the same bank, then take TOPK_FINAL ---
            hits = [h for h in hits if h[2].get("bank_folder") == target_bank][:TOPK_FINAL]


            print("\n=== TOPK ===", flush=True)
            for rnk, score, m in hits:
                print(f"[{rnk}] score={score:.4f} bank={m.get('bank_folder')} stem={m.get('stem')} chunk={m.get('chunk_id')}", flush=True)

            context = build_context(hits)

            prompt = (
    "You are a financial information extraction engine.\n\n"
    "Task: Extract metrics ONLY if explicitly stated as numbers in the Context.\n"
    "Target fiscal year: 2024\n\n"
    "Hard rules:\n"
    "1) Do NOT infer, summarize, or generalize.\n"
    "2) Use ONLY the Context as the source of truth.\n"
    "3) Each found metric MUST include the source_chunk_id copied EXACTLY from a [k=...] header.\n"
    "4) Normalize synonyms: ROAA -> ROA, ROAE -> ROE.\n"
    "5) If not explicitly stated, keep NOT FOUND.\n\n"
    "You MUST output EXACTLY the following JSON object and nothing else:\n\n"
    "{\n"
    "  \"results\": [\n"
    "    {\"metric_name\":\"ROA\",\"value\":\"NOT FOUND\",\"unit\":\"NOT FOUND\",\"fiscal_year\":2024,\"source_chunk_id\":\"NOT FOUND\"},\n"
    "    {\"metric_name\":\"ROE\",\"value\":\"NOT FOUND\",\"unit\":\"NOT FOUND\",\"fiscal_year\":2024,\"source_chunk_id\":\"NOT FOUND\"},\n"
    "    {\"metric_name\":\"NIM\",\"value\":\"NOT FOUND\",\"unit\":\"NOT FOUND\",\"fiscal_year\":2024,\"source_chunk_id\":\"NOT FOUND\"},\n"
    "    {\"metric_name\":\"NII\",\"value\":\"NOT FOUND\",\"unit\":\"NOT FOUND\",\"fiscal_year\":2024,\"source_chunk_id\":\"NOT FOUND\"},\n"
    "    {\"metric_name\":\"Provision for Credit Losses\",\"value\":\"NOT FOUND\",\"unit\":\"NOT FOUND\",\"fiscal_year\":2024,\"source_chunk_id\":\"NOT FOUND\"}\n"
    "  ]\n"
    "}\n\n"
    "Question:\n"
    + q + "\n\n"
    "Context:\n"
    + context + "\n\n"
    "Answer (JSON only):"
)

            # Limit to 3 evidence blocks to keep the context short. 
            context = build_context(hits)

            prompt = (
                "You are a financial metric extractor.\n"
                "Return ONLY JSON. No extra text.\n"
                "Language: English.\n\n"
                "Task: Extract Net Interest Margin (NIM) for fiscal year 2024.\n"
                "Rules:\n"
                "1) Only extract if an explicit numeric NIM value appears in the context.\n"
                "2) Do NOT infer.\n"
                "3) source_chunk_id must be copied from the nearest [k=...|stem=...|chunk=...] header.\n\n"
                "Output JSON schema (exact keys):\n"
                "{\n"
                "  \"metric_name\": \"NIM\",\n"
                "  \"val\": \"NOT FOUND\",\n"
                "  \"unit\": \"NOT FOUND\",\n"
                "  \"fiscal_year\": 2024,\n"
                "  \"source_chunk_id\": \"NOT FOUND\"\n"
                "}\n\n"
                "Context:\n"
                f"{context}\n"
            )

            dprint(f"[DEBUG] prompt length = {len(prompt)} chars")
            print("[EXTRACT] calling ollama ...", flush=True)
            from src.rag.llm_extract import _ollama_generate
            ans = _ollama_generate(prompt)
            print("\n=== ANSWER ===", flush=True)
            print(ans.strip(), flush=True)


        except Exception as e:
            print("\n[ERROR] Exception happened, but program will continue.", flush=True)
            print("type:", type(e).__name__, flush=True)
            print("msg :", str(e), flush=True)
            print("--- traceback ---", flush=True)
            traceback.print_exc()
            print("---------------", flush=True)
            # Continue to the next input item instead of exiting on error.

if __name__ == "__main__":
    main()
