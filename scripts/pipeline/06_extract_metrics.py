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
AUDIT_ROWS = []
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# Constrain CPU threads to reduce oversubscription and improve run-to-run stability (FAISS / BLAS).
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import csv
import json
import re
from pathlib import Path
from collections import Counter

from pathlib import Path
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

def find_repo_root(start: Path) -> Path:
    """
    Return the repository root directory.
    This resolves the project root used for consistent relative-path handling across scripts.
    """
    p = start.resolve()
    for _ in range(10):  # Search up to 10 parent directories for repo root markers
        if (p / ".git").exists() or (p / "README.md").exists() or (p / "data").exists():
            return p
        p = p.parent
    raise RuntimeError(f"Cannot locate repo root from: {start}")

ROOT = find_repo_root(Path(__file__))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

YEAR = "2024"

INDEX_DIR = ROOT / "data" / "interim" / "index" / f"faiss_{YEAR}_full"
INDEX_PATH = INDEX_DIR / "faiss.index"
META_PATH  = INDEX_DIR / "meta.jsonl"

EMB_MODEL = "BAAI/bge-m3"
EMB_DEVICE = "cuda"

TOPK_SEARCH = 50   # Initial retrieval size from the full index (before bank filtering)
TOPK_FINAL  = 20   # Final number of chunks kept after filtering by bank

TOPK = 10

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "qwen3:4b"   # Model name must match a local Ollama model (see `ollama list`)
TEMPERATURE = 0.2

def load_meta(path: Path):
    """
    Load FAISS metadata records from meta.jsonl.
    Returns a list of dict objects, one per indexed chunk, used for bank/stem/chunk lookup and context assembly.
    """
    meta = []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            meta.append(json.loads(line))
    return meta

def build_context(hits):
    """
    Accept two hit formats:
    A) list[(rank, score, meta_dict)]  where meta_dict has bank_folder/stem/chunk_id/text
    B) list[dict] from search_faiss()  where dict has bank/stem/chunk_id/text/score
    """
    blocks = []
    if not hits:
        return ""

    for idx, h in enumerate(hits, 1):
        # --- normalize ---
        if isinstance(h, (tuple, list)) and len(h) == 3:
            rank, score, m = h
            score = float(score)
            m = m or {}
            text = (m.get("text") or "").strip()
            bank = m.get("bank_folder") or m.get("bank")  # tolerate both
            stem = m.get("stem")
            chunk_id = m.get("chunk_id")
        elif isinstance(h, dict):
            rank = idx
            score = float(h.get("score", 0.0))
            text = (h.get("text") or "").strip()
            bank = h.get("bank") or h.get("bank_folder")
            stem = h.get("stem")
            chunk_id = h.get("chunk_id")
        else:
            # unknown format: skip instead of crash
            continue

        # --- head+tail truncate (table-aware) ---
        t = text.lower()

        # Default truncation settings for general text blocks
        MAXC = 1400
        HEAD = 700
        TAIL = 700

        # For ratio tables and structured financial summaries, apply a larger context window
        # to avoid truncating fiscal-year numeric columns (e.g., 2024 values)
        is_ratio_table = any(k in t for k in [
            "selected performance ratios",
            "return on assets",
            "return on equity",
            "equity to assets",
            "dividend payout",
            "net interest margin",
        ])
        if is_ratio_table:
            MAXC = 4200
            HEAD = 1600
            TAIL = 2600

        if len(text) > MAXC:
            text = text[:HEAD] + "\n...[middle truncated]...\n" + text[-TAIL:]

# Contract: LLM must copy source_chunk_id exactly from this header for traceability.
        header = f"[k={bank}|stem={stem}|chunk={chunk_id}]"
        blocks.append(
            f"{header}\n"
            f"(rank={rank} score={score:.4f})\n"
            f"{text}"
        )

    return "\n\n---\n\n".join(blocks)


def ollama_generate(prompt: str) -> str:
    """
    Call the Ollama HTTP API and return the raw model response text.
    This function is intentionally thin: it does not interpret schema, and callers handle JSON parsing/repair.
    """
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "format": "json",  # Enforce JSON-formatted output from the LLM 
        "options": {
            "temperature": 0,
            # Remove stop tokens that may prematurely truncate the model output
            "stop": ["\nQ (empty to exit):"]
        }
    }
    r = requests.post(OLLAMA_URL, json=payload, timeout=180)

    debug_text = r.text or ""
    if debug_text.strip():
        print("[DEBUG] Ollama raw HTTP body (head):")
        print(debug_text[:500])
    else:
        print("[WARN] Ollama returned empty HTTP body")

    if r.status_code != 200:
        raise RuntimeError(f"Ollama HTTP {r.status_code}: {debug_text[:500]}")

    j = r.json()

    # Prefer the 'response' field; fall back to 'thinking' if response is empty
    resp = (j.get("response") or "").strip()
    if resp:
        return resp
    think = (j.get("thinking") or "").strip()
    if think:
        return think

    msg = j.get("message") or {}
    content = (msg.get("content") or "").strip() if isinstance(msg, dict) else ""
    return content


import re
from datetime import datetime

DEFAULT_RETRIEVAL_QUERY = "FY2024 ROA ROE NIM NII net interest income net interest margin return on assets return on equity provision for credit losses"

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

import re, json

def parse_or_fallback(raw_text: str, year: int = 2024):
    """
    Try `parse_json_loose` first; if it fails, fall back to regex extraction from non-JSON explanatory outputs.
    Returns:
      - A compliant object (e.g., {"results":[...]})
      - Or a flat dict: {"value": "...", "unit": "...", "source_chunk_id": "...", "fiscal_year": 2024}
    """
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

    # 3) Fallback: extract NII and source chunk information from non-JSON model outputs using regex.
    #    value: prefer numeric patterns like 190,591 / 190591 / $190,591 / 190.591
    m_val = re.search(r"(?i)\bvalue\b[^0-9$]*\$?\s*`?\s*([0-9][0-9,\.]*)", s)
    value = m_val.group(1) if m_val else "NOT FOUND"
    value = value.replace(",", "") if value != "NOT FOUND" else value

    #    unit: infer scale from phrases like "(in thousands)" / "thousand dollars" / "million"; fallback to NOT FOUND
    m_unit = re.search(r"(?i)\bunit\b[^A-Za-z]*`?\s*([A-Za-z][A-Za-z \-\(\)%]+)", s)
    unit = m_unit.group(1).strip() if m_unit else "NOT FOUND"

    #    chunk: prefer structured ids like [k=...|stem=...|chunk=12]; fallback to a numeric chunk id when available
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

def retrieve_hits(index, meta, qvec, topk_search=TOPK_SEARCH, target_bank=None, topk_final=TOPK_FINAL):
    """
    Retrieve top hits for a bank using a single retrieval query.
    This is the baseline retrieval path (single query -> bank-filtered hits), used by higher-level retrieval helpers.
    """
    D, I = index.search(qvec, topk_search)

    hits = []
    for rnk, (score, idx) in enumerate(zip(D[0], I[0]), 1):
        hits.append((rnk, float(score), meta[int(idx)]))

    if not hits:
        return [], None

    # Default behavior: use the bank identifier from the top-ranked hit
    if target_bank is None:
        target_bank = hits[0][2].get("bank_folder")

    # Keep only hits from the target bank
    hits = [h for h in hits if h[2].get("bank_folder") == target_bank][:topk_final]
    return hits, target_bank

def _prefer_year_hits(hits, year: str):
    """
    hits: list[(rnk, score, meta)]
    Prefer hits whose stem contains the target year; if none match, return the original set to avoid empty evidence.
    """
    y = str(year)
    good = []
    for h in hits:
        m = h[2]
        stem = (m.get("stem") or "")
        if y in stem or f"FY{y[-2:]}" in stem or "12-31-24" in stem:
            good.append(h)
    return good if good else hits

def _hit_key(h: dict):
    """
    Return a stable deduplication key for a hit.
    Keys are based on (bank_folder/bank, stem, chunk_id) to ensure one evidence block per unique chunk.
    """
    try:
        return (h.get("bank"), h.get("stem"), int(h.get("chunk_id")))
    except Exception:
        return None

def _build_meta_lookup(meta: list):
    """
    Build a lookup table for meta records by (bank, stem, chunk_id).
    Used to quickly resolve neighbor chunks and to normalize references during context construction.
    """
    # (bank_folder, stem, chunk_id) -> meta_record
    lookup = {}
    for m in meta:
        b = m.get("bank_folder")
        s = m.get("stem")
        cid = m.get("chunk_id")
        if b and s and cid is not None:
            try:
                lookup[(b, s, int(cid))] = m
            except Exception:
                pass
    return lookup

def retrieve_hits_multiquery(
    index,
    meta,
    emb,
    target_bank: str,
    year: str,
    per_metric_topk: int = 10, # Reduced per-metric retrieval size for efficiency
    topk_final: int = 20, # Final number of hits retained after bank filtering
    k0: int = 200,
    kmax: int = 20000,
    MIN_SCORE = 0.50,  # Minimum similarity score threshold for enabling filtering
):
    """
    Key behavior:
    - Retrieve per metric using metric-specific queries
    - Adaptively increase K until enough bank-specific hits are obtained
    - Prefer stems that match the target fiscal year when selecting evidence
    """

    def _expand_neighbors(dedup_hits, meta, window=1):
        # dedup_hits: list[dict] in the same schema as FAISS search results
        # meta: full list loaded from meta.jsonl
        keyset = set()
        out = []

        def _key(h):
            # h can be dict hit or tuple hit: (rank, score, meta_dict)
            if isinstance(h, tuple) and len(h) >= 3 and isinstance(h[2], dict):
                m = h[2]
                bank = m.get("bank_folder", m.get("bank"))
                stem = m.get("stem")
                cid = m.get("chunk_id")
            else:
                bank = h.get("bank_folder", h.get("bank"))
                stem = h.get("stem")
                cid = h.get("chunk_id")

            try:
                cid = int(cid)
            except Exception:
                cid = -1

            return (bank, stem, cid)


        # Seed with original hits
        for h in dedup_hits:
            k = _key(h)
            if k in keyset:
                continue
            keyset.add(k)
            out.append(h)

        # Build a fast index: (bank, stem, chunk_id) -> meta_item
        idx = {}
        for m in meta:
            try:
                bk = m.get("bank") or m.get("bank_folder")
                st = m.get("stem")
                cid = int(m.get("chunk_id"))
                idx[(bk, st, cid)] = m
            except Exception:
                continue

        # Expand each hit by adding neighboring chunks as additional evidence
        for h in list(out):
            bk = h.get("bank") or h.get("bank_folder")
            st = h.get("stem")
            try:
                cid = int(h.get("chunk_id"))
            except Exception:
                continue

            for d in range(-window, window + 1):
                if d == 0:
                    continue
                nk = (bk, st, cid + d)
                m2 = idx.get(nk)
                if not m2:
                    continue
                # Create a synthetic neighboring hit with a slightly reduced score
                hh = {
                    "bank": bk,
                    "stem": st,
                    "chunk_id": cid + d,
                    "text": m2.get("text", ""),
                    "score": float(h.get("score", 0.0)) - 1e-6,
                }
                kk = _key(hh)
                if kk in keyset:
                    continue
                keyset.add(kk)
                out.append(hh)

        return out

    # Robust bank identifier matching:
    # batch-level bank_id may not exactly match meta['bank_folder']
    def _norm_bank(s: str) -> str:
        return (s or "").strip().lower()

    def _bank_match(meta_bank: str, target_bank: str) -> bool:
        mb = _norm_bank(meta_bank)
        tb = _norm_bank(target_bank)
        if not mb or not tb:
            return False
        if mb == tb:
            return True
        # common cases: one side has extra suffix/prefix (e.g., spaces, ids)
        if mb.startswith(tb) or tb.startswith(mb):
            return True
        # last resort: substring match (guarded by length)
        if len(tb) >= 8 and tb in mb:
            return True
        if len(mb) >= 8 and mb in tb:
            return True
        return False

    # try to resolve an exact bank_folder from meta to make matching faster/stable
    tb_norm = _norm_bank(target_bank)
    if tb_norm:
        for mm in meta:
            mb = _norm_bank(mm.get("bank_folder"))
            if mb == tb_norm:
                target_bank = mm.get("bank_folder")
                break
    metric_queries = {
        "ROA": f"FY{year} return on assets ROA ROAA return on average assets",
        "ROE": f"FY{year} return on equity ROE ROAE return on average equity",
        "NIM": (
            f"FY{year} net interest margin NIM "
            f"interest margin margin (%) "
            f"tax-equivalent net interest margin FTE net interest margin "
            f"\"net interest margin\" \"interest margin\""
        ),
        "NII": f"FY{year} net interest income NII net interest revenue",
        "Provision for Credit Losses": f"FY{year} provision for credit losses PCL provision credit losses",
    }

    # Aggregate candidates across per-metric queries, deduplicated by (bank_folder, stem, chunk_id).
    seen = set()
    pooled = []

    for metric, rq in metric_queries.items():
        topk = 12 if metric == "NIM" else per_metric_topk
        want_topk = 12 if metric == "NIM" else per_metric_topk
        qvec = emb.encode([rq], batch_size=1, normalize_embeddings=True, convert_to_numpy=True).astype(np.float32)

        topk = 12 if metric == "NIM" else per_metric_topk

        k = k0
        bank_hits = []
        while True:
            D, I = index.search(qvec, k)
            tmp = []
            for rnk, (score, idx) in enumerate(zip(D[0], I[0]), 1):
                m = meta[int(idx)]
                if _bank_match(m.get("bank_folder"), target_bank):
                    tmp.append((rnk, float(score), m))
                    if len(tmp) >= want_topk:
                        break

            if tmp:
                bank_hits = tmp
                break

            if k >= kmax:
                bank_hits = []
                break
            k *= 2

        # Prefer hits whose document stem matches the target fiscal year when available.    
        bank_hits = _prefer_year_hits(bank_hits, year)

        # Merge and deduplicate: keep at most one hit per unique chunk key.
        for h in bank_hits:
            m = h[2]
            key = (m.get("bank_folder"), m.get("stem"), m.get("chunk_id"))
            if key in seen:
                continue
            seen.add(key)
            pooled.append(h)

    # NOTE: We sort by similarity score (higher is more relevant) before final selection.
    # pooled.sort(key=lambda x: x[1], reverse=True)

    # Final evidence passed to the LLM (top-k chunks after ranking/filtering).
    # final_hits = pooled[:topk_final]
    # return final_hits
    print("[DEBUG] pooled before bank-filter =", len(pooled), flush=True)
    print("[DEBUG] example bank_folder =", pooled[0][2].get("bank_folder") if pooled else None, flush=True)

    # Sort pooled hits by similarity score in descending order (higher cosine similarity = more relevant).
    pooled.sort(key=lambda x: x[1], reverse=True)

    # Apply similarity score threshold to reduce low-relevance chunks
    filtered = [h for h in pooled if h[1] >= MIN_SCORE]
    if len(filtered) >= 3:   # Only enforce MIN_SCORE if it does not collapse evidence (keep at least 3 chunks when possible).
        pooled = filtered

    # Final evidence passed to the LLM (top-k after optional thresholding).
    final_hits = pooled[:topk_final]

    # Convert tuple-based hits to dict format for downstream processing
    def _tuple_hit_to_dict(h):
        # h: (rnk, score, meta_dict)
        try:
            rnk, score, m = h
            return {
                "score": float(score),
                "bank": (m.get("bank_folder") or m.get("bank") or ""),
                "stem": (m.get("stem") or ""),
                "chunk_id": m.get("chunk_id"),
                "text": (m.get("text") or ""),
            }
        except Exception:
            return None

    if any(isinstance(h, tuple) for h in final_hits):
        final_hits = [x for x in (_tuple_hit_to_dict(h) for h in final_hits) if x]

    before = len(final_hits)
    final_hits = _expand_neighbors(final_hits, meta, window=1)
    print(f"[NEI] multiquery: before={before} after={len(final_hits)}")

    return final_hits


def _expand_neighbors(hits, meta, window=1):
    """
    hits: list[dict] from search_faiss() -> {bank, stem, chunk_id, text, score}
    meta: list[dict] loaded from meta.jsonl -> should contain (bank_folder/bank, stem, chunk_id, text)
    Return: hits + neighbor chunks (+/- window) per hit, deduped by (bank, stem, chunk_id)
    """
    if not hits or not meta:
        return hits or []

    def _norm_bank(x):
        return (x or "").strip()

    def _get_bank(d):
        return _norm_bank(d.get("bank")) or _norm_bank(d.get("bank_folder"))

    def _get_stem(d):
        return (d.get("stem") or "").strip()

    def _get_cid(d):
        try:
            return int(d.get("chunk_id"))
        except Exception:
            return None

    # build meta index
    meta_idx = {}
    for m in meta:
        bk = _get_bank(m)
        st = _get_stem(m)
        cid = _get_cid(m)
        if not bk or not st or cid is None:
            continue
        meta_idx[(bk, st, cid)] = m

    # add originals first
    out = []
    seen = set()
    for h in hits:
        bk = _get_bank(h)
        st = _get_stem(h)
        cid = _get_cid(h)
        if not bk or not st or cid is None:
            continue
        k = (bk, st, cid)
        if k in seen:
            continue
        seen.add(k)
        out.append(h)

    # add neighbors
    for h in list(out):
        bk = _get_bank(h)
        st = _get_stem(h)
        cid = _get_cid(h)
        if cid is None:
            continue
        base_score = float(h.get("score", 0.0) or 0.0)

        for d in range(-window, window + 1):
            if d == 0:
                continue
            nk = (bk, st, cid + d)
            m2 = meta_idx.get(nk)
            if not m2:
                continue
            if nk in seen:
                continue
            seen.add(nk)
            out.append({
                "bank": bk,
                "stem": st,
                "chunk_id": cid + d,
                "text": m2.get("text", ""),
                "score": base_score - 1e-6,  # keep near original
            })

    return out

def search_faiss(index, meta, emb, query: str, topk: int = 50):
    """
    Return list[dict] with keys:
      - score
      - bank (meta bank_folder)
      - stem
      - chunk_id
      - text
    """
    if not query:
        return []
    qvec = emb.encode([query], batch_size=1, normalize_embeddings=True,
                      convert_to_numpy=True).astype(np.float32)
    D, I = index.search(qvec, int(topk))

    out = []
    for score, idx in zip(D[0], I[0]):
        if idx < 0:
            continue
        m = meta[int(idx)]
        out.append({
            "score": float(score),
            "bank": m.get("bank_folder"),
            "stem": m.get("stem"),
            "chunk_id": m.get("chunk_id"),
            "text": m.get("text") or "",
        })
    return out

def retrieve_hits_per_metric(index, meta, emb, bank_id: str, year: int,
                            metrics: list,
                            topk_per_query: int = 40,
                            topk_per_metric: int = 25,
                            min_score: float = 0.50):
    """
    Return: dict(metric_name -> list[hits])  where hits are list[dict] from search_faiss()
    """

    def _norm_bank(x: str) -> str:
        return (x or "").strip().lower()

    def _bank_match(hit_bank: str, target_bank: str) -> bool:
        hb = _norm_bank(hit_bank)
        tb = _norm_bank(target_bank)
        if not hb or not tb:
            return False
        if hb == tb:
            return True
        # tolerate leading underscore or suffix/prefix
        hb2 = hb.lstrip("_")
        tb2 = tb.lstrip("_")
        if hb2 == tb2:
            return True
        if hb2.startswith(tb2) or tb2.startswith(hb2):
            return True
        # guarded substring match
        if len(tb2) >= 8 and tb2 in hb2:
            return True
        if len(hb2) >= 8 and hb2 in tb2:
            return True
        return False

    # --- resolve to an exact bank_folder from meta if possible (fix _GermanAmerican_37640 issue) ---
    target_bank = bank_id
    tb_norm = _norm_bank(bank_id).lstrip("_")
    if tb_norm:
        for mm in meta:
            mb = _norm_bank(mm.get("bank_folder")).lstrip("_")
            if mb == tb_norm:
                target_bank = mm.get("bank_folder")  # use the canonical bank_folder
                break

    # 1) Per-metric multi-query retrieval (keep the original QUERY_BANK behavior).
    QUERY_BANK = {
        "NIM": [
            f"{bank_id} net interest margin {year}",
            f"net interest margin {year}",
            "NIM net interest margin",
            "tax-equivalent net interest margin",
            "net interest margin (nim)",
        ],
        "NII": [
            f"{bank_id} net interest income {year}",
            f"net interest income {year}",
            "NII net interest income",
            "net interest income (nii)",
        ],
        "ROA": [
            f"{bank_id} return on assets {year}",
            f"return on assets {year}",
            "ROA return on assets",
            "ROAA return on average assets",
            "return on average assets",
            "Selected Performance Ratios return on assets 2024",
            "Selected Performance Ratios return on equity 2024",
            "Other Data at Year-end Selected Performance Ratios",
            "Equity to Assets Dividend Payout Return on Assets Return on Equity",
            "ROAA ROAE Selected Performance Ratios",
            "FY2024 return on average assets ROA performance ratios selected performance ratios",
        ],
        "ROE": [
            f"{bank_id} return on equity {year}",
            f"return on equity {year}",
            "ROE return on equity",
            "ROAE return on average equity",
            "return on average equity",
            "Selected Performance Ratios return on assets 2024",
            "Selected Performance Ratios return on equity 2024",
            "Other Data at Year-end Selected Performance Ratios",
            "Equity to Assets Dividend Payout Return on Assets Return on Equity",
            "ROAA ROAE Selected Performance Ratios",
            "FY2024 return on average equity ROE performance ratios selected performance ratios",
        ],
        "Provision for Credit Losses": [
            f"{bank_id} provision for credit losses {year}",
            f"provision for credit losses {year}",
            "provision for credit losses",
            "provision for loan losses",
            "credit loss expense",
            "allowance for credit losses provision",
        ],
    }

    # Extract bank identifier from a hit dict (supports multiple key names).
    def _get_hit_bank(h):
        return h.get("bank") or h.get("bank_id") or h.get("k", "")

    # Return a stable dedup key for a hit (prefers chunk_id).
    def _get_hit_key(h):
        return str(h.get("chunk_id") or h.get("id") or h.get("k") or "")

    # Convenience wrapper: parse similarity score as float.
    def _score(h):
        return float(h.get("score", 0.0))

    out = {}

    for metric in metrics:

        # ROA/ROE often appear in "Selected Performance Ratios" tables; similarity scores can be lower, so we widen retrieval and relax thresholds.
        topk_per_query_eff = topk_per_query + 20 if metric in ("ROA", "ROE") else topk_per_query
        topk_per_metric_eff = 40 if metric in ("ROA", "ROE") else topk_per_metric
        min_score_eff = (min_score - 0.03) if metric in ("ROA", "ROE") else min_score
        queries = QUERY_BANK.get(metric, [f"{bank_id} {metric} {year}", f"{metric} {year}", metric])

        pooled = []
        for q in queries:
            hits = search_faiss(index, meta, emb, query=q, topk=topk_per_query_eff)
            if hits:
                pooled.extend(hits)

        # Bank filter uses tolerant matching to handle minor bank_folder/bank_id formatting differences.
        pooled = [h for h in pooled if _bank_match(_get_hit_bank(h), target_bank)]

        strong = [h for h in pooled if _score(h) >= min_score_eff]

        if not strong and pooled:
            strong = [h for h in pooled if _score(h) >= (min_score_eff - 0.05)]
        if not strong and pooled:
            strong = pooled[:]

        strong.sort(key=_score, reverse=True)
        seen = set()
        dedup = []
        for h in strong:
            key = _get_hit_key(h)
            if not key or key in seen:
                continue
            seen.add(key)
            dedup.append(h)
            if len(dedup) >= topk_per_metric_eff:
                break

        before = len(dedup)
        dedup = _expand_neighbors(dedup, meta, window=2)
        after = len(dedup)
        print(f"[NEI] {metric}: before={before} after={after}", flush=True)
        out[metric] = dedup

    return out

# Prompt enforces an output schema. Downstream normalization assumes this contract; parsing includes a fallback for non-compliant outputs.
def make_prompt(q: str, context: str):
    """
    Build the strict JSON-only extraction prompt for a single metric.
    The prompt enforces: extract only explicit values from context, no inference, and strict citation copying.
    """
    # Full strict prompt for multi-metric extraction
    return (
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


def make_prompt_loose(context: str):
    """
    Build a slightly more tolerant extraction prompt.
    Used when strict prompting is too brittle; still requires JSON-only output and prohibits inference.
    """
    return (
        "You are a financial metric extractor.\n"
        "Return ONLY a JSON object. No extra text.\n"
        "Language: English.\n\n"

        "Task:\n"
        "Extract Net Interest Income (NII) for fiscal year 2024 ONLY.\n\n"

        "Rules:\n"
        "1) ONLY extract if an explicit numeric NII value appears in the Context.\n"
        "2) Do NOT infer, do NOT calculate.\n"
        "3) Copy source_chunk_id EXACTLY from the nearest header like:\n"
        "   [k=...|stem=...|chunk=...]\n"
        "   If the header includes brackets [], keep them.\n"
        "4) If you cannot find NII explicitly, set value/unit/source_chunk_id to \"NOT FOUND\".\n\n"

        "IMPORTANT OUTPUT REQUIREMENTS:\n"
        "- You MUST output EXACTLY this JSON schema with key \"results\".\n"
        "- The JSON must include ALL keys shown below.\n"
        "- Do NOT add any other keys.\n\n"

        "Output JSON (EXACT):\n"
        "{\n"
        "  \"results\": [\n"
        "    {\n"
        "      \"metric_name\": \"NII\",\n"
        "      \"value\": \"NOT FOUND\",\n"
        "      \"unit\": \"NOT FOUND\",\n"
        "      \"fiscal_year\": 2024,\n"
        "      \"source_chunk_id\": \"NOT FOUND\"\n"
        "    }\n"
        "  ]\n"
        "}\n\n"

        "Context:\n"
        + context
    )

def make_repair_prompt(bad_json_text: str, year: int = 2024) -> str:
    """
    Build a prompt that repairs model output into the required results schema.
    This is used as a second pass when the first LLM response is not valid JSON or not in the expected schema.
    """
    return (
        "You MUST output ONLY valid JSON. No markdown. No extra text.\n"
        "Convert the previous answer into EXACTLY this schema.\n\n"
        "IMPORTANT:\n"
        "- Do NOT invent placeholders like 'header_string'.\n"
        "- If the previous answer contains a source_chunk_id, COPY IT EXACTLY.\n"
        "- If missing, set source_chunk_id to \"NOT FOUND\".\n"
        "- Keep fiscal_year as the given year.\n\n"
        "OUTPUT JSON (EXACT):\n"
        "{\n"
        "  \"results\": [\n"
        f"    {{\"metric_name\":\"ROA\",\"value\":\"NOT FOUND\",\"unit\":\"NOT FOUND\",\"fiscal_year\":{year},\"source_chunk_id\":\"NOT FOUND\"}},\n"
        f"    {{\"metric_name\":\"ROE\",\"value\":\"NOT FOUND\",\"unit\":\"NOT FOUND\",\"fiscal_year\":{year},\"source_chunk_id\":\"NOT FOUND\"}},\n"
        f"    {{\"metric_name\":\"NIM\",\"value\":\"NOT FOUND\",\"unit\":\"NOT FOUND\",\"fiscal_year\":{year},\"source_chunk_id\":\"NOT FOUND\"}},\n"
        f"    {{\"metric_name\":\"NII\",\"value\":\"NOT FOUND\",\"unit\":\"NOT FOUND\",\"fiscal_year\":{year},\"source_chunk_id\":\"NOT FOUND\"}},\n"
        f"    {{\"metric_name\":\"Provision for Credit Losses\",\"value\":\"NOT FOUND\",\"unit\":\"NOT FOUND\",\"fiscal_year\":{year},\"source_chunk_id\":\"NOT FOUND\"}}\n"
        "  ]\n"
        "}\n\n"
        "Previous answer:\n"
        + (bad_json_text or "")
    )


def make_prompt_multi_metrics(context: str, year: int = 2024) -> str:
    """
    Build a prompt to extract multiple metrics in one call.
    Prefer per-metric extraction for stability; this function exists for experiments and backward compatibility.
    """
    return (
        "You are a financial metric extractor.\n"
        "Return ONLY a JSON object. No markdown. No extra text.\n\n"
        f"Task: Extract the following metrics for fiscal year {year} ONLY:\n"
        "- ROA\n- ROE\n- NIM\n- NII\n- Provision for Credit Losses\n\n"
        "Rules:\n"
        "1) ONLY extract if an explicit numeric value appears in the Context.\n"
        "2) Do NOT infer or calculate.\n"
        "3) source_chunk_id MUST be copied EXACTLY from the nearest header like: [k=...|stem=...|chunk=...]\n"
        "4) If not found, keep value/unit/source_chunk_id as \"NOT FOUND\".\n"
        "5) For NIM: unit is usually \"%\". If the value appears as 3.63 or 3.63% in the context, set unit to \"%\".\n"
        "6) If value contains a trailing \"%\", remove it from value and keep unit=\"%\".\n\n"
        "Output JSON (EXACT schema):\n"
        "{\n"
        "  \"results\": [\n"
        f"    {{\"metric_name\":\"ROA\",\"value\":\"NOT FOUND\",\"unit\":\"NOT FOUND\",\"fiscal_year\":{year},\"source_chunk_id\":\"NOT FOUND\"}},\n"
        f"    {{\"metric_name\":\"ROE\",\"value\":\"NOT FOUND\",\"unit\":\"NOT FOUND\",\"fiscal_year\":{year},\"source_chunk_id\":\"NOT FOUND\"}},\n"
        f"    {{\"metric_name\":\"NIM\",\"value\":\"NOT FOUND\",\"unit\":\"NOT FOUND\",\"fiscal_year\":{year},\"source_chunk_id\":\"NOT FOUND\"}},\n"
        f"    {{\"metric_name\":\"NII\",\"value\":\"NOT FOUND\",\"unit\":\"NOT FOUND\",\"fiscal_year\":{year},\"source_chunk_id\":\"NOT FOUND\"}},\n"
        f"    {{\"metric_name\":\"Provision for Credit Losses\",\"value\":\"NOT FOUND\",\"unit\":\"NOT FOUND\",\"fiscal_year\":{year},\"source_chunk_id\":\"NOT FOUND\"}}\n"
        "  ]\n"
        "}\n\n"
        "Context:\n"
        + context
    )

METRICS = ["ROA", "ROE", "NIM", "NII", "Provision for Credit Losses"]

def make_prompt_one_metric(metric: str, context: str, year: int = 2024) -> str:
    """
    Build a prompt to extract exactly one metric and return a single-item results list.
    This aligns the model output with downstream CSV writing (results schema).
    """
    # metric: "NIM"/"ROA"/"ROE"/"Provision for Credit Losses"
    return (
        "You are a financial metric extraction engine.\n"
        "Return ONLY a JSON object. No markdown. No extra text.\n\n"
        f"Task: Extract {metric} for fiscal year {year} ONLY.\n\n"
        "Hard rules:\n"
        "1) ONLY extract if an explicit numeric value appears in the Context.\n"
        "2) Do NOT infer, summarize, or calculate.\n"
        "3) source_chunk_id MUST be copied EXACTLY from the nearest header like: [k=...|stem=...|chunk=...]\n"
        "4) If not found, keep value/unit/source_chunk_id as \"NOT FOUND\".\n\n"
        "Output JSON (EXACT schema):\n"
        "{\n"
        "  \"results\": [\n"
        f"    {{\"metric_name\":\"{metric}\",\"value\":\"NOT FOUND\",\"unit\":\"NOT FOUND\",\"fiscal_year\":{year},\"source_chunk_id\":\"NOT FOUND\"}}\n"
        "  ]\n"
        "}\n\n"
        "Context:\n"
        + context
    )

def make_template(year: int):
    """
    Create a default results template for a given fiscal year.
    All metrics are initialized to NOT FOUND to support merge-in of partial extractions.
    """
    return {
        "results": [
            {"metric_name": m, "value": "NOT FOUND", "unit": "NOT FOUND", "fiscal_year": year, "source_chunk_id": "NOT FOUND"}
            for m in METRICS
        ]
    }

import re

# def _valid_cite(x: str) -> bool:
#     if not isinstance(x, str):
#         return False
#           Expected citation format: k=<bank>|stem=<doc>|chunk=<id>.
#     return bool(re.match(r"^k=.+\|stem=.+\|chunk=\d+$", x.strip()))

def _valid_cite(cid: str) -> bool: # Relaxed validation: accept either full header-style citations or a plain numeric chunk id.
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

import re

def try_regex_extract_nii_from_context(context: str, neighbor_k: int = 3, head_scan_chars: int = 2200):
    """
    Improvements:
    1) Collect multiple candidates and select the best-scoring match instead of returning the first hit
    2) Stronger unit inference: current chunk + neighbor chunks + context header + local window near the matched span
    3) Filter obvious false positives: pure years, unrealistically small values without nearby scale indicators
    """
    if not context:
        return None

    blocks = re.split(r"\n\n---\n\n", context)

    header_pat = re.compile(r"^\[k=.*?\|stem=.*?\|chunk=(\d+)\]", flags=re.M)
    # Capture NII numeric values (allow $, commas, and parenthesized negatives)
    val_pat = re.compile(
        r"(net\s+interest\s+income|NII)\b.{0,160}?"
        r"(\(?-?\$?\s*\d{1,3}(?:,\d{3})+(?:\.\d+)?\)?|\(?-?\$?\s*\d+(?:\.\d+)?\)?)",
        flags=re.I | re.S
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

    candidates = []

    head = context[:head_scan_chars]

    for i, blk in enumerate(blocks):
        m_header = header_pat.search(blk)
        if not m_header:
            continue
        chunk_id = m_header.group(1)

        for m in val_pat.finditer(blk):
            val = m.group(2).strip()
            x = _num(val)
            if x is None:
                continue

            # Filter year-like false positives (e.g., 2024/2025)
            if 1900 <= x <= 2100 and abs(x - int(x)) < 1e-9:
                continue

            # 1) Unit signals in the current chunk
            unit = _guess_unit(blk)

            # 2）Local context window around the matched value to infer scale indicators
            # (e.g., thousand / million / billion)
            span_l = max(0, m.start() - 180)
            span_r = min(len(blk), m.end() + 220)
            near = blk[span_l:span_r]
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

            candidates.append((score, val, unit, chunk_id))

    if not candidates:
        return None

    candidates.sort(key=lambda z: z[0], reverse=True)
    best = candidates[0]
    return (best[1], best[2], best[3])


def try_regex_extract_nim_from_context(context: str, head_scan_chars: int = 2400):
    """
    Extract NIM from context using regex. Returns (value, unit, source_chunk_id) or None.
    Looks for patterns like:
      - "net interest margin ... 3.63%"
      - "NIM ... 3.63%"
      - tables with "Net interest margin" and nearby percentage
    """
    if not context:
        return None

    blocks = re.split(r"\n\n---\n\n", context)

    header_pat = re.compile(r"^\[k=.*?\|stem=.*?\|chunk=(\d+)\]", flags=re.M)

    # percentage like 3.63% or 3.63 %
    pct_pat = re.compile(r"(?P<val>\d{1,2}(?:\.\d{1,4})?)\s*%")
    # "net interest margin" or "NIM" within window
    nim_pat = re.compile(r"(net\s+interest\s+margin|interest\s+margin|\bNIM\b)", flags=re.I)

    def _chunk_header(blk: str):
        m = header_pat.search(blk)
        if not m:
            return None
        # return full header line (we want the whole [k=...|chunk=..] string if present)
        line = blk.splitlines()[0].strip()
        return line if line.startswith("[k=") else None

    candidates = []
    head = context[:head_scan_chars]

    for blk in blocks:
        if not nim_pat.search(blk):
            continue
        # find pct near nim keyword
        for m in pct_pat.finditer(blk):
            # use a local window around the match to ensure it's about margin
            s = max(0, m.start() - 180)
            e = min(len(blk), m.end() + 180)
            window = blk[s:e]
            if not nim_pat.search(window):
                continue
            val = m.group("val")
            cid = _chunk_header(blk) or "NOT FOUND"
            # Basic sanity: NIM usually between 0 and 20
            try:
                fv = float(val)
                if fv <= 0 or fv > 50:
                    continue
            except:
                pass
            candidates.append((val, "%", cid, len(window)))

    # fallback: search head only (sometimes margin appears in first blocks)
    if not candidates and nim_pat.search(head):
        for m in pct_pat.finditer(head):
            s = max(0, m.start() - 200)
            e = min(len(head), m.end() + 200)
            window = head[s:e]
            if nim_pat.search(window):
                val = m.group("val")
                candidates.append((val, "%", "NOT FOUND", len(window)))

    if not candidates:
        return None

    # pick best by window length (more context) then earliest (stable)
    candidates.sort(key=lambda x: (-x[3],))
    val, unit, cid, _ = candidates[0]
    return (val, unit, cid)


def try_regex_extract_roa_roe_from_context(context: str, head_scan_chars: int = 2600):
    """
    Try to extract ROA/ROE (usually percentages) directly from context.

    Handles prose like:
      - "Return on average assets and equity were 1.99% and 8.69%, respectively, for the year ended December 31, 2024."
    And table-ish blocks containing "Return on Assets"/"Return on Equity" near percentage values.

    Return: dict with optional keys 'ROA'/'ROE' each as (value, unit, source_chunk_id_headerline)
    """
    if not context:
        return {}

    blocks = re.split(r"\n\n---\n\n", context)
    header_pat = re.compile(r"^\[k=.*?\|stem=.*?\|chunk=\d+\]", flags=re.M)

    pct_pat = re.compile(r"(?P<val>\d{1,2}(?:\.\d{1,4})?)\s*%")

    # keywords
    roa_kw = re.compile(r"(return\s+on\s+(?:average\s+)?(?:total\s+)?assets|\bROA\b|\bROAA\b)", flags=re.I)
    roe_kw = re.compile(r"(return\s+on\s+(?:average\s+)?(?:common\s+)?(?:shareholders[’']?\s+)?equity|\bROE\b|\bROAE\b)", flags=re.I)

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

    for blk in blocks:
        cid = _header_line(blk)

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

        # 2) ROA: pct near keyword window
        if roa_kw.search(blk):
            for m in pct_pat.finditer(blk):
                s = max(0, m.start() - 220)
                e = min(len(blk), m.end() + 220)
                window = blk[s:e]
                # require ROA keyword appears BEFORE the percentage (avoid stealing ROE/other values)
                if not _kw_before(roa_kw, window, m.start()-s, 220):
                    continue
                val = m.group("val")
                sc = _score_candidate(val, window, "ROA")
                prev = best.get("ROA")
                if (prev is None) or (sc > prev[0]):
                    best["ROA"] = (sc, val, "%", cid)

        # 3) ROE: pct near keyword window
        if roe_kw.search(blk):
            for m in pct_pat.finditer(blk):
                s = max(0, m.start() - 240)
                e = min(len(blk), m.end() + 240)
                window = blk[s:e]
                # require ROE keyword appears BEFORE the percentage
                if not _kw_before(roe_kw, window, m.start()-s, 240):
                    continue
                val = m.group("val")
                sc = _score_candidate(val, window, "ROE")
                prev = best.get("ROE")
                if (prev is None) or (sc > prev[0]):
                    best["ROE"] = (sc, val, "%", cid)

    out = {}
    for metric in ("ROA", "ROE"):
        if metric in best:
            _, val, unit, cid = best[metric]
            out[metric] = (val, unit, cid)

    return out

def normalize_extraction(obj, year: int):
    """
    Normalize model outputs into a standard schema:
    {"results": [...]}
    """
    out = make_template(year)
    
    # Compatibility patch:
    # accept flat extraction outputs without an explicit 'results' schema
    if isinstance(obj, dict) and ("value" in obj) and ("source_chunk_id" in obj) and ("results" not in obj):
        out = make_template(int(year))
        metric = "NII"

        val = str(obj.get("value", "")).strip()
        cid = str(obj.get("source_chunk_id", "")).strip()
        fy  = obj.get("fiscal_year", year)

        # find the right row by metric_name
        for row in out["results"]:
            if row.get("metric_name") == metric:
                row["value"] = val if val else "NOT FOUND"
                row["unit"] = obj.get("unit", "NOT FOUND")
                row["fiscal_year"] = int(fy) if str(fy).isdigit() else int(year)
                row["source_chunk_id"] = cid if cid else "NOT FOUND"
                break
        return out

    # --- end patch ---

    # Case A: Already in the expected {'results': [...]} schema.
    if isinstance(obj, dict) and isinstance(obj.get("results"), list) and len(obj["results"]) > 0:
        # Fill into the template when possible by aligning on metric_name.
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

    # Case B: Model returned a dict without 'results' (common example: {'Return on Average Assets': '0.75%'}).  
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

    # Case C: unexpected / unsupported output shape
    
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

# Validate and normalize source_chunk_id before returning
    for row in out["results"]:
        cid = row.get("source_chunk_id", "")
        cid = (cid or "").strip()
        # Allow bracket-wrapped cite IDs and strip surrounding brackets
        if cid.startswith("[") and cid.endswith("]"):
            cid = cid[1:-1].strip()

        if not _valid_cite(cid):
            row["source_chunk_id"] = "NOT FOUND"
        else:
            row["source_chunk_id"] = cid
    
    # ---- Post clean: normalize percent formatting (e.g., value="1.99%" + unit="%") ----
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

def normalize_value_unit(val, unit):
    """
    Normalize (value, unit) pairs.

    Examples:
    - value="1.99%" and unit="%" -> ("1.99", "%")
    - value="8.69 %" -> ("8.69", "%")
    - Empty / missing fields -> "NOT FOUND"
    """
    if val is None:
        return "NOT FOUND", unit or "NOT FOUND"

    s = str(val).strip()
    u = (unit or "").strip()

    if s.endswith("%"):
        s = s[:-1].strip()
        u = "%"

    if not s:
        s = "NOT FOUND"
    if not u:
        u = "NOT FOUND"

    return s, u

# Helper functions for derived metric computation (non-direct extraction)
import re

def _to_intish(s: str):
    if s is None:
        return None
    x = str(s).strip()
    if not x:
        return None
    # remove $ and commas
    x = x.replace("$", "").replace(",", "")
    # handle parentheses negative
    neg = False
    if x.startswith("(") and x.endswith(")"):
        neg = True
        x = x[1:-1].strip()
    # sometimes OCR has weird like "2,/75" -> give up
    if not re.fullmatch(r"-?\d+(\.\d+)?", x):
        return None
    v = float(x)
    if neg:
        v = -v
    return v

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
        if lines[i] == ystr:
            year_idx = i
            break
        # stop if next section encountered before year
        if any(t.lower() in lines[i].lower() for t in stop_titles):
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

import re

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


def maybe_compute_roa_roe_from_context(final_obj: dict, context: str, year: int):
    """
    Optionally compute ROA/ROE from context when explicit values are missing.
    This function is conservative: it only computes when required inputs are confidently extracted,
    otherwise it returns None to avoid introducing inferred values into the output table.
    """
    if not isinstance(final_obj, dict) or "results" not in final_obj:
        return

    # quick check: only compute when missing
    def _get_row(name):
        for r in final_obj.get("results", []):
            if r.get("metric_name") == name:
                return r
        return None

    roa_row = _get_row("ROA")
    roe_row = _get_row("ROE")
    if roa_row is None and roe_row is None:
        return

    need_roa = (roa_row is not None and str(roa_row.get("value", "")).strip().upper() in ("NOT FOUND", "NOT_FOUND", ""))
    need_roe = (roe_row is not None and str(roe_row.get("value", "")).strip().upper() in ("NOT FOUND", "NOT_FOUND", ""))
    if not (need_roa or need_roe):
        return

    # Define before use (avoid referencing variables prior to assignment).
    def _pick_by_contains(d: dict, must_have: list):
        if not d:
            return None
        for k, v in d.items():
            kl = (k or "").lower()
            if all(x in kl for x in must_have):
                if v is not None:
                    return v
        return None

    summary = _parse_vertical_year_table(
        context, section_title="Summary of Operations", year=year,
        stop_titles=["Year-end Balances", "Average Balances", "Per Share Data", "Selected Performance Ratios"]
    )
    avg = _parse_vertical_year_table(
        context, section_title="Average Balances", year=year,
        stop_titles=["Per Share Data", "Selected Performance Ratios", "Other Data at Year-end", "Year-end Balances"]
    )

    print(f"[ENH] parsed keys summary={list(summary.keys())[:8]} avg={list(avg.keys())[:8]}", flush=True)

    net_income = summary.get("Net Income") or _pick_by_contains(summary, ["net", "income"])

    avg_assets = None
    avg_equity = None
    if avg:
        avg_assets = (
            _pick_by_contains(avg, ["average", "assets"])
            or _pick_by_contains(avg, ["avg", "assets"])
            or _pick_by_contains(avg, ["assets"])
        )
        avg_equity = (
            _pick_by_contains(avg, ["average", "equity"])
            or _pick_by_contains(avg, ["avg", "equity"])
            or _pick_by_contains(avg, ["equity"])
        )

    else:
        avg_assets = _find_first_money_after_label(context, [
            "Average Total Assets", "Average total assets", "Avg Total Assets", "Average Assets"
        ])
        avg_equity = _find_first_money_after_label(context, [
            "Average Shareholders' Equity", "Average Shareholders’ Equity",
            "Average Stockholders' Equity", "Average stockholders’ equity", "Average equity"
        ])

    m = re.search(r"\[k=.*?\|stem=.*?\|chunk=\d+\]", context)
    cite = m.group(0) if m else "NOT FOUND"

    if net_income is None or avg_assets is None or avg_equity is None:
        print(f"[ENH] compute ROA/ROE skipped: net_income={net_income} avg_assets={avg_assets} avg_equity={avg_equity}", flush=True)
        return

    # Plausibility checks to prevent invalid financial ratio computation
    try:
        if avg_assets is None or avg_equity is None:
            return None
        # assets/equity should be much larger than net income; also usually > 1e6 in dollars
        if (avg_assets < 1e6) or (avg_equity < 1e6) or (net_income is not None and avg_assets <= max(1.0, net_income)):
            print(f"[ENH] skip ROA/ROE due to implausible avg values: net_income={net_income} avg_assets={avg_assets} avg_equity={avg_equity}", flush=True)
            return None
    except Exception:
        return None
    roa = (net_income / avg_assets) * 100.0 if avg_assets else None
    roe = (net_income / avg_equity) * 100.0 if avg_equity else None

    def _fmt_pct(x):
        if x is None:
            return "NOT FOUND"
        return f"{x:.2f}"

    if need_roa and roa_row is not None and roa is not None:
        roa_row["value"] = _fmt_pct(roa)
        roa_row["unit"] = "%"
        roa_row["fiscal_year"] = int(year)
        roa_row["source_chunk_id"] = f"COMPUTED_FROM {cite} (Net Income / Avg Assets)"

    if need_roe and roe_row is not None and roe is not None:
        roe_row["value"] = _fmt_pct(roe)
        roe_row["unit"] = "%"
        roe_row["fiscal_year"] = int(year)
        roe_row["source_chunk_id"] = f"COMPUTED_FROM {cite} (Net Income / Avg Equity)"


def extract_for_bank(index, meta, emb, bank_id=None, year=YEAR, retrieval_query=DEFAULT_RETRIEVAL_QUERY):
    """
    Run metric extraction for a single bank and fiscal year.
    Pipeline: retrieve evidence -> build context -> regex prefill (where applicable) -> per-metric LLM extraction ->
    schema repair/normalization -> merge into final results dict.
    """
    print(f"[EXTRACT] start bank={bank_id} year={year}", flush=True)
    # 1) Use retrieval-optimized queries to gather evidence (avoid free-form user phrasing).
    target_bank = bank_id  # The batch input bank_id is the target bank identifier.
    hits = retrieve_hits_multiquery(
        index=index,
        meta=meta,
        emb=emb,
        target_bank=target_bank,
        year=year,
        per_metric_topk=10, # Reduced per-metric retrieval size for efficiency
        topk_final=TOPK_FINAL,
        k0=200,
        kmax=20000,
    )

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


    context = build_context(hits)
    
    # DEBUG: persist the assembled context for inspection and reproducibility.
    debug_dir = ROOT / "data" / "outputs" / "debug"
    debug_dir.mkdir(parents=True, exist_ok=True)

    (debug_dir / f"{bank_id}_{year}_context.txt").write_text(
        context,
        encoding="utf-8",
        errors="ignore"
    )
    print("[EXTRACT] context saved", flush=True)
    # ===== END DEBUG =====

    # Stability-first: prefill NII via regex directly from the context before calling the LLM.
    prefill = {}  # metric_name -> {value/unit/source_chunk_id}

    if "NII" in METRICS:
        got = try_regex_extract_nii_from_context(context)
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
    # ====== END ======

    if "NIM" in METRICS:
        got = try_regex_extract_nim_from_context(context)
        if got:
            val, unit, cid = got
            prefill["NIM"] = {
                "metric_name": "NIM",
                "value": val,
                "unit": unit,
                "fiscal_year": int(year),
                "source_chunk_id": cid,
            }
            print("[EXTRACT] regex prefill NIM ok, continue to LLM for other metrics", flush=True)

    # ====== NEW: regex prefill ROA/ROE (prose + ratio table) ======
    got_rr = try_regex_extract_roa_roe_from_context(context)
    if got_rr:
        if "ROA" in got_rr:
            val, unit, cid = got_rr["ROA"]
            prefill["ROA"] = {
                "metric_name": "ROA",
                "value": val,
                "unit": unit,
                "fiscal_year": int(year),
                "source_chunk_id": cid,
            }
            print("[EXTRACT] regex prefill ROA ok", flush=True)
        if "ROE" in got_rr:
            val, unit, cid = got_rr["ROE"]
            prefill["ROE"] = {
                "metric_name": "ROE",
                "value": val,
                "unit": unit,
                "fiscal_year": int(year),
                "source_chunk_id": cid,
            }
            print("[EXTRACT] regex prefill ROE ok", flush=True)
    # ====== END ======

    # 2) LLM extraction per metric (stabilize NIM first, then expand to ROA/ROE/PCL).
    debug_dir = ROOT / "data" / "outputs" / "debug"
    debug_dir.mkdir(parents=True, exist_ok=True)

    # Retrieve evidence per metric (cap chunks per metric to keep prompts short).
    hits_by_metric = retrieve_hits_per_metric(
        index=index,
        meta=meta,
        emb=emb,
        bank_id=target_bank,     # Explicit naming: pass the target bank id into retrieval.
        year=int(year),
        metrics=METRICS,         # Required: enumerate metrics for per-metric retrieval.
        topk_per_query=40,
        topk_per_metric=8,
        min_score=0.50,
    )

    final = make_template(int(year))

    # Apply regex prefills (e.g., NII/NIM) before merging LLM outputs.
    for mname, pobj in prefill.items():
        for row in final["results"]:
            if row["metric_name"] == mname:
                row.update(pobj)
                break

    # Ask the LLM to fill remaining metrics (NIM/ROA/ROE/PCL); NII is considered stable via regex.
    llm_metrics = [m for m in ["NIM", "ROA", "ROE", "Provision for Credit Losses"] if m not in prefill]

    for metric in llm_metrics:
        mhits = hits_by_metric.get(metric, [])
        if not mhits:
            continue

        # Cap the number of evidence blocks to avoid overly long contexts.
        mhits = mhits[:12] if metric in ("ROA", "ROE") else mhits[:8]
        mctx = build_context(mhits)

        (debug_dir / f"{bank_id}_{year}_context_{metric}.txt").write_text(
            mctx, encoding="utf-8", errors="ignore"
        )

        prompt = make_prompt_one_metric(metric, mctx, year=int(year))
        print(f"[DEBUG] prompt({metric}) length = {len(prompt)} chars", flush=True)
        print(f"[EXTRACT] calling ollama metric={metric} ...", flush=True)

        try:
            raw = ollama_generate(prompt)
        except requests.exceptions.ReadTimeout as e:
            print(f"[WARN] ollama timeout bank={bank_id} metric={metric} -> {e}", flush=True)
            continue
        except Exception as e:
            print(f"[WARN] ollama error bank={bank_id} metric={metric} -> {repr(e)}", flush=True)
            continue

        (debug_dir / f"{bank_id}_{year}_raw_{metric}.txt").write_text(
            raw if raw is not None else "",
            encoding="utf-8",
            errors="ignore"
        )

        if not (raw or "").strip():
            continue

        obj = parse_or_fallback(raw, year=int(year))

        # Schema repair: if the model output is not in the expected 'results' format, attempt a repair pass.
        # Some model outputs are non-JSON; repair/fallback keeps the pipeline robust.
        if isinstance(obj, dict) and (not is_results_schema(obj)):
            repair_prompt = make_repair_prompt(raw, year=int(year))
            try:
                raw2 = ollama_generate(repair_prompt)
                (debug_dir / f"{bank_id}_{year}_raw_repair_{metric}.txt").write_text(
                    raw2 if raw2 is not None else "",
                    encoding="utf-8",
                    errors="ignore"
                )
                obj = parse_or_fallback(raw2, year=int(year))
            except Exception as e:
                print(f"[WARN] repair failed bank={bank_id} metric={metric} -> {repr(e)}", flush=True)

        norm = normalize_extraction(obj, int(year))

        # If normalization yields a single-item results list (single metric), merge it back into the final results.
        if isinstance(norm, dict) and isinstance(norm.get("results"), list):
            for it in norm["results"]:
                if not isinstance(it, dict):
                    continue
                if it.get("metric_name") != metric:
                    continue
                # Merge into final results (do not overwrite unrelated metrics).
                for row in final["results"]:
                    if row["metric_name"] == metric:
                        row.update(it)
                        break

    # ====== 3) ENHANCEMENT: compute ROA/ROE if not directly disclosed ======
    try:
        maybe_compute_roa_roe_from_context(final, context, int(year))
    except Exception as e:
        print(f"[ENH][ERROR] compute ROA/ROE failed: {e!r}", flush=True)

    # ===== meta =====
    final["_meta"] = {
        "bank": target_bank,
        "year": int(year),
        "retrieval_query": "PER_METRIC_MULTIQUERY",
        "topk": sum(len(v or []) for v in hits_by_metric.values()),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "prefill": list(prefill.keys()) if prefill else [],
    }
    return final


import csv

def write_jsonl(path: Path, records):
    """
    Append one JSON record per line to a JSONL file.
    Used for audit/debug outputs to keep a durable trace of per-bank extraction results.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def flatten_metrics(records):
    """
    Flatten a normalized results dict into rows for CSV writing.
    Outputs the canonical columns used by the pipeline (bank, year, metric_name, value, unit, source_chunk_id).
    """
    rows = []
    for rec in records:
        meta0 = rec.get("_meta", {})
        bank = meta0.get("bank")
        year = meta0.get("year")

        for item in rec.get("results", []):
            metric = item.get("metric_name", "UNKNOWN")
            val = item.get("value")
            unit = item.get("unit")
            val, unit = normalize_value_unit(val, unit)

            # Audit logging: record only metrics with a concrete extracted value.
            if not _is_not_found(val):
                AUDIT_ROWS.append({
                    "bank": bank,
                    "year": year,
                    "metric": metric,
                    "val": val,
                    "unit": unit,
                    "chunk": item.get("source_chunk_id"),
                })

            rows.append({
                "bank": bank,
                "year": year,
                "metric_name": metric,
                "val": val,
                "unit": unit,
                "source_chunk_id": item.get("source_chunk_id"),
            })
    return rows


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
                    year_mismatch.append(m or "UNKNOWN")

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


def write_metrics_csv(path: Path, rows):
    """
    Write the final metrics table to CSV.
    Consumes flattened rows and writes a stable schema for downstream analysis (pivot tables, QA, reporting).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = ["bank", "year", "metric_name", "val", "unit", "source_chunk_id"]
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)


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
            parts = q.split(maxsplit=1)
            if len(parts) != 2:
                print("Usage: :batch <banks.txt>", flush=True)
                continue
            bank_file = Path(parts[1].strip())

            # If a relative path is provided, resolve it relative to the repository root.
            if not bank_file.is_absolute():
                bank_file = (ROOT / bank_file).resolve()

            if not bank_file.exists():
                raise FileNotFoundError(f"banks file not found: {bank_file}")

            banks = [
                ln.strip()
                for ln in bank_file.read_text(encoding="utf-8").splitlines()
                if ln.strip()
            ]

            out_jsonl = ROOT / "data" / "outputs" / "log" / f"extractions_{YEAR}.jsonl"
            out_csv   = ROOT / "data" / "outputs" / "processed" / f"metrics_{YEAR}.csv"

            records = []
            for i, b in enumerate(banks, 1):
                print(f"[BATCH] {i}/{len(banks)} bank={b}", flush=True)
                try:
                    rec = extract_for_bank(index, meta, emb, bank_id=b, year=YEAR)
                except Exception as e:
                    print("[BATCH][ERROR]", b, "->", repr(e), flush=True)
                    traceback.print_exc()
                    rec = make_template(int(YEAR))
                    rec["_meta"] = {"bank": b, "year": int(YEAR)}
                    rec["error"] = repr(e)
                records.append(rec)

            write_jsonl(out_jsonl, records)
            rows = flatten_metrics(records)
            write_metrics_csv(out_csv, rows)
            print(f"[DONE] wrote: {out_jsonl}", flush=True)
            print(f"[DONE] wrote: {out_csv}", flush=True)
            audit_csv = ROOT / "data" / "outputs" / "logs" / f"write_audit_{YEAR}.csv"
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
            print("[DEBUG] encoding query ...", flush=True)
            qvec = emb.encode([q], batch_size=1, normalize_embeddings=True,
            convert_to_numpy=True).astype(np.float32)

            print("[DEBUG] searching faiss ...", flush=True)
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

            print(f"[DEBUG] prompt length = {len(prompt)} chars", flush=True)
            print("[DEBUG] calling ollama ...", flush=True)
            ans = ollama_generate(prompt)
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