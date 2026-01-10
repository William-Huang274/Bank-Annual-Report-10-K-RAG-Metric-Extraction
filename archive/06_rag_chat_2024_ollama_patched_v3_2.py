# rag_chat_2024_ollama.py
# pip install -U faiss-cpu sentence-transformers numpy requests
AUDIT_ROWS = []
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
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

PROJECT_ROOT = Path(r"D:\Annual report LLM project\LLM project_20251207")
YEAR = "2024"

INDEX_DIR = PROJECT_ROOT / "data" / "index" / f"faiss_{YEAR}_full"
INDEX_PATH = INDEX_DIR / "faiss.index"
META_PATH  = INDEX_DIR / "meta.jsonl"

EMB_MODEL = r"D:\LKY SCH OF PUBLIC POLICY RA program\LLM_models\models--BAAI--bge-m3\snapshots\5617a9f61b028005a4858fdac845db406aefb181"
EMB_DEVICE = "cuda"

TOPK_SEARCH = 50   # 先从全库取更大的候选，新增
TOPK_FINAL  = 20   # 再过滤到同一家 bank 后取前 10，新增/20260106修改为20

TOPK = 10

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "qwen3:4b"   # 改成你 ollama list 里存在的
TEMPERATURE = 0.2

def load_meta(path: Path):
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

        # 默认截断（短上下文）
        MAXC = 1400
        HEAD = 700
        TAIL = 700

        # ✅ 表格/比率表：放宽截断，避免把 2024 数值列截掉
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

        header = f"[k={bank}|stem={stem}|chunk={chunk_id}]"
        blocks.append(
            f"{header}\n"
            f"(rank={rank} score={score:.4f})\n"
            f"{text}"
        )

    return "\n\n---\n\n".join(blocks)


def ollama_generate(prompt: str) -> str:
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "format": "json",  # ✅ 强制 JSON
        "options": {
            "temperature": 0,
            # ✅ 删掉 ---，避免把输出截断成空 response
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

    # ✅ 关键：优先 response；为空再用 thinking（你已经有这个逻辑了）
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
    s = (s or "").strip()
    # 1) 直接 json.loads
    try:
        return json.loads(s)
    except Exception:
        pass

    # 2) 用 raw_decode：从任意位置找到第一个合法 JSON
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
    先尝试 parse_json_loose；失败则从“非 JSON 解释性回答”里兜底提取 value + chunk header。
    返回：
      - 合规 obj（可能是 {"results":[...]}）
      - 或扁平 dict {"value": "...", "unit": "...", "source_chunk_id": "...", "fiscal_year": 2024}
    """
    s = (raw_text or "").strip()

    # 1) 正常 JSON
    try:
        return parse_json_loose(s)
    except Exception:
        pass

    # 2) 尝试从文本里抠一个 JSON 子串（有些模型会先说一堆，再给 { ... }）
    #    找到第一个 "{" 开始，直到最后一个 "}" 结束
    l = s.find("{")
    r = s.rfind("}")
    if l != -1 and r != -1 and r > l:
        candidate = s[l:r+1]
        try:
            return parse_json_loose(candidate)
        except Exception:
            pass

    # 3) 彻底不是 JSON：用 regex 兜底抽取 NII（以及 chunk header）
    #    value: 优先抓 190,591 / 190591 / $190,591 / 190.591 等
    m_val = re.search(r"(?i)\bvalue\b[^0-9$]*\$?\s*`?\s*([0-9][0-9,\.]*)", s)
    value = m_val.group(1) if m_val else "NOT FOUND"
    value = value.replace(",", "") if value != "NOT FOUND" else value

    #    unit: 很多时候模型会写 (in thousands) / thousand dollars / million 等；抓不到就 NOT FOUND
    m_unit = re.search(r"(?i)\bunit\b[^A-Za-z]*`?\s*([A-Za-z][A-Za-z \-\(\)%]+)", s)
    unit = m_unit.group(1).strip() if m_unit else "NOT FOUND"

    #    chunk: 抓 [k=...|stem=...|chunk=12] 这种；抓不到再退化成纯数字 chunk
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
    D, I = index.search(qvec, topk_search)

    hits = []
    for rnk, (score, idx) in enumerate(zip(D[0], I[0]), 1):
        hits.append((rnk, float(score), meta[int(idx)]))

    if not hits:
        return [], None

    # 默认：用 top1 bank 作为目标 bank（你现在就是这么做的 :contentReference[oaicite:1]{index=1}）
    if target_bank is None:
        target_bank = hits[0][2].get("bank_folder")

    # 过滤到同一 bank
    hits = [h for h in hits if h[2].get("bank_folder") == target_bank][:topk_final]
    return hits, target_bank

def _prefer_year_hits(hits, year: str):
    """
    hits: list[(rnk, score, meta)]
    优先保留 stem 含 year 的命中；如果一个都没有，则原样返回（避免全空）
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
    try:
        return (h.get("bank"), h.get("stem"), int(h.get("chunk_id")))
    except Exception:
        return None

def _build_meta_lookup(meta: list):
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
    per_metric_topk: int = 10, # 新修改为10，原为30
    topk_final: int = 20, #新修改为20，原为10
    k0: int = 200,
    kmax: int = 20000,
    MIN_SCORE = 0.50,  # 相似度阈值，0.52~0.60 之间可调,新修为0.50
):
    """
    关键改动：
    - 5 个指标分开检索（更容易命中指标段）
    - 全库检索 K 自适应翻倍，直到捞到该 bank 的足够 hits
    - year/stem 过滤优先 2024
    """

    def _expand_neighbors(dedup_hits, meta, window=1):
        # dedup_hits: list[dict] (search_faiss 返回那种)
        # meta: 你加载的 meta.jsonl 全量列表
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


        # 先把原 hit 放进去
        for h in dedup_hits:
            k = _key(h)
            if k in keyset:
                continue
            keyset.add(k)
            out.append(h)

        # 建一个快速索引： (bank, stem, chunk_id) -> meta_item
        idx = {}
        for m in meta:
            try:
                bk = m.get("bank") or m.get("bank_folder")
                st = m.get("stem")
                cid = int(m.get("chunk_id"))
                idx[(bk, st, cid)] = m
            except Exception:
                continue

        # 给每个 hit 补邻居
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
                # 伪造一个 hit dict，分数略低一点防止排序乱
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

    # --- robust bank id matching (batch bank_id may not equal meta['bank_folder'] 1:1) ---
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

    # 聚合候选（按 chunk 唯一键去重）
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

        # year 优先
        bank_hits = _prefer_year_hits(bank_hits, year)

        # 合并去重（同一个 chunk 只保留一次）
        for h in bank_hits:
            m = h[2]
            key = (m.get("bank_folder"), m.get("stem"), m.get("chunk_id"))
            if key in seen:
                continue
            seen.add(key)
            pooled.append(h)

    # # pooled 里按 score 排一下（score 越大越相关；你这里是 cosine 相似度）
    # pooled.sort(key=lambda x: x[1], reverse=True)

    # # 最终给 LLM 的证据
    # final_hits = pooled[:topk_final]
    # return final_hits
    print("[DEBUG] pooled before bank-filter =", len(pooled), flush=True)
    print("[DEBUG] example bank_folder =", pooled[0][2].get("bank_folder") if pooled else None, flush=True)

    # pooled 里按 score 排一下（score 越大越相关；你这里是 cosine 相似度）
    pooled.sort(key=lambda x: x[1], reverse=True)

    # ===== 新增：相似度阈值过滤（非常重要）=====（202601062040）
    filtered = [h for h in pooled if h[1] >= MIN_SCORE]
    if len(filtered) >= 3:   # 至少保留 3 个证据再启用阈值
        pooled = filtered

    # 最终给 LLM 的证据
        # 最终给 LLM 的证据
    final_hits = pooled[:topk_final]

    # ===== 修复：tuple hit -> dict hit，才能喂给 _expand_neighbors =====
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
    Return: hits + neighbor chunks (±window) per hit, deduped by (bank, stem, chunk_id)
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

    # 1) per-metric multi queries（保留你原来的 QUERY_BANK）
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

    def _get_hit_bank(h):
        return h.get("bank") or h.get("bank_id") or h.get("k", "")

    def _get_hit_key(h):
        return str(h.get("chunk_id") or h.get("id") or h.get("k") or "")

    def _score(h):
        return float(h.get("score", 0.0))

    out = {}

    for metric in metrics:
        queries = QUERY_BANK.get(metric, [f"{bank_id} {metric} {year}", f"{metric} {year}", metric])

        pooled = []
        for q in queries:
            hits = search_faiss(index, meta, emb, query=q, topk=topk_per_query)
            if hits:
                pooled.extend(hits)

        # ✅ bank filter：用容错 match（关键修复点）
        pooled = [h for h in pooled if _bank_match(_get_hit_bank(h), target_bank)]

        strong = [h for h in pooled if _score(h) >= min_score]

        if not strong and pooled:
            strong = [h for h in pooled if _score(h) >= (min_score - 0.05)]
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
            if len(dedup) >= topk_per_metric:
                break

        before = len(dedup)
        dedup = _expand_neighbors(dedup, meta, window=2)
        after = len(dedup)
        print(f"[NEI] {metric}: before={before} after={after}", flush=True)
        out[metric] = dedup

    return out

def make_prompt(q: str, context: str):
    # 复用你现在的强模板 prompt（逻辑不变 :contentReference[oaicite:2]{index=2}）
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
#     # 必须像: k=Bank_xxx|stem=xxx|chunk=123
#     return bool(re.match(r"^k=.+\|stem=.+\|chunk=\d+$", x.strip()))

def _valid_cite(cid: str) -> bool: # 新增，放宽规则
    if not isinstance(cid, str):
        return False
    cid = cid.strip().strip("[]")
    return (
        "chunk=" in cid
        or cid.isdigit()
    )

def is_results_schema(obj) -> bool:
    if not isinstance(obj, dict):
        return False
    rs = obj.get("results")
    if not isinstance(rs, list):
        return False
    # 至少包含一个 dict，且有 metric_name/value 字段（你模板就是这样）
    for it in rs:
        if isinstance(it, dict) and ("metric_name" in it) and ("value" in it):
            return True
    return False

def _guess_unit(text: str) -> str:
    t = (text or "").lower()

    # --- NEW: $ heuristic (many tables omit wording but show $) ---
    # 如果出现 $，但没明确 thousand/million，先给 dollars（后续邻域/头部再细化成 thousand/million）
    if "$" in (text or ""):
        # 先不直接返回 thousand，避免误判；先标记 dollars
        # 如果文本里同时出现 in thousands/millions，会在下面覆盖
        unit_dollar = "dollars"
    else:
        unit_dollar = None

    # 常见写法： (dollars in thousands) / ($ in thousands) / (in thousands)
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

    # NEW: 更广一些的表头写法
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
    改进点：
    1) 不再“命中第一条就 return”，而是收集候选并打分选最佳
    2) unit 推断更强：当前块 + 邻域块 + context 头部 + 命中位置附近窗口
    3) 过滤明显伪命中：纯年份、过小金额且附近无 million/billion
    """
    if not context:
        return None

    blocks = re.split(r"\n\n---\n\n", context)

    header_pat = re.compile(r"^\[k=.*?\|stem=.*?\|chunk=(\d+)\]", flags=re.M)
    # 捕捉 NII 数字（允许 $、逗号、括号负数）
    val_pat = re.compile(
        r"(net\s+interest\s+income|NII)\b.{0,160}?"
        r"(\(?-?\$?\s*\d{1,3}(?:,\d{3})+(?:\.\d+)?\)?|\(?-?\$?\s*\d+(?:\.\d+)?\)?)",
        flags=re.I | re.S
    )

    def _num(s: str):
        # 变成 float（去掉 $,逗号,括号）
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
        """更局部的规模词识别（比 _guess_unit 更贴近命中位置）"""
        t = (text or "").lower()
        # 常见隐式写法
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

            # 过滤年份伪命中：2024/2025
            if 1900 <= x <= 2100 and abs(x - int(x)) < 1e-9:
                continue

            # 1) 当前块单位
            unit = _guess_unit(blk)

            # 2) 命中位置附近窗口（非常关键：抓 billion/million 就靠这个）
            span_l = max(0, m.start() - 180)
            span_r = min(len(blk), m.end() + 220)
            near = blk[span_l:span_r]
            unit_near = _scale_from_near(near)
            if unit_near != "NOT FOUND":
                unit = unit_near

            # 3) 邻域块（表头通常在前面）
            if unit in ("NOT FOUND", "dollars"):
                neigh = "\n\n".join(blocks[max(0, i-neighbor_k): min(len(blocks), i+neighbor_k+1)])
                unit2 = _guess_unit(neigh)
                if unit2 != "NOT FOUND":
                    unit = unit2
                else:
                    unit2b = _scale_from_near(neigh)
                    if unit2b != "NOT FOUND":
                        unit = unit2b

            # 4) context 头部扫一把（很多表格标题都在前面）
            if unit in ("NOT FOUND", "dollars"):
                unit3 = _guess_unit(head)
                if unit3 != "NOT FOUND":
                    unit = unit3
                else:
                    unit3b = _scale_from_near(head)
                    if unit3b != "NOT FOUND":
                        unit = unit3b

            # 小额过滤：如果 unit 仍然只是 dollars 且数值太小（<1000）且没有逗号，基本就是误命中
            # 除非附近抓到了 million/billion（那就不算小）
            if unit == "dollars":
                has_comma = ("," in val)
                if (not has_comma) and abs(x) < 1000:
                    # 把它当成低可信候选，但不立刻返回
                    penalty_small = 1
                else:
                    penalty_small = 0
            else:
                penalty_small = 0

            # 评分：优先 scale 明确 > 只剩 dollars；优先大数/有逗号；惩罚小额伪命中
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


def normalize_extraction(obj, year: int):
    """
    把模型的各种乱输出，统一规整成标准 {"results":[5条]}.
    """
    out = make_template(year)
    
    # --- PATCH: accept flat extraction like {"value": "...", "source_chunk_id": "..."} ---
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

    # Case A: 已经是合规 results list
    if isinstance(obj, dict) and isinstance(obj.get("results"), list) and len(obj["results"]) > 0:
        # 尽量把它填回 template（按 metric_name 对齐）
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

    # Case B: 模型返回的是一个 dict，但没有 results（常见：{"Return on Average Assets":"0.75%"}）
    if isinstance(obj, dict):
        keymap = {
            # 常见同义词/乱 key
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

        # 把这些键值塞到 template 的 value 里（unit/source_chunk_id 还是 NOT FOUND）
        for k, v in obj.items():
            if k in keymap:
                metric = keymap[k]
                for row in out["results"]:
                    if row["metric_name"] == metric:
                        row["value"] = str(v)
                        # unit/source_chunk_id 没证据就先 NOT FOUND
                        break
        return out

    # Case C: 其它情况（解析失败/空/乱七八糟）
    
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

# ... 在 normalize_extraction() 返回 out 之前做校验
    for row in out["results"]:
        cid = row.get("source_chunk_id", "")
        cid = (cid or "").strip()
        # 允许模型返回带方括号的也行：把 [ ] 去掉
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
    - 把 '1.99%' + '%' → ('1.99', '%')
    - 把 '8.69 %' → ('8.69', '%')
    - 兜底 NOT FOUND
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

#新增函数，做模糊计算，不是直接抽取数据
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
    if not text:
        return None
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]

    # 更严格：优先抓 1,234,567 这种
    money_commas = re.compile(r"(?<!\d)(\(?\$?\d{1,3}(?:,\d{3})+(?:\.\d+)?\)?)(?!\d)")
    # 兜底抓普通数字，但要过滤年份/百分数
    num_plain = re.compile(r"(?<!\d)(\(?\$?\d+(?:\.\d+)?\)?)(?!\d)")

    def _clean_to_float(s: str):
        s = s.strip()
        if "%" in s:
            return None
        # 去 $, 逗号, 括号负数
        neg = s.startswith("(") and s.endswith(")")
        s2 = s.replace("$", "").replace(",", "").strip("()")
        try:
            v = float(s2)
        except:
            return None
        if neg:
            v = -v
        # 过滤年份
        if 1900 <= v <= 2100 and abs(v - int(v)) < 1e-9:
            return None
        return v

    labels_l = [x.lower() for x in labels if x]
    for i, ln in enumerate(lines):
        lnl = ln.lower()
        if not any(lb in lnl for lb in labels_l):
            continue

        # 在本行+后续几行找数（避免只抓到年份）
        window = " ".join(lines[i:i+6])

        # 1) 先抓带逗号的大数
        for m in money_commas.finditer(window):
            v = _clean_to_float(m.group(1))
            if v is not None:
                return v

        # 2) 再兜底抓普通数（但过滤严格）
        for m in num_plain.finditer(window):
            v = _clean_to_float(m.group(1))
            if v is not None and abs(v) >= 1000:  # 额外约束：资产/权益至少应很大
                return v

    return None


def maybe_compute_roa_roe_from_context(final_obj: dict, context: str, year: int):
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

    # ✅ 先定义再使用
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
    print(f"[EXTRACT] start bank={bank_id} year={year}", flush=True)
    # 1) 用“检索专用 query”取证据（不要用用户随口问法）
    target_bank = bank_id  # batch 里你传进来的就是目标 bank
    hits = retrieve_hits_multiquery(
        index=index,
        meta=meta,
        emb=emb,
        target_bank=target_bank,
        year=year,
        per_metric_topk=10, # 新修改为10，原为30
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
    
    # ===== DEBUG: save context =====新增
    debug_dir = PROJECT_ROOT / "data" / "outputs" / "debug"
    debug_dir.mkdir(parents=True, exist_ok=True)

    (debug_dir / f"{bank_id}_{year}_context.txt").write_text(
        context,
        encoding="utf-8",
        errors="ignore"
    )
    print("[EXTRACT] context saved", flush=True)
    # ===== END DEBUG =====

    # ====== 稳定性优先：先 regex 直接从 context 抠 NII ======
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

    # ====== 2) LLM：按指标单独抽取（先稳 NIM，再扩 ROA/ROE/PCL） ======
    debug_dir = PROJECT_ROOT / "data" / "outputs" / "debug"
    debug_dir.mkdir(parents=True, exist_ok=True)

    # 先按指标拿证据（每个指标最多 6~8 块，prompt 立刻变短）
    hits_by_metric = retrieve_hits_per_metric(
        index=index,
        meta=meta,
        emb=emb,
        bank_id=target_bank,     # ✅ 改名
        year=int(year),
        metrics=METRICS,         # ✅ 必须传
        topk_per_query=40,
        topk_per_metric=8,
        min_score=0.50,
    )

    final = make_template(int(year))

    # 先把 regex prefill（NII / NIM 等）写进去
    for mname, pobj in prefill.items():
        for row in final["results"]:
            if row["metric_name"] == mname:
                row.update(pobj)
                break

    # 只让 LLM 补：NIM / ROA / ROE / PCL（NII 已经 regex 稳了）
    llm_metrics = [m for m in ["NIM", "ROA", "ROE", "Provision for Credit Losses"] if m not in prefill]

    for metric in llm_metrics:
        mhits = hits_by_metric.get(metric, [])
        if not mhits:
            continue

        # 控制证据块数量，避免超长
        mhits = mhits[:6]
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

        # schema repair（如果不是 results schema）
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

        # norm 只有 1 条 results（单指标），把它 merge 回 final
        if isinstance(norm, dict) and isinstance(norm.get("results"), list):
            for it in norm["results"]:
                if not isinstance(it, dict):
                    continue
                if it.get("metric_name") != metric:
                    continue
                # 合并到 final
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
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def flatten_metrics(records):
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

            # audit: 只记录真的找到值的
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
    if x is None:
        return True
    s = str(x).strip().upper()
    return s in ("NOT FOUND", "NOT_FOUND", "")

# def _valid_cite(cid: str) -> bool:
#     if not isinstance(cid, str):
#         return False
#     cid = cid.strip()
#     if cid.startswith("[") and cid.endswith("]"):
#         cid = cid[1:-1].strip()
#     # 你现在 build_context 的 cite 形如：k=Bank|stem=...|chunk=123
#     return bool(re.match(r"^k=.+\|stem=.+\|chunk=\d+$", cid))

def _extract_stem(cid: str) -> str:
    if not isinstance(cid, str):
        return ""
    cid = cid.strip()
    if cid.startswith("[") and cid.endswith("]"):
        cid = cid[1:-1].strip()
    m = re.search(r"stem=([^|]+)", cid)
    return m.group(1) if m else ""

def analyze_extractions(jsonl_path: Path, out_csv_path: Path, year: int):
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

            # 分类统计
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

            # 结果是 list
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

                # 引用合规统计：只要不是 NOT FOUND，就要求符合 cite 格式
                if not _is_not_found(cid) and (not _valid_cite(cid)):
                    bad_cite += 1

                # 年份 mismatch：stem 明显是 2023/2022 但你在跑 2024
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

    # 写 diagnostics.csv
    out_csv_path.parent.mkdir(parents=True, exist_ok=True)
    cols = ["bank", "year", "topk", "status", "error", "n_found", "found_metrics", "n_bad_cite", "year_mismatch_metrics"]
    with out_csv_path.open("w", encoding="utf-8", newline="") as fw:
        w = csv.DictWriter(fw, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    # 终端汇总
    print("\n=== STATS SUMMARY ===", flush=True)
    print(f"[STATS] jsonl: {jsonl_path}", flush=True)
    print(f"[STATS] csv : {out_csv_path}", flush=True)
    for k, v in cnt.most_common():
        print(f"{k}: {v}", flush=True)
    print("=====================\n", flush=True)


def write_metrics_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = ["bank", "year", "metric_name", "val", "unit", "source_chunk_id"]
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main():
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

            # 如果是相对路径，统一相对于项目根目录
            if not bank_file.is_absolute():
                bank_file = (PROJECT_ROOT / bank_file).resolve()

            if not bank_file.exists():
                raise FileNotFoundError(f"banks file not found: {bank_file}")

            banks = [
                ln.strip()
                for ln in bank_file.read_text(encoding="utf-8").splitlines()
                if ln.strip()
            ]

            out_jsonl = PROJECT_ROOT / "data" / "outputs" / f"extractions_{YEAR}.jsonl"
            out_csv   = PROJECT_ROOT / "data" / "outputs" / f"metrics_{YEAR}.csv"

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
            audit_csv = PROJECT_ROOT / "data" / "outputs" / f"write_audit_{YEAR}.csv"
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
            # 用法：:stats [optional_jsonl_path]
            parts = q.split(maxsplit=1)
            if len(parts) == 2:
                jsonl_path = Path(parts[1].strip())
            else:
                jsonl_path = PROJECT_ROOT / "data" / "outputs" / f"extractions_{YEAR}.jsonl"

            out_csv = PROJECT_ROOT / "data" / "outputs" / f"diagnostics_{YEAR}.csv"
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

            # 只取 3 个证据，避免 ctx 太长
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
            # 继续下一轮 input，不要退出

if __name__ == "__main__":
    main()
