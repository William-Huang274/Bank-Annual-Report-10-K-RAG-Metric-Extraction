# rag_chat_2024_ollama.py
# pip install -U faiss-cpu sentence-transformers numpy requests
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
    blocks = []
    for rank, score, m in hits:
        text = (m.get("text") or "").strip()
        MAXC = 1200
        HEAD = 600
        TAIL = 600
        if len(text) > MAXC:
            text = text[:HEAD] + "\n...[middle truncated]...\n" + text[-TAIL:]


        bank = m.get("bank_folder")
        stem = m.get("stem")
        chunk_id = m.get("chunk_id")

        # 关键：引用头，LLM 直接复制这个作为 source_chunk_id
        header = f"[k={bank}|stem={stem}|chunk={chunk_id}]"

        blocks.append(
            f"{header}\n"
            f"(rank={rank} score={score:.4f})\n"
            f"{text}"
        )
    return "\n\n---\n\n".join(blocks)

def ollama_generate(prompt: str):
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        # ⚠️ 先不要强制 json
        # "format": "json",
        "options": {"temperature": 0, "stop": ["\n\n---\n\n", "\nQ (empty to exit):"] }
    }
    r = requests.post(OLLAMA_URL, json=payload, timeout=120)

    # ===== 新增 DEBUG =====
    debug_text = r.text
    if not debug_text.strip():
        print("[WARN] Ollama returned empty HTTP body")
    else:
        print("[DEBUG] Ollama raw HTTP body (head):")
        print(debug_text[:500])
    # ======================

    if r.status_code != 200:
        raise RuntimeError(f"Ollama HTTP {r.status_code}: {debug_text[:500]}")

    j = r.json()
    return j.get("response", "")

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
    metric_queries = {
        "ROA": f"FY{year} return on assets ROA ROAA return on average assets",
        "ROE": f"FY{year} return on equity ROE ROAE return on average equity",
        "NIM": f"FY{year} net interest margin NIM",
        "NII": f"FY{year} net interest income NII net interest revenue",
        "Provision for Credit Losses": f"FY{year} provision for credit losses PCL provision credit losses",
    }

    # 聚合候选（按 chunk 唯一键去重）
    seen = set()
    pooled = []

    for metric, rq in metric_queries.items():
        qvec = emb.encode([rq], batch_size=1, normalize_embeddings=True, convert_to_numpy=True).astype(np.float32)

        k = k0
        bank_hits = []
        while True:
            D, I = index.search(qvec, k)
            tmp = []
            for rnk, (score, idx) in enumerate(zip(D[0], I[0]), 1):
                m = meta[int(idx)]
                if m.get("bank_folder") == target_bank:
                    tmp.append((rnk, float(score), m))
                    if len(tmp) >= per_metric_topk:
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
    print("[DEBUG] example meta bank =", pooled[0][0].get("bank") if pooled else None, flush=True)

    # pooled 里按 score 排一下（score 越大越相关；你这里是 cosine 相似度）
    pooled.sort(key=lambda x: x[1], reverse=True)

    # ===== 新增：相似度阈值过滤（非常重要）=====（202601062040）
    filtered = [h for h in pooled if h[1] >= MIN_SCORE]
    if len(filtered) >= 3:   # 至少保留 3 个证据再启用阈值
        pooled = filtered

    # 最终给 LLM 的证据
    final_hits = pooled[:topk_final]
    return final_hits

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
        "Return ONLY JSON. No extra text.\n"
        "Language: English.\n\n"
        "Task: Extract Net Interest Margin (NIM) for fiscal year 2024 ONLY.\n"
        "Rules:\n"
        "1) Only extract if an explicit numeric NIM value appears in the context.\n"
        "2) Do NOT infer.\n"
        "3) source_chunk_id must be copied EXACTLY from the nearest [k=...|stem=...|chunk=...] header.\n\n"
        "You MUST output EXACTLY this JSON and nothing else:\n"
        "{\n"
        "  \"results\": [\n"
        "    {\"metric_name\":\"NIM\",\"value\":\"NOT FOUND\",\"unit\":\"NOT FOUND\",\"fiscal_year\":2024,\"source_chunk_id\":\"NOT FOUND\"}\n"
        "  ]\n"
        "}\n\n"
        "Context:\n"
        + context
    )

METRICS = ["NIM"]

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

def normalize_extraction(obj, year: int):
    """
    把模型的各种乱输出，统一规整成标准 {"results":[5条]}.
    """
    out = make_template(year)

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
    return out

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


    context = build_context(hits[:3])
    
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

    # 2) LLM 抽取（你也可以把 q 设成固定抽取指令）
    q = "Extract explicitly stated financial metrics for fiscal year 2024, including NII, NIM, ROA, ROE, and provision for credit losses."
    # 抽取阶段：用宽 prompt
    prompt = make_prompt_loose(context)
    print(f"[DEBUG] prompt length = {len(prompt)} chars", flush=True)
    #新增
    print("[EXTRACT] calling ollama...", flush=True)
    raw = ollama_generate(prompt)
    print(f"[EXTRACT] ollama returned len={len(raw or '')}", flush=True)


    # ===== DEBUG: save raw LLM output =====
    (debug_dir / f"{bank_id}_{year}_raw.txt").write_text(
    raw if raw is not None else "",
    encoding="utf-8",
    errors="ignore"
    )

    if not (raw or "").strip():
        obj = make_template(int(year))
        obj["_meta"] = {"bank": target_bank, "year": int(year)}
        obj["error"] = "EMPTY_LLM_RESPONSE"
        return obj
    # ===== END DEBUG =====

    obj = parse_json_loose(raw)
    obj = normalize_extraction(obj, int(year))
    # 补充元信息，便于落盘/后处理
    obj["_meta"] = {
        "bank": target_bank,
        "year": int(year),
        "retrieval_query": retrieval_query,
        "topk": len(hits),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }
    return obj
import csv

def write_jsonl(path: Path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def flatten_metrics(records):
    rows = []
    for rec in records:
        meta = rec.get("_meta", {})
        bank = meta.get("bank")
        year = meta.get("year")
        for item in rec.get("results", []):
            rows.append({
                "bank": bank,
                "year": year,
                "metric_name": item.get("metric_name"),
                "value": item.get("value"),
                "unit": item.get("unit"),
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
                v = it.get("value", None)
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
    cols = ["bank", "year", "metric_name", "value", "unit", "source_chunk_id"]
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
        q = input("\nQ (empty to exit): ").strip()
        if not q:
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
            context = build_context(hits[:3])

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
                "  \"value\": \"NOT FOUND\",\n"
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
