# embed_stage1_2024.py

from pathlib import Path
import os

PROJECT_ROOT = Path(r"D:\Annual report LLM project\LLM project_20251207")

TMP_DIR = PROJECT_ROOT / "_tmp"
TMP_DIR.mkdir(parents=True, exist_ok=True)
os.environ["TMP"] = str(TMP_DIR)
os.environ["TEMP"] = str(TMP_DIR)
os.environ["TMPDIR"] = str(TMP_DIR)

HF_HOME = PROJECT_ROOT / "_hf_cache"
HF_HOME.mkdir(parents=True, exist_ok=True)
os.environ["HF_HOME"] = str(HF_HOME)
os.environ["HUGGINGFACE_HUB_CACHE"] = str(HF_HOME / "hub")
os.environ["TRANSFORMERS_CACHE"] = str(HF_HOME / "transformers")
os.environ["TORCH_HOME"] = str(HF_HOME / "torch")

# —— 到这里才开始 import 其它库 ——
import re, json, csv, time, traceback
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer


# ====== 配置区 ======
PROJECT_ROOT = Path(r"D:\Annual report LLM project\LLM project_20251207")
YEAR = "2024"

TXT_ROOT = PROJECT_ROOT / "txt" / YEAR
OUT_ROOT = PROJECT_ROOT / "embeddings" / YEAR   # 第一阶段输出：按银行落盘
LOG_CSV  = PROJECT_ROOT / f"embed_log_{YEAR}.csv"

MODEL_NAME = r"D:\LKY SCH OF PUBLIC POLICY RA program\LLM_models\models--BAAI--bge-m3\snapshots\5617a9f61b028005a4858fdac845db406aefb181"   # 或你的本地路径
DEVICE = "cuda"             # RTX 4060
BATCH_SIZE = 8

CHUNK_SIZE = 1400
OVERLAP = 200

MAX_CHARS_PER_FILE = 400_000   #新增硬上限，防止内存不足
MAX_CHUNKS_PER_FILE = 2000

# 跑量控制：None(全量) / 10 / 100 / 1000 ...
LIMIT = None
# ====================


def clean_text(s: str) -> str:
    s = s.replace("\r\n", "\n")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()

def pick_windows(n_chars: int, window_size: int, max_windows: int):
    """
    返回一组窗口 (start, end)，尽量覆盖整篇：头/中/尾 + 均匀分布
    """
    if n_chars <= window_size:
        return [(0, n_chars)]

    # 至少 3 个：头、中、尾
    positions = [0, max(0, (n_chars - window_size)//2), max(0, n_chars - window_size)]

    # 如果还允许更多窗口，均匀撒点
    if max_windows > 3:
        extra = max_windows - 3
        step = max(1, (n_chars - window_size) // (extra + 1))
        for k in range(1, extra + 1):
            positions.append(k * step)

    # 去重 + 排序
    positions = sorted(set(max(0, min(p, n_chars - window_size)) for p in positions))

    windows = [(p, min(p + window_size, n_chars)) for p in positions]
    return windows


def build_chunks_full_coverage(text: str, chunk_size: int, overlap: int, window_size: int, max_windows: int, max_chunks_total: int):
    """
    对整篇做覆盖：多个窗口分块 + 去重（按 char_start ）
    """
    n = len(text)
    windows = pick_windows(n, window_size, max_windows)

    chunks = []
    seen = set()  # (global_start) 去重
    for ws, we in windows:
        sub = text[ws:we]
        for st, ed, ch in chunk_iter(sub, chunk_size, overlap):
            gst = ws + st
            ged = ws + ed
            if gst in seen:
                continue
            seen.add(gst)
            chunks.append((gst, ged, ch))
            if len(chunks) >= max_chunks_total:
                return chunks
    return chunks

def chunk_iter(text: str, chunk_size: int, overlap: int):
    n = len(text)
    start = 0
    while start < n:
        end = min(start + chunk_size, n)
        ch = text[start:end]          # 不在这里 strip，省内存
        if ch:
            yield start, end, ch
        start = end - overlap
        if start < 0:
            start = 0
        if start >= n:
            break

def chunk_by_chars(text: str, chunk_size: int, overlap: int, max_chunks: int):
    chunks = []
    for st, ed, ch in chunk_iter(text, chunk_size, overlap):
        chunks.append((st, ed, ch))
        if len(chunks) >= max_chunks:
            break
    return chunks

def iter_txt_files(txt_root: Path):
    for bank_dir in sorted(txt_root.glob("*")):
        if not bank_dir.is_dir():
            continue
        for p in sorted(bank_dir.glob("*.txt")):
            yield bank_dir.name, p

import torch

import torch
import numpy as np

def encode_to_memmap(model, texts, out_npy_path: Path, batch_size: int):
    if not texts:
        raise ValueError("texts empty")

    # 先跑一小批拿 dim
    first_n = min(batch_size, len(texts))
    first = model.encode(
        texts[:first_n],
        batch_size=first_n,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32)

    dim = first.shape[1]

    # ✅ 写标准 .npy（带 header），以后 np.load 直接读
    emb = np.lib.format.open_memmap(
        str(out_npy_path),
        mode="w+",
        dtype="float32",
        shape=(len(texts), dim),
    )
    emb[:first_n] = first

    for i in range(first_n, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        vec = model.encode(
            batch,
            batch_size=len(batch),
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype(np.float32)
        emb[i:i+len(batch)] = vec

    # flush + 清显存碎片
    del emb
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def safe_mkdir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def main():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    files = list(iter_txt_files(TXT_ROOT))
    if LIMIT is not None:
        files = files[: int(LIMIT)]

    print(f"[INFO] TXT_ROOT = {TXT_ROOT}")
    print(f"[INFO] files = {len(files)}  LIMIT={LIMIT}")

    print(f"[INFO] loading model: {MODEL_NAME} device={DEVICE}")
    model = SentenceTransformer(MODEL_NAME, device=DEVICE)

    # 写日志
    with LOG_CSV.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "year", "bank_folder", "txt_file",
            "txt_path", "out_dir",
            "num_chars", "num_chunks",
            "status", "error"
        ])

        for idx, (bank_folder, txt_path) in enumerate(files, 1):
            t0 = time.time()
            out_dir = OUT_ROOT / bank_folder
            safe_mkdir(out_dir)

            stem = txt_path.stem
            out_meta = out_dir / f"chunks__{stem}.jsonl"
            out_emb  = out_dir / f"emb__{stem}.npy"
            out_done = out_dir / f"_DONE__{stem}.ok"


            # 已经做过就跳过（断点续跑）
            if out_done.exists() and out_meta.exists() and out_emb.exists():
                print(f"[{idx}/{len(files)}] [SKIP] {bank_folder} already done")
                w.writerow([YEAR, bank_folder, txt_path.name, str(txt_path), str(out_dir),
                            "", "", "skip_exist", ""])
                continue

            try:
                raw = txt_path.read_text(encoding="utf-8", errors="ignore")
                text = clean_text(raw)
                if len(text) < 50:
                    raise ValueError("text too short after cleaning")
                # 覆盖整篇：窗口采样
                WINDOW_SIZE = 300_000        # 每个窗口 30万字符
                MAX_WINDOWS = 6              # 最多 6 个窗口（头/中/尾+均匀）
                MAX_CHUNKS_TOTAL = 3000      # 单个 txt 最多 3000 chunks（别太大）

                chunks = build_chunks_full_coverage(
                    text=text,
                    chunk_size=CHUNK_SIZE,
                    overlap=OVERLAP,
                    window_size=WINDOW_SIZE,
                    max_windows=MAX_WINDOWS,
                    max_chunks_total=MAX_CHUNKS_TOTAL,
                )
                
                if not chunks:
                    raise ValueError("no chunks produced")
                # if len(chunks) > MAX_CHUNKS_PER_FILE:
                #     chunks = chunks[:MAX_CHUNKS_PER_FILE]

                chunk_texts = [c[2] for c in chunks]

                print(f"[{idx}/{len(files)}] [RUN] {bank_folder} chars={len(text)} chunks={len(chunks)}")
                # DEBUG：验证切块是否正常
                for j in range(min(2, len(chunks))):
                    st, ed, ch = chunks[j]
                    print(f"    [CHUNK{j}] st={st} ed={ed} len={len(ch)} head={repr(ch[:60])}")

                encode_to_memmap(model, chunk_texts, out_emb, batch_size=BATCH_SIZE)

                # 落盘：meta + embeddings
                with out_meta.open("w", encoding="utf-8") as mf:
                    for cid, (st, ed, ch) in enumerate(chunks):
                        rec = {
                            "year": YEAR,
                            "bank_folder": bank_folder,
                            "source_txt": str(txt_path),
                            "chunk_id": cid,
                            "char_start": st,
                            "char_end": ed,
                            "text": ch,
                        }
                        mf.write(json.dumps(rec, ensure_ascii=False) + "\n")


                # 完成标记（tweet 项目同款：DONE flag）
                out_done.write_text(f"ok elapsed={time.time()-t0:.2f}s\n", encoding="utf-8")

                w.writerow([YEAR, bank_folder, txt_path.name, str(txt_path), str(out_dir),
                            len(text), len(chunks), "ok", ""])

            except Exception as e:
                err = f"{type(e).__name__}: {e}"
                print(f"[{idx}/{len(files)}] [FAIL] {bank_folder} -> {err}")
                # 记录更详细 traceback 方便你排查
                tb_path = (OUT_ROOT / bank_folder / "_ERROR.txt")
                try:
                    tb_path.write_text(traceback.format_exc(), encoding="utf-8")
                except Exception:
                    pass

                w.writerow([YEAR, bank_folder, txt_path.name, str(txt_path), str(out_dir),
                            "", "", "fail", err])
                continue

    print(f"[DONE] log saved: {LOG_CSV}")


if __name__ == "__main__":
    main()
