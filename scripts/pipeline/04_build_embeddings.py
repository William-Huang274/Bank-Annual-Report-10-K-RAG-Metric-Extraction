# embed_stage1_2024.py  (FULL / STREAMING)
# pip install -U sentence-transformers numpy tqdm
import sys
from pathlib import Path
import os

def find_repo_root(start: Path) -> Path:
    p = start.resolve()
    for _ in range(10):  # 最多向上找 10 层
        if (p / ".git").exists() or (p / "README.md").exists() or (p / "data").exists():
            return p
        p = p.parent
    raise RuntimeError(f"Cannot locate repo root from: {start}")

ROOT = find_repo_root(Path(__file__))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ====== 环境变量区 ======
FORCE_RERUN = True  # True = 强制重跑（忽略 _DONE__*.ok）


TMP_DIR = ROOT/ "_tmp"
TMP_DIR.mkdir(parents=True, exist_ok=True)
os.environ["TMP"] = str(TMP_DIR)
os.environ["TEMP"] = str(TMP_DIR)
os.environ["TMPDIR"] = str(TMP_DIR)

HF_HOME = ROOT / "_hf_cache"
HF_HOME.mkdir(parents=True, exist_ok=True)
os.environ["HF_HOME"] = str(HF_HOME)
os.environ["HUGGINGFACE_HUB_CACHE"] = str(HF_HOME / "hub")
# TRANSFORMERS_CACHE 未来会弃用；不设也行（你想保留 warning 就删掉这行）
# os.environ["TRANSFORMERS_CACHE"] = str(HF_HOME / "transformers")
os.environ["TORCH_HOME"] = str(HF_HOME / "torch")

# —— 到这里才 import 重库 ——
import re, json, csv, time, traceback
import numpy as np
from tqdm import tqdm
import torch
from sentence_transformers import SentenceTransformer


# ====== 配置区 ======
YEAR = "2024"

TXT_ROOT = ROOT / "data" / "interim" / "txt" / YEAR
OUT_ROOT = ROOT / "artifacts" / "embeddings" / YEAR
LOG_CSV  = ROOT / "data" / "interim" / "output" / "logs" / f"embed_log_{YEAR}.csv"

MODEL_NAME = "BAAI/bge-m3"
DEVICE = "cuda"
BATCH_SIZE = 8

CHUNK_SIZE = 1400
OVERLAP = 200

# ✅ 全量：不截断字符
# 但仍建议给一个“最大chunk数安全阀”，防止极端大文件把你硬盘写爆
MAX_CHUNKS_PER_FILE = 50000   # 你可以先 20000，稳定后再加

# 跑量控制：None(全量) / 10 / 100 / ...
LIMIT = None
# ====================


def clean_text(s: str) -> str:
    s = s.replace("\r\n", "\n")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def safe_mkdir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def iter_txt_files(txt_root: Path):
    for bank_dir in sorted(txt_root.glob("*")):
        if not bank_dir.is_dir():
            continue
        for p in sorted(bank_dir.glob("*.txt")):
            yield bank_dir.name, p


def chunk_iter(text: str, chunk_size: int, overlap: int):
    """
    流式产出 (char_start, char_end, chunk_text)
    ✅ 关键：保证 start 一定前进，避免任何死循环
    """
    n = len(text)
    start = 0
    step = chunk_size - overlap
    if step <= 0:
        raise ValueError(f"BAD PARAM: chunk_size={chunk_size} overlap={overlap} (need overlap < chunk_size)")

    while start < n:
        end = min(start + chunk_size, n)
        ch = text[start:end]
        if ch:
            yield start, end, ch

        new_start = start + step
        if new_start <= start:
            new_start = end  # 兜底强制前进
        start = new_start


def estimate_num_chunks(n_chars: int, chunk_size: int, overlap: int, max_chunks: int):
    """
    只用长度估算 chunk 数（无需生成列表，省内存）
    """
    step = chunk_size - overlap
    if step <= 0:
        raise ValueError(f"BAD PARAM: chunk_size={chunk_size} overlap={overlap}")
    if n_chars <= 0:
        return 0
    # 生成 start = 0, step, 2*step... < n_chars
    est = (n_chars + step - 1) // step  # ceil(n/step)
    return int(min(est, max_chunks))


def main():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    LOG_CSV.parent.mkdir(parents=True, exist_ok=True)

    files = list(iter_txt_files(TXT_ROOT))
    if LIMIT is not None:
        files = files[: int(LIMIT)]

    print(f"[INFO] TXT_ROOT = {TXT_ROOT}")
    print(f"[INFO] files = {len(files)}  LIMIT={LIMIT}")
    print(f"[INFO] CHUNK_SIZE={CHUNK_SIZE} OVERLAP={OVERLAP} STEP={CHUNK_SIZE-OVERLAP} MAX_CHUNKS_PER_FILE={MAX_CHUNKS_PER_FILE}")

    print(f"[INFO] loading model: {MODEL_NAME} device={DEVICE}")
    model = SentenceTransformer(MODEL_NAME, device=DEVICE)

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

            # 断点续跑：按 bank+stem
            if (not FORCE_RERUN) and out_done.exists() and out_meta.exists() and out_emb.exists():
                print(f"[{idx}/{len(files)}] [SKIP] {bank_folder} / {stem} already done")
                w.writerow([YEAR, bank_folder, txt_path.name, str(txt_path), str(out_dir),
                            "", "", "skip_exist", ""])
                continue

            try:
                raw = txt_path.read_text(encoding="utf-8", errors="ignore")
                text = clean_text(raw)
                n_chars = len(text)
                if n_chars < 50:
                    raise ValueError("text too short after cleaning")

                # 先估算 chunk 数（不生成列表）
                n_chunks = estimate_num_chunks(n_chars, CHUNK_SIZE, OVERLAP, MAX_CHUNKS_PER_FILE)
                if n_chunks <= 0:
                    raise ValueError("no chunks (after estimate)")

                print(f"[{idx}/{len(files)}] [RUN] {bank_folder} / {stem} chars={n_chars} est_chunks={n_chunks}")

                # ==========
                # 1) 先拿第一批，探 dim，并创建标准 .npy memmap
                # ==========
                it = chunk_iter(text, CHUNK_SIZE, OVERLAP)

                first_texts = []
                first_metas = []
                for _ in range(min(BATCH_SIZE, n_chunks)):
                    st, ed, ch = next(it)  # 可能抛 StopIteration
                    first_texts.append(ch)
                    first_metas.append((st, ed, ch))

                # debug 打印前2块确认切块正确
                for j in range(min(2, len(first_metas))):
                    st, ed, ch = first_metas[j]
                    print(f"    [CHUNK{j}] st={st} ed={ed} len={len(ch)} head={repr(ch[:60])}")

                first_vec = model.encode(
                    first_texts,
                    batch_size=len(first_texts),
                    show_progress_bar=False,
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                ).astype(np.float32)
                dim = int(first_vec.shape[1])

                emb = np.lib.format.open_memmap(
                    str(out_emb),
                    mode="w+",
                    dtype="float32",
                    shape=(n_chunks, dim),
                )

                # 打开 meta，流式写
                with out_meta.open("w", encoding="utf-8") as mf:
                    # 写第一批
                    cur = 0
                    emb[cur:cur + len(first_texts)] = first_vec
                    for k, (st, ed, ch) in enumerate(first_metas):
                        rec = {
                            "year": YEAR,
                            "bank_folder": bank_folder,
                            "source_txt": str(txt_path),
                            "chunk_id": cur + k,
                            "char_start": st,
                            "char_end": ed,
                            "text": ch,
                        }
                        mf.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    cur += len(first_texts)

                    # ==========
                    # 2) 后续批次：边生成 chunk 边 encode 边写入
                    # ==========
                    while cur < n_chunks:
                        batch_texts = []
                        batch_metas = []
                        need = min(BATCH_SIZE, n_chunks - cur)

                        # 收集 need 个 chunk
                        for _ in range(need):
                            try:
                                st, ed, ch = next(it)
                            except StopIteration:
                                # 实际 chunk 数比估算少：缩容并截断 memmap（最简单：记录并 break）
                                # 这种情况一般不会发生（除非 text 很短但 n_chunks 估算偏大）
                                need = len(batch_texts)
                                break
                            batch_texts.append(ch)
                            batch_metas.append((st, ed, ch))

                        if not batch_texts:
                            break

                        vec = model.encode(
                            batch_texts,
                            batch_size=len(batch_texts),
                            show_progress_bar=False,
                            convert_to_numpy=True,
                            normalize_embeddings=True,
                        ).astype(np.float32)

                        emb[cur:cur + len(batch_texts)] = vec

                        for k, (st, ed, ch) in enumerate(batch_metas):
                            rec = {
                                "year": YEAR,
                                "bank_folder": bank_folder,
                                "source_txt": str(txt_path),
                                "chunk_id": cur + k,
                                "char_start": st,
                                "char_end": ed,
                                "text": ch,
                            }
                            mf.write(json.dumps(rec, ensure_ascii=False) + "\n")

                        cur += len(batch_texts)

                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()

                # flush memmap
                del emb
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                out_done.write_text(f"ok elapsed={time.time()-t0:.2f}s\n", encoding="utf-8")
                w.writerow([YEAR, bank_folder, txt_path.name, str(txt_path), str(out_dir),
                            n_chars, n_chunks, "ok", ""])

            except Exception as e:
                err = f"{type(e).__name__}: {e}"
                print(f"[{idx}/{len(files)}] [FAIL] {bank_folder} / {txt_path.stem} -> {err}")

                tb_path = (OUT_ROOT / bank_folder / f"_ERROR__{txt_path.stem}.txt")
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
