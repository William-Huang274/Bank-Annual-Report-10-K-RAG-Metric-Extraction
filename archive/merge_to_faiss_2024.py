# merge_to_faiss_2024.py
# pip install -U faiss-cpu numpy tqdm

from pathlib import Path
import os, json, time, csv
import numpy as np
from tqdm import tqdm
import faiss


# ====== 配置区：按你项目改 ======
PROJECT_ROOT = Path(r"D:\Annual report LLM project\LLM project_20251207")
YEAR = "2024"

# Stage1 输出目录（每家银行一个子目录）
STAGE1_ROOT = PROJECT_ROOT / "embeddings" / YEAR

# Stage2 输出目录（全局索引）
OUT_DIR = PROJECT_ROOT / "index" / f"faiss_{YEAR}_allbanks"
OUT_DIR.mkdir(parents=True, exist_ok=True)

INDEX_PATH = OUT_DIR / "faiss.index"
META_PATH  = OUT_DIR / "meta.jsonl"
LOG_CSV    = OUT_DIR / "merge_log.csv"

# 可选：只合并前 N 家银行（调试用）；None=全量
LIMIT_BANKS = None

# 每次往 faiss add 多少向量（越大越快，但内存会高）
ADD_BATCH = 20000
# =================================


def iter_bank_dirs(stage1_root: Path):
    for bank_dir in sorted(stage1_root.glob("*")):
        if not bank_dir.is_dir():
            continue
        done = bank_dir / "_DONE.ok"
        emb  = bank_dir / "embeddings.npy"
        meta = bank_dir / "chunks.jsonl"
        if done.exists() and emb.exists() and meta.exists():
            yield bank_dir.name, bank_dir


def count_lines(p: Path) -> int:
    n = 0
    with p.open("r", encoding="utf-8", errors="ignore") as f:
        for _ in f:
            n += 1
    return n


def main():
    t0 = time.time()

    banks = list(iter_bank_dirs(STAGE1_ROOT))
    if LIMIT_BANKS is not None:
        banks = banks[: int(LIMIT_BANKS)]

    print(f"[INFO] STAGE1_ROOT = {STAGE1_ROOT}")
    print(f"[INFO] ready banks = {len(banks)} (LIMIT_BANKS={LIMIT_BANKS})")
    if not banks:
        print("[WARN] 没有找到任何可合并的银行目录（缺 _DONE.ok / embeddings.npy / chunks.jsonl）。")
        return

    # 读第一家拿 dim
    first_bank, first_dir = banks[0]
    first_emb_path = first_dir / "embeddings.npy"
    first_emb = np.load(first_emb_path, mmap_mode="r")
    if first_emb.ndim != 2:
        raise ValueError(f"bad embeddings shape for {first_bank}: {first_emb.shape}")
    dim = int(first_emb.shape[1])
    print(f"[INFO] embedding dim = {dim}")

    # 建索引（因为你 embeddings 已经 normalize 过，所以用 IP = cosine）
    index = faiss.IndexFlatIP(dim)

    # 断点：如果已有 index/meta，则从上次继续（简单策略：重新生成更稳）
    # 为避免复杂状态错乱，这里默认“每次重新生成”
    if INDEX_PATH.exists():
        INDEX_PATH.unlink()
    if META_PATH.exists():
        META_PATH.unlink()

    # 全局 id 计数
    global_id = 0

    with LOG_CSV.open("w", newline="", encoding="utf-8") as lf, META_PATH.open("w", encoding="utf-8") as mf:
        w = csv.writer(lf)
        w.writerow(["bank_folder", "num_vec", "num_meta_lines", "status", "error"])

        for bank_folder, bank_dir in tqdm(banks, desc="merge banks"):
            emb_path = bank_dir / "embeddings.npy"
            meta_path = bank_dir / "chunks.jsonl"

            try:
                emb = np.load(emb_path, mmap_mode="r")  # 不把全部读入内存
                if emb.ndim != 2 or emb.shape[1] != dim:
                    raise ValueError(f"dim mismatch: {emb.shape} expected (*,{dim})")

                num_vec = int(emb.shape[0])

                # meta 行数检查（可选，但强烈建议）
                num_meta = count_lines(meta_path)
                if num_meta != num_vec:
                    # 不直接 fail，先记 warning，因为你未来可能会做过滤导致不一致
                    print(f"[WARN] {bank_folder}: meta lines({num_meta}) != vecs({num_vec})")

                # 分批 add 到 faiss
                for i in range(0, num_vec, ADD_BATCH):
                    batch = np.asarray(emb[i:i+ADD_BATCH], dtype=np.float32)
                    index.add(batch)

                # 写 meta：给每条 chunk 加一个 global_id（全局可回溯）
                with meta_path.open("r", encoding="utf-8", errors="ignore") as f:
                    for line in f:
                        obj = json.loads(line)
                        obj["global_id"] = global_id
                        obj["bank_folder"] = bank_folder  # 再写一次确保一致
                        mf.write(json.dumps(obj, ensure_ascii=False) + "\n")
                        global_id += 1

                w.writerow([bank_folder, num_vec, num_meta, "ok", ""])

            except Exception as e:
                w.writerow([bank_folder, "", "", "fail", f"{type(e).__name__}: {e}"])
                continue

    faiss.write_index(index, str(INDEX_PATH))
    print(f"[INFO] saved faiss index: {INDEX_PATH}")
    print(f"[INFO] saved meta: {META_PATH}")
    print(f"[INFO] saved log: {LOG_CSV}")
    print(f"[DONE] total_vectors={index.ntotal} elapsed={time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
