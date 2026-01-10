# merge_to_faiss_2024_full.py
# pip install -U faiss-cpu numpy tqdm

from pathlib import Path
import json, time, csv
import numpy as np
from tqdm import tqdm
import faiss
import sys

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

YEAR = "2024"

STAGE1_ROOT = ROOT / "artifacts" / "embeddings" / YEAR
OUT_DIR = ROOT / "data" / "interim" / "index" / f"faiss_{YEAR}_full"
OUT_DIR.mkdir(parents=True, exist_ok=True)

INDEX_PATH = OUT_DIR / "faiss.index"
META_PATH  = OUT_DIR / "meta.jsonl"
LOG_CSV    = OUT_DIR / "merge_log.csv"

LIMIT_ITEMS = None      # None=全量；也可以设 5 测试
ADD_BATCH = 20000       # 每次往 faiss add 的向量数（内存够可调大）


def iter_items(stage1_root: Path):
    """
    产出每个 emb 文件及其对应 meta
    emb__X.npy 对 chunks__X.jsonl，且存在 _DONE__X.ok
    """
    for bank_dir in sorted(stage1_root.glob("*")):
        if not bank_dir.is_dir():
            continue
        for emb_path in sorted(bank_dir.glob("emb__*.npy")):
            stem = emb_path.stem.replace("emb__", "", 1)
            meta_path = bank_dir / f"chunks__{stem}.jsonl"
            done_path = bank_dir / f"_DONE__{stem}.ok"
            if meta_path.exists() and done_path.exists():
                yield bank_dir.name, stem, emb_path, meta_path


def count_lines(p: Path) -> int:
    n = 0
    with p.open("r", encoding="utf-8", errors="ignore") as f:
        for _ in f:
            n += 1
    return n


def main():
    t0 = time.time()
    items = list(iter_items(STAGE1_ROOT))
    if LIMIT_ITEMS is not None:
        items = items[: int(LIMIT_ITEMS)]

    print(f"[INFO] items to merge = {len(items)}")
    if not items:
        print("[WARN] no items found")
        return

    # dim from first
    bank0, stem0, emb0, meta0 = items[0]
    e0 = np.load(emb0, mmap_mode="r")
    dim = int(e0.shape[1])
    print(f"[INFO] dim={dim}  first={bank0}/{stem0} vecs={e0.shape[0]}")

    # 用 cosine (normalize_embeddings=True) 的话，内积=cosine
    index = faiss.IndexFlatIP(dim)

    # reset outputs
    if INDEX_PATH.exists():
        INDEX_PATH.unlink()
    if META_PATH.exists():
        META_PATH.unlink()

    global_id = 0

    with LOG_CSV.open("w", newline="", encoding="utf-8") as lf, META_PATH.open("w", encoding="utf-8") as mf:
        w = csv.writer(lf)
        w.writerow(["bank_folder", "stem", "num_vec", "num_meta", "status", "error"])

        for bank_folder, stem, emb_path, meta_path in tqdm(items, desc="merge"):
            try:
                emb = np.load(emb_path, mmap_mode="r")
                if emb.ndim != 2 or emb.shape[1] != dim:
                    raise ValueError(f"dim mismatch: {emb.shape} expected (*,{dim})")

                num_vec = int(emb.shape[0])
                num_meta = count_lines(meta_path)
                if num_meta != num_vec:
                    raise ValueError(f"meta({num_meta}) != vec({num_vec})")

                # add vectors
                for i in range(0, num_vec, ADD_BATCH):
                    batch = np.asarray(emb[i:i+ADD_BATCH], dtype=np.float32)
                    index.add(batch)

                # write meta with global_id + source info
                with meta_path.open("r", encoding="utf-8", errors="ignore") as f:
                    for line in f:
                        obj = json.loads(line)
                        obj["global_id"] = global_id
                        obj["bank_folder"] = bank_folder
                        obj["stem"] = stem
                        obj["source_emb"] = str(emb_path)
                        obj["source_meta"] = str(meta_path)
                        mf.write(json.dumps(obj, ensure_ascii=False) + "\n")
                        global_id += 1

                w.writerow([bank_folder, stem, num_vec, num_meta, "ok", ""])

            except Exception as e:
                w.writerow([bank_folder, stem, "", "", "fail", f"{type(e).__name__}: {e}"])
                continue

    faiss.write_index(index, str(INDEX_PATH))
    print(f"[INFO] saved index: {INDEX_PATH}")
    print(f"[INFO] saved meta : {META_PATH}")
    print(f"[DONE] vectors={index.ntotal} elapsed={time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
