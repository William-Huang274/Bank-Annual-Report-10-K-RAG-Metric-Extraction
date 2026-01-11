"""
Merge per-file embedding shards into a single FAISS index.

This script:
- Iterates over embedding shards and their corresponding metadata
- Validates vector dimensions and alignment with metadata
- Incrementally adds vectors to a FAISS index in batches
- Writes a consolidated metadata JSONL with global IDs

Designed for large-scale indexing with bounded memory usage.
"""
# Dependencies: faiss (CPU), numpy, tqdm
# This script is named 05_build_faiss_index.py in the pipeline; older names may exist in archive/history.

from pathlib import Path
import json, time, csv
import numpy as np
from tqdm import tqdm
import faiss
import sys

def find_repo_root(start: Path) -> Path:
    """
    Locate the repository root directory.
    The root is identified by the presence of one of: .git, README.md, or data/.
    Raises RuntimeError if no root is found within 10 parent levels.
    """
    p = start.resolve()
    for _ in range(10):  # Search up to 10 parent directories for the project root
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

LIMIT_ITEMS = None      # Optional limit on number of items (None for full run) 
ADD_BATCH = 20000       # Number of vectors added to FAISS per batch
# Tuning: increase ADD_BATCH for faster indexing (more RAM), decrease for lower peak memory.


def iter_items(stage1_root: Path):
    """
    Yield embedding shards and their corresponding metadata.
    Expected layout:
    - emb__X.npy          : embedding vectors
    - chunks__X.jsonl     : per-chunk metadata
    - _DONE__X.ok         : completion marker for shard X
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
    """Count lines in a text file (streaming; tolerant to encoding errors)."""
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

    # Infer embedding dimension from the first shard
    bank0, stem0, emb0, meta0 = items[0]
    e0 = np.load(emb0, mmap_mode="r")
    dim = int(e0.shape[1])
    print(f"[INFO] dim={dim}  first={bank0}/{stem0} vecs={e0.shape[0]}")

    # Use inner product as similarity metric.
    # When embeddings are L2-normalized, inner product is equivalent to cosine similarity.
    index = faiss.IndexFlatIP(dim)

    # Ensure a clean rerun by removing any existing output artifacts.
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

                # Add vectors to the FAISS index in batches
                for i in range(0, num_vec, ADD_BATCH):
                    batch = np.asarray(emb[i:i+ADD_BATCH], dtype=np.float32)
                    index.add(batch)

                # Write metadata with assigned global_id and source provenance
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

            # Fail-fast per shard, but continue merging other shards to maximize progress in long runs.
            except Exception as e:
                w.writerow([bank_folder, stem, "", "", "fail", f"{type(e).__name__}: {e}"])
                continue

    faiss.write_index(index, str(INDEX_PATH))
    print(f"[INFO] saved index: {INDEX_PATH}")
    print(f"[INFO] saved meta : {META_PATH}")
    print(f"[DONE] vectors={index.ntotal} elapsed={time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
