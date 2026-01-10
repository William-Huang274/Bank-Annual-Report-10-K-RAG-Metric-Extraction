from pathlib import Path
import numpy as np
import sys
# repo root: .../<repo>/scripts/debug/query_faiss.py -> parents[2] = <repo>
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
EMB_ROOT = ROOT / "artifacts" / "embeddings" / "2024"

def count_lines(p: Path) -> int:
    n=0
    with p.open("r", encoding="utf-8", errors="ignore") as f:
        for _ in f: n+=1
    return n

items = 0
bad = 0
for bank_dir in sorted(EMB_ROOT.glob("*")):
    if not bank_dir.is_dir():
        continue
    for emb in bank_dir.glob("emb__*.npy"):
        stem = emb.stem.replace("emb__", "", 1)
        meta = bank_dir / f"chunks__{stem}.jsonl"
        done = bank_dir / f"_DONE__{stem}.ok"
        if not (meta.exists() and done.exists()):
            continue

        vec = np.load(emb, mmap_mode="r")
        nvec = int(vec.shape[0])
        nmeta = count_lines(meta)
        items += 1
        if nvec != nmeta:
            bad += 1
            print(f"[MISMATCH] {bank_dir.name}/{stem}: vec={nvec} meta={nmeta}")

print(f"[DONE] items={items} bad={bad}")
