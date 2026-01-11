"""
Lightweight consistency check for embedding outputs.
This script validates that each embedding file (.npy) has the same number
of vectors as its corresponding chunk metadata (.jsonl), ensuring
embedding generation completed correctly before downstream indexing.
"""
from pathlib import Path
import numpy as np
import sys

def find_repo_root(start: Path) -> Path:
    """
    Locate the repository root by walking up parent directories and
    checking for common project markers (e.g., .git, README.md, data/).
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

# Default location produced by 04_build_embeddings.py
YEAR = "2024"
EMB_ROOT = ROOT / "artifacts" / "embeddings" / YEAR

def count_lines(p: Path) -> int:
    """
    Count lines in a text file efficiently.
    For JSONL metadata, each line corresponds to one chunk entry.
    Line counting is faster and more robust than full JSON parsing.
    """
    n=0
    with p.open("r", encoding="utf-8", errors="ignore") as f:
        for _ in f: n+=1
    return n

def main():
    # Iterate over all bank embedding outputs and validate vector/metadata alignment.
    items = 0
    bad = 0
    # Each subdirectory under EMB_ROOT corresponds to one bank.
    for bank_dir in sorted(EMB_ROOT.glob("*")):
        if not bank_dir.is_dir():
            continue
        # For each embedding shard:
        #   - emb__*.npy     : embedding vectors
        #   - chunks__*.jsonl: corresponding chunk metadata
        #   - _DONE__*.ok    : completion marker for successful generation
        for emb in bank_dir.glob("emb__*.npy"):
            stem = emb.stem.replace("emb__", "", 1)
            meta = bank_dir / f"chunks__{stem}.jsonl"
            done = bank_dir / f"_DONE__{stem}.ok"
            # Skip incomplete or partially generated outputs.
            if not (meta.exists() and done.exists()):
                continue

            # Memory-map embeddings to avoid loading large arrays fully into RAM.
            vec = np.load(emb, mmap_mode="r")
            nvec = int(vec.shape[0])
            nmeta = count_lines(meta)
            items += 1
            # Any mismatch indicates a corrupted or interrupted embedding run.
            if nvec != nmeta:
                bad += 1
                print(f"[MISMATCH] {bank_dir.name}/{stem}: vec={nvec} meta={nmeta}")

    # Final summary for quick inspection or CI-style sanity checks.
    print(f"[DONE] items={items} bad={bad}")

if __name__ == "__main__":
    main()
