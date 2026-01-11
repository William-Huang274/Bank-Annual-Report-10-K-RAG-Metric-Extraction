"""
Interactive FAISS index query tool for debugging retrieval quality.
This script:
- Loads a FAISS index and its meta.jsonl sidecar
- Encodes a user query using the embedding model (CUDA preferred, CPU fallback)
- Prints the top-k matched chunks with basic metadata for inspection
"""
from pathlib import Path
import json
import numpy as np
import faiss
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

YEAR = "2024"

# Pre-built FAISS index and metadata (read-only; no re-indexing in this script).
INDEX_DIR = ROOT /"data" /"interim" /"index" / f"faiss_{YEAR}_full"
INDEX_PATH = INDEX_DIR / "faiss.index"
META_PATH  = INDEX_DIR / "meta.jsonl"

EMB_MODEL = "BAAI/bge-m3"

TOPK = 8

def load_meta(meta_path: Path):
    """
    Load FAISS metadata stored as JSONL.
    Each line corresponds to one vector in the index and must align with index.ntotal.
    """ 
    meta = []
    with meta_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            meta.append(json.loads(line))
    return meta

def load_st_model(prefer_cuda=True):
    """
    Load sentence-transformer embedding model with optional CUDA preference.
    Automatically falls back to CPU if CUDA initialization fails or runs out of memory.
    """
    from sentence_transformers import SentenceTransformer
    import torch

    def _load(device: str):
        torch.set_grad_enabled(False)
        if device == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()
        return SentenceTransformer(EMB_MODEL, device=device), device

    if prefer_cuda:
        try:
            m, dev = _load("cuda")
            print("[INFO] embedding device=cuda")
            return m, dev
        except Exception as e:
            print(f"[WARN] cuda load failed -> fallback cpu: {type(e).__name__}: {e}")

    m, dev = _load("cpu")
    print("[INFO] embedding device=cpu")
    return m, dev

def main():
    """Run an interactive REPL to query the FAISS index and inspect top-k hits."""
    print(f"[INFO] loading index: {INDEX_PATH}")
    index = faiss.read_index(str(INDEX_PATH))
    print(f"[INFO] ntotal={index.ntotal} dim={index.d}")

    print(f"[INFO] loading meta: {META_PATH}")
    meta = load_meta(META_PATH)
    assert len(meta) == index.ntotal, f"meta({len(meta)}) != index({index.ntotal})"

    model, dev = load_st_model(prefer_cuda=True)

    # Interactive loop for ad-hoc semantic search against the FAISS index.
    while True:
        q = input("\nQuery (empty to exit): ").strip()
        if not q:
            break

        try:
            qvec = model.encode([q], batch_size=1, normalize_embeddings=True, convert_to_numpy=True).astype(np.float32)
        except RuntimeError as e:
            # Encoding may also trigger OOM; perform an explicit CPU fallback and retry once.
            if "out of memory" in str(e).lower():
                print("[WARN] encode OOM -> switching to cpu")
                model, dev = load_st_model(prefer_cuda=False)
                qvec = model.encode([q], batch_size=1, normalize_embeddings=True, convert_to_numpy=True).astype(np.float32)
            else:
                raise

        D, I = index.search(qvec, TOPK)

        print("\n=== TOPK ===")
        for rank, (score, idx) in enumerate(zip(D[0], I[0]), 1):
            m = meta[int(idx)]
            text = m.get("text", "")
            # Print a short preview only to keep console output readable.
            head = text[:300].replace("\n", "\\n")
            print(f"[{rank}] score={score:.4f}  bank={m.get('bank_folder')}  stem={m.get('stem')}  chunk_id={m.get('chunk_id')}")
            print(f"     char=[{m.get('char_start')},{m.get('char_end')}]")
            print(f"     {head}\n")

if __name__ == "__main__":
    main()
