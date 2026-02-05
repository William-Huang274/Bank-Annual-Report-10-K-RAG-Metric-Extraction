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
import argparse

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
TOPK = 2000
TARGET_BANK = "Huntington_Bank_12311"
TARGET_CHUNK = 170

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
    ap = argparse.ArgumentParser()
    ap.add_argument("--query", type=str, default=None, help="Run a single query then exit.")
    ap.add_argument("--topk", type=int, default=TOPK, help="Number of final hits to print.")
    ap.add_argument("--bank", type=str, default=None, help="Filter hits by bank_folder substring. Default: target bank.")
    ap.add_argument("--scan-k", type=int, default=5000, help="Initial retrieval size before bank filter.")
    args = ap.parse_args()

    topk = int(args.topk)

    def run_one(q: str):
        nonlocal model, dev
        q = (q or "").strip()
        if not q:
            return

        try:
            qvec = model.encode([q], batch_size=1, normalize_embeddings=True, convert_to_numpy=True).astype(np.float32)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print("[WARN] encode OOM -> switching to cpu", flush=True)
                model, dev = load_st_model(prefer_cuda=False)
                qvec = model.encode([q], batch_size=1, normalize_embeddings=True, convert_to_numpy=True).astype(np.float32)
            else:
                raise

        if qvec.ndim == 1:
            qvec = qvec.reshape(1, -1)

        scan_k = int(args.scan_k)

        # 1) retrieve a larger candidate set
        D, I = index.search(qvec, scan_k)

        # 2) filter by bank (optional) then take final topk
        # default to TARGET_BANK so we don't mix other banks unless explicitly requested
        bank_pat = (args.bank or TARGET_BANK or "").strip().lower()
        bank_total = (
            sum(1 for mm in meta if bank_pat in str(mm.get("bank_folder", "")).lower())
            if bank_pat else len(meta)
        )
        hits = []

        for score, idx in zip(D[0], I[0]):
            m = meta[int(idx)]
            b = str(m.get("bank_folder", "")).lower()
            if bank_pat and (bank_pat not in b):
                continue
            hits.append((float(score), int(idx), m))
            if len(hits) >= topk:
                break

        if len(hits) < topk:
            print(f"[NOTE] hits={len(hits)} < topk={topk} (scan_k={scan_k}, bank_matches={bank_total})")

        print("\n=== TOPK ===")
        for rank, (score, idx, m) in enumerate(hits, 1):
            text = m.get("text", "")
            head = text[:300].replace("\n", "\\n")
            if m.get("bank_folder") == TARGET_BANK and int(m.get("chunk_id", -1)) == TARGET_CHUNK:
                print(f"[FOUND170] rank={rank} score={score:.4f}")
            print(f"[{rank}] score={score:.4f}  bank={m.get('bank_folder')}  stem={m.get('stem')}  chunk_id={m.get('chunk_id')}")
            print(f"     char=[{m.get('char_start')},{m.get('char_end')}]")
            print(f"     {head}\n")


    if args.query is not None:
        run_one(args.query)
        return

    while True:
        try:
            q = input("\nQuery (empty to exit): ").strip()
        except EOFError:
            print("[INFO] stdin is not interactive (EOF). Exit.", flush=True)
            break
        if not q:
            break
        run_one(q)

if __name__ == "__main__":
    main()
