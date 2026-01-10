# query_faiss_2024.py (auto fallback cuda->cpu)
from pathlib import Path
import json
import numpy as np
import faiss
import sys
from pathlib import Path

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

INDEX_DIR = ROOT /"data" /"interim" /"index" / f"faiss_{YEAR}_full"
INDEX_PATH = INDEX_DIR / "faiss.index"
META_PATH  = INDEX_DIR / "meta.jsonl"

EMB_MODEL = "BAAI/bge-m3"

TOPK = 8

def load_meta(meta_path: Path):
    meta = []
    with meta_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            meta.append(json.loads(line))
    return meta

def load_st_model(prefer_cuda=True):
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
    print(f"[INFO] loading index: {INDEX_PATH}")
    index = faiss.read_index(str(INDEX_PATH))
    print(f"[INFO] ntotal={index.ntotal} dim={index.d}")

    print(f"[INFO] loading meta: {META_PATH}")
    meta = load_meta(META_PATH)
    assert len(meta) == index.ntotal, f"meta({len(meta)}) != index({index.ntotal})"

    model, dev = load_st_model(prefer_cuda=True)

    while True:
        q = input("\nQuery (empty to exit): ").strip()
        if not q:
            break

        try:
            qvec = model.encode([q], batch_size=1, normalize_embeddings=True, convert_to_numpy=True).astype(np.float32)
        except RuntimeError as e:
            # encode 时也可能 OOM：再 fallback 一次
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
            head = text[:300].replace("\n", "\\n")
            print(f"[{rank}] score={score:.4f}  bank={m.get('bank_folder')}  stem={m.get('stem')}  chunk_id={m.get('chunk_id')}")
            print(f"     char=[{m.get('char_start')},{m.get('char_end')}]")
            print(f"     {head}\n")

if __name__ == "__main__":
    main()
