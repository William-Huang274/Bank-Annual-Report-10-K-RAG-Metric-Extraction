"""
Build text embeddings from OCR-extracted annual report text.

This stage:
- Loads cleaned text files
- Chunks text with overlap
- Generates dense embeddings using a SentenceTransformer model
- Streams embeddings to disk using NumPy memmap
- Writes chunk-level metadata in JSONL format

Designed for large-scale batch processing with bounded memory usage.
"""

# embed_stage1_2024.py  (FULL / STREAMING)
# pip install -U sentence-transformers numpy tqdm

import sys
from pathlib import Path
import os

def find_repo_root(start: Path) -> Path:
    """
    Locate the repository root directory.
    The root is identified by the presence of one of: .git, README.md, or data/.
    Raises RuntimeError if not found within 10 parent levels.
    """
    p = start.resolve()
    for _ in range(10):  # Search up to 10 parent directories for repo root.
        if (p / ".git").exists() or (p / "README.md").exists() or (p / "data").exists():
            return p
        p = p.parent
    raise RuntimeError(f"Cannot locate repo root from: {start}")

ROOT = find_repo_root(Path(__file__))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Environment variables
FORCE_RERUN = True  # When True, ignore _DONE__ markers and recompute outputs for reproducibility.


TMP_DIR = ROOT/ "_tmp"
TMP_DIR.mkdir(parents=True, exist_ok=True)
os.environ["TMP"] = str(TMP_DIR)
os.environ["TEMP"] = str(TMP_DIR)
os.environ["TMPDIR"] = str(TMP_DIR)

HF_HOME = ROOT / "_hf_cache"
HF_HOME.mkdir(parents=True, exist_ok=True)
os.environ["HF_HOME"] = str(HF_HOME)
os.environ["HUGGINGFACE_HUB_CACHE"] = str(HF_HOME / "hub")
# TRANSFORMERS_CACHE 
# os.environ["TRANSFORMERS_CACHE"] = str(HF_HOME / "transformers")
os.environ["TORCH_HOME"] = str(HF_HOME / "torch")

# Heavy dependencies are imported after environment configuration
import re, json, csv, time, traceback
import numpy as np
from tqdm import tqdm
import torch
from sentence_transformers import SentenceTransformer


# Config
YEAR = "2024"

TXT_ROOT = ROOT / "data" / "interim" / "txt" / YEAR
OUT_ROOT = ROOT / "artifacts" / "embeddings" / YEAR
LOG_CSV  = ROOT / "data" / "interim" / "output" / "logs" / f"embed_log_{YEAR}.csv"

MODEL_NAME = "BAAI/bge-m3"
DEVICE = "cuda"
BATCH_SIZE = 8

CHUNK_SIZE = 1400
OVERLAP = 200

# Full pass (no truncation). Keep a safety cap on max chunks per file.
# number of chunks per file and prevent excessive disk and memory usage.
MAX_CHUNKS_PER_FILE = 50000   # Safety cap to bound disk usage and runtime on unusually long documents.

# Optional limit on the number of input files processed (None for full run)
LIMIT = None
# ====================


def clean_text(s: str) -> str:
    s = s.replace("\r\n", "\n")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def safe_mkdir(p: Path):
    """Create a directory if it does not exist (including parents)."""
    p.mkdir(parents=True, exist_ok=True)


def iter_txt_files(txt_root: Path):
    """Yield (bank_folder, txt_path) pairs under the given txt_root."""
    for bank_dir in sorted(txt_root.glob("*")):
        if not bank_dir.is_dir():
            continue
        for p in sorted(bank_dir.glob("*.txt")):
            yield bank_dir.name, p


def chunk_iter(text: str, chunk_size: int, overlap: int):
    """
    Stream chunks as (char_start, char_end, chunk_text).

    Design note:
    The start offset is guaranteed to advance on each iteration
    to avoid infinite loops under all parameter settings.
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
            new_start = end  # Fallback to enforce forward progress
        start = new_start


def estimate_num_chunks(n_chars: int, chunk_size: int, overlap: int, max_chunks: int):
    # The estimate may exceed the actual number of emitted chunks.
    # If the iterator is exhausted early, stop and keep the effective chunk count for this file.
    step = chunk_size - overlap
    if step <= 0:
        raise ValueError(f"BAD PARAM: chunk_size={chunk_size} overlap={overlap}")
    if n_chars <= 0:
        return 0
    # Starts are 0, step, 2*step, ... while start < n_chars.
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

            # Resume capability: skip processing based on (bank_folder, stem) if outputs already exist.
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

                # Estimate the number of chunks without materializing them into a list.
                n_chunks = estimate_num_chunks(n_chars, CHUNK_SIZE, OVERLAP, MAX_CHUNKS_PER_FILE)
                if n_chunks <= 0:
                    raise ValueError("no chunks (after estimate)")

                print(f"[{idx}/{len(files)}] [RUN] {bank_folder} / {stem} chars={n_chars} est_chunks={n_chunks}")

                # ==========
                # 1) Process the first batch to infer embedding dimension and create the .npy memmap
                # ==========
                it = chunk_iter(text, CHUNK_SIZE, OVERLAP)

                first_texts = []
                first_metas = []
                for _ in range(min(BATCH_SIZE, n_chunks)):
                    st, ed, ch = next(it)  # May raise StopIteration if the iterator is exhausted early
                    first_texts.append(ch)
                    first_metas.append((st, ed, ch))

                # Debug: print the first few chunks to validate chunking behavior
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

                # Open metadata file and write records in a streaming manner
                with out_meta.open("w", encoding="utf-8") as mf:
                    # Write the first batch of chunks and embeddings.
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
                    # 2) Subsequent batches: generate chunks, encode embeddings, and write outputs incrementally
                    # ==========
                    while cur < n_chunks:
                        batch_texts = []
                        batch_metas = []
                        need = min(BATCH_SIZE, n_chunks - cur)

                        # Collect need chunks
                        for _ in range(need):
                            try:
                                st, ed, ch = next(it)
                            except StopIteration:
                                # Actual number of chunks may be smaller than the estimate.
                                # In this case, stop early and record the effective number of chunks.
                                # This scenario is rare and typically occurs only for very short texts.
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
                        # Optionally clear cached CUDA memory between batches to reduce peak usage in long runs.
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
