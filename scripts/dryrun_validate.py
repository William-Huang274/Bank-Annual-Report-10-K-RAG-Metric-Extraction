# scripts/dryrun_validate.py
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple, Optional


# -----------------------------
# Helpers
# -----------------------------
def guess_repo_root() -> Path:
    # scripts/dryrun_validate.py -> repo_root = parents[1]
    return Path(__file__).resolve().parents[1]


def human(n: int) -> str:
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if n < 1024:
            return f"{n:.0f}{unit}" if unit == "B" else f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}PB"


def check_path_exists(p: Path) -> Tuple[bool, str]:
    if p.exists():
        return True, "OK"
    return False, "MISSING"


def check_file(p: Path) -> Tuple[bool, str]:
    if not p.exists():
        return (False, "MISSING")
    if p.is_dir():
        return (True, "OK (dir)")
    try:
        s = p.stat().st_size
        return (True, f"OK ({human(s)})")
    except Exception as e:
        return (True, f"OK (stat failed: {e})")


def sample_lines(p: Path, k: int = 3) -> List[str]:
    out: List[str] = []
    if not p.exists() or p.is_dir():
        return out
    with p.open("r", encoding="utf-8", errors="ignore") as f:
        for _ in range(k):
            line = f.readline()
            if not line:
                break
            out.append(line.strip()[:220])
    return out


def rglob_first(root: Path, patterns: List[str]) -> Optional[Path]:
    """
    Find first match for any of patterns under root.
    Prefer shorter path (more canonical), then lexicographic.
    """
    hits: List[Path] = []
    for pat in patterns:
        hits.extend(list(root.rglob(pat)))
    if not hits:
        return None
    hits.sort(key=lambda x: (len(str(x)), str(x)))
    return hits[0]


def rglob_count(root: Path, patterns: List[str], cap: int = 200000) -> int:
    """
    Count matches for patterns under root. Stops early if too many.
    """
    cnt = 0
    for pat in patterns:
        for _ in root.rglob(pat):
            cnt += 1
            if cnt >= cap:
                return cnt
    return cnt


def try_load_faiss(index_dir: Path) -> Tuple[bool, str]:
    """
    Light smoke test:
    - import faiss
    - load index file if found in index_dir
    Does NOT search / embed, only loads.
    """
    if not index_dir.exists() or not index_dir.is_dir():
        return (False, "MISSING_DIR")

    candidates = [
        index_dir / "index.faiss",
        index_dir / "faiss.index",
        index_dir / "index.bin",
        index_dir / "faiss_index.bin",
    ]
    idx_file = next((p for p in candidates if p.exists()), None)
    if idx_file is None:
        # fallback: any file that looks like an index
        any_bin = list(index_dir.glob("*.faiss")) + list(index_dir.glob("*.index")) + list(index_dir.glob("*.bin"))
        idx_file = any_bin[0] if any_bin else None

    if idx_file is None:
        return (False, "NO_INDEX_FILE_FOUND")

    try:
        import faiss  # type: ignore

        _ = faiss.read_index(str(idx_file))
        return (True, f"OK (loaded {idx_file.name})")
    except Exception as e:
        return (False, f"FAIL (faiss load error: {e})")


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Dry-run validator: verify pipeline artifacts exist without recomputation."
    )
    ap.add_argument("--root", type=str, default="", help="Repo root. Default: auto-detect from this file.")
    ap.add_argument("--year", type=int, default=2024, help="Fiscal year to validate.")
    ap.add_argument(
        "--index-dir",
        type=str,
        default="",
        help="Optional: explicit index dir (folder containing meta.jsonl / index file).",
    )
    ap.add_argument(
        "--artifacts-root",
        type=str,
        default="",
        help="Optional: extra folder to also search (if your large data lives outside repo).",
    )
    ap.add_argument("--verbose", action="store_true", help="Print sample lines for jsonl/meta.")
    args = ap.parse_args()

    root = Path(args.root).resolve() if args.root else guess_repo_root()
    year = args.year

    # Search roots: repo root (+ optional external artifacts root)
    search_roots: List[Path] = [root]
    if args.artifacts_root:
        ar = Path(args.artifacts_root).resolve()
        if ar.exists():
            search_roots.append(ar)

    def find_across_roots(patterns: List[str]) -> Optional[Path]:
        best: Optional[Path] = None
        for r in search_roots:
            hit = rglob_first(r, patterns)
            if hit is None:
                continue
            if best is None:
                best = hit
            else:
                # prefer shorter
                if len(str(hit)) < len(str(best)):
                    best = hit
        return best

    def count_across_roots(patterns: List[str]) -> int:
        total = 0
        for r in search_roots:
            total += rglob_count(r, patterns)
        return total

    # -----------------------------
    # Auto-detect important artifacts
    # -----------------------------
    # Index dir:
    index_dir: Path
    if args.index_dir:
        index_dir = Path(args.index_dir).resolve()
    else:
        meta = find_across_roots(["meta.jsonl"])
        if meta:
            index_dir = meta.parent
        else:
            # fallback: locate any index-like file then take its parent
            idx = find_across_roots(["*.faiss", "*.index", "index.faiss", "faiss.index", "*.bin"])
            index_dir = idx.parent if idx else (root / "data" / "processed" / "index" / f"faiss_{year}_full")

    meta_jsonl = index_dir / "meta.jsonl"

    # Extractions + metrics outputs:
    extractions_jsonl = find_across_roots(
        [f"extractions_{year}.jsonl", f"*extractions*{year}*.jsonl", f"*extract*{year}*.jsonl"]
    )
    metrics_csv = find_across_roots(
        [f"metrics_{year}.csv", f"*metrics*{year}*.csv", f"*kpi*{year}*.csv", f"*results*{year}*.csv"]
    )

    # “Stage-ish” artifact hints (not strict)
    any_pdf_cnt = count_across_roots(["*.pdf"])
    any_txt_cnt = count_across_roots(["*.txt"])
    any_embed_cnt = count_across_roots(["*.npy", "*.npz", "*.pt"])

    # entry pages file (guess)
    entry_pages = find_across_roots(["entry_pages.csv", "*entry*pages*.csv", "*entry_pages*.csv", "*links*.csv"])

    # -----------------------------
    # Print
    # -----------------------------
    print("=" * 80)
    print("[DRYRUN VALIDATE] repo_root =", root)
    if args.artifacts_root:
        print("[DRYRUN VALIDATE] artifacts_root =", Path(args.artifacts_root).resolve())
    print("[DRYRUN VALIDATE] year =", year)
    print("[DRYRUN VALIDATE] detected index_dir =", index_dir)
    print("=" * 80)

    overall_ok = True

    # Stage 01 (entry pages)
    print("\n## 01_collect_entry_pages (artifact presence)")
    if entry_pages:
        ok, msg = check_file(entry_pages)
        print(f"- entry pages file: {msg}  ->  {entry_pages}")
    else:
        overall_ok = False
        print("- entry pages file: MISSING (could not find entry_pages.csv / links.csv variants)")

    # Stage 02 (pdf)
    print("\n## 02_download_reports (artifact presence)")
    if any_pdf_cnt > 0:
        print(f"- pdf files (*.pdf): OK (count={any_pdf_cnt})")
    else:
        overall_ok = False
        print("- pdf files (*.pdf): MISSING (count=0)")

    # Stage 03 (txt)
    print("\n## 03_ocr_to_text (artifact presence)")
    if any_txt_cnt > 0:
        print(f"- text files (*.txt): OK (count={any_txt_cnt})")
    else:
        overall_ok = False
        print("- text files (*.txt): MISSING (count=0)")

    # Stage 04 (embeddings)
    print("\n## 04_build_embeddings (artifact presence)")
    if any_embed_cnt > 0:
        print(f"- embeddings (*.npy/*.npz/*.pt): OK (count={any_embed_cnt})")
    else:
        overall_ok = False
        print("- embeddings (*.npy/*.npz/*.pt): MISSING (count=0)")

    # Stage 05 (faiss index)
    print("\n## 05_build_faiss_index (smoke test: load index)")
    ok_dir, msg_dir = check_path_exists(index_dir)
    print(f"- index dir exists: {msg_dir}  ->  {index_dir}")
    if not ok_dir:
        overall_ok = False

    ok_meta, msg_meta = check_file(meta_jsonl)
    print(f"- meta.jsonl exists: {msg_meta}  ->  {meta_jsonl}")
    if not ok_meta:
        # meta.jsonl not mandatory for FAISS load, but usually required in your pipeline
        overall_ok = False

    ok_faiss, msg_faiss = try_load_faiss(index_dir)
    print(f"- faiss index load: {'OK' if ok_faiss else 'FAIL'}  ->  {msg_faiss}")
    if not ok_faiss:
        overall_ok = False

    if args.verbose and meta_jsonl.exists():
        print("  * meta.jsonl sample:")
        for line in sample_lines(meta_jsonl, 3):
            print("    -", line)

    # Stage 06 (extractions/metrics outputs)
    print("\n## 06_extract_metrics (outputs)")
    if extractions_jsonl:
        ok, msg = check_file(extractions_jsonl)
        print(f"- extractions jsonl: {msg}  ->  {extractions_jsonl}")
        if args.verbose and extractions_jsonl.exists():
            print("  * extractions sample:")
            for line in sample_lines(extractions_jsonl, 3):
                print("    -", line)
    else:
        overall_ok = False
        print(f"- extractions jsonl: MISSING (searched for extractions_{year}.jsonl variants)")

    if metrics_csv:
        ok, msg = check_file(metrics_csv)
        print(f"- metrics csv: {msg}  ->  {metrics_csv}")
    else:
        overall_ok = False
        print(f"- metrics csv: MISSING (searched for metrics/kpi/results {year} csv variants)")

    print("\n" + "=" * 80)
    if overall_ok:
        print("[RESULT] ✅ Key artifacts found; FAISS index loads. Safe to proceed without recomputation.")
    else:
        print("[RESULT] ⚠️ Some key artifacts not found (or index failed to load).")
        print("         This usually means your artifacts live outside repo, or filenames differ.")
        print("         Fix by either:")
        print("         1) run with --artifacts-root <path_to_big_data>, or")
        print("         2) pass --index-dir <folder_with_meta_jsonl_and_index_file>")
    print("=" * 80)


if __name__ == "__main__":
    main()