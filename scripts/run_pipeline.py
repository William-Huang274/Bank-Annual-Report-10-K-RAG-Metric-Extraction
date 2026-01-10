# scripts/run_pipeline.py
import argparse
import subprocess
import sys
from pathlib import Path

def run_script(script_path: Path, args: list[str]) -> int:
    if not script_path.exists():
        print(f"[ERROR] Script not found: {script_path}")
        return 2
    cmd = [sys.executable, str(script_path)] + args
    print("[RUN]", " ".join(cmd))
    return subprocess.call(cmd)

def main():
    scripts_dir = Path(__file__).resolve().parent          # .../scripts
    root = scripts_dir.parent                               # project root

    mapping = {
        "collect": scripts_dir / "pipeline" / "01_collect_entry_pages.py",
        "download": scripts_dir / "pipeline" / "02_download_reports.py",
        "ocr": scripts_dir / "pipeline" / "03_ocr_to_text.py",
        "embed": scripts_dir / "pipeline" / "04_build_embeddings.py",
        "index": scripts_dir / "pipeline" / "05_build_faiss_index.py",
        "extract": scripts_dir / "pipeline" / "06_extract_metrics.py",
        "check": scripts_dir / "check" / "check_embeddings_outputs.py",
        # tools
        "query": scripts_dir / "debug" / "query_faiss.py",
    }

    parser = argparse.ArgumentParser(
        description="Run the bank KPI extraction pipeline stages."
    )
    parser.add_argument("command", choices=list(mapping.keys()))

    # common args (mainly for extract)
    parser.add_argument("--year", type=int, default=None)
    parser.add_argument("--banks", type=str, default=None)

    # allow passing any extra args through to the stage scripts
    args, extra = parser.parse_known_args()

    forwarded = []
    if args.year is not None:
        forwarded += ["--year", str(args.year)]
    if args.banks is not None:
        forwarded += ["--banks", args.banks]

    forwarded += extra

    return_code = run_script(mapping[args.command], forwarded)
    raise SystemExit(return_code)

if __name__ == "__main__":
    main()
