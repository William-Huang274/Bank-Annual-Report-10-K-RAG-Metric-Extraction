from __future__ import annotations
from pathlib import Path

def get_project_root(from_file: str | None = None) -> Path:
    """
    Return project root path robustly.
    Assumes this file is under <root>/src/, or scripts under <root>/scripts/.
    """
    if from_file is None:
        # default: this file is <root>/src/paths.py
        return Path(__file__).resolve().parents[1]
    p = Path(from_file).resolve()
    # if called from scripts/<x>.py -> parents[1] is root
    # if called from root/<x>.py -> parents[0] is root
    if p.parent.name == "scripts":
        return p.parents[1]
    return p.parents[0]

def resolve_under_root(root: Path, maybe_rel: str | Path) -> Path:
    p = Path(maybe_rel)
    return p if p.is_absolute() else (root / p)


# Compatibility helpers for pipeline scripts
def get_repo_root(from_file: str) -> Path:
    """
    Mirror the inference used in scripts/pipeline/06_extract_metrics_patched_v2_final.py:
    Path(__file__).resolve().parents[2]
    """
    return Path(from_file).resolve().parents[2]


def default_paths(root: Path, year: str | int):
    """
    Return commonly used output/input paths for the metrics pipeline.
    The layout mirrors the in-script construction in 06_extract_metrics_patched_v2_final.py.
    """
    y = str(year)
    out_jsonl = root / "data" / "outputs" / "logs" / f"extractions_{y}.jsonl"
    out_csv = root / "data" / "outputs" / "processed" / f"metrics_{y}.csv"
    audit_csv = root / "data" / "outputs" / "logs" / f"write_audit_{y}.csv"
    sidecar_path = root / "data" / "interim" / "tables" / f"table_sidecar_{y}.jsonl"
    pdf_sidecar_path = root / "data" / "interim" / "tables" / f"table_sidecar_pdf_{y}.jsonl"
    pdf_sidecar_dir = root / "data" / "interim" / "tables" / "pdf_sidecar" / y
    return {
        "out_jsonl": out_jsonl,
        "out_csv": out_csv,
        "audit_csv": audit_csv,
        "sidecar_path": sidecar_path,
        "pdf_sidecar_path": pdf_sidecar_path,
        "pdf_sidecar_dir": pdf_sidecar_dir,
    }
