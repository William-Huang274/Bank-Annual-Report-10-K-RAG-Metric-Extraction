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
