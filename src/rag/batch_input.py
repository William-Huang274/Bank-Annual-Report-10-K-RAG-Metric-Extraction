from __future__ import annotations

from pathlib import Path
from typing import Optional, List

from src.rag.paths import resolve_under_root


def parse_batch_command(line: str) -> Optional[str]:
    """
    Parse a ':batch <file>' command; return the path string or None if usage is invalid.
    """
    parts = (line or "").split(maxsplit=1)
    if len(parts) != 2:
        return None
    return parts[1].strip()


def read_bank_list(path: Path) -> List[str]:
    """
    Read bank ids from a text file, stripping blanks.
    """
    text = path.read_text(encoding="utf-8")
    return [ln.strip() for ln in text.splitlines() if ln.strip()]


def resolve_batch_path(root: Path, maybe_rel: str | Path) -> Path:
    """
    Resolve batch file path relative to repo root (mirrors previous in-script logic).
    """
    return resolve_under_root(root, maybe_rel)
