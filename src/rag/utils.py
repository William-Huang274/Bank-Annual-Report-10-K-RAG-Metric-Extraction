# src/rag/utils.py
from __future__ import annotations

import json
from pathlib import Path


# def find_repo_root(start: Path) -> Path:
#     """
#     Return the repository root directory.

#     This resolves the project root used for consistent relative-path handling across scripts.
#     """
#     p = start.resolve()
#     for _ in range(10):  # Search up to 10 parent directories for repo root markers
#         if (p / ".git").exists() or (p / "README.md").exists() or (p / "data").exists():
#             return p
#         p = p.parent
#     raise RuntimeError(f"Cannot locate repo root from: {start}")


def load_meta(path: Path) -> list[dict]:
    """
    Load FAISS metadata records from meta.jsonl.

    Returns a list of dict objects, one per indexed chunk, used for bank/stem/chunk lookup and context assembly.
    """
    meta: list[dict] = []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            meta.append(json.loads(line))
    return meta
