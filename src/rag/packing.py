# src/rag/packing.py
from __future__ import annotations


def build_context(hits) -> str:
    """
    Build an evidence context string from retrieval hits.

    Accept two hit formats:
      A) list[(rank, score, meta_dict)]  where meta_dict has bank_folder/stem/chunk_id/text
      B) list[dict] from search_faiss()  where dict has bank/stem/chunk_id/text/score
    """
    blocks = []
    if not hits:
        return ""

    for idx, h in enumerate(hits, 1):
        # --- normalize ---
        if isinstance(h, (tuple, list)) and len(h) == 3:
            rank, score, m = h
            score = float(score)
            m = m or {}
            text = (m.get("text") or "").strip()
            bank = m.get("bank_folder") or m.get("bank")  # tolerate both
            stem = m.get("stem")
            chunk_id = m.get("chunk_id")
        elif isinstance(h, dict):
            rank = idx
            score = float(h.get("score", 0.0))
            text = (h.get("text") or "").strip()
            bank = h.get("bank") or h.get("bank_folder")
            stem = h.get("stem")
            chunk_id = h.get("chunk_id")
        else:
            # Unknown format: skip instead of crash
            continue

        # --- head+tail truncate (table-aware) ---
        t = text.lower()

        # Default truncation settings for general text blocks
        MAXC = 1400
        HEAD = 700
        TAIL = 700

        # For ratio tables and structured financial summaries, apply a larger context window
        # to avoid truncating fiscal-year numeric columns (e.g., 2024 values)
        is_ratio_table = any(k in t for k in [
            "selected performance ratios",
            "return on assets",
            "return on equity",
            "equity to assets",
            "dividend payout",
            "net interest margin",
        ])
        if is_ratio_table:
            MAXC = 4200
            HEAD = 1600
            TAIL = 2600

        if len(text) > MAXC:
            text = text[:HEAD] + "\n...[middle truncated]...\n" + text[-TAIL:]

        # Contract: LLM must copy source_chunk_id exactly from this header for traceability.
        header = f"[k={bank}|stem={stem}|chunk={chunk_id}]"
        blocks.append(
            f"{header}\n"
            f"(rank={rank} score={score:.4f})\n"
            f"{text}"
        )

    return "\n\n---\n\n".join(blocks)
