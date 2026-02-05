from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List
from collections import Counter

CID_BANK_RE = re.compile(r"\[k=([^|\]]+)\|")


def _bank_from_cid(cid: str) -> str | None:
    if not cid:
        return None
    m = CID_BANK_RE.search(cid)
    return m.group(1) if m else None


def _canon_bank(x: str | None) -> str:
    s = (x or "").strip()
    while s.startswith("_"):
        s = s[1:]
    return s

def enforce_bank_filter(hits: list[dict], bank: str, logger):
    expected = _canon_bank(bank)

    banks = []
    for h in hits:
        cid = h.get("cid") or h.get("chunk_id") or h.get("k") or ""
        b_raw = h.get("bank") or _bank_from_cid(cid) or "UNKNOWN"
        b = _canon_bank(b_raw)
        banks.append(b)

    cnt = Counter(banks)
    if len(cnt) > 1 or (len(cnt) == 1 and expected not in cnt):
        logger.warning("[BANK_MIX] expected=%s dist=%s", expected, dict(cnt))

    filtered = []
    for h in hits:
        cid = h.get("cid") or h.get("chunk_id") or h.get("k") or ""
        b_raw = h.get("bank") or _bank_from_cid(cid)
        b = _canon_bank(b_raw)
        if b == expected:
            filtered.append(h)

    if not filtered:
        logger.warning("[BANK_FILTER_EMPTY] expected=%s keep_original=%d", expected, len(hits))
        return hits  

    return filtered


from src.rag.packing import build_context
from src.rag.retrieval import (
    retrieve_hits_multiquery,
    retrieve_hits_per_metric,
    expand_neighbors_from_meta,
    rerank_hits_by_metric_keywords,
)
def _build_context_for_metric(
    *,
    bank: str,
    year: int,
    metric_name: str,
    hits,
    meta_lookup,
    neighbor_k: int,
    max_chars: int,
    mode: str,
    logger,
    dump_path: Path | None,
    **kwargs,
):
    if metric_name == "NIM":
        hits = expand_neighbors_from_meta(hits, meta_lookup, window=10, max_add=60)
    elif metric_name in ("NII", "ROA", "ROE", "Provision for Credit Losses"):
        hits = expand_neighbors_from_meta(hits, meta_lookup, window=3, max_add=60)
    ctx = build_context(hits)
    if dump_path is not None:
        dump_path.write_text(ctx, encoding="utf-8", errors="ignore")
    return ctx, hits


def retrieve_and_build_context_for_bank(
    *,
    index,
    meta,
    emb,
    bank_id: str,
    year: int,
    metrics: List[str],
    topk_final: int,
    max_context_chars_per_metric: int,
    debug_dir: Path,
    dprint,
    logger,
):
    debug_dir.mkdir(parents=True, exist_ok=True)

    hits = retrieve_hits_multiquery(
        index=index,
        meta=meta,
        emb=emb,
        target_bank=bank_id,
        year=str(year),
        per_metric_topk=10,
        topk_final=topk_final,
        k0=200,
        kmax=20000,
    )

    context = build_context(hits)

    # Persist assembled context for reproducible debugging (e.g., evidence inspection).
    (debug_dir / f"{bank_id}_{year}_context.txt").write_text(
        context,
        encoding="utf-8",
        errors="ignore"
    )

    hits_by_metric = retrieve_hits_per_metric(
        index=index,
        meta=meta,
        emb=emb,
        bank_id=bank_id,
        year=int(year),
        metrics=metrics,
        topk_per_query=40,
        topk_per_metric=20,
        min_score=0.50,
    )
    # keep original (06patch.py) bank matching behavior inside retrieve_hits_per_metric; do not re-filter here

    # Helper merge
    def _merge_hits_keep_best(a: list, b: list | None = None, max_keep: int = 50) -> list:
        best = {}
        for h in (a or []) + (b or []):
            cid = str(h.get("chunk_id") or "")
            if not cid:
                continue
            prev = best.get(cid)
            if (prev is None) or (float(h.get("score", 0.0)) > float(prev.get("score", 0.0))):
                best[cid] = h
        out = sorted(best.values(), key=lambda x: float(x.get("score", 0.0)), reverse=True)
        return out[:max_keep]

    def _prepend_hits(primary: list, base: list, max_keep: int = 80) -> list:
        out = []
        seen = set()
        for h in (primary or []) + (base or []):
            k = (str(h.get("bank") or h.get("bank_folder") or ""), str(h.get("stem") or ""), str(h.get("chunk_id") or ""))
            if not k[2] or (k in seen):
                continue
            seen.add(k)
            out.append(h)
            if len(out) >= int(max_keep):
                break
        return out

    def _is_roa_roe_compute_anchor(metric_name: str, text: str) -> bool:
        tx = (text or "").lower()
        if metric_name == "ROA":
            return (
                ("total average assets" in tx)
                or ("return on average total assets" in tx)
                or ("average avg average avg" in tx and "assets" in tx)
            )
        if metric_name == "ROE":
            return (
                ("return on average common shareholders" in tx)
                or ("return on average equity" in tx)
                or ("average stockholders" in tx and "equity" in tx)
                or ("average shareholders" in tx and "equity" in tx)
                or ("average tangible equity" in tx)
                or ("roace" in tx)
            )
        return False

    def _force_add_following_chunks(mhits, meta, base_hit, b, k=12):
        s = base_hit.get("stem")
        c0 = int(base_hit.get("chunk_id"))

        key2mm = {}
        for mm in meta:
            try:
                key2mm[(mm.get("bank_folder"), mm.get("stem"), int(mm.get("chunk_id")))] = mm
            except Exception:
                pass

        extra = []
        for i in range(1, k + 1):
            mm = key2mm.get((b, s, c0 + i))
            if mm:
                extra.append({
                    "bank": b,
                    "stem": s,
                    "chunk_id": mm.get("chunk_id"),
                    "score": base_hit.get("score", 0.0) - 1e-6,
                    "text": mm.get("text", "")
                })
        return mhits + extra

    def _has_metric_keywords(hits: list, metric: str) -> bool:
        m = (metric or "").upper()
        for h in hits or []:
            t = (h.get("text") or "").lower()
            if m == "ROA":
                if ("return on average assets" in t) or ("return on assets" in t) or ("roaa" in t) or (" roa" in t):
                    return True
            if m == "ROE":
                if ("return on average equity" in t) or ("return on equity" in t) or ("roae" in t) or (" roe" in t):
                    return True
        return False

    # Prefill contexts from the shared hit pool
    nim_hits0 = hits[:20]
    nim_hits0 = expand_neighbors_from_meta(nim_hits0, meta, window=3, max_add=60, dprint=dprint)
    nim_ctx = build_context(nim_hits0)

    pcl_hits0 = hits[:20]
    pcl_hits0 = expand_neighbors_from_meta(pcl_hits0, meta, window=3, max_add=60, dprint=dprint)
    pcl_ctx = build_context(pcl_hits0)

    nii_hits0 = hits_by_metric.get("NII") or []
    nii_ctx, _ = _build_context_for_metric(
        bank=bank_id,
        year=int(year),
        metric_name="NII",
        hits=expand_neighbors_from_meta(nii_hits0, meta, window=3, max_add=60, dprint=dprint)[:25],
        meta_lookup=meta,
        neighbor_k=3,
        max_chars=0,
        mode="default",
        logger=logger,
        dump_path=debug_dir / f"{bank_id}_context_NII.txt",
    )

    metric_contexts: Dict[str, Dict] = {}

    TOPK_CTX = 20

    for metric in metrics:
        mhits = hits_by_metric.get(metric, [])
        if not mhits:
            mhits = hits[:12]

        if metric in ("ROA", "ROE"):
            peer = "ROE" if metric == "ROA" else "ROA"
            peer_hits = hits_by_metric.get(peer, [])
            if peer_hits and (not _has_metric_keywords(mhits, metric)):
                mhits = _merge_hits_keep_best(mhits, peer_hits, max_keep=30)

        mhits = rerank_hits_by_metric_keywords(mhits, metric)

        if metric == "NIM":
            hdr = next((h for h in mhits if "Net Interest Margin Table" in (h.get("text","") or "")), None)
            if hdr:
                forced = _force_add_following_chunks([], meta, hdr, bank_id, k=12)
                mhits = _merge_hits_keep_best(forced, mhits, max_keep=200)

        mhits_for_ctx = mhits[:TOPK_CTX]

        mctx = build_context(mhits_for_ctx)

        if metric in ("ROA", "ROE"):
            kw_hits = [h for h in mhits if _has_metric_keywords([h], metric)]
            if kw_hits:
                mhits_for_ctx = _merge_hits_keep_best(kw_hits[:5], mhits_for_ctx, max_keep=TOPK_CTX)
                mctx = build_context(mhits_for_ctx)

        if metric == "NIM":
            mhits = mhits[:16]
        elif metric in ("ROA", "ROE"):
            # Keep a wider candidate set for ROA/ROE so average-balance blocks survive.
            mhits = mhits[:60]
        else:
            mhits = mhits[:10]

        if metric == "NIM" and len(mhits) < 3:
            mhits = hits[:8]
            mhits = rerank_hits_by_metric_keywords(mhits, metric)

        if metric in ("NII", "NIM"):
            mhits = expand_neighbors_from_meta(mhits, meta, window=3, max_add=60, dprint=dprint)
        else:
            mhits = expand_neighbors_from_meta(mhits, meta, window=3, max_add=60, dprint=dprint)

        MAX_BLOCKS_AFTER_EXPAND = 30
        if len(mhits) > MAX_BLOCKS_AFTER_EXPAND:
            mhits = mhits[:MAX_BLOCKS_AFTER_EXPAND]

        def cap_context_by_blocks(mhits, max_chars: int) -> str:
            parts = []
            total = 0
            for h in mhits:
                block = build_context([h])
                if not block:
                    continue
                add = len(block) + (5 if parts else 0)
                if total + add > max_chars:
                    break
                parts.append(block)
                total += add
            return "\n\n---\n\n".join(parts)

        mhits = rerank_hits_by_metric_keywords(mhits, metric)
        if metric in ("ROA", "ROE"):
            # Promote compute anchors (e.g., total average assets / average stockholders equity)
            # before context capping so these blocks are less likely to be truncated out.
            anchor_hits = [h for h in mhits if _is_roa_roe_compute_anchor(metric, h.get("text") or "")]
            if anchor_hits:
                mhits = _prepend_hits(anchor_hits[:8], mhits, max_keep=max(len(mhits), 40))
        mctx = cap_context_by_blocks(mhits, max_context_chars_per_metric)

        (debug_dir / f"{bank_id}_{year}_context_{metric}.txt").write_text(
            mctx, encoding="utf-8", errors="ignore"
        )

        metric_contexts[metric] = {
            "context": mctx,
            "hits": mhits,
        }

    return {
        "hits": hits,
        "context": context,
        "nim_ctx": nim_ctx,
        "pcl_ctx": pcl_ctx,
        "nii_ctx": nii_ctx,
        "hits_by_metric": hits_by_metric,
        "metric_contexts": metric_contexts,
    }
