# src/rag/retrieval.py
from __future__ import annotations

import numpy as np
import re
import logging
logger = logging.getLogger("extract")


def retrieve_hits(index, meta, qvec, topk_search: int = 50, target_bank=None, topk_final: int = 20):
    """
    Retrieve top hits for a bank using a single retrieval query.
    Baseline retrieval path (single query -> bank-filtered hits).
    """
    D, I = index.search(qvec, int(topk_search))

    hits = []
    for rnk, (score, idx) in enumerate(zip(D[0], I[0]), 1):
        hits.append((rnk, float(score), meta[int(idx)]))

    if not hits:
        return [], None

    # Default behavior: use the bank identifier from the top-ranked hit
    if target_bank is None:
        target_bank = hits[0][2].get("bank_folder")

    # Keep only hits from the target bank
    hits = [h for h in hits if h[2].get("bank_folder") == target_bank][: int(topk_final)]
    return hits, target_bank


def search_faiss(index, meta, emb, query: str, topk: int = 50):
    """
    Return list[dict] with keys:
    - score
    - bank (meta bank_folder)
    - stem
    - chunk_id
    - text
    """
    if not query:
        return []
    qvec = emb.encode([query], batch_size=1, normalize_embeddings=True, show_progress_bar=False,
            convert_to_numpy=True).astype(np.float32)
    D, I = index.search(qvec, int(topk))

    out = []
    for score, idx in zip(D[0], I[0]):
        if idx < 0:
            continue
        m = meta[int(idx)]
        out.append({
            "score": float(score),
            "bank": m.get("bank_folder"),
            "stem": m.get("stem"),
            "chunk_id": m.get("chunk_id"),
            "text": m.get("text") or "",
        })
    return out


def retrieve_hits_per_metric(index, meta, emb, bank_id: str, year: int,
                            metrics: list,
                            topk_per_query: int = 40,
                            topk_per_metric: int = 25,
                            min_score: float = 0.50):
    """
    Return: dict(metric_name -> list[hits]) where hits are list[dict] from search_faiss()
    """

    def _norm_bank(x: str) -> str:
        return (x or "").strip().lower()

    def _bank_match(hit_bank: str, target_bank: str) -> bool:
        hb = _norm_bank(hit_bank)
        tb = _norm_bank(target_bank)
        if not hb or not tb:
            return False
        if hb == tb:
            return True
        # tolerate leading underscore or suffix/prefix
        hb2 = hb.lstrip("_")
        tb2 = tb.lstrip("_")
        if hb2 == tb2:
            return True
        if hb2.startswith(tb2) or tb2.startswith(hb2):
            return True
        # guarded substring match
        if len(tb2) >= 8 and tb2 in hb2:
            return True
        if len(hb2) >= 8 and hb2 in tb2:
            return True
        return False

    # --- resolve to a canonical bank_folder from meta if possible ---
    target_bank = bank_id
    tb_norm = _norm_bank(bank_id).lstrip("_")
    if tb_norm:
        for mm in meta:
            mb = _norm_bank(mm.get("bank_folder")).lstrip("_")
            if mb == tb_norm:
                target_bank = mm.get("bank_folder")
                break

    QUERY_BANK = {
        "NIM": [
            f"{bank_id} net interest margin {year}",
            f"{bank_id} net interest margin on a tax-equivalent basis {year}",
            f"{bank_id} NIM {year}",
            "Selected Performance Ratios net interest margin",
            "net interest margin on a tax-equivalent basis was",
            "net interest margin was",
            "net interest margin for 2024 was",
            "net interest margin on a tax-equivalent basis was",
            "tax-equivalent net interest margin was",
            "Net interest margin (GAAP-derived)",
            "NIM net interest margin",
            "full year NIM was",
            "net interest margin was",
            "NIM was",
            "down * bps / up * bps",
            "tax-equivalent net interest margin",
            "fully taxable-equivalent net interest margin",
            "Net interest margin - FTE",
            "GAAP-derived",
            "net yield on interest-earning assets",
            "net yield on earning assets",
            "net yield",
            "net interest margin was % 2024",
            "NIM net interest margin 2024",
            "tax-equivalent net interest margin 2024",
            "net interest spread",
            "net interest margin table",
            "tax-equivalent",
            "FTE",
            "performance ratio table",
        ],
        "NII": [
            f"{bank_id} net interest income {year}",
            f"net interest income {year}",
            "NII net interest income",
            "net interest income (nii)",
            "net financing revenue",
            "financing revenue",
            "net interest income",
            "net interest revenue",
            "interest income net",
            "increased/decreased",
            "interest income",
            "interest expense",
            "net interest revenue",
            "net financing revenue and other interest income",
            "net interest income was $ 2024",
            "NII net interest income 2024",
            "tax-equivalent net interest income 2024",
            "other interest income",
        ],
        "ROA": [
            "return on average total assets",
            "ROAA return on average assets",
            f"Selected Performance Ratios return on assets {year}",
            f"{bank_id} return on assets {year}",
            f"return on assets {year}",
            "ROA return on assets",
            "ROA was",
            "return on assets",
            "return on average assets",
            "Other Data at Year-end Selected Performance Ratios",
            "ROAA ROAE Selected Performance Ratios",
            "FY2024 return on average assets ROA performance ratios selected performance ratios",
            "ROA",
            # "net income"
            # "average assets",
            # "average total assets",
            
            # section/title anchors
            f"Financial ratios Return on average assets {year}",
            f"Average Balances {year}",
            f"Average Balances average total assets {year}",
            f"Average Balances average assets {year}",
            # f"Summary of Operations Net Income {year}",
            f"Five year summary Selected Performance Ratios {year}",
            f"Financial Highlights return on assets {year}",
            f"Key performance indicators return on assets {year}",
            f"Key performance ratios ROAA ROAE {year}",
            "Return on average assets (a)",
            "Return on average assets",

            # computation anchors
            f"net income {year} average total assets",
            f"net income {year} average assets",
            f"average total assets {year}",
            f"average assets {year}",
            f"Selected financial data return on average assets {year}",
            f"average total assets and return on assets {year}",
            f"net income divided by average assets {year}", 
        ],
        "ROE": [
            f"{bank_id} return on equity {year}",
            f"return on equity {year}",
            "return on average equity",
            "return on equity",
            "ROE return on equity",
            "ROAE return on average equity",
            "return on average shareholders' equity",
            "return on average stockholders' equity",
            "return on avg equity",
            f"Selected Performance Ratios return on equity {year}",
            "Other Data at Year-end Selected Performance Ratios",
            "Equity to Assets Dividend Payout Return on Equity",
            "ROAA ROAE Selected Performance Ratios",
            "FY2024 return on average equity ROE performance ratios selected performance ratios",
            "GAAP Return on Equity",
            "Return (net income) on ... shareholder's equity",
            "return on common equity",
            "return on shareholders'equity",
            "ROE",
            
            # section/title anchors
            # f"Summary of Operations Net Income {year}",
            f"Five year summary Selected Performance Ratios {year}",
            f"Financial Highlights return on equity {year}",
            f"Key performance indicators return on equity {year}",
            f"Selected financial data return on average equity {year}",
            f"Selected financial data return on average equity {year}",
            
            # computation anchors
            f"net income {year} average equity",
            f"net income {year} average shareholders' equity",
            f"average equity {year}",
            f"average shareholders' equity {year}",
            f"average equity and return on equity {year}",
        ],
        "Provision for Credit Losses": [
            f"{bank_id} provision for credit losses {year}",
            f"provision for credit losses {year}",
            "provision for credit losses",
            "provision (benefit) for credit losses",
            "reversal of provision for credit losses",
            "provision for credit losses was",
            "credit loss expense",
            "credit loss expense was",
            "provision for loan losses",
            "provision for loan and lease losses",
            "allowance for credit losses",
            "allowance for credit losses provision",
            "ACL",

            # income statement neighbors (helps pull the right statement block)
            "consolidated statements of income",
            "income before income taxes",
            "noninterest expense",
            "non-interest expense",
            "income tax expense",

            # light year anchors (optional)
            f"provision for credit losses in {year}",
            f"provision for credit losses {year}",
        ],
    }

    def _get_hit_bank(h):
        return h.get("bank") or h.get("bank_id") or h.get("k", "")

    def _get_hit_key(h):
        # Dedup within a bank/metric must be per-(stem, chunk_id)
        return (str(h.get("stem") or ""), str(h.get("chunk_id") or h.get("id") or ""))

    results = {}
    for metric in metrics:
        # Per-metric cap: NII/NIM often split keyword/value across neighbor chunks,
        # so we keep more candidates for these two metrics.
        local_topk_per_query = topk_per_query
        local_topk_per_metric = topk_per_metric
        if metric in ("NII", "NIM"):
            local_topk_per_query = max(int(local_topk_per_query), 160)
            local_topk_per_metric = max(int(local_topk_per_metric), 12)

        queries = QUERY_BANK.get(metric, [])
        pooled = []

        # ROA/ROE need deeper retrieval to catch ratio-table blocks like "Financial ratios: Return on average assets ..."
        _topk_q = int(local_topk_per_query)
        if metric in ("ROA", "ROE"):
            _topk_q = max(_topk_q, 200)   # Increase depth to 200 only for ROA/ROE; leave others unchanged.

        for q in queries:
            pooled.extend(search_faiss(index, meta, emb, q, topk=_topk_q))

        # bank filter + score filter + dedup
        filtered = []
        seen = set()
        # DEBUG: inspect whether ROA/ROE has near chunks BEFORE min_score gate
        if metric in ("ROA", "ROE"):
            near = [
                (str(h.get("chunk_id")), float(h.get("score", 0.0)))
                for h in pooled
                if str(h.get("chunk_id", "")).isdigit()
                and 260 <= int(h["chunk_id"]) <= 280
            ]
            if near:
                near_sorted = sorted(near, key=lambda x: x[1], reverse=True)[:30]
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug("[DEBUG][RETR_PRE_GATE] bank=%s metric=%s near_top=%s total=%d",
                                target_bank, metric, near_sorted[:10], len(near_sorted))

        # DEBUG: inspect pooled hits before min_score gate
        if metric in ("ROA", "ROE"):
            near = [
                (str(h.get("chunk_id")), float(h.get("score", 0.0)))
                for h in pooled
                if str(h.get("chunk_id", "")).isdigit()
                and 260 <= int(h["chunk_id"]) <= 280
            ]
            if near:
                if logger.isEnabledFor(logging.DEBUG):
                    near_sorted_all = sorted(near, key=lambda x: x[1], reverse=True)
                    logger.debug("[DEBUG][RETR_PRE_GATE] metric=%s near_chunks_top=%s total=%d",
                                metric, near_sorted_all[:10], len(near_sorted_all))

        def _kw_hit(metric: str, text: str) -> bool:
            tx = (text or "").lower()
            if metric == "ROA":
                return ("return on average assets" in tx) or ("return on assets" in tx) or ("roaa" in tx) or ("selected performance ratios" in tx)
            if metric == "ROE":
                return ("return on average equity" in tx) or ("return on equity" in tx) or ("roae" in tx) or ("selected performance ratios" in tx)
            return False

        dropped_dbg = []
        for h in sorted(
                pooled,
                key=lambda x: (1 if _kw_hit(metric, x.get("text")) else 0, float(x.get("score", 0.0))),
                reverse=True
            ):
            sc = float(h.get("score", 0.0))
            cid = str(h.get("chunk_id", ""))

            if sc < float(min_score):
                # DEBUG: see whether ROA/ROE key chunks are dropped by min_score
                if metric in ("ROA", "ROE") and cid.isdigit():
                    ci = int(cid)
                    if 260 <= ci <= 280:
                        dropped_dbg.append((cid, sc))
                continue

            if not _bank_match(_get_hit_bank(h), target_bank):
                continue
            k = _get_hit_key(h)
            if not k or k in seen:
                continue
            seen.add(k)
            filtered.append(h)
            if len(filtered) >= int(local_topk_per_metric):
                break

        if dropped_dbg and logger.isEnabledFor(logging.DEBUG):
            dropped_dbg.sort(key=lambda x: x[1], reverse=True)
            logger.debug("[DEBUG][RETR_GATE_DROP_SUM] metric=%s dropped=%d min_score=%s top=%s",
                        metric, len(dropped_dbg), min_score, dropped_dbg[:10])

        # Ensure we keep at least one "ratio table" evidence block for ROA/ROE if present
        if metric in ("ROA", "ROE"):
            def _is_ratio_table_hit(h: dict) -> bool:
                tx = (h.get("text") or "").lower()
                if "financial ratios" in tx:
                    return True
                if "selected performance ratios" in tx:
                    return True
                if metric == "ROA" and "return on average assets" in tx:
                    return True
                if metric == "ROE" and "return on average equity" in tx:
                    return True
                return False

            if filtered and (not any(_is_ratio_table_hit(h) for h in filtered)):
                for h in sorted(pooled, key=lambda x: x.get("score", 0.0), reverse=True):
                    if h.get("score", 0.0) < float(min_score):
                        continue
                    if not _bank_match(_get_hit_bank(h), target_bank):
                        continue
                    if _is_ratio_table_hit(h):
                        # replace the last one to keep list size stable
                        filtered[-1] = h
                        break

        results[metric] = filtered
    return results


def retrieve_hits_multiquery(
    index,
    meta,
    emb,
    target_bank: str,
    year: str,
    per_metric_topk: int = 10,
    topk_final: int = 20,
    k0: int = 200,
    kmax: int = 20000,
    MIN_SCORE: float = 0.50,
    neighbor_window: int = 2,
):
    """
    A-2: multiquery retrieval + neighbor expansion.
    - Keep your QUERY_BANK + retrieve_hits_per_metric() (high value).
    - Merge hits across metrics -> dedup -> optionally expand neighbors (chunk +/- window)
    - Return list[dict] sorted by score desc.
    """

    metrics = ["NII", "NIM", "ROA", "ROE", "Provision for Credit Losses"]

    # 1) Per-metric multiquery (reuse your existing logic)
    by_metric = retrieve_hits_per_metric(
        index=index,
        meta=meta,
        emb=emb,
        bank_id=target_bank,
        year=int(year),
        metrics=metrics,
        topk_per_query=max(int(k0), 40),                 # now k0 actually matters
        topk_per_metric=max(int(per_metric_topk), 10),   # per metric cap
        min_score=float(MIN_SCORE),
    )

    # 2) Merge + dedup
    merged = []
    seen = set()
    for m in metrics:
        for h in by_metric.get(m, []):
            key = (h.get("bank"), h.get("stem"), str(h.get("chunk_id")))
            if key in seen:
                continue
            seen.add(key)

            # add cite key used by downstream context/prompt
            if "k" not in h:
                h["k"] = f'k={h.get("bank")}|stem={h.get("stem")}|chunk={h.get("chunk_id")}'
            merged.append(h)

    merged.sort(key=lambda x: x.get("score", 0.0), reverse=True)

    # 3) Neighbor expansion (same bank+stem, chunk_id +/- neighbor_window)
    if neighbor_window and neighbor_window > 0 and merged:
        # build lookup: (bank, stem, chunk_id) -> meta idx
        key2idx = {}
        for i, mm in enumerate(meta):
            b = mm.get("bank_folder")
            stem = mm.get("stem")
            cid = mm.get("chunk_id")
            if b is None or stem is None or cid is None:
                continue
            key2idx[(b, stem, str(cid))] = i

        def _add_neighbor(bank, stem, chunk_id, base_score, metric):
            kk = (bank, stem, str(chunk_id))
            if kk in seen:
                return
            mi = key2idx.get(kk)
            if mi is None:
                return
            mm = meta[mi]
            penalty = 0.0 if metric in ("NIM", "NII") else 0.02
            
            nh = {
                "score": float(max(base_score - penalty, 0.0)),
                "bank": mm.get("bank_folder"),
                "stem": mm.get("stem"),
                "chunk_id": mm.get("chunk_id"),
                "text": mm.get("text") or "",
                "k": f'k={mm.get("bank_folder")}|stem={mm.get("stem")}|chunk={mm.get("chunk_id")}',
                "metric": metric,
                "neighbor_of": str(chunk_id)
            }
            seen.add((nh["bank"], nh["stem"], str(nh["chunk_id"])))
            merged.append(nh)

        # expand around current top pool (cap expansion)
        cap = min(len(merged), int(kmax))
        base_list = list(merged[:cap])
        for h in base_list:
            bank = h.get("bank")
            stem = h.get("stem")
            cid = h.get("chunk_id")
            if bank is None or stem is None or cid is None:
                continue
            try:
                cid_int = int(cid)
            except Exception:
                continue
            base_score = float(h.get("score", 0.0))
            metric = h.get("metric")
            for d in range(-int(neighbor_window), int(neighbor_window) + 1):
                if d == 0:
                    continue
                _add_neighbor(bank, stem, cid_int + d, base_score, metric)

        # ---- must-keep: ensure NIM/NII neighbors survive final topk truncation
        must_keep = []
        rest = []
        for h in merged:
            if h.get("metric") in ("NIM", "NII") and h.get("neighbor_of"):
                must_keep.append(h)
            else:
                rest.append(h)

        # keep order by score inside each group
        must_keep.sort(key=lambda x: float(x.get("score", 0.0)), reverse=True)
        rest.sort(key=lambda x: float(x.get("score", 0.0)), reverse=True)

        merged = must_keep + rest

        # re-sort after expansion
        merged.sort(key=lambda x: x.get("score", 0.0), reverse=True)

    # 4) Final trim
    return merged[: int(topk_final)]

def expand_neighbors_from_meta(mhits, meta, window=2, max_add=40, dprint=None):
    """
    Expand retrieval hits by adding neighbor chunks using the meta.jsonl lookup.

    Args:
        mhits: List[dict]. Retrieval hits from FAISS (typically from search_faiss()).
                Each hit should contain bank/bank_folder, stem, chunk_id, and score.
        meta:  List[dict]. Rows from meta.jsonl with at least:
                bank_folder, stem, chunk_id, text (and optional other fields).
        window: Neighbor radius. For each hit chunk_id=c, try to add [c-window, ..., c+window].
        max_add: Maximum number of neighbor chunks to add (runtime/size guardrail).

    Returns:
        List[dict]: Original hits + neighbor hits (deduplicated by (bank_folder, stem, chunk_id)).
                    Neighbor scores are derived from the base hit score with a distance penalty.

    Notes:
        This is used to recover table rows/columns that spill across chunk boundaries,
        especially for ratio tables (e.g., NIM/ROA/ROE) where headers and values may not co-locate.
    """
    if dprint is None:
        dprint = lambda *a, **k: None

    key2mm = {}
    for mm in meta:
        b = mm.get("bank_folder")
        s = mm.get("stem")
        c = mm.get("chunk_id")
        if b is None or s is None or c is None:
            continue
        key2mm[(b, s, int(c))] = mm

    out = list(mhits)
    seen = set(((h.get("bank") or h.get("bank_folder") or h.get("bank_id")), h.get("stem"), int(h.get("chunk_id"))) 
                for h in mhits if h.get("chunk_id") is not None)


    for h in list(mhits):
        if h.get("chunk_id") in ("497", 497):
            dprint("[DEBUG][NIM_NEI] hit keys:",
                "bank=", h.get("bank"),
                "bank_folder=", h.get("bank_folder"),
                "stem=", h.get("stem"),
                "chunk=", h.get("chunk_id"))
        try:
            b, s, c = (h.get("bank") or h.get("bank_folder") or h.get("bank_id")), h.get("stem"), int(h.get("chunk_id"))
        except Exception:
            continue
        base_score = float(h.get("score", 0.0))
        for d in range(-window, window + 1):
            if d == 0:
                continue
            kk = (b, s, c + d)
            if kk in seen:
                continue
            mm = key2mm.get(kk)
            if (c in (497,498,499,500,501)) and (d in (1,2,3,4)):
                dprint("[DEBUG][NIM_NEI_TRY] kk=", kk, "found=", bool(mm))
            if not mm:
                continue
            seen.add(kk)
            out.append({
                "score": max(base_score - 0.02, 0.0),
                "bank": mm.get("bank_folder"),
                "stem": mm.get("stem"),
                "chunk_id": mm.get("chunk_id"),
                "text": mm.get("text") or "",
                "k": f'k={mm.get("bank_folder")}|stem={mm.get("stem")}|chunk={mm.get("chunk_id")}',
            })
            if len(out) >= len(mhits) + max_add:
                return out
    return out

def rerank_hits_by_metric_keywords(mhits, metric: str):
    """
    Re-rank FAISS hits using cheap keyword bonuses/penalties so the evidence
    for a metric (ROA/ROE/NIM/PCL) appears early in the prompt.
    """
    if not mhits:
        return mhits

    metric = (metric or "").strip()

    # Per-metric positive patterns (strong -> higher bonus)
    POS = {
        "ROA": [
            r"\breturn on (average )?assets\b",
            r"\broa\b",
            r"\breturn on assets\b",
            r"net income.{0,40}average (total )?assets",
            r"ratio of net income.{0,40}average (total )?assets",
        ],
        "ROE": [
            r"\breturn on (average )?equity\b",
            r"\broe\b",
            r"\breturn on equity\b",
            r"net income.{0,40}average (total )?equity",
            r"ratio of net income.{0,40}average (total )?equity",
        ],
        "NIM": [
            r"\bnet interest margin\b",
            r"\bnim\b",
            r"\bnet yield on (average )?interest[- ]earning assets\b",
            r"\bnet interest income\b.{0,80}\baverage\b.{0,40}assets",
        ],
        "Provision for Credit Losses": [
            r"\bprovision for credit losses\b",
            r"\bprovision for loan losses\b",
            r"\bcredit loss(es)? provision\b",
            r"\ballowance for credit losses\b.{0,80}\bprovision\b",
        ],
    }

    # Cross-metric penalties (to prevent NIM chunks hijacking ROA/ROE, etc.)
    NEG = {
        "ROA": [r"\bnet interest margin\b", r"\bnim\b", r"\bnet interest income\b"],
        "ROE": [r"\bnet interest margin\b", r"\bnim\b", r"\bnet interest income\b"],
        "NIM": [r"\breturn on\b", r"\broa\b", r"\broe\b"],
        "Provision for Credit Losses": [r"\bnet interest margin\b", r"\bnim\b"],
    }

    pos_pats = [re.compile(p, re.I) for p in POS.get(metric, [])]
    neg_pats = [re.compile(p, re.I) for p in NEG.get(metric, [])]

    def _bonus(text: str) -> float:
        t = (text or "")
        b = 0.0
        # stronger bonus for strong phrases
        for j, pat in enumerate(pos_pats):
            if pat.search(t):
                b += 0.18 if j < 2 else 0.10
        for pat in neg_pats:
            if pat.search(t):
                b -= 0.12
        return b

    scored = []
    for h in mhits:
        base = float(h.get("score", 0.0))
        b = _bonus(h.get("text", ""))
        h2 = dict(h)
        h2["_rerank"] = base + b
        scored.append(h2)

    scored.sort(key=lambda x: x.get("_rerank", x.get("score", 0.0)), reverse=True)
    for h in scored:
        if "_rerank" in h:
            del h["_rerank"]
    return scored
