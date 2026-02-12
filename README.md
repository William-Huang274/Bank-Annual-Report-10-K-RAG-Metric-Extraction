# Bank Annual Report / 10-K RAG Metric Extraction (Batch CSV)

English | [简体中文](README.zh-CN.md)

Evidence-first RAG pipeline to extract key financial metrics from U.S. community banks' Annual Reports / 10-K PDFs into a **reproducible CSV**.

This is **not** a chat demo. The primary deliverable is a **batch metrics table** with **evidence traceability** (`source_chunk_id`) and **failure buckets** to support debugging and evaluation.

### Core algorithm contributions (recall + ranking)

- **Entry-page recall**: BFS crawling with domain constraints plus weighted page scoring to find Annual Report / 10-K hubs from noisy bank websites.
- **PDF candidate ranking**: year-aware and intent-aware PDF scoring (annual report / 10-K boosts, false-positive penalties) to prioritize target fiscal-year filings.
- **Metric retrieval recall**: per-metric multi-query FAISS retrieval + bank-constrained filtering + neighbor chunk expansion to recover split table evidence.
- **Lightweight reranking**: metric-specific keyword bonus/penalty reranker to reduce cross-metric hijacking (for example NIM chunks outranking ROA/ROE evidence).
- **Measured impact (A/B, 25 banks x 5 metrics)**: `NOT FOUND` **32 -> 29**, found rows with valid citations **93 -> 96**.

## Update (2026-02-05): v1.0 Evidence-Grounded LLM Review

This release upgrades Stage 06 from a single-pass extractor to a two-layer, evidence-bounded decision pipeline.

- **Two LLM calls with different roles**:
  - Candidate-selection judge (ROA/ROE, only when still missing): select from mined candidates only, no value invention.
  - Unified post-extraction review (all 5 metrics): `keep` / `replace` / `reject` with citation constraints.
- **Confidence-gated execution**: only low-confidence rows are actively reviewed by default.
- **Auditability upgrade**: outputs now include `confidence_*`, `review_action`, `review_note`, `orig_*`, and `review_model`.
- **Evidence quality and traceability** improved in A/B runs:
  - `NOT FOUND`: **32 -> 29**
  - found rows with valid citations: **93 -> 96**
  - value changes are usually citation-linked (21 value changes, 18 with citation updates)

For full technical details and breakdowns, see [Section 6.1](#61-v10-update-evidence-grounded-llm-review-in-stage-06).

---

## 1. What you get

### 1.1 Output artifacts

- **Final CSVs** (default output directory):
  - `data/outputs/processed/metrics_2024.csv`
  - `data/outputs/processed/metrics_2024_tablefilled.csv` (optional)
  - `data/outputs/processed/metrics_2024_comparison.csv` (optional)
- **Logs**: `data/outputs/logs/`
- **Debug dumps** (contexts / intermediate traces): `data/outputs/debug/`

### 1.2 Output schema (typical)

Each metric record is stored with:

- `bank`, `fiscal_year`, `metric`
- `value`, `unit`
- `source_chunk_id` (evidence pointer for audit/debug)
- `failure_reason` (e.g., `value_missing`, `unit_missing`, `semantic_ambiguous`, `no_candidates`, ...)

### 1.3 Metrics covered (baseline)

- NII (Net Interest Income)
- NIM (Net Interest Margin)
- ROA (Return on Assets)
- ROE (Return on Equity)
- PCL (Provision for Credit Losses)

---

## 2. System overview (Why this is an algorithmic retrieval/ranking project)

### 2.1 Architecture (Evidence -> Retrieval -> Context -> Extraction)

1. **Chunk & embed** report text (default embedding model: `BAAI/bge-m3`)
2. **FAISS retrieval** to gather evidence chunks for each metric
   - per-metric multi-query retrieval (different query templates for NII/NIM/ROA/ROE/PCL)
   - bank-aware filtering to prevent cross-bank contamination
   - neighbor chunk expansion to recover values split across chunk boundaries
   - lightweight keyword reranker for metric relevance
3. **Context packing** to build metric-specific evidence context blocks
4. **Hybrid extraction (deterministic-first)**:
   - **Table-first**: parse financial tables and backfill values when available
   - **Regex fallback**: extract from narrative text when tables miss or are incomplete
   - **LLM gated fallback / judge (optional)**: used only for hard cases (ambiguity, conflict arbitration, schema repair)

### 2.2 Design rationale (Deterministic-first + Gated LLM)

For batch financial metric extraction, a naive "LLM-first everywhere" approach is often:

- expensive at scale,
- hard to reproduce,
- difficult to audit (silent failures / hallucinations),
- brittle when output format drifts.

This repo intentionally uses LLM as a **bounded component**:

- ambiguity resolution / conflict arbitration,
- schema-constrained extraction for hard buckets,
- JSON/schema repair to stabilize downstream ingestion.

This is a common production pattern: **deterministic-first for stability + gated LLM for coverage**.

### 2.3 Optimization target and error model

The main optimization target is not "more LLM calls"; it is better retrieval quality:

- improve **recall** for evidence-bearing chunks (reduce `no_candidates` / `value_missing` caused by miss retrieval),
- improve **ranking precision** so relevant chunks appear earlier in packed context,
- preserve **auditability** by keeping evidence citation constraints.

In practice, this project treats retrieval and ranking as first-class algorithm components, and treats LLM as a bounded resolver for hard ambiguity cases.

---

## 3. Pipeline stages (01-06)

- **01** Collect report entry pages
- **02** Download Annual Report / 10-K PDFs
- **03** OCR / text extraction -> plain text
- **03a/03b** Table sidecar extraction (structured table artifacts)
- **04** Build embeddings
- **05** Build FAISS index
- **06** Extract metrics (table-first -> regex fallback -> LLM judge)

---

## 4. Quickstart (Sample index)

The repo includes a **small sample** (1 bank) so you can run an end-to-end smoke test without downloading/OCR-ing large PDFs or rebuilding the full index.

### 4.1 Setup (Windows PowerShell)

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1

python -m pip install -U pip setuptools wheel
pip install -e .
```

> Why a venv? It isolates dependencies (FAISS, PyTorch, sentence-transformers, etc.) so the project is reproducible and won't conflict with your system Python. If you prefer, you can install into your existing environment, but venv is the safe default.

### 4.2 Use the bundled sample index

By default, the extraction script points to the **full** index:

- `data/interim/index/faiss_2024_full/`

For the sample smoke test, switch to:

- `data/sample/index/faiss_2024_sample/`

**Recommended (edit one line):** in `scripts/pipeline/06_extract_metrics_patched_v2_final.py`, set:

- `INDEX_DIR = ROOT / "data" / "sample" / "index" / "faiss_2024_sample"`

**Optional (only if you added env support):**

```powershell
$env:FAISS_INDEX_DIR="data/sample/index/faiss_2024_sample"
```

### 4.3 Run extraction on the sample bank(batch mode, interactive)

This project supports an interactive batch runner. Start the extractor, then type a
`.batch` command at the prompt.

1. Prepare a bank list (one per line), e.g. `data/input/banks_one.txt`

2. Start the extractor:

```powershell
python scripts/pipeline/06_extract_metrics_patched_v2_final.py
# When you see the prompt (e.g., Q (empty to exit):), run:

#   :batch .\data\input\banks_one.txt

# Press Enter on an empty line to exit.

### Expected outputs
```

- `data/outputs/processed/metrics_2024.csv` (final batch table)
- `data/outputs/debug/` (contexts, raw model outputs, traces for debugging)
- `data/outputs/logs/` (run logs)

## 5. Full pipeline (from bank website)

The end-to-end pipeline can be reproduced locally. Expect **large** intermediate artifacts if you run this on many banks (PDFs, OCR text, embeddings, index).

### 5.1 Typical run order

```powershell
python scripts/pipeline/01_collect_entry_pages.py --year 2024
python scripts/pipeline/02_download_reports.py --year 2024
python scripts/pipeline/03_ocr_to_text.py --year 2024

python scripts/pipeline/03a_extract_tables_from_pdf.py --year 2024
python scripts/pipeline/03b_extract_tables.py --year 2024

python scripts/pipeline/04_build_embeddings.py --year 2024
python scripts/pipeline/05_build_faiss_index.py --year 2024

python scripts/pipeline/06_extract_metrics_patched_v2_final.py --year 2024
# When you see the prompt (e.g., `Q (empty to exit):`), run one of:
#   :batch .\data\input\banks_one.txt
#   :batch .\data\input\banks_three.txt
#   :batch .\data\input\banks_25.txt
```

### 5.2 What each step produces (high-level)

#### 01) Collect entry pages (bank website discovery)

- Script: `scripts/pipeline/01_collect_entry_pages.py`
- Purpose: maximize annual-report hub recall under noisy navigation using BFS + rule-based page scoring + PDF candidate ranking.
- Outputs (typical):
  - `data/interim/entry_pages/<YEAR>/*.jsonl` (ranked candidates + scores)
  - `data/outputs/logs/` + `data/outputs/debug/` (timeouts, 403/404, redirects, scoring traces)

#### 02) Download annual reports / 10-K PDFs

- Script: `scripts/pipeline/02_download_reports.py`
- Purpose: follow the selected entry page(s) and download report PDFs.
- Outputs (typical):
  - `data/raw/pdfs/<YEAR>/<bank_id>/*.pdf`
  - `data/outputs/logs/` (success/failure reason, final URL, content-type)

#### 03) OCR / parse PDFs to text

- Script: `scripts/pipeline/03_ocr_to_text.py`
- Purpose: convert PDFs (native text or scanned) into normalized text for chunking.
- Outputs (typical):
  - `data/interim/txt/<YEAR>/<bank_id>/...` (extracted text)
  - `data/outputs/logs/` + `data/outputs/debug/` (OCR stats, failures)

#### 03a/03b) Extract tables (table sidecar)

- Scripts:
  - `scripts/pipeline/03a_extract_tables_from_pdf.py` (PDF-to-table extraction)
  - `scripts/pipeline/03b_extract_tables.py` (post-process / consolidate sidecar)
- Purpose: produce a table sidecar to support **table-first** metric extraction.
- Outputs (typical):
  - `data/interim/tables/table_sidecar_<...>.jsonl` (tables per bank / per PDF)

#### 04) Build embeddings

- Script: `scripts/pipeline/04_build_embeddings.py`
- Purpose: chunk text and compute embeddings (default: `BAAI/bge-m3` on GPU).
- Outputs (typical):
  - `data/interim/embeddings/<YEAR>/<bank_id>/embeddings.npy`
  - `data/interim/embeddings/<YEAR>/<bank_id>/chunks.jsonl` (chunk metadata + text pointers)

#### 05) Build FAISS index

- Script: `scripts/pipeline/05_build_faiss_index.py`
- Purpose: merge embeddings and build the FAISS index + metadata.
- Outputs (typical):
  - `data/interim/index/faiss_<YEAR>_full/faiss.index`
  - `data/interim/index/faiss_<YEAR>_full/meta.jsonl`
  - `data/interim/index/faiss_<YEAR>_full/merge_log.csv`

#### 06) Extract metrics (hybrid: table-first -> regex fallback -> LLM judge)

- Script: `scripts/pipeline/06_extract_metrics_patched_v2_final.py`
- Purpose: per-metric multi-query retrieval + rerank + neighbor expansion, then extract metrics and write a **batch CSV** with evidence IDs.
- Outputs (typical):
  - `data/outputs/processed/metrics_<YEAR>.csv`
  - `data/outputs/debug/` (contexts per metric, raw model outputs, repair traces)
  - `data/outputs/logs/` (run logs)

### 5.3 Index paths (important)

- Default **full** index path (typical):
  - `data/interim/index/faiss_2024_full/`
- Sample index path:
  - `data/sample/index/faiss_2024_sample/`

---

## 6. LLM integration (gated)

LLM is **not** used as the default extractor. It is gated to reduce cost and variance.

Typical use cases:

- resolving conflicting candidates (multiple values for one metric),
- disambiguating definitions (GAAP vs non-GAAP),
- enforcing output schema (JSON repair / schema-constrained extraction),
- extracting values from hard narrative cases when deterministic methods fail.

If you enable LLM fallback:

- ensure your local inference endpoint (e.g., Ollama) is running,
- keep temperature low for schema stability,
- prefer short, evidence-bounded prompts (context is packed from retrieved chunks).

---


### 6.1 v1.0 update: Evidence-grounded LLM review in Stage 06

This upgrade is not just "one more LLM call." It turns Stage 06 from a single-pass extractor into a two-layer, auditable decision pipeline.

#### Two LLM calls, two different responsibilities

1. **Candidate-selection judge (targeted, ROA/ROE only when still missing)**
   - Trigger: only when ROA/ROE remains `NOT FOUND` after deterministic steps.
   - Flow: mine candidates from raw retrieval hits (`mine_ratio_candidates_from_hits`) and let `judge_select_candidate` choose an index.
   - Constraint: the judge must select from provided candidates or return `-1` (no value invention).

2. **Unified post-extraction reviewer (all 5 metrics together)**
   - Flow: `review_all_metrics_after_extract` issues `keep` / `replace` / `reject` decisions.
   - Constraint: `replace` must use an allowed evidence citation for that metric; invalid citations are discarded.
   - Default policy: review only low-confidence rows (`REVIEW_ONLY_LOW_CONFIDENCE=1`) to control cost and minimize unnecessary churn.

#### Why this improves reliability (not just coverage)

- **Evidence credibility**: replacements are evidence-bounded and citation-validated, which reduces free-form hallucination risk.
- **Traceability**: every review override keeps both new and original fields (`orig_val`, `orig_unit`, `orig_source_chunk_id`).
- **Debuggability**: confidence and review metadata (`confidence_*`, `review_action`, `review_note`, `review_model`) expose why each decision happened.
- **Controlled risk posture**: low-confidence rows get active review; medium/high-confidence rows are left stable by default.

#### A/B comparison on the same dataset

- Baseline: `data/outputs/metrics_2024.csv`
- v1.0: `data/outputs/metrics_2024_llm_full_version1.0.csv`
- Scope: 25 banks x 5 metrics = 125 rows (same keys in both files)

Key results:

- **Coverage**: `NOT FOUND` reduced from **32 -> 29** (net +3).
- **Foundness transitions**: 6 rows improved (`NOT FOUND -> FOUND`), 3 rows regressed (`FOUND -> NOT FOUND`).
- **Confidence-gated review**: 33 rows flagged low confidence (`needs_review=1`), 92 rows skipped by confidence gate.
- **Review actions**: `reject=27`, `replace=1`, `keep=5`, `skip_by_confidence=92`.
- **Evidence traceability improved**:
  - found rows with valid citations: **93 -> 96**
  - among rows that are found in both versions (90 rows), citation changed in 23 rows
  - among 21 value-changed rows, 18 also changed citation (value updates stayed evidence-linked)
- **Audit completeness**: for all 28 `replace/reject` rows, `orig_val` and `orig_source_chunk_id` are present (100%).
- **Bucket shift**:
  - old: `ok=86`, `value_missing=32`, `table_prefill=7`
  - new: `ok=89`, `llm_review_rejected=27`, `value_missing=2`, `table_prefill=6`, `llm_review_replaced=1`

Interpretation: the main gain is a shift from opaque extraction output to an evidence-grounded, reviewable, and rollback-friendly pipeline.

## 7. Evaluation & debugging

### 7.1 Failure buckets (typical)

- `value_missing`: no value found after retrieval + extraction
- `unit_missing`: value found but unit missing (often in table header / neighbor chunks)
- `semantic_ambiguous`: multiple candidates / definition mismatch
- `no_candidates`: retrieval returned no usable evidence

### 7.2 Debugging playbook (recommended)

1. **Evidence exists?**  
   Check if the metric's value appears in retrieved contexts.
2. **Recall sufficient?**  
   If evidence is missing, improve retrieval (multi-query, rerank, thresholds).
3. **Parsing correct?**  
   If evidence exists but not extracted, adjust table parsing / regex patterns / unit inference.
4. **Conflicts?**  
   If multiple candidates exist, enable LLM judge path and keep schema constraints strict.

### 7.3 Where to look

- `data/outputs/processed/` - final CSVs
- `data/outputs/logs/` - runtime logs
- `data/outputs/debug/` - debug artifacts (retrieval contexts, intermediate dumps)

---

## 8. Repo policy (Artifacts & Git)

This repo follows a **lightweight + reproducible demo** policy:

- Commit: code, configs, and small sample artifacts under `data/sample/...`
- Do **not** commit: OCR outputs, full embeddings, full FAISS indexes, or other large intermediate artifacts

---

## 9. Roadmap (next upgrades)

- Retrieval ranking: move from heuristic keyword bonuses to a trainable reranker with hard-negative mining.
- Adaptive retrieval policy: dynamic `topk` / score thresholds by metric and bank report style.
- Extraction: stronger unit inference (header/neighbor scan; normalization to consistent units).
- Learning-based extraction: weak supervision to train a structured extractor for hard buckets.
- Derived metrics: compute ROA/ROE when reports require calculation (avg assets/equity)

---

## 10. Disclaimer

PDFs and reports are owned by their respective publishers.
This repository contains code and small derived sample artifacts for demonstration and reproducibility.

