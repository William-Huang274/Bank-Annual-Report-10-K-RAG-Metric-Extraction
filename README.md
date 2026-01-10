# Bank Annual Report / 10-K RAG Metric Extraction Pipeline

> **End-to-end, reproducible pipeline for extracting structured financial metrics from U.S. bank annual reports (10-K / Annual Reports) using OCR, embeddings, FAISS, and local LLMs.**

---

## 1. Project Overview

This project builds a **fully automated, reproducible RAG-style pipeline** to extract key financial metrics from **unstructured bank annual reports** (PDF / HTML), and deliver them as **structured CSV outputs** for downstream analysis.

### Target Metrics (FY2024)

- **Net Interest Income (NII)**
- **Net Interest Margin (NIM)**
- **Return on Assets (ROA)**
- **Return on Equity (ROE)**
- **Provision for Credit Losses (PCL)**

### Scale & Validation

- **25 U.S. banks**
- **10-K / Annual Report PDFs (100+ documents)**
- **End-to-end batch extraction with coverage & sanity checks**

This project is designed as an **engineering system**, not a demo notebook:

- Reproducible directory structure
- Deterministic batch runs
- Debug-friendly intermediate artifacts
- Portable across machines

---

## 2. Why This Project Exists (Problem Statement)

Bank annual reports are:

- Long (200–400 pages)
- Semi-structured (tables, prose, footnotes)
- Highly inconsistent across institutions

Traditional rule-based parsing is brittle, while naïve LLM prompting:

- Does not scale
- Lacks traceability
- Produces unverifiable results

**Goal:**  
Build a system that can **reliably extract financial metrics at scale**, with **clear provenance** and **reproducible outputs**.

---

## 3. System Architecture

```text
PDF / HTML
   ↓
OCR & Text Normalization
   ↓
Chunking (overlap + metadata)
   ↓
Embedding (BGE-M3, GPU)
   ↓
FAISS Vector Index
   ↓
Multi-Query Retrieval
   ↓
Hybrid Extraction
  ├─ Regex-first (high precision)
  └─ LLM fallback (contextual metrics)
   ↓
Structured CSV Output
```

**Key design choice:**  
> Batch table extraction → CSV delivery, **not conversational QA**.

---

## 4. Key Engineering Decisions

### 4.1 Hybrid Extraction Strategy

- **Regex-first** for stable, numeric-heavy metrics (e.g. NII, NIM)
- **LLM fallback** for contextual or computed metrics (ROA, ROE)
- Minimizes hallucination and improves determinism

### 4.2 Retrieval Quality over Prompt Engineering

- Multi-query expansion per metric
- Conservative similarity thresholds with fallback
- Context truncation using **head + tail** strategy to preserve numeric evidence

### 4.3 Strict Output Contract

All outputs are normalized into a fixed schema:

```text
bank | fiscal_year | metric | value | unit | source_chunk_id
```

Every extracted value is traceable back to an indexed text chunk.

### 4.4 Local, Controllable Models

- **Embeddings:** `BAAI/bge-m3` (GPU-accelerated)
- **LLM inference:** local Ollama (`qwen3:4b`)
- No external API dependency → reproducible and cost-free runs

---

## 5. Directory Structure

```text
.
├─ data/
│  ├─ input/            # bank lists, entry files
│  ├─ raw/              # OCR outputs, normalized text
│  ├─ interim/
│  │   └─ index/        # FAISS index + metadata
│  └─ processed/
│      └─ metrics_2024.csv
├─ scripts/
│  ├─ pipeline/         # 01–06 pipeline stages
│  ├─ debug/            # retrieval & extraction tools
│  └─ dryrun_validate.py
├─ outputs/
│  ├─ logs/
│  └─ debug/
├─ run_pipeline.py
└─ README.md
```

---

## 6. Reproducibility & Smoke Test

To validate the pipeline **without recomputation**:

```bash
python scripts/dryrun_validate.py
```

This checks:

- FAISS index loadability
- Embedding model availability
- LLM initialization
- Output path consistency

---

## 7. Results Summary

- **25 banks processed**
- **NII / NIM:** stable, high-confidence extraction
- **ROA / ROE:** improved via prose-first + LLM fallback
- **CSV outputs ready** for downstream analysis

This project emphasizes **engineering reliability** over absolute metric completeness.

---

## 8. What This Project Demonstrates

- Large-scale unstructured document processing
- Vector retrieval system design (FAISS)
- Hybrid information extraction (rules + LLMs)
- Debug-driven ML system development
- Production-oriented Python project structure

---

## 9. Possible Extensions

- Table structure detection for improved unit inference
- Reranker integration
- Multi-year trend extraction
- Downstream analytics / dashboarding

---

## 10. Tech Stack

- Python 3.10
- FAISS (IVF-PQ)
- sentence-transformers / BGE-M3
- Ollama (Qwen3-4B)
- OCR & PDF processing tools
