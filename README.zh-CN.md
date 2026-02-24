# 银行年报 / 10-K RAG 指标抽取（批处理 CSV）

[English](README.md) | 简体中文

本项目是一套以证据为核心（evidence-first）的 RAG 流水线，用于从美国社区银行 Annual Report / 10-K PDF 中批量提取关键财务指标，并产出**可复现的 CSV**。

它**不是**聊天演示。核心交付是一张**批量指标结果表**，并为每条结果保留**证据可追踪性**（`source_chunk_id`）和**失败分桶**，便于调试、评估与复盘。

### 核心算法贡献（召回 + 排序）

- **入口页召回**：通过域名约束下的 BFS 抓取与页面加权评分，在噪声较高的银行官网中定位 Annual Report / 10-K 入口。
- **PDF 候选排序**：结合年份与意图对 PDF 打分（annual report / 10-K 加权、误报惩罚），优先命中目标财年文件。
- **指标检索召回**：按指标执行 multi-query FAISS 检索，并配合银行范围过滤与邻接 chunk 扩展，补回被切分表格里的关键证据。
- **轻量重排**：利用指标关键词加减分，降低跨指标“劫持”（例如 NIM 片段压过 ROA/ROE 证据）。
- **可量化效果（A/B，25 家银行 x 5 个指标）**：`NOT FOUND` **32 -> 29**，有有效引用的命中行 **93 -> 96**。

## 更新（2026-02-05）：v1.0 证据约束 LLM 复核

本次版本将 Stage 06 从单次抽取升级为双层、证据边界明确的决策流程。

- **两次 LLM 调用，各司其职**：
  - 候选选择裁决（ROA/ROE，仅在仍缺失时触发）：只能从已挖掘候选中选择，不允许编造数值。
  - 抽取后统一复核（5 个指标一起判断）：在引用约束下输出 `keep` / `replace` / `reject`。
- **按置信度门控**：默认仅主动复核低置信度行。
- **可审计性增强**：输出新增 `confidence_*`、`review_action`、`review_note`、`orig_*`、`review_model`。
- **A/B 中证据质量与可追踪性提升**：
  - `NOT FOUND`：**32 -> 29**
  - 有有效引用的命中行：**93 -> 96**
  - 数值变更通常伴随引用变更（21 行数值变化中，18 行同步更新引用）

完整技术细节见第 6.1 节。

---

## 1. 你将得到什么

### 1.1 输出产物

- **最终 CSV**（默认输出目录）：
  - `data/outputs/processed/metrics_2024.csv`
  - `data/outputs/processed/metrics_2024_tablefilled.csv`（可选）
  - `data/outputs/processed/metrics_2024_comparison.csv`（可选）
- **日志目录**：`data/outputs/logs/`
- **调试产物**（上下文 / 中间轨迹）：`data/outputs/debug/`

### 1.2 输出字段（典型）

每条指标记录通常包含：

- `bank`、`fiscal_year`、`metric`
- `value`、`unit`
- `source_chunk_id`（审计 / 调试用证据指针）
- `failure_reason`（如 `value_missing`、`unit_missing`、`semantic_ambiguous`、`no_candidates` 等）

### 1.3 覆盖指标（基线）

- NII（Net Interest Income）
- NIM（Net Interest Margin）
- ROA（Return on Assets）
- ROE（Return on Equity）
- PCL（Provision for Credit Losses）

---

## 2. 系统概览（为什么这是检索/排序算法项目）

### 2.1 架构（Evidence -> Retrieval -> Context -> Extraction）

1. **文本切块并向量化**（默认向量模型：`BAAI/bge-m3`）
2. **FAISS 检索**，按指标聚合证据 chunk
   - 按指标执行 multi-query（NII/NIM/ROA/ROE/PCL 使用不同查询模板）
   - 银行范围过滤，避免跨银行污染
   - 邻接 chunk 扩展，补回跨 chunk 边界的数值
   - 轻量关键词重排，强化指标相关性
3. **上下文打包**，构建按指标组织的证据块
4. **混合抽取（deterministic-first）**
   - **表格优先（table-first）**：优先解析财务表并回填数值
   - **正则兜底（regex fallback）**：表格缺失或不完整时，从叙述文本提取
   - **LLM 门控兜底 / 裁决（可选）**：仅用于难例（歧义、冲突仲裁、schema 修复）

### 2.2 设计思路（Deterministic-first + Gated LLM）

在批量财务指标抽取中，直接“LLM-first everywhere”往往会带来：

- 规模化成本偏高，
- 结果复现性差，
- 审计困难（静默失败 / 幻觉），
- 输出格式漂移时更脆弱。

因此本仓库将 LLM 放在**有边界的环节**，主要用于：

- 歧义消解与冲突仲裁，
- 困难样本的 schema 约束抽取，
- JSON/schema 修复，稳定下游处理。

这也是常见的生产策略：**先用确定性方法稳住主流程，再用门控 LLM 补覆盖**。

### 2.3 优化目标与错误模型

本项目优化重点不是“多调 LLM”，而是提升检索质量：

- 提高证据 chunk **召回率**（降低检索遗漏导致的 `no_candidates` / `value_missing`），
- 提升排序精度，让更相关的 chunk 更早进入上下文，
- 用证据引用约束保持**可审计性**。

实践上，本项目把检索与排序视为一等算法组件，把 LLM 作为复杂歧义场景下的受限决策器。

---

## 3. 流水线阶段（01-06）

- **01** 收集报告入口页
- **02** 下载 Annual Report / 10-K PDF
- **03** OCR / 文本抽取 -> 纯文本
- **03a/03b** 表格 sidecar 抽取（结构化表格产物）
- **04** 构建向量
- **05** 构建 FAISS 索引
- **06** 抽取指标（table-first -> regex fallback -> LLM judge）

---

## 4. 快速开始（示例索引）

仓库自带一个**小型样例**（1 家银行），可在不下载 / OCR 大量 PDF、也不重建全量索引的情况下完成端到端 smoke test。

### 4.1 环境准备（Windows PowerShell）

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1

python -m pip install -U pip setuptools wheel
pip install -e .
```

> 为什么建议 venv？它能隔离依赖（FAISS、PyTorch、sentence-transformers 等），降低环境冲突，保证可复现。你也可以用已有环境，但 venv 更稳妥。

### 4.2 使用仓库自带 sample 索引

默认情况下，抽取脚本指向**全量**索引：

- `data/interim/index/faiss_2024_full/`

若要跑 sample smoke test，请切换到：

- `data/sample/index/faiss_2024_sample/`

**推荐方式（改 1 行）：**在 `scripts/pipeline/06_extract_metrics_patched_v2_final.py` 中设置：

- `INDEX_DIR = ROOT / "data" / "sample" / "index" / "faiss_2024_sample"`

**可选（仅当你已加环境变量支持）：**

```powershell
$env:FAISS_INDEX_DIR="data/sample/index/faiss_2024_sample"
```

### 4.3 在 sample 银行上运行抽取（批量模式，交互式）

项目支持交互式批处理。启动抽取器后，在提示符输入 `.batch` 命令。

1. 准备银行列表（每行一个），如 `data/input/banks_one.txt`
2. 启动抽取器：

```powershell
python scripts/pipeline/06_extract_metrics_patched_v2_final.py
# 看到提示（例如 Q (empty to exit):）后，执行：
#   :batch .\data\input\banks_one.txt
# 直接回车空行即可退出。
```

### 预期输出

- `data/outputs/processed/metrics_2024.csv`（最终批量表）
- `data/outputs/debug/`（上下文、模型原始输出、调试轨迹）
- `data/outputs/logs/`（运行日志）

## 5. 全量流水线（从银行网站开始）

该端到端流程可在本地复现。若覆盖多家银行，预计会产生**较大**中间产物（PDF、OCR 文本、向量、索引）。

### 5.1 典型运行顺序

```powershell
python scripts/pipeline/01_collect_entry_pages.py --year 2024
python scripts/pipeline/02_download_reports.py --year 2024
python scripts/pipeline/03_ocr_to_text.py --year 2024

python scripts/pipeline/03a_extract_tables_from_pdf.py --year 2024
python scripts/pipeline/03b_extract_tables.py --year 2024

python scripts/pipeline/04_build_embeddings.py --year 2024
python scripts/pipeline/05_build_faiss_index.py --year 2024

python scripts/pipeline/06_extract_metrics_patched_v2_final.py --year 2024
# 看到提示（例如 `Q (empty to exit):`）后，执行以下之一：
#   :batch .\data\input\banks_one.txt
#   :batch .\data\input\banks_three.txt
#   :batch .\data\input\banks_25.txt
```

### 5.2 各步骤产物（高层）

#### 01) 收集入口页（银行网站发现）

- 脚本：`scripts/pipeline/01_collect_entry_pages.py`
- 目的：在嘈杂站点结构中，通过 BFS + 规则打分 + PDF 候选排序，尽可能提高年报入口召回率。
- 典型输出：
  - `data/interim/entry_pages/<YEAR>/*.jsonl`（候选及评分）
  - `data/outputs/logs/` + `data/outputs/debug/`（超时、403/404、跳转、评分轨迹）

#### 02) 下载年报 / 10-K PDF

- 脚本：`scripts/pipeline/02_download_reports.py`
- 目的：沿入口页链路下载目标报告 PDF。
- 典型输出：
  - `data/raw/pdfs/<YEAR>/<bank_id>/*.pdf`
  - `data/outputs/logs/`（成功/失败原因、最终 URL、content-type）

#### 03) OCR / 解析 PDF 为文本

- 脚本：`scripts/pipeline/03_ocr_to_text.py`
- 目的：将 PDF（原生文本或扫描件）转换为可切块的规范化文本。
- 典型输出：
  - `data/interim/txt/<YEAR>/<bank_id>/...`（抽取文本）
  - `data/outputs/logs/` + `data/outputs/debug/`（OCR 统计、失败记录）

#### 03a/03b) 抽取表格（table sidecar）

- 脚本：
  - `scripts/pipeline/03a_extract_tables_from_pdf.py`（PDF 到表格）
  - `scripts/pipeline/03b_extract_tables.py`（后处理 / sidecar 汇总）
- 目的：生成表格 sidecar，支持**table-first** 指标抽取。
- 典型输出：
  - `data/interim/tables/table_sidecar_<...>.jsonl`（按银行 / PDF 组织的表格）

#### 04) 构建向量

- 脚本：`scripts/pipeline/04_build_embeddings.py`
- 目的：文本切块并计算向量（默认：GPU 上的 `BAAI/bge-m3`）。
- 典型输出：
  - `data/interim/embeddings/<YEAR>/<bank_id>/embeddings.npy`
  - `data/interim/embeddings/<YEAR>/<bank_id>/chunks.jsonl`（chunk 元数据 + 文本指针）

#### 05) 构建 FAISS 索引

- 脚本：`scripts/pipeline/05_build_faiss_index.py`
- 目的：合并向量并构建 FAISS 索引及元数据。
- 典型输出：
  - `data/interim/index/faiss_<YEAR>_full/faiss.index`
  - `data/interim/index/faiss_<YEAR>_full/meta.jsonl`
  - `data/interim/index/faiss_<YEAR>_full/merge_log.csv`

#### 06) 抽取指标（hybrid: table-first -> regex fallback -> LLM judge）

- 脚本：`scripts/pipeline/06_extract_metrics_patched_v2_final.py`
- 目的：按指标执行 multi-query 检索 + 重排 + 邻接扩展，再抽取并写出带证据 ID 的**批量 CSV**。
- 典型输出：
  - `data/outputs/processed/metrics_<YEAR>.csv`
  - `data/outputs/debug/`（按指标上下文、模型原始输出、修复轨迹）
  - `data/outputs/logs/`（运行日志）

### 5.3 索引路径（重要）

- 默认**全量**索引路径（典型）：
  - `data/interim/index/faiss_2024_full/`
- Sample 索引路径：
  - `data/sample/index/faiss_2024_sample/`

---

## 6. LLM 集成（门控）

LLM **不是**默认抽取器，而是在必要场景下按条件触发，以控制成本和结果波动。

典型用途：

- 处理冲突候选（同一指标出现多个候选值），
- 消解定义歧义（GAAP vs non-GAAP），
- 约束输出 schema（JSON 修复 / schema 约束抽取），
- 在确定性方法失效时补齐困难叙述文本中的数值。

若启用 LLM fallback：

- 确保本地推理端点（如 Ollama）已启动，
- 将 temperature 设为较低值以保持 schema 稳定，
- 尽量使用短且证据边界清晰的 prompt（上下文来自检索打包 chunk）。

---

### 6.1 v1.0 更新：Stage 06 的证据约束 LLM 复核

这次升级不是“简单再加一次 LLM 调用”，而是把 Stage 06 改造成双层、可审计的决策流水线。

#### 两次 LLM 调用，两类职责

1. **候选选择裁决器（定向触发，仅 ROA/ROE 且仍缺失时）**
   - 触发条件：确定性步骤后，ROA/ROE 仍为 `NOT FOUND`。
   - 流程：先从原始检索命中中挖掘候选（`mine_ratio_candidates_from_hits`），再由 `judge_select_candidate` 选择索引。
   - 约束：只能在给定候选中选择，或返回 `-1`（禁止数值编造）。

2. **抽取后统一复核器（5 个指标联合）**
   - 流程：`review_all_metrics_after_extract` 输出 `keep` / `replace` / `reject`。
   - 约束：`replace` 必须使用该指标允许的证据引用，非法引用会被丢弃。
   - 默认策略：仅复核低置信度行（`REVIEW_ONLY_LOW_CONFIDENCE=1`），兼顾成本和稳定性。

#### 为什么提升的是可靠性，而不仅是覆盖率

- **证据可信度更高**：替换结果受证据边界与引用校验约束，降低自由生成幻觉风险。
- **可追溯性更完整**：每次复核覆盖都会保留新旧字段（`orig_val`、`orig_unit`、`orig_source_chunk_id`）。
- **可调试性更强**：置信度与复核元数据（`confidence_*`、`review_action`、`review_note`、`review_model`）可以解释决策原因。
- **风险控制更稳**：低置信度行主动复核，中高置信度行默认保持稳定。

#### 同数据集 A/B 对比

- Baseline：`data/outputs/metrics_2024.csv`
- v1.0：`data/outputs/metrics_2024_llm_full_version1.0.csv`
- 范围：25 家银行 x 5 个指标 = 125 行（两文件键一致）

关键结果：

- **覆盖率**：`NOT FOUND` 从 **32 -> 29**（净增 +3）。
- **命中状态迁移**：6 行改善（`NOT FOUND -> FOUND`），3 行退化（`FOUND -> NOT FOUND`）。
- **置信度门控复核**：33 行标记低置信度（`needs_review=1`），92 行被门控跳过。
- **复核动作分布**：`reject=27`、`replace=1`、`keep=5`、`skip_by_confidence=92`。
- **证据可追踪性提升**：
  - 有有效引用的命中行：**93 -> 96**
  - 两版本均命中的 90 行中，23 行发生引用变更
  - 21 行数值变化中，18 行同步变更引用（数值更新与证据绑定）
- **审计完整性**：28 行 `replace/reject` 全部保留 `orig_val` 与 `orig_source_chunk_id`（100%）。
- **分桶迁移**：
  - old：`ok=86`、`value_missing=32`、`table_prefill=7`
  - new：`ok=89`、`llm_review_rejected=27`、`value_missing=2`、`table_prefill=6`、`llm_review_replaced=1`

结论：最大的收益，是把“黑盒式抽取结果”升级为“证据约束、可复核、可回滚”的流程。

## 7. 评估与调试

### 7.1 失败分桶（典型）

- `value_missing`：检索 + 抽取后仍找不到值
- `unit_missing`：值已找到但单位缺失（常见于表头 / 邻接 chunk）
- `semantic_ambiguous`：多候选冲突 / 定义不一致
- `no_candidates`：检索未返回可用证据

### 7.2 调试手册（推荐）

1. **先确认是否有证据**
   检查目标指标值是否出现在检索上下文中。
2. **再看召回是否足够**
   如果证据缺失，优先优化检索（multi-query、rerank、阈值）。
3. **检查解析是否正确**
   如果证据存在但未抽取，调整表格解析 / 正则规则 / 单位推断。
4. **处理冲突候选**
   如果候选冲突明显，启用 LLM judge 路径并保持严格 schema 约束。

### 7.3 关键排查位置

- `data/outputs/processed/` - 最终 CSV
- `data/outputs/logs/` - 运行日志
- `data/outputs/debug/` - 调试产物（检索上下文、中间 dump）

---

## 8. 仓库策略（产物与 Git）

本仓库遵循**轻量 + 可复现 Demo**策略：

- 建议提交：代码、配置，以及 `data/sample/...` 下的小型样例产物
- 不建议提交：OCR 全量输出、全量向量、全量 FAISS 索引及其他大体量中间产物

---

## 9. 路线图（后续升级）

- 检索排序：从启发式关键词加分升级为可训练 reranker（结合 hard-negative mining）。
- 自适应检索策略：按指标和银行报告风格动态调整 `topk` / 分数阈值。
- 抽取层：强化单位推断（表头 / 邻接扫描；统一单位归一化）。
- 学习式抽取：用弱监督训练结构化抽取器，覆盖困难分桶。
- 派生指标：对需要计算的场景补充 ROA/ROE 推导（平均资产 / 权益）。

---

## 10. 免责声明

PDF 与报告版权归其发布机构所有。
本仓库仅包含用于演示与可复现的代码及小型衍生样例产物。
