import json
import re

MAX_PROMPT_CHARS = 20000
MAX_CONTEXT_CHARS_PER_METRIC = 8000
METRICS = ["ROA", "ROE", "NIM", "NII", "Provision for Credit Losses"]


def _slim_text(text: str, max_chars: int) -> str:
    """Keep head+tail to preserve both headers and numeric rows."""
    if not text:
        return ""
    if len(text) <= max_chars:
        return text
    head = text[: max_chars // 2]
    tail = text[-(max_chars - len(head)) :]
    return head + "\n\n...[TRUNCATED]...\n\n" + tail


# Prompt enforces an output schema. Downstream normalization assumes this contract; parsing includes a fallback for non-compliant outputs.
def make_prompt(q: str, context: str):
    """
    Build the strict JSON-only extraction prompt for a single metric.
    The prompt enforces: extract only explicit values from context, no inference, and strict citation copying.
    """
    # Full strict prompt for multi-metric extraction
    return (
        "You are a financial information extraction engine.\n\n"
        "Task: Extract metrics ONLY if explicitly stated as numbers in the Context.\n"
        "Target fiscal year: 2024\n\n"
        "Hard rules:\n"
        "1) Do NOT infer, summarize, or generalize.\n"
        "2) Use ONLY the Context as the source of truth.\n"
        "3) Each found metric MUST include the source_chunk_id copied EXACTLY from a [k=...] header.\n"
        "4) Normalize synonyms: ROAA -> ROA, ROAE -> ROE.\n"
        "5) If not explicitly stated, keep NOT FOUND.\n\n"
        "You MUST output EXACTLY the following JSON object and nothing else:\n\n"
        "{\n"
        "  \"results\": [\n"
        "    {\"metric_name\":\"ROA\",\"value\":\"NOT FOUND\",\"unit\":\"NOT FOUND\",\"fiscal_year\":2024,\"source_chunk_id\":\"NOT FOUND\"},\n"
        "    {\"metric_name\":\"ROE\",\"value\":\"NOT FOUND\",\"unit\":\"NOT FOUND\",\"fiscal_year\":2024,\"source_chunk_id\":\"NOT FOUND\"},\n"
        "    {\"metric_name\":\"NIM\",\"value\":\"NOT FOUND\",\"unit\":\"NOT FOUND\",\"fiscal_year\":2024,\"source_chunk_id\":\"NOT FOUND\"},\n"
        "    {\"metric_name\":\"NII\",\"value\":\"NOT FOUND\",\"unit\":\"NOT FOUND\",\"fiscal_year\":2024,\"source_chunk_id\":\"NOT FOUND\"},\n"
        "    {\"metric_name\":\"Provision for Credit Losses\",\"value\":\"NOT FOUND\",\"unit\":\"NOT FOUND\",\"fiscal_year\":2024,\"source_chunk_id\":\"NOT FOUND\"}\n"
        "  ]\n"
        "}\n\n"
        "Question:\n"
        + q + "\n\n"
        "Context:\n"
        + context + "\n\n"
        "Answer (JSON only):"
    )


def make_prompt_loose(context: str):
    """
    Build a slightly more tolerant extraction prompt.
    Used when strict prompting is too brittle; still requires JSON-only output and prohibits inference.
    """
    return (
        "You are a financial metric extractor.\n"
        "Return ONLY a JSON object. No extra text.\n"
        "Language: English.\n\n"

        "Task:\n"
        "Extract Net Interest Income (NII) for fiscal year 2024 ONLY.\n\n"

        "Rules:\n"
        "1) ONLY extract if an explicit numeric NII value appears in the Context.\n"
        "2) Do NOT infer, do NOT calculate.\n"
        "3) Copy source_chunk_id EXACTLY from the nearest header like:\n"
        "   [k=...|stem=...|chunk=...]\n"
        "   If the header includes brackets [], keep them.\n"
        "4) If you cannot find NII explicitly, set value/unit/source_chunk_id to \"NOT FOUND\".\n\n"

        "IMPORTANT OUTPUT REQUIREMENTS:\n"
        "- You MUST output EXACTLY this JSON schema with key \"results\".\n"
        "- The JSON must include ALL keys shown below.\n"
        "- Do NOT add any other keys.\n\n"

        "Output JSON (EXACT):\n"
        "{\n"
        "  \"results\": [\n"
        "    {\n"
        "      \"metric_name\": \"NII\",\n"
        "      \"value\": \"NOT FOUND\",\n"
        "      \"unit\": \"NOT FOUND\",\n"
        "      \"fiscal_year\": 2024,\n"
        "      \"source_chunk_id\": \"NOT FOUND\"\n"
        "    }\n"
        "  ]\n"
        "}\n\n"

        "Context:\n"
        + context
    )


def make_repair_prompt(bad_json_text: str, year: int = 2024) -> str:
    """
    Build a prompt that repairs model output into the required results schema.
    This is used as a second pass when the first LLM response is not valid JSON or not in the expected schema.
    """
    return (
        "You MUST output ONLY valid JSON. No markdown. No extra text.\n"
        "Convert the previous answer into EXACTLY this schema.\n\n"
        "IMPORTANT:\n"
        "- Do NOT invent placeholders like 'header_string'.\n"
        "- If the previous answer contains a source_chunk_id, COPY IT EXACTLY.\n"
        "- If missing, set source_chunk_id to \"NOT FOUND\".\n"
        "- Keep fiscal_year as the given year.\n\n"
        "OUTPUT JSON (EXACT):\n"
        "{\n"
        "  \"results\": [\n"
        f"    {{\"metric_name\":\"ROA\",\"value\":\"NOT FOUND\",\"unit\":\"NOT FOUND\",\"fiscal_year\":{year},\"source_chunk_id\":\"NOT FOUND\"}},\n"
        f"    {{\"metric_name\":\"ROE\",\"value\":\"NOT FOUND\",\"unit\":\"NOT FOUND\",\"fiscal_year\":{year},\"source_chunk_id\":\"NOT FOUND\"}},\n"
        f"    {{\"metric_name\":\"NIM\",\"value\":\"NOT FOUND\",\"unit\":\"NOT FOUND\",\"fiscal_year\":{year},\"source_chunk_id\":\"NOT FOUND\"}},\n"
        f"    {{\"metric_name\":\"NII\",\"value\":\"NOT FOUND\",\"unit\":\"NOT FOUND\",\"fiscal_year\":{year},\"source_chunk_id\":\"NOT FOUND\"}},\n"
        f"    {{\"metric_name\":\"Provision for Credit Losses\",\"value\":\"NOT FOUND\",\"unit\":\"NOT FOUND\",\"fiscal_year\":{year},\"source_chunk_id\":\"NOT FOUND\"}}\n"
        "  ]\n"
        "}\n\n"
        "Previous answer:\n"
        + (bad_json_text or "")
    )


def make_repair_prompt_one_metric(metric: str, bad_json_text: str, year: int = 2024) -> str:
    """
    Build a strict JSON-only "repair" prompt for a single metric.

    Purpose:
    - Used when the model output is not valid JSON or does not follow the required schema.
    - This function ONLY repairs formatting/schema; it should not introduce new facts.

    Output schema:
    {"results":[{"metric_name","value","unit","fiscal_year","source_chunk_id"}]}

    Notes:
    - If the previous answer contains a value or source_chunk_id, we instruct the model to copy them verbatim.
    - Missing fields must be "NOT FOUND".
    """

    return (
        "You MUST output ONLY valid JSON. No markdown. No extra text.\n"
        "Convert the previous answer into EXACTLY this schema.\n\n"
        "IMPORTANT:\n"
        "- If the previous answer contains a numeric value for the metric, copy it.\n"
        "- If the previous answer contains a source_chunk_id, COPY IT EXACTLY.\n"
        "- If missing, set fields to \"NOT FOUND\".\n\n"
        "OUTPUT JSON (EXACT):\n"
        "{\n"
        "  \"results\": [\n"
        f"    {{\"metric_name\":\"{metric}\",\"value\":\"NOT FOUND\",\"unit\":\"NOT FOUND\",\"fiscal_year\":{year},\"source_chunk_id\":\"NOT FOUND\"}}\n"
        "  ]\n"
        "}\n\n"
        "Previous answer:\n"
        + (bad_json_text or "")
    )


def make_prompt_all_metrics_from_context(context: str, year: int = 2024) -> str:
    """
    Build a prompt to extract multiple metrics in one call.
    Prefer per-metric extraction for stability; this function exists for experiments and backward compatibility.
    """
    context = _slim_text(context or "", MAX_PROMPT_CHARS)
    return (
        "You are a financial metric extractor.\n"
        "Return ONLY a JSON object. No markdown. No extra text.\n\n"
        f"Task: Extract the following metrics for fiscal year {year} ONLY:\n"
        "- ROA\n- ROE\n- NIM\n- NII\n- Provision for Credit Losses\n\n"
        "Rules:\n"
        "1) ONLY extract if an explicit numeric value appears in the Context.\n"
        "2) Do NOT infer or calculate.\n"
        "3) source_chunk_id MUST be copied EXACTLY from the nearest header like: [k=...|stem=...|chunk=...]\n"
        "4) If not found, keep value/unit/source_chunk_id as \"NOT FOUND\".\n"
        "5) For NIM: unit is usually \"%\". If the value appears as 3.63 or 3.63% in the context, set unit to \"%\".\n"
        "6) If value contains a trailing \"%\", remove it from value and keep unit=\"%\".\n\n"
        "Output JSON (EXACT schema):\n"
        "{\n"
        "  \"results\": [\n"
        f"    {{\"metric_name\":\"ROA\",\"value\":\"NOT FOUND\",\"unit\":\"NOT FOUND\",\"fiscal_year\":{year},\"source_chunk_id\":\"NOT FOUND\"}},\n"
        f"    {{\"metric_name\":\"ROE\",\"value\":\"NOT FOUND\",\"unit\":\"NOT FOUND\",\"fiscal_year\":{year},\"source_chunk_id\":\"NOT FOUND\"}},\n"
        f"    {{\"metric_name\":\"NIM\",\"value\":\"NOT FOUND\",\"unit\":\"NOT FOUND\",\"fiscal_year\":{year},\"source_chunk_id\":\"NOT FOUND\"}},\n"
        f"    {{\"metric_name\":\"NII\",\"value\":\"NOT FOUND\",\"unit\":\"NOT FOUND\",\"fiscal_year\":{year},\"source_chunk_id\":\"NOT FOUND\"}},\n"
        f"    {{\"metric_name\":\"Provision for Credit Losses\",\"value\":\"NOT FOUND\",\"unit\":\"NOT FOUND\",\"fiscal_year\":{year},\"source_chunk_id\":\"NOT FOUND\"}}\n"
        "  ]\n"
        "}\n\n"
        "Context:\n"
        + context
    )


def make_prompt_multi_metrics(metrics, ctx_by_metric: dict, year: int) -> str:
    """Build a strict JSON-only prompt for extracting multiple metrics in one call.

    IMPORTANT: This can easily become too long and trigger non-schema outputs on small models.
    If you use this function, keep ctx_by_metric values short (or rely on the internal truncation).
    """
    blocks = []
    for m in metrics:
        c = _slim_text((ctx_by_metric.get(m) or "").strip(), MAX_CONTEXT_CHARS_PER_METRIC)
        blocks.append(f"### METRIC={m}\n{c}\n")
    evidence = "\n".join(blocks)

    schema = {
        "results": [
            {"metric_name": "ROA", "value": "NOT FOUND", "unit": "NOT FOUND", "fiscal_year": year, "source_chunk_id": "NOT FOUND"},
            {"metric_name": "ROE", "value": "NOT FOUND", "unit": "NOT FOUND", "fiscal_year": year, "source_chunk_id": "NOT FOUND"},
            {"metric_name": "NIM", "value": "NOT FOUND", "unit": "NOT FOUND", "fiscal_year": year, "source_chunk_id": "NOT FOUND"},
            {"metric_name": "NII", "value": "NOT FOUND", "unit": "NOT FOUND", "fiscal_year": year, "source_chunk_id": "NOT FOUND"},
            {"metric_name": "Provision for Credit Losses", "value": "NOT FOUND", "unit": "NOT FOUND", "fiscal_year": year, "source_chunk_id": "NOT FOUND"},
        ]
    }

    prompt = (
        "You are a financial metric extraction engine.\n"
        "Return ONLY a JSON object. No markdown. No extra text.\n\n"
        f"Task: Extract the following metrics for fiscal year {year} ONLY: {', '.join(metrics)}\n\n"
        "Hard rules:\n"
        "1) ONLY extract if an explicit numeric value appears in the Evidence.\n"
        "2) Do NOT infer, summarize, or calculate.\n"
        "3) metric_name must be exactly one of: ROA, ROE, NIM, NII, Provision for Credit Losses.\n"
        "4) source_chunk_id MUST be copied EXACTLY from evidence headers like [k=...|stem=...|chunk=...].\n"
        "5) If not found, keep value/unit/source_chunk_id as \"NOT FOUND\".\n\n"
        "Output JSON (EXACT schema):\n"
        + json.dumps(schema, ensure_ascii=False)
        + "\n\nEVIDENCE:\n"
        + _slim_text(evidence, MAX_PROMPT_CHARS)
    )
    return prompt


def make_prompt_one_metric(metric: str, context: str, year: int = 2024) -> str:
    """
    Build a prompt to extract exactly one metric and return a single-item results list.
    This aligns the model output with downstream CSV writing (results schema).
    """
    # hard cap evidence; long evidence makes small local models drift to summaries
    context = _slim_text((context or "").strip(), MAX_CONTEXT_CHARS_PER_METRIC)
    # metric: "NIM"/"ROA"/"ROE"/"Provision for Credit Losses"
    return (
        "You are a financial metric extraction engine.\n"
        "Return ONLY a JSON object. No markdown. No extra text.\n\n"
        f"Task: Extract {metric} for fiscal year {year} ONLY.\n\n"
        "Hard rules:\n"
        "1) ONLY extract if an explicit numeric value appears in the Context.\n"
        "2) Do NOT infer, summarize, or calculate.\n"
        "3) source_chunk_id MUST be copied EXACTLY from the nearest header like: [k=...|stem=...|chunk=...]\n"
        "4) If not found, keep value/unit/source_chunk_id as \"NOT FOUND\".\n\n"
        "Output JSON (EXACT schema):\n"
        "{\n"
        "  \"results\": [\n"
        f"    {{\"metric_name\":\"{metric}\",\"value\":\"NOT FOUND\",\"unit\":\"NOT FOUND\",\"fiscal_year\":{year},\"source_chunk_id\":\"NOT FOUND\"}}\n"
        "  ]\n"
        "}\n\n"
        "Context:\n"
        + context
    )


def make_template(year: int):
    """
    Create a default results template for a given fiscal year.
    All metrics are initialized to NOT FOUND to support merge-in of partial extractions.
    """
    return {
        "results": [
            {"metric_name": m, "value": "NOT FOUND", "unit": "NOT FOUND", "fiscal_year": int(year), "source_chunk_id": "NOT FOUND"}
            for m in METRICS
        ]
    }
