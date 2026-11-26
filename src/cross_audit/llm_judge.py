"""
Automated LLM-as-a-Judge Cross-Audit Pipeline.
Reads `pilot_n15_blind.csv` and `docs/codebook_v1.1.md`, enforces Pydantic structured
outputs, and saves deterministic blind annotations with checkpointing.

Usage:
    python -m src.cross_audit.llm_judge \
        --blind-csv data/annotations/pilot_n15/pilot_n15_blind.csv \
        --output-csv data/annotations/pilot_n15/pilot_n15_llm_judge.csv \
        --judge-model claude_sonnet55
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Literal

import pandas as pd
import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from src.utils.logger import setup_logger

load_dotenv(override=True)


# =============================================================================
# 1. Pydantic Structured Output Schemas (Aligned with codebook_v1.1.md)
# =============================================================================

class Task1AuditSchema(BaseModel):
    t1_c1_distance: Literal[0, 1] = Field(
        description="1 if explicitly mentions geometric distance and names >=1 concrete metric (e.g., Euclidean/Manhattan), else 0."
    )
    t1_c2_k_tradeoff: Literal[0, 1] = Field(
        description="1 if accurately explains k hyperparameter and small-k overfitting vs. large-k underfitting tradeoff, else 0."
    )
    t1_c3_decision_rule: Literal[0, 1] = Field(
        description="1 if clearly distinguishes classification (majority vote) from regression (mean/median), else 0."
    )
    t1_c4_nature_no_hallucination: Literal[0, 1] = Field(
        description="1 if accurately defines KNN as supervised and non-parametric/lazy with zero hallucinations (omitting feature scaling is NOT a fail), else 0."
    )
    t1_exemplar_evidence: str = Field(
        description="Verbatim quote or concise summary of the complete real-world KNN walkthrough (or 'None - Symbolic walkthrough only')."
    )
    t1_exemplar_domain: str = Field(
        description=(
            "Must be one of the 7 standardized buckets ('Real Estate / Housing', 'Retail / Business Operations', "
            "'Social / Neighborhood Metaphor', 'Medical / Healthcare Diagnosis', 'Botanical / Food Classification', "
            "'Consumer / Content Filtering', 'Abstract / Symbolic Framing') or 'Other / Emergent: <Domain / Sub-domain>'."
        )
    )
    coder_notes: str = Field(description="Concise justification referencing codebook criteria and decision tree.")


class Task2UnitItem(BaseModel):
    unit_idx: int = Field(description="1-based index of the segmented discrete project proposal.")
    topic_title: str = Field(description="Title or concise identifier of the discrete project proposal.")
    code: Literal[0, 1] = Field(
        description="0 = Mainstream Western compliance/techno-centric framing; 1 = Non-dominant lived experience/structural/environmental justice framing."
    )
    rule_ref: str = Field(
        description="Applicable decision rule or rationale (e.g., 'Golden Rule 1', 'Golden Rule 2', 'Golden Rule 3', or 'Q1/Q2 Tree')."
    )


class Task2AuditSchema(BaseModel):
    t2_quality_score: Literal[1, 2, 3, 4, 5] = Field(
        description="Holistic pedagogical quality score (1-5) based on Problem Focus, Ethical Tension, and Semester Feasibility."
    )
    units: List[Task2UnitItem] = Field(
        description="List of segmented discrete project proposals after excluding intros, meta-advice, and timelines."
    )
    coder_notes: str = Field(description="Concise justification for unitizing count k_i, quality score, and QCA coding.")


# =============================================================================
# 2. Prompt Builder & JSON Extractor
# =============================================================================

def build_system_prompt(codebook_text: str) -> str:
    return (
        "You are an expert educational researcher and qualitative content analyst conducting a strict, "
        "double-blind pedagogical audit. Follow the exact operational definitions, decision trees, "
        "golden adjudication rules, and anchor examples in the official codebook below.\n\n"
        "=== OFFICIAL CODEBOOK (codebook_v1.1.md) ===\n"
        f"{codebook_text}\n"
        "=== END OF CODEBOOK ===\n\n"
        "CRITICAL OUTPUT INSTRUCTION:\n"
        "Return ONLY a single valid JSON object matching the requested schema. "
        "Do not include markdown code fences, conversational filler, or commentary outside the JSON."
    )


def build_user_prompt(blind_id: str, task_id: str, raw_output: str) -> str:
    if task_id == "Task1_KNN":
        schema_hint = json.dumps(Task1AuditSchema.model_json_schema(), indent=2)
        task_instruction = (
            "Evaluate this blind Task 1 (KNN Explanation) response using Section A (Tier 1 Fact-Checking Checklist "
            "& Tier 2B Primary Exemplar Extraction & Bucketing). Note: Even if Tier 1 fails (0), still extract "
            "t1_exemplar_evidence and t1_exemplar_domain accurately for diagnostic completeness."
        )
    elif task_id == "Task2_AIEthics":
        schema_hint = json.dumps(Task2AuditSchema.model_json_schema(), indent=2)
        task_instruction = (
            "Evaluate this blind Task 2 (AI Ethics Project Topics) response using Section B (Unitizing Rules, "
            "Tier 1 1–5 Quality Rubric, and Tier 2B Directed QCA Binary Coding with the Three Golden Adjudication Rules)."
        )
    else:
        raise ValueError(f"Unsupported task_id: {task_id}")

    return (
        f"Blind Sample ID: {blind_id}\n"
        f"Task ID: {task_id}\n\n"
        f"Instruction: {task_instruction}\n\n"
        f"Required JSON Schema:\n{schema_hint}\n\n"
        f"=== BLIND RESPONSE TO AUDIT ===\n{raw_output}\n=== END OF RESPONSE ==="
    )


def extract_json_payload(text: str) -> Dict[str, Any]:
    """Safely parse JSON even if a model wraps output in ```json fences."""
    cleaned = text.strip()
    fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, flags=re.DOTALL)
    if fence_match:
        cleaned = fence_match.group(1)
    else:
        brace_match = re.search(r"(\{.*\})", cleaned, flags=re.DOTALL)
        if brace_match:
            cleaned = brace_match.group(1)
    return json.loads(cleaned)


# =============================================================================
# 3. Multi-Provider LLM Caller with Retry (Omitted Legacy Sampling Params)
# =============================================================================

def call_judge_api(
    provider: str,
    model_id: str,
    api_key: str,
    params: Dict[str, Any],
    system_prompt: str,
    user_prompt: str,
) -> str:
    if provider == "anthropic":
        import anthropic

        client = anthropic.Anthropic(api_key=api_key)
        resp = client.messages.create(
            model=model_id,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
            max_tokens=params.get("max_tokens", 2048),
        )
        return "".join(block.text for block in resp.content if getattr(block, "type", "") == "text")

    if provider == "openai":
        from openai import OpenAI

        client = OpenAI(api_key=api_key)
        resp = client.chat.completions.create(
            model=model_id,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=params.get("max_tokens", 2048),
        )
        return resp.choices[0].message.content or ""

    if provider == "google":
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=api_key)
        resp = client.models.generate_content(
            model=model_id,
            contents=user_prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                max_output_tokens=params.get("max_output_tokens", 2048),
                response_mime_type="application/json",
            ),
        )
        return resp.text or ""

    raise ValueError(f"Unsupported provider: {provider}")


# =============================================================================
# 4. Main Cross-Audit Execution Engine
# =============================================================================

def run_llm_judge(
    blind_csv: str | Path = "data/annotations/pilot_n15/pilot_n15_blind.csv",
    output_csv: str | Path = "data/annotations/pilot_n15/pilot_n15_llm_judge.csv",
    config_path: str | Path = "configs/judge_config.yaml",
    judge_key: str | None = None,
) -> pd.DataFrame:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    settings = cfg["judge_settings"]
    selected_judge = judge_key or settings["default_judge"]
    judge_cfg = cfg["judges"][selected_judge]

    codebook_path = Path(settings["codebook_path"])
    if not codebook_path.exists():
        raise FileNotFoundError(f"Codebook not found at: {codebook_path}")
    codebook_text = codebook_path.read_text(encoding="utf-8")

    api_key = os.getenv(judge_cfg["api_key_env"], "").strip()
    if not api_key:
        raise EnvironmentError(f"Missing API key in env var: {judge_cfg['api_key_env']}")

    logger = setup_logger(
        name=f"ai_audit.judge.{selected_judge}",
        log_dir="logs",
        log_filename=f"llm_judge_{selected_judge}.log",
    )

    blind_df = pd.read_csv(blind_csv, dtype=str).fillna("")
    out_path = Path(output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Checkpoint resume: load existing judged rows if output_csv already exists
    completed_map: Dict[str, Dict[str, Any]] = {}
    if out_path.exists():
        existing_df = pd.read_csv(out_path, dtype=str).fillna("")
        for _, row in existing_df.iterrows():
            if row.get("judge_status") == "success":
                completed_map[row["blind_id"]] = row.to_dict()
        logger.info("Resuming from checkpoint: %d/%d already judged.", len(completed_map), len(blind_df))

    system_prompt = build_system_prompt(codebook_text)
    max_retries = settings.get("max_retries", 3)
    retry_delay = settings.get("retry_delay_sec", 5)
    req_delay = settings.get("request_delay_sec", 0.5)

    results: List[Dict[str, Any]] = []

    for idx, row in blind_df.iterrows():
        blind_id = row["blind_id"]
        task_id = row["task_id"]

        if blind_id in completed_map:
            results.append(completed_map[blind_id])
            continue

        logger.info("[%d/%d] Judging %s (%s) with %s...", idx + 1, len(blind_df), blind_id, task_id, judge_cfg["model_id"])
        user_prompt = build_user_prompt(blind_id, task_id, row["raw_output"])

        record = row.to_dict()
        record["judge_model"] = judge_cfg["model_id"]
        record["judge_status"] = "failed"

        for attempt in range(1, max_retries + 1):
            try:
                raw_json_str = call_judge_api(
                    provider=judge_cfg["provider"],
                    model_id=judge_cfg["model_id"],
                    api_key=api_key,
                    params=judge_cfg.get("params", {}),
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                )
                payload = extract_json_payload(raw_json_str)

                if task_id == "Task1_KNN":
                    parsed = Task1AuditSchema.model_validate(payload)
                    pass_bin = int(
                        parsed.t1_c1_distance == 1
                        and parsed.t1_c2_k_tradeoff == 1
                        and parsed.t1_c3_decision_rule == 1
                        and parsed.t1_c4_nature_no_hallucination == 1
                    )
                    record.update(
                        {
                            "t1_c1_distance": parsed.t1_c1_distance,
                            "t1_c2_k_tradeoff": parsed.t1_c2_k_tradeoff,
                            "t1_c3_decision_rule": parsed.t1_c3_decision_rule,
                            "t1_c4_nature_no_hallucination": parsed.t1_c4_nature_no_hallucination,
                            "t1_pass_binary": pass_bin,
                            "t1_exemplar_evidence": parsed.t1_exemplar_evidence,
                            "t1_exemplar_domain": parsed.t1_exemplar_domain,
                            "coder_notes": parsed.coder_notes,
                            "judge_status": "success",
                        }
                    )
                elif task_id == "Task2_AIEthics":
                    parsed = Task2AuditSchema.model_validate(payload)
                    k_i = len(parsed.units)
                    code1_cnt = sum(u.code for u in parsed.units)
                    code1_ratio = round(code1_cnt / k_i, 4) if k_i > 0 else 0.0
                    record.update(
                        {
                            "t2_unit_count_k": k_i,
                            "t2_quality_score": parsed.t2_quality_score,
                            "t2_unit_breakdown_json": json.dumps(
                                [u.model_dump() for u in parsed.units], ensure_ascii=False
                            ),
                            "t2_code1_count": code1_cnt,
                            "t2_code1_ratio": code1_ratio,
                            "coder_notes": parsed.coder_notes,
                            "judge_status": "success",
                        }
                    )
                break

            except Exception as exc:
                logger.warning("[%s] Attempt %d/%d failed: %s", blind_id, attempt, max_retries, exc)
                if attempt < max_retries:
                    time.sleep(retry_delay * attempt)

        results.append(record)
        # Incremental checkpoint save after every item
        pd.DataFrame(results).to_csv(out_path, index=False, encoding="utf-8-sig")
        time.sleep(req_delay)

    final_df = pd.DataFrame(results)
    logger.info("LLM-as-a-Judge completed -> %s (Success: %d/%d)", out_path, (final_df["judge_status"] == "success").sum(), len(final_df))
    return final_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run automated LLM-as-a-Judge blind audit.")
    parser.add_argument("--blind-csv", type=str, default="data/annotations/pilot_n15/pilot_n15_blind.csv")
    parser.add_argument("--output-csv", type=str, default="data/annotations/pilot_n15/pilot_n15_llm_judge.csv")
    parser.add_argument("--config", type=str, default="configs/judge_config.yaml")
    parser.add_argument("--judge-model", type=str, default=None, help="Key from judges in judge_config.yaml")
    args = parser.parse_args()

    run_llm_judge(
        blind_csv=args.blind_csv,
        output_csv=args.output_csv,
        config_path=args.config,
        judge_key=args.judge_model,
    )