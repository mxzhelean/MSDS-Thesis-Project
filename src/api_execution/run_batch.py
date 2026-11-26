"""
Stateless, Crash-Safe Multi-Model Batch Execution Engine with Unified Logging.

Enforces:
1. Stateless single-turn calls (no history carryover, no system_instruction, no tools/grounding).
2. Strict client-side HTTP timeouts + exponential backoff retries on transient/rate-limit errors.
3. Incremental per-request JSONL checkpointing with automatic resume via `sample_id`.

Usage (from project root):
    # Smoke test (N=2 per task/model):
    python -m src.api_execution.run_batch --n 2 --batch-name smoke_n2

    # Pilot full run (N=15 per task/model):
    python -m src.api_execution.run_batch --n 15 --batch-name pilot_n15

    # Run a single model only:
    python -m src.api_execution.run_batch --n 15 --batch-name pilot_n15 --models gpt6_luna
"""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List
from venv import logger

import yaml
from dotenv import load_dotenv

# Provider SDKs
from anthropic import Anthropic
from google import genai
from google.genai import types as genai_types
from openai import OpenAI

from src.schema import (
    AuditRecord,
    append_record_jsonl,
    get_completed_sample_ids,
)
from src.utils.logger import setup_logger


def load_yaml(path: str | Path) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def call_openai_stateless(
    client: OpenAI,
    model_id: str,
    prompt_text: str,
    native_params: Dict[str, Any],
) -> str:
    """Single-turn stateless call to OpenAI Responses/Chat Completions API."""
    kwargs: Dict[str, Any] = {
        "model": model_id,
        "messages": [{"role": "user", "content": prompt_text}],
    }
    if native_params.get("temperature") is not None:
        kwargs["temperature"] = native_params["temperature"]
    if native_params.get("top_p") is not None:
        kwargs["top_p"] = native_params["top_p"]
    if native_params.get("max_tokens") is not None:
        kwargs["max_tokens"] = native_params["max_tokens"]

    resp = client.chat.completions.create(**kwargs)
    return (resp.choices[0].message.content or "").strip()


def call_anthropic_stateless(
    client: Anthropic,
    model_id: str,
    prompt_text: str,
    native_params: Dict[str, Any],
) -> str:
    """Single-turn stateless call to Anthropic Messages API (no system prompt, no tools)."""
    kwargs: Dict[str, Any] = {
        "model": model_id,
        "max_tokens": native_params.get("max_tokens") or 4096,
        "messages": [{"role": "user", "content": prompt_text}],
    }
    if native_params.get("temperature") is not None:
        kwargs["temperature"] = native_params["temperature"]
    if native_params.get("top_p") is not None:
        kwargs["top_p"] = native_params["top_p"]

    resp = client.messages.create(**kwargs)
    text_blocks = [block.text for block in resp.content if getattr(block, "type", "") == "text"]
    return "\n".join(text_blocks).strip()


def call_google_stateless(
    client: genai.Client,
    model_id: str,
    prompt_text: str,
    native_params: Dict[str, Any],
) -> str:
    """Single-turn stateless call to Google GenAI SDK (no system_instruction, no grounding/tools)."""
    config_kwargs: Dict[str, Any] = {
        "tools": [],
    }
    if native_params.get("temperature") is not None:
        config_kwargs["temperature"] = native_params["temperature"]
    if native_params.get("top_p") is not None:
        config_kwargs["top_p"] = native_params["top_p"]
    if native_params.get("max_output_tokens") is not None:
        config_kwargs["max_output_tokens"] = native_params["max_output_tokens"]

    resp = client.models.generate_content(
        model=model_id,
        contents=prompt_text,
        config=genai_types.GenerateContentConfig(**config_kwargs),
    )
    return (resp.text or "").strip()


def build_client(provider: str, api_key: str, timeout_sec: int) -> Any:
    """Instantiate provider SDK client with strict request-level timeout."""
    if provider == "openai":
        return OpenAI(api_key=api_key, timeout=float(timeout_sec), max_retries=0)
    if provider == "anthropic":
        return Anthropic(api_key=api_key, timeout=float(timeout_sec), max_retries=0)
    if provider == "google":
        # google-genai HttpOptions timeout is specified in milliseconds
        return genai.Client(
            api_key=api_key,
            http_options=genai_types.HttpOptions(timeout=timeout_sec * 1000),
        )
    raise ValueError(f"Unsupported provider: {provider}")


def execute_with_retry(
    provider: str,
    client: Any,
    model_id: str,
    prompt_text: str,
    native_params: Dict[str, Any],
    max_retries: int,
    backoff_base_sec: int,
) -> tuple[str, float]:
    """Execute a stateless API call with exponential backoff on transient failures."""
    last_err: Exception | None = None
    for attempt in range(1, max_retries + 1):
        t0 = time.perf_counter()
        try:
            if provider == "openai":
                text = call_openai_stateless(client, model_id, prompt_text, native_params)
            elif provider == "anthropic":
                text = call_anthropic_stateless(client, model_id, prompt_text, native_params)
            elif provider == "google":
                text = call_google_stateless(client, model_id, prompt_text, native_params)
            else:
                raise ValueError(f"Unsupported provider: {provider}")

            if not text:
                raise RuntimeError("Received empty response payload from API.")

            latency = round(time.perf_counter() - t0, 3)
            return text, latency

        except Exception as exc:  # noqa: BLE001
            last_err = exc
            if attempt < max_retries:
                sleep_sec = backoff_base_sec * (2 ** (attempt - 1))
                logger.warning(
                    "[%s] Attempt %d/%d failed (%s: %s). Retrying in %ds...",
                    sample_id,
                    attempt,
                    max_retries,
                    type(exc).__name__,
                    exc,
                    sleep_sec,
                )
                time.sleep(sleep_sec)

    raise RuntimeError(f"Exhausted {max_retries} retries. Last error: {last_err}") from last_err


def run_batch(
    n_iterations: int,
    batch_name: str,
    prompts_cfg_path: str = "configs/prompts_v1.0.yaml",
    models_cfg_path: str = "configs/models_baseline.yaml",
    target_models: List[str] | None = None,
    inter_call_delay_sec: float = 1.0,
) -> None:
    load_dotenv()  # Load environment variables from .env file (for API keys)
    # Setup unified logger for this batch run
    logger = setup_logger(
            name=f"ai_audit.{batch_name}",
            log_dir="logs",
            log_filename=f"{batch_name}_execution.log",
        )    

    prompts_cfg = load_yaml(prompts_cfg_path)
    models_cfg = load_yaml(models_cfg_path)

    protocol = models_cfg["baseline_protocol"]
    timeout_sec: int = protocol["timeout_sec"]
    max_retries: int = protocol["max_retries"]
    backoff_base_sec: int = protocol["backoff_base_sec"]

    raw_dir = Path("data/raw") / batch_name
    raw_dir.mkdir(parents=True, exist_ok=True)
    fail_log_path = Path("logs") / f"{batch_name}_failed_requests.jsonl"

    tasks: Dict[str, Any] = prompts_cfg["tasks"]
    models: List[Dict[str, Any]] = models_cfg["models"]

    if target_models:
        models = [m for m in models if m["model_short"] in target_models]
        if not models:
            raise ValueError(f"No matching models found for filter: {target_models}")

    logger.info(
            "Batch started | batch=%s | n=%d | models=%s | timeout=%ds | max_retries=%d",
            batch_name,
            n_iterations,
            [m["model_short"] for m in models],
            timeout_sec,
            max_retries,
    )

    stats = {"completed": 0, "skipped": 0, "failed": 0}

    for model_spec in models:
        model_short = model_spec["model_short"]
        provider = model_spec["provider"]
        model_id = model_spec["model_id"]
        api_key_env = model_spec["api_key_env"]
        native_params = model_spec.get("native_params", {})

        api_key = os.getenv(api_key_env)
        if not api_key:
            logger.error("Missing environment variable %s for %s. Skipping model.", api_key_env, model_short)
            continue

        client = build_client(provider=provider, api_key=api_key, timeout_sec=timeout_sec)
        output_jsonl = raw_dir / f"{model_short}.jsonl"
        completed_ids = get_completed_sample_ids(output_jsonl)

        api_params_snapshot = json.dumps(
            {
                "protocol": protocol,
                "native_params": native_params,
                "prompt_schema": prompts_cfg["schema_version"],
            },
            ensure_ascii=False,
        )

        logger.info("=== Model: %s (%s) -> %s ===", model_spec["display_name"], model_id, output_jsonl)

        for task_id, task_spec in tasks.items():
            task_short = task_spec["task_short"]
            prompt_text = task_spec["prompt_text"].strip()

            for iteration in range(1, n_iterations + 1):
                sample_id = f"{task_short}_{model_short}_{iteration:03d}"

                if sample_id in completed_ids:
                    stats["skipped"] += 1
                    logger.info("  [RESUME-SKIP] %s already completed.", sample_id)
                    continue

                logger.info("  [RUN] %s ...", sample_id)
                try:
                    raw_output, latency = execute_with_retry(
                        provider=provider,
                        client=client,
                        model_id=model_id,
                        prompt_text=prompt_text,
                        native_params=native_params,
                        max_retries=max_retries,
                        backoff_base_sec=backoff_base_sec,
                        logger=logger,
                        sample_id=sample_id
                    )

                    record = AuditRecord(
                        sample_id=sample_id,
                        task_id=task_id,
                        model_id=model_id,
                        iteration=iteration,
                        timestamp_utc=datetime.now(timezone.utc).isoformat(),
                        latency_sec=latency,
                        status="success",
                        prompt_text=prompt_text,
                        api_params=api_params_snapshot,
                        raw_output=raw_output,
                    )
                    append_record_jsonl(record, output_jsonl)
                    stats["completed"] += 1
                    logger.info(
                        "[%s] SUCCESS | latency=%.2fs | words=%d",
                        sample_id,
                        latency,
                        record.word_count,
                    )

                except Exception as exc:  # noqa: BLE001
                    stats["failed"] += 1
                    fail_record = AuditRecord(
                        sample_id=sample_id,
                        task_id=task_id,
                        model_id=model_id,
                        iteration=iteration,
                        timestamp_utc=datetime.now(timezone.utc).isoformat(),
                        latency_sec=None,
                        status="failed",
                        prompt_text=prompt_text,
                        api_params=api_params_snapshot,
                        raw_output="",
                        adjudication_notes=f"API_ERROR: {exc}",
                    )
                    append_record_jsonl(fail_record, fail_log_path)
                    logger.error(
                        "[%s] FAILED -> logged to %s",
                        sample_id,
                        fail_log_path,
                        exc_info=True,
                    )

                if inter_call_delay_sec > 0:
                    time.sleep(inter_call_delay_sec)
    logger.info(
        "Batch finished | batch=%s | completed=%d | skipped=%d | failed=%d",
        batch_name,
        stats["completed"],
        stats["skipped"],
        stats["failed"],
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run stateless multi-model batch API audit.")
    parser.add_argument("--n", type=int, default=15, help="Iterations per task per model (default: 15)")
    parser.add_argument("--batch-name", type=str, default="pilot_n15", help="Subdirectory under data/raw/")
    parser.add_argument("--prompts", type=str, default="configs/prompts_v1.0.yaml", help="Path to prompts YAML")
    parser.add_argument("--models-cfg", type=str, default="configs/models_baseline.yaml", help="Path to models YAML")
    parser.add_argument(
        "--models",
        nargs="*",
        default=None,
        help="Optional subset of model_short names (e.g., gpt6_luna claude_haiku45)",
    )
    parser.add_argument("--delay", type=float, default=1.0, help="Polite sleep seconds between requests")
    args = parser.parse_args()

    run_batch(
        n_iterations=args.n,
        batch_name=args.batch_name,
        prompts_cfg_path=args.prompts,
        models_cfg_path=args.models_cfg,
        target_models=args.models,
        inter_call_delay_sec=args.delay,
    )
    # python -m src.api_execution.run_batch --n 15 --batch-name pilot_n15