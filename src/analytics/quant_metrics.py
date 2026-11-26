"""
Automated Quantitative Metrics Pipeline for Pedagogical AI Audit.

Computes across (Model x Task):
  1. Length Compliance & Stability: Word count Mean, SD, CV (SD/Mean), and target-window hit rate.
  2. Lexical Diversity: Sample-level TTR and length-invariant MTLD (via `lexical-diversity`).
  3. Cross-Iteration Semantic Homogenization: Mean pairwise cosine similarity across all N*(N-1)/2
     iteration pairs using Sentence-Transformers (`all-MiniLM-L6-v2`).

Usage:
    python -m src.analytics.quant_metrics --batch-name pilot_n15
"""

from __future__ import annotations

import argparse
import re
from itertools import combinations
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from lexical_diversity import lex_div as ld
from sentence_transformers import SentenceTransformer

from src.schema import load_records_jsonl
from src.utils.logger import setup_logger

MODEL_DISPLAY_MAP: Dict[str, str] = {
    "gpt-6-luna": "GPT-6 Luna",
    "claude-haiku-4-5": "Claude Haiku 4.5",
    "gemini-3.5-flash-lite": "Gemini 3.5 Flash Lite",
}

TASK_DISPLAY_MAP: Dict[str, str] = {
    "Task1_KNN": "Task 1: KNN Explanation (Closed)",
    "Task2_AIEthics": "Task 2: AI Ethics Topics (Open)",
}


def tokenize_words(text: str) -> List[str]:
    """Extract lowercase alphabetic tokens for clean lexical diversity calculation."""
    return re.findall(r"[a-zA-Z]+", text.lower())


def compute_sample_lexical_metrics(text: str) -> tuple[float, float]:
    """Return (TTR, MTLD) for a single text response."""
    tokens = tokenize_words(text)
    if len(tokens) < 10:
        return 0.0, 0.0
    ttr = float(ld.ttr(tokens))
    mtld = float(ld.mtld(tokens))
    return round(ttr, 4), round(mtld, 2)


def compute_pairwise_cosine_stats(embeddings: np.ndarray) -> tuple[float, float, np.ndarray]:
    """
    Compute upper-triangle pairwise cosine similarity statistics for L2-normalized embeddings.
    Returns (mean_sim, sd_sim, all_pair_sims).
    """
    n = embeddings.shape[0]
    if n < 2:
        return np.nan, np.nan, np.array([])

    # Embeddings are L2-normalized, so dot product equals cosine similarity
    sim_matrix = np.matmul(embeddings, embeddings.T)
    triu_indices = np.triu_indices(n, k=1)
    pair_sims = sim_matrix[triu_indices]
    return float(np.mean(pair_sims)), float(np.std(pair_sims, ddof=1)), pair_sims


def run_quantitative_analysis(
    batch_name: str = "pilot_n15",
    embed_model_name: str = "all-MiniLM-L6-v2",
    target_min_words: int = 400,
    target_max_words: int = 450,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    logger = setup_logger(
        name=f"ai_audit.quant.{batch_name}",
        log_dir="logs",
        log_filename=f"{batch_name}_quant_metrics.log",
    )

    raw_dir = Path("data/raw") / batch_name
    out_dir = Path("data/processed") / batch_name
    out_dir.mkdir(parents=True, exist_ok=True)

    jsonl_files = sorted(raw_dir.glob("*.jsonl"))
    if not jsonl_files:
        raise FileNotFoundError(f"No JSONL files found in {raw_dir}")

    records = []
    for path in jsonl_files:
        records.extend([r.to_dict() for r in load_records_jsonl(path) if r.status == "success"])

    df = pd.DataFrame(records)
    logger.info("Loaded %d successful records from %s", len(df), raw_dir)

    # 1. Sample-level Word Count & Lexical Diversity (TTR, MTLD)
    lex_results = df["raw_output"].apply(compute_sample_lexical_metrics)
    df["ttr"] = [x[0] for x in lex_results]
    df["mtld"] = [x[1] for x in lex_results]
    df["in_target_window"] = df["word_count"].between(target_min_words, target_max_words)
    df["model_label"] = df["model_id"].map(MODEL_DISPLAY_MAP).fillna(df["model_id"])
    df["task_label"] = df["task_id"].map(TASK_DISPLAY_MAP).fillna(df["task_id"])

    # 2. Encode all responses with Sentence-Transformers (L2-normalized)
    logger.info("Loading embedding model '%s' and encoding %d outputs...", embed_model_name, len(df))
    st_model = SentenceTransformer(embed_model_name)
    embeddings = st_model.encode(
        df["raw_output"].tolist(),
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    # 3. Group-level Aggregation: (Model x Task)
    summary_rows = []
    pairwise_rows = []

    for (model_id, task_id), group in df.groupby(["model_id", "task_id"], sort=False):
        idx = group.index.to_numpy()
        group_emb = embeddings[idx]

        wc_mean = group["word_count"].mean()
        wc_sd = group["word_count"].std(ddof=1)
        wc_cv = (wc_sd / wc_mean) if wc_mean > 0 else np.nan

        mean_cos, sd_cos, pair_sims = compute_pairwise_cosine_stats(group_emb)

        # Save pairwise similarity distribution for boxplot/violin visualization
        sample_ids = group["sample_id"].tolist()
        for (i, j), sim_val in zip(combinations(range(len(sample_ids)), 2), pair_sims):
            pairwise_rows.append(
                {
                    "model_id": model_id,
                    "model_label": MODEL_DISPLAY_MAP.get(model_id, model_id),
                    "task_id": task_id,
                    "task_label": TASK_DISPLAY_MAP.get(task_id, task_id),
                    "sample_a": sample_ids[i],
                    "sample_b": sample_ids[j],
                    "cosine_similarity": round(float(sim_val), 5),
                }
            )

        summary_rows.append(
            {
                "model_id": model_id,
                "model_label": MODEL_DISPLAY_MAP.get(model_id, model_id),
                "task_id": task_id,
                "task_label": TASK_DISPLAY_MAP.get(task_id, task_id),
                "n_samples": len(group),
                "wc_mean": round(float(wc_mean), 1),
                "wc_sd": round(float(wc_sd), 1),
                "wc_cv": round(float(wc_cv), 4),
                "window_hit_rate": round(float(group["in_target_window"].mean()), 3),
                "ttr_mean": round(float(group["ttr"].mean()), 4),
                "ttr_sd": round(float(group["ttr"].std(ddof=1)), 4),
                "mtld_mean": round(float(group["mtld"].mean()), 2),
                "mtld_sd": round(float(group["mtld"].std(ddof=1)), 2),
                "pairwise_cos_mean": round(mean_cos, 4),
                "pairwise_cos_sd": round(sd_cos, 4),
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    pairwise_df = pd.DataFrame(pairwise_rows)

    # Save tabular outputs
    sample_out_csv = out_dir / f"{batch_name}_sample_metrics.csv"
    summary_out_csv = out_dir / f"{batch_name}_group_summary.csv"
    pairwise_out_csv = out_dir / f"{batch_name}_pairwise_cosine.csv"

    df.drop(columns=["raw_output", "prompt_text", "api_params"]).to_csv(sample_out_csv, index=False)
    summary_df.to_csv(summary_out_csv, index=False)
    pairwise_df.to_csv(pairwise_out_csv, index=False)

    logger.info("Saved sample metrics -> %s", sample_out_csv)
    logger.info("Saved group summary  -> %s", summary_out_csv)
    logger.info("Saved pairwise cos   -> %s", pairwise_out_csv)

    return df, summary_df, pairwise_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute quantitative audit metrics.")
    parser.add_argument("--batch-name", type=str, default="pilot_n15", help="Batch folder name")
    parser.add_argument("--embed-model", type=str, default="all-MiniLM-L6-v2", help="SentenceTransformer model")
    args = parser.parse_args()

    _, summary, _ = run_quantitative_analysis(
        batch_name=args.batch_name,
        embed_model_name=args.embed_model,
    )
    print("\n=== Group Quantitative Summary ===")
    print(summary.to_string(index=False))