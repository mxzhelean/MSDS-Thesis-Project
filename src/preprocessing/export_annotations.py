"""
Export Raw JSONL Audit Records to Master & Double-Blind Annotation CSVs.
Aligned with `codebook_v1.1.md` structured annotation schema.

Usage:
    python -m src.preprocessing.export_annotations \
        --input-dir data/raw/pilot_n15 \
        --output-dir data/annotations/pilot_n15 \
        --prefix pilot_n15 \
        --seed 42
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Dict, List

import pandas as pd

from src.schema import load_records_jsonl
from src.utils.logger import setup_logger

# Structured annotation columns matching codebook_v1.1.md
ANNOTATION_COLUMNS: List[str] = [
    # Task 1: Tier 1 Fact-Checking Checklist (4 sub-criteria + binary pass)
    "t1_c1_distance",
    "t1_c2_k_tradeoff",
    "t1_c3_decision_rule",
    "t1_c4_nature_no_hallucination",
    "t1_pass_binary",
    # Task 1: Tier 2B Primary Exemplar Extraction & Standardized Domain Bucket
    "t1_exemplar_evidence",
    "t1_exemplar_domain",
    # Task 2: Unitizing & Tier 1 Holistic Quality (1-5)
    "t2_unit_count_k",
    "t2_quality_score",
    # Task 2: Tier 2B Unit-Level Directed QCA Breakdown & Aggregated Counts
    "t2_unit_breakdown_json",
    "t2_code1_count",
    "t2_code1_ratio",
    # General Audit Notes
    "coder_notes",
]


def jsonl_to_csv(
    input_paths: str | Path | List[str | Path],
    output_dir: str | Path = "data/annotations/pilot_n15",
    batch_prefix: str = "pilot_n15",
    seed: int = 42,
) -> Dict[str, str]:
    logger = setup_logger(
        name=f"ai_audit.export.{batch_prefix}",
        log_dir="logs",
        log_filename=f"{batch_prefix}_export.log",
    )

    # Resolve input JSONL files
    if isinstance(input_paths, (str, Path)):
        p = Path(input_paths)
        jsonl_files = sorted(p.glob("*.jsonl")) if p.is_dir() else [p]
    else:
        jsonl_files = [Path(p) for p in input_paths]

    if not jsonl_files:
        raise FileNotFoundError(f"No JSONL files found at: {input_paths}")

    records = []
    for fpath in jsonl_files:
        records.extend([r.to_dict() for r in load_records_jsonl(fpath) if r.status == "success"])

    if not records:
        raise ValueError("No successful audit records found to export.")

    df = pd.DataFrame(records)
    logger.info("Loaded %d successful records from %d files.", len(df), len(jsonl_files))

    # Deterministic shuffle for double-blind annotation
    indices = list(range(len(df)))
    rng = random.Random(seed)
    rng.shuffle(indices)

    df_shuffled = df.iloc[indices].reset_index(drop=True)
    df_shuffled.insert(0, "blind_id", [f"BLIND_{i:03d}" for i in range(1, len(df_shuffled) + 1)])

    # Initialize empty structured annotation columns
    for col in ANNOTATION_COLUMNS:
        df_shuffled[col] = ""

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    master_csv = out_path / f"{batch_prefix}_master.csv"
    blind_csv = out_path / f"{batch_prefix}_blind.csv"

    # 1. Master CSV: retains full provenance (model_id, sample_id, iteration, params)
    df_shuffled.to_csv(master_csv, index=False, encoding="utf-8-sig")

    # 2. Blind CSV: strips model_id, sample_id, iteration, latency, and API params
    blind_cols = ["blind_id", "task_id", "word_count", "raw_output"] + ANNOTATION_COLUMNS
    df_shuffled[blind_cols].to_csv(blind_csv, index=False, encoding="utf-8-sig")

    logger.info("Exported Master CSV -> %s (%d rows)", master_csv, len(df_shuffled))
    logger.info("Exported Blind CSV  -> %s (%d rows, %d cols)", blind_csv, len(df_shuffled), len(blind_cols))

    return {
        "master_csv": str(master_csv),
        "blind_csv": str(blind_csv),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export JSONL records to Master & Blind CSVs.")
    parser.add_argument("--input-dir", type=str, default="data/raw/pilot_n15", help="Directory containing raw JSONL files")
    parser.add_argument("--output-dir", type=str, default="data/annotations/pilot_n15", help="Output directory for CSVs")
    parser.add_argument("--prefix", type=str, default="pilot_n15", help="Filename prefix")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for blind shuffling")
    args = parser.parse_args()

    jsonl_to_csv(
        input_paths=args.input_dir,
        output_dir=args.output_dir,
        batch_prefix=args.prefix,
        seed=args.seed,
    )