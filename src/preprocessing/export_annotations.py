"""
Lightweight JSONL-to-CSV Exporter for Blind & Master Annotation Sheets.

Usage (from project root):
    python -m src.preprocessing.export_annotations \
        --input-dir data/raw/pilot_n15 \
        --output-dir data/annotations/pilot_n15
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

from src.schema import (
    AuditRecord,
    export_records_to_csv,
    load_records_jsonl,
)


def jsonl_to_csv(
    input_paths: str | Path | List[str | Path],
    output_dir: str | Path,
    batch_prefix: str = "pilot_n15",
    only_success: bool = True,
    seed: int = 42,
) -> dict[str, Path]:
    """
    Convert one or more raw JSONL files into:
      1. Master CSV (all metadata & model IDs intact, for LLM cross-audit & final merge)
      2. Blinded CSV (shuffled rows, hidden model_id/sample_id, with BLIND_XXX IDs)
      3. Blind Key CSV (private mapping between BLIND_XXX and sample_id/model_id)
    """
    if isinstance(input_paths, (str, Path)):
        p = Path(input_paths)
        files = sorted(p.glob("*.jsonl")) if p.is_dir() else [p]
    else:
        files = [Path(p) for p in input_paths]

    records: List[AuditRecord] = []
    for file_path in files:
        records.extend(load_records_jsonl(file_path))

    if only_success:
        records = [
            r for r in records if r.status == "success" and r.raw_output.strip()
        ]

    if not records:
        raise ValueError(f"No valid records found in {files}.")

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    master_csv = out_dir / f"{batch_prefix}_master.csv"
    blind_csv = out_dir / f"{batch_prefix}_blind.csv"

    # 1. Export full master sheet (for LLM-as-a-Judge cross-audit & post-blind merging)
    export_records_to_csv(records, master_csv, blind=False)

    # 2. Export shuffled, anonymized sheet + *_blind_key.csv (for human blind coding)
    export_records_to_csv(records, blind_csv, blind=True, seed=seed)

    return {
        "master_csv": master_csv,
        "blind_csv": blind_csv,
        "blind_key_csv": blind_csv.with_name(f"{blind_csv.stem}_blind_key.csv"),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert raw JSONL batch to annotation CSVs.")
    parser.add_argument("--input-dir", type=str, default="data/raw/pilot_n15", help="Directory or file path of raw JSONL(s)")
    parser.add_argument("--output-dir", type=str, default="data/annotations/pilot_n15", help="Output directory for CSVs")
    parser.add_argument("--prefix", type=str, default="pilot_n15", help="Filename prefix for exported CSVs")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for deterministic blind shuffling")
    args = parser.parse_args()

    outputs = jsonl_to_csv(
        input_paths=args.input_dir,
        output_dir=args.output_dir,
        batch_prefix=args.prefix,
        seed=args.seed,
    )
    for label, path in outputs.items():
        print(f"[OK] {label}: {path}")