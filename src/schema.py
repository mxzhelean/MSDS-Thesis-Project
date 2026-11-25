"""
Unified Data Schema & Serialization Utilities for Pedagogical AI Audit.

Defines the canonical data contract (`AuditRecord`) shared across:
- `src/api_execution`: Stateless batch collection & incremental JSONL checkpointing
- `src/preprocessing`: CSV export & blinded annotation sheet generation
- `src/cross_audit`: Deterministic (temperature=0) LLM-as-a-Judge auditing
- `src/analytics`: Quantitative & QCA statistical evaluation
"""

from __future__ import annotations

import csv
import json
import random
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional


TaskID = Literal["Task1_KNN", "Task2_AIEthics"]


@dataclass
class AuditRecord:
    """Canonical record schema aligning raw API payloads with Human & LLM coding columns."""

    # 1. Provenance & Execution Metadata
    sample_id: str  # Format: "{task_short}_{model_short}_{iter:03d}", e.g., "T1_gemini_001"
    task_id: str  # "Task1_KNN" or "Task2_AIEthics"
    model_id: str  # Exact model snapshot identifier
    iteration: int  # 1-based iteration index
    timestamp_utc: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    latency_sec: Optional[float] = None
    status: str = "success"  # "success" | "failed"

    # 2. Prompt, Baseline Config & Raw Output
    prompt_text: str = ""
    api_params: str = "{}"  # JSON-serialized baseline config (system_instruction=None, tools=[])
    raw_output: str = ""
    word_count: int = 0

    # 3. Human Primary Coding Columns (Populated during blind annotation)
    # Task 1 (KNN Explanation)
    t1_human_tier1_pass: Optional[int] = None  # Binary 0/1 checklist
    t1_human_tier2b_bucket: Optional[str] = None  # Primary exemplar domain bucket

    # Task 2 (AI Ethics Project Topics)
    t2_human_option_set_size: Optional[int] = None  # Unconstrained cardinality k_i
    t2_human_tier1_score: Optional[int] = None  # Holistic quality 1-5 scale
    t2_human_qca_codes: Optional[str] = None  # JSON list of 0/1 codes per project, e.g., "[0, 1, 1]"
    t2_human_code1_ratio: Optional[float] = None  # Proportion r_i = sum(codes) / k_i

    # 4. Deterministic LLM Cross-Audit Columns (Populated at temperature=0)
    judge_model_id: Optional[str] = None
    llm_judge_rationale: Optional[str] = None
    t1_llm_tier1_pass: Optional[int] = None
    t1_llm_tier2b_bucket: Optional[str] = None
    t2_llm_option_set_size: Optional[int] = None
    t2_llm_tier1_score: Optional[int] = None
    t2_llm_qca_codes: Optional[str] = None
    t2_llm_code1_ratio: Optional[float] = None

    # 5. Adjudication & Audit Notes
    adjudication_notes: Optional[str] = None

    def __post_init__(self) -> None:
        if self.raw_output and self.word_count == 0:
            self.word_count = len(self.raw_output.split())

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_jsonl_line(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AuditRecord":
        valid_keys = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in data.items() if k in valid_keys}
        return cls(**filtered)

    @classmethod
    def get_column_names(cls) -> List[str]:
        return [f.name for f in fields(cls)]


def append_record_jsonl(record: AuditRecord, jsonl_path: str | Path) -> None:
    """Append a single AuditRecord immediately to a JSONL file (crash-safe checkpointing)."""
    path = Path(jsonl_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(record.to_jsonl_line() + "\n")


def load_records_jsonl(jsonl_path: str | Path) -> List[AuditRecord]:
    """Load all AuditRecords from an existing JSONL file."""
    path = Path(jsonl_path)
    if not path.exists():
        return []
    records: List[AuditRecord] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(AuditRecord.from_dict(json.loads(line)))
    return records


def get_completed_sample_ids(jsonl_path: str | Path) -> set[str]:
    """Return sample_ids that already succeeded so batch runs can resume cleanly."""
    return {
        rec.sample_id
        for rec in load_records_jsonl(jsonl_path)
        if rec.status == "success" and rec.raw_output.strip()
    }


def export_schema_template_csv(csv_path: str | Path) -> None:
    """Write an empty CSV header template matching AuditRecord fields."""
    path = Path(csv_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=AuditRecord.get_column_names())
        writer.writeheader()


def export_records_to_csv(
    records: List[AuditRecord],
    csv_path: str | Path,
    blind: bool = False,
    seed: int = 42,
) -> None:
    """
    Export AuditRecords to CSV for annotation or analysis.

    If `blind=True`, shuffles row order deterministically and strips model identity
    columns (`model_id`, `sample_id`, `timestamp_utc`, `api_params`) while attaching
    a `blind_id` and writing a private mapping key (`*_blind_key.csv`) alongside it.
    """
    path = Path(csv_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    rows = [rec.to_dict() for rec in records]
    if not blind:
        with path.open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=AuditRecord.get_column_names())
            writer.writeheader()
            writer.writerows(rows)
        return

    rng = random.Random(seed)
    rng.shuffle(rows)

    hidden_cols = {"sample_id", "model_id", "iteration", "timestamp_utc", "latency_sec", "api_params"}
    visible_cols = ["blind_id"] + [c for c in AuditRecord.get_column_names() if c not in hidden_cols]

    blind_rows = []
    key_rows = []
    for idx, row in enumerate(rows, start=1):
        blind_id = f"BLIND_{idx:03d}"
        key_rows.append(
            {
                "blind_id": blind_id,
                "sample_id": row["sample_id"],
                "task_id": row["task_id"],
                "model_id": row["model_id"],
                "iteration": row["iteration"],
            }
        )
        filtered_row = {"blind_id": blind_id}
        for col in visible_cols:
            if col != "blind_id":
                filtered_row[col] = row[col]
        blind_rows.append(filtered_row)

    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=visible_cols)
        writer.writeheader()
        writer.writerows(blind_rows)

    key_path = path.with_name(f"{path.stem}_blind_key.csv")
    with key_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["blind_id", "sample_id", "task_id", "model_id", "iteration"]
        )
        writer.writeheader()
        writer.writerows(key_rows)


if __name__ == "__main__":
    # Generate default empty schema template when run directly
    export_schema_template_csv("data/annotations/schema_template.csv")
    print("Generated data/annotations/schema_template.csv successfully.")
