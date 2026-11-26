"""
Unblind and Analyze LLM-as-a-Judge Pilot Results (N=90).
Merges `pilot_n15_llm_judge.csv` with `pilot_n15_master.csv`, computes Task 1 & Task 2
pedagogical metrics (including Shannon entropy H), and exports summary tables & figures.

Usage:
    python -m src.analysis.analyze_llm_judge
"""

from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def shannon_entropy(labels: list[str]) -> tuple[float, float, int]:
    """Return (Shannon Entropy H in bits, Normalized Evenness J, Unique Bucket Count K)."""
    counts = Counter(labels)
    total = sum(counts.values())
    k = len(counts)
    if total == 0:
        return 0.0, 0.0, 0
    probs = [c / total for c in counts.values()]
    h = -sum(p * math.log2(p) for p in probs if p > 0)
    j = h / math.log2(k) if k > 1 else 0.0
    return round(h, 4), round(j, 4), k


def run_judge_analysis(
    master_csv: str | Path = "data/annotations/pilot_n15/pilot_n15_master.csv",
    judge_csv: str | Path = "data/annotations/pilot_n15/pilot_n15_llm_judge.csv",
    out_dir: str | Path = "data/processed/pilot_n15",
) -> dict[str, pd.DataFrame]:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    master_df = pd.read_csv(master_csv)
    judge_df = pd.read_csv(judge_csv)

    # 1. Verify completeness
    assert len(judge_df) == 90, f"Expected 90 rows, found {len(judge_df)}"
    failed = judge_df[judge_df["judge_status"] != "success"]
    if not failed.empty:
        raise ValueError(f"Found {len(failed)} incomplete rows: {failed['blind_id'].tolist()}")

    # 2. Unblind merge on blind_id
    prov_cols = ["blind_id", "model_id", "sample_id", "iteration"]
    judge_cols = [c for c in judge_df.columns if c not in ["task_id", "word_count", "raw_output"]]
    df = pd.merge(master_df[prov_cols + ["task_id", "word_count", "raw_output"]], judge_df[judge_cols], on="blind_id")

    unblinded_csv = out_path / "llm_judge_unblinded.csv"
    df.to_csv(unblinded_csv, index=False, encoding="utf-8-sig")

    # =========================================================================
    # 3. Task 1 Analysis (KNN Explanation)
    # =========================================================================
    t1 = df[df["task_id"] == "Task1_KNN"].copy()
    num_t1_cols = ["t1_c1_distance", "t1_c2_k_tradeoff", "t1_c3_decision_rule", "t1_c4_nature_no_hallucination", "t1_pass_binary"]
    for c in num_t1_cols:
        t1[c] = pd.to_numeric(t1[c], errors="coerce")

    t1_rows = []
    for model_id, grp in t1.groupby("model_id", sort=True):
        # Compute entropy across all responses AND Tier-1 passing responses
        h_all, j_all, k_all = shannon_entropy(grp["t1_exemplar_domain"].tolist())
        passed_grp = grp[grp["t1_pass_binary"] == 1]
        h_pass, j_pass, k_pass = shannon_entropy(passed_grp["t1_exemplar_domain"].tolist())

        t1_rows.append({
            "model_id": model_id,
            "n": len(grp),
            "t1_pass_rate": round(grp["t1_pass_binary"].mean(), 4),
            "c1_distance_rate": round(grp["t1_c1_distance"].mean(), 4),
            "c2_k_tradeoff_rate": round(grp["t1_c2_k_tradeoff"].mean(), 4),
            "c3_decision_rule_rate": round(grp["t1_c3_decision_rule"].mean(), 4),
            "c4_no_hallucination_rate": round(grp["t1_c4_nature_no_hallucination"].mean(), 4),
            "domain_entropy_H_all": h_all,
            "domain_evenness_J_all": j_all,
            "unique_domains_all": k_all,
            "domain_entropy_H_pass": h_pass,
            "top_domain": grp["t1_exemplar_domain"].value_counts().index[0],
            "top_domain_share": round(grp["t1_exemplar_domain"].value_counts().iloc[0] / len(grp), 4),
        })
    t1_summary = pd.DataFrame(t1_rows)

    # Cross-tab of Domain Buckets by Model
    t1_domain_ct = pd.crosstab(t1["t1_exemplar_domain"], t1["model_id"], margins=True)

    # =========================================================================
    # 4. Task 2 Analysis (AI Ethics Project Topics)
    # =========================================================================
    t2 = df[df["task_id"] == "Task2_AIEthics"].copy()
    for c in ["t2_unit_count_k", "t2_quality_score", "t2_code1_count", "t2_code1_ratio"]:
        t2[c] = pd.to_numeric(t2[c], errors="coerce")

    t2_rows = []
    unit_records = []
    for model_id, grp in t2.groupby("model_id", sort=True):
        total_units = int(grp["t2_unit_count_k"].sum())
        total_code1 = int(grp["t2_code1_count"].sum())
        micro_code1_rate = round(total_code1 / total_units, 4) if total_units > 0 else 0.0
        macro_code1_rate = round(grp["t2_code1_ratio"].mean(), 4)

        t2_rows.append({
            "model_id": model_id,
            "n": len(grp),
            "k_mean": round(grp["t2_unit_count_k"].mean(), 2),
            "k_sd": round(grp["t2_unit_count_k"].std(), 2),
            "quality_mean": round(grp["t2_quality_score"].mean(), 2),
            "quality_sd": round(grp["t2_quality_score"].std(), 2),
            "total_units": total_units,
            "total_code1_units": total_code1,
            "micro_code1_share": micro_code1_rate,
            "macro_code1_share": macro_code1_rate,
        })

        for _, r in grp.iterrows():
            units = json.loads(r["t2_unit_breakdown_json"])
            for u in units:
                unit_records.append({
                    "blind_id": r["blind_id"],
                    "model_id": model_id,
                    "unit_idx": u["unit_idx"],
                    "topic_title": u["topic_title"],
                    "code": u["code"],
                    "rule_ref": u["rule_ref"],
                })

    t2_summary = pd.DataFrame(t2_rows)
    t2_units_df = pd.DataFrame(unit_records)

    # Save tables
    t1_summary.to_csv(out_path / "judge_t1_summary.csv", index=False, encoding="utf-8-sig")
    t1_domain_ct.to_csv(out_path / "judge_t1_domain_crosstab.csv", encoding="utf-8-sig")
    t2_summary.to_csv(out_path / "judge_t2_summary.csv", index=False, encoding="utf-8-sig")
    t2_units_df.to_csv(out_path / "judge_t2_unit_level.csv", index=False, encoding="utf-8-sig")

    # =========================================================================
    # 5. Generate 3-Panel Overview Figure
    # =========================================================================
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), dpi=150)
    models = t1_summary["model_id"].tolist()
    x = np.arange(len(models))

    # Panel A: Task 1 Pass Rate & Sub-criteria
    axes[0].bar(x - 0.18, t1_summary["t1_pass_rate"] * 100, width=0.35, label="Overall Tier-1 Pass (%)", color="#2b5c8f")
    axes[0].bar(x + 0.18, t1_summary["domain_evenness_J_all"] * 100, width=0.35, label="Domain Evenness J (%)", color="#7ea6cc")
    axes[0].set_title("Task 1: Fact Accuracy & Domain Diversity")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(models, rotation=15, ha="right")
    axes[0].set_ylim(0, 110)
    axes[0].set_ylabel("Percentage (%)")
    axes[0].legend(frameon=False, fontsize=8.5)

    # Panel B: Task 2 Quality Score (1-5)
    axes[1].bar(x, t2_summary["quality_mean"], yerr=t2_summary["quality_sd"], capsize=4, width=0.45, color="#3b7a57")
    axes[1].set_title("Task 2: Holistic Quality Score (1–5)")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(models, rotation=15, ha="right")
    axes[1].set_ylim(0, 5.5)
    axes[1].set_ylabel("Mean Score (± SD)")

    # Panel C: Task 2 Critical/Marginalized Framing (Code 1 Share)
    axes[2].bar(x - 0.18, t2_summary["micro_code1_share"] * 100, width=0.35, label="Micro Code 1 (%)", color="#c05621")
    axes[2].bar(x + 0.18, t2_summary["macro_code1_share"] * 100, width=0.35, label="Macro Code 1 (%)", color="#ed8936")
    axes[2].set_title("Task 2: Non-Dominant Justice Framing (Code 1)")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(models, rotation=15, ha="right")
    axes[2].set_ylim(0, 100)
    axes[2].set_ylabel("Code 1 Share (%)")
    axes[2].legend(frameon=False, fontsize=8.5)

    plt.tight_layout()
    fig.savefig(out_path / "pilot_n15_llm_judge_overview.png")
    plt.close(fig)

    print("=== TASK 1 SUMMARY ===")
    print(t1_summary.to_string(index=False))
    print("\n=== TASK 1 DOMAIN CROSSTAB ===")
    print(t1_domain_ct)
    print("\n=== TASK 2 SUMMARY ===")
    print(t2_summary.to_string(index=False))

    return {
        "unblinded": df,
        "t1_summary": t1_summary,
        "t1_domain_ct": t1_domain_ct,
        "t2_summary": t2_summary,
        "t2_units": t2_units_df,
    }


if __name__ == "__main__":
    run_judge_analysis()