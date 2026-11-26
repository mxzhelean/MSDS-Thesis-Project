"""
Publication-Ready Visualization Suite for Pedagogical AI Audit.

Generates 300-DPI comparative figures in `reports/figures/{batch_name}/`:
  1. fig1_length_stability.png       - Word count distribution & 400-450 target compliance
  2. fig2_lexical_diversity.png      - TTR & MTLD across closed vs. open pedagogical tasks
  3. fig3_semantic_homogenization.png - Pairwise cross-iteration cosine similarity distribution

Usage:
    python -m src.analytics.plot_metrics --batch-name pilot_n15
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.utils.logger import setup_logger

PALETTE = {
    "GPT-6 Luna": "#2B5B84",
    "Claude Haiku 4.5": "#D97757",
    "Gemini 3.5 Flash Lite": "#3B8EA5",
}


def configure_style() -> None:
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
    plt.rcParams.update(
        {
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": True,
            "legend.framealpha": 0.9,
        }
    )


def generate_figures(batch_name: str = "pilot_n15") -> None:
    logger = setup_logger(
        name=f"ai_audit.plots.{batch_name}",
        log_dir="logs",
        log_filename=f"{batch_name}_plots.log",
    )
    configure_style()

    proc_dir = Path("data/processed") / batch_name
    fig_dir = Path("doc/reports/figures") / batch_name
    fig_dir.mkdir(parents=True, exist_ok=True)

    sample_df = pd.read_csv(proc_dir / f"{batch_name}_sample_metrics.csv")
    summary_df = pd.read_csv(proc_dir / f"{batch_name}_group_summary.csv")
    pairwise_df = pd.read_csv(proc_dir / f"{batch_name}_pairwise_cosine.csv")

    model_order = [m for m in PALETTE if m in sample_df["model_label"].unique()]

    # -------------------------------------------------------------------------
    # Figure 1: Output Length Distribution & Target Window Compliance (400-450w)
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 5.2))
    ax.axhspan(400, 450, color="#2ECC71", alpha=0.15, label="Target Window (400–450 words)")
    ax.axhline(400, color="#27AE60", linestyle="--", linewidth=1)
    ax.axhline(450, color="#27AE60", linestyle="--", linewidth=1)

    sns.boxplot(
        data=sample_df,
        x="task_label",
        y="word_count",
        hue="model_label",
        hue_order=model_order,
        palette=PALETTE,
        width=0.55,
        fliersize=3,
        ax=ax,
    )
    sns.stripplot(
        data=sample_df,
        x="task_label",
        y="word_count",
        hue="model_label",
        hue_order=model_order,
        palette=PALETTE,
        dodge=True,
        alpha=0.65,
        size=5,
        edgecolor="white",
        linewidth=0.6,
        legend=False,
        ax=ax,
    )

    ax.set_title("Output Length Compliance & Cross-Iteration Stability (N=15)", fontweight="bold", pad=12)
    ax.set_xlabel("")
    ax.set_ylabel("Word Count")
    ax.legend(title="Model", loc="upper right")
    fig.tight_layout()
    fig1_path = fig_dir / "fig1_length_stability.png"
    fig.savefig(fig1_path)
    plt.close(fig)
    logger.info("Generated %s", fig1_path)

    # -------------------------------------------------------------------------
    # Figure 2: Lexical Diversity (TTR & MTLD)
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    sns.boxplot(
        data=sample_df,
        x="task_label",
        y="ttr",
        hue="model_label",
        hue_order=model_order,
        palette=PALETTE,
        width=0.55,
        ax=axes[0],
    )
    axes[0].set_title("Type-Token Ratio (TTR)", fontweight="bold")
    axes[0].set_xlabel("")
    axes[0].set_ylabel("TTR (Higher = More Diverse)")
    axes[0].get_legend().remove()

    sns.boxplot(
        data=sample_df,
        x="task_label",
        y="mtld",
        hue="model_label",
        hue_order=model_order,
        palette=PALETTE,
        width=0.55,
        ax=axes[1],
    )
    axes[1].set_title("MTLD (Length-Invariant Lexical Diversity)", fontweight="bold")
    axes[1].set_xlabel("")
    axes[1].set_ylabel("MTLD Score (Higher = More Diverse)")
    axes[1].legend(title="Model", loc="upper right")

    fig.suptitle("Lexical Diversity Across Closed vs. Open Pedagogical Tasks", fontweight="bold", y=1.02)
    fig.tight_layout()
    fig2_path = fig_dir / "fig2_lexical_diversity.png"
    fig.savefig(fig2_path, bbox_inches="tight")
    plt.close(fig)
    logger.info("Generated %s", fig2_path)

    # -------------------------------------------------------------------------
    # Figure 3: Cross-Iteration Semantic Homogenization (Pairwise Cosine Sim)
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9.5, 5.4))
    sns.violinplot(
        data=pairwise_df,
        x="task_label",
        y="cosine_similarity",
        hue="model_label",
        hue_order=model_order,
        palette=PALETTE,
        inner="box",
        cut=0,
        linewidth=1.1,
        ax=ax,
    )

    ax.set_title(
        "Cross-Iteration Semantic Homogenization (Pairwise Cosine Similarity, 105 Pairs/Group)",
        fontweight="bold",
        pad=12,
    )
    ax.set_xlabel("")
    ax.set_ylabel("Pairwise Cosine Similarity (Higher = More Homogenized)")
    ax.legend(title="Model", loc="lower right")
    fig.tight_layout()
    fig3_path = fig_dir / "fig3_semantic_homogenization.png"
    fig.savefig(fig3_path)
    plt.close(fig)
    logger.info("Generated %s", fig3_path)

    # Print quick CV & Homogenization summary table to console
    print("\n=== Key Stability & Homogenization Metrics ===")
    cols = ["model_label", "task_id", "wc_mean", "wc_cv", "window_hit_rate", "mtld_mean", "pairwise_cos_mean"]
    print(summary_df[cols].to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate quantitative audit figures.")
    parser.add_argument("--batch-name", type=str, default="pilot_n15", help="Batch folder name")
    args = parser.parse_args()
    generate_figures(batch_name=args.batch_name)