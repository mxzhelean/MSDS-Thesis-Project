# MSDS Thesis Project
# __A Task-Anchored Audit of Epistemic Enclosure in Higher Education LLMs__

This repository contains the complete computational pipeline, raw data corpora, and modular LaTeX source for the Master's Thesis by **Mengxi Zhong** (Master of Science in Data Science, Boston University).

---

## Repository Structure

```text
MSDS-Thesis-Project/
├── README.md                      # Primary documentation and build instructions
├── .gitignore                     # Git rules for TeX build artifacts & Python caches
├── LICENSE                        # Project MIT license
│
├── data/                          # Experimental dataset storage
│   ├── raw/                       # 300 stateless API raw JSON execution payloads
│   ├── processed/                 # Normalised text outputs & metadata matrices
│   └── embeddings/                # Sentence embedding vectors & BERTopic outputs
│
├── src/                           # Computational pipeline & empirical scripts
│   ├── api_execution/             # Automated stateless zero-shot API callers
│   ├── preprocessing/             # Markdown strippers & text normalization
│   ├── embeddings/                # High-dimensional sentence vector generators
│   └── analytics/                 # Cosine distance variance, BERTopic & QCA pipelines
│
├── notebooks/                     # Exploratory Data Analysis & visual plots
│   ├── 01_data_exploration.ipynb  # Summary statistics & token distributions
│   ├── 02_semantic_variance.ipynb # Cosine distance variance analysis (Task 1 vs 2)
│   └── 03_topic_clustering.ipynb  # BERTopic cluster diversity visualization
│
└── thesis/                        # Modular LaTeX source for the manuscript
    ├── main.tex                   # Master entry point (Preamble & \include statements)
    ├── references.bib             # BibLaTeX reference database (APA 7th Edition)
    │
    ├── config/                    # Global document styling
    │   ├── packages.tex           # Package imports & dependency management
    │   └── commands.tex           # Custom commands, macros & math operators
    │
    ├── frontmatter/               # Dissertation front matter
    │   ├── titlepage.tex          # Formal title page
    │   ├── abstract.tex           # Abstract (150–250 words)
    │   ├── acknowledgments.tex    # Acknowledgments
    │   └── abbreviations.tex      # List of Abbreviations (GTA, LLM, QCA, etc.)
    │
    ├── chapters/                  # Main manuscript body (Chapter-by-Chapter)
    │   ├── ch01_introduction.tex  # Ch 1: Motivation, Problem Statement, RQs & H1/H2
    │   ├── ch02_lit_review.tex    # Ch 2: Theoretical Framework & Literature Review
    │   ├── ch03_methodology.tex   # Ch 3: Task Matrix & Computational Methodology
    │   ├── ch04_results.tex       # Ch 4: Empirical Findings & Statistical Outputs
    │   ├── ch05_discussion.tex    # Ch 5: Discussion, Theoretical Linkages & Limitations
    │   └── ch06_conclusion.tex    # Ch 6: Conclusion, Policy & Future Extensions
    │
    └── appendices/                # Appendices & Supplementary Materials
        ├── app_a_prompts.tex      # Appendix A: Zero-Shot Prompt Framing Templates
        ├── app_b_params.tex       # Appendix B: Endpoint Decoding Hyperparameters
        ├── app_c_raw_samples.tex  # Appendix C: Representative API JSON Payloads
        ├── app_d_bertopic.tex     # Appendix D: BERTopic Cluster Keyword Detail Tables
        └── app_e_stats.tex        # Appendix E: Complete Statistical ANOVA Tables
```

---

## Reproduction & Compilation Guide

### 1. Compiling the LaTeX Manuscript

The LaTeX project is decoupled for fast modular compilation. From the `thesis/` directory:

```bash
cd thesis/

# Standard multi-pass build (PDFLaTeX + Biber)
pdflatex main.tex
biber main
pdflatex main.tex
pdflatex main.tex

# Or using latexmk (Automated)
latexmk -pdf -bibtex main.tex
```

> **Tip for Overleaf / Fast Iteration:** To compile a single chapter in isolation while maintaining cross-references and page numbers, uncomment `\includeonly{chapters/ch03_methodology}` in `thesis/main.tex`.

### 2. Running the Data Analysis Pipeline

Ensure you have Python 3.10+ installed:

```bash
# Clone the repository
git clone https://github.com/your-username/master-thesis-epistemic-enclosure.git
cd master-thesis-epistemic-enclosure

# Set up virtual environment and dependencies
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Run the complete analytical pipeline
python src/main_pipeline.py
```

---

## Citation & License

If you reference this dataset, codebase, or theoretical framework in your research, please cite:

```bibtex
@mastersthesis{Zhong2026EpistemicEnclosure,
  author  = {Zhong, Mengxi},
  title   = {A Task-Anchored Audit of Epistemic Enclosure in Higher Education {LLMs}},
  school  = {Boston University},
  department = {Department of Computer Science / Data Science},
  year    = {2026},
  month   = {September}
}
```
