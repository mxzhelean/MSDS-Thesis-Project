# Repository Structure

```text
MSDS-Thesis-Project/
├── README.md                      # Project Overview, environment setup & reproduction instructions
├── requirements.txt               # Locked Python dependencies 
├── .gitignore                     # Git rules for TeX build artifacts & Python caches
├── LICENSE                        # Project MIT license
├── configs/                       # Immutable baseline configurations & prompts
│   ├── prompts_v1.0.json          # Locked symmetric Task 1 & Task 2 zero-shot prompts
│   └── models_baseline.json       # Exact model snapshot IDs, native params & no-tool settings
│
├── docs/                          # [TBD] Methodological protocols & coding guidelines
│   └── CODEBOOK.md                # Detailed Tier 1 & Tier 2B rubrics, decision trees & anchors
│
├── data/                          # Experimental dataset storage (gitignored except schema samples)
│   ├── raw/                       # Append-only stateless API raw JSONL payloads
│   │   ├── pilot_n15/             # Current N=15 batch validation (90 responses)
│   │   └── full_n50/              # Full N=50 production run (300 responses)
│   ├── processed/                 # Cleaned, Markdown-stripped texts & word-count metadata
│   ├── annotations/               # Blinded coding sheets (Human Primary + LLM Cross-Audit)
│   └── embeddings/                # Response-level dense sentence embeddings (.npy / .parquet)
│
├── src/                           # Computational pipeline & empirical scripts
│   ├── api_execution/             # Automated stateless zero-shot API callers
│   ├── preprocessing/             # Markdown strippers & text normalization
│   ├── cross_audit/               # Deterministic (temp=0) LLM-as-a-Judge audit scripts
│   ├── embeddings/                # Response-level sentence vector generators (Tier 2A)
│   └── analytics/                 # Cosine distance/centroid variance (Tier 2A), QCA chi-square/KW & ICR (Tier 2B)
│
├── notebooks/                     # Exploratory Data Analysis & visual plots
│   ├── 01_data_exploration.ipynb  # Summary statistics & token distributions
│   ├── 02_semantic_variance.ipynb # Cosine distance variance analysis (Task 1 vs 2)
│   └── 03_qca_and_icr_audit.ipynb # Tier 2B Shannon entropy, QCA proportions & Human-LLM Cohen's kappa
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
