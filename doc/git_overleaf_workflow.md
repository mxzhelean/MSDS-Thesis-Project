# Git & Overleaf Synchronization Workflow

This repository uses a __nested architecture__ where the root directory manages the entire data science project (`src/`, `data/`, pipelines), while the `thesis/` subfolder houses the modular LaTeX codebase synchronized with Overleaf.

```text
┌─────────────────────────────────────────────────────────────┐
│                    GitHub Root Repository                   │
│  ├── data/                                                  │
│  ├── src/                                                   │
│  └── thesis/ ─── (Git Subtree) ───► Overleaf Remote (main)  │
└─────────────────────────────────────────────────────────────┘
```

## Authentication Setup (Overleaf)

Overleaf requires a Git Authentication Token for CLI operations.

Go to Overleaf Account Settings $\rightarrow$ Git Authentication Tokens $\rightarrow$ Create Token.

Configure the dual-remote locally:

```bash
git remote add overleaf https://git:<YOUR_OVERLEAF_TOKEN>@[git.overleaf.com/](https://git.overleaf.com/)<YOUR_PROJECT_ID>
```

## Daily Development & Sync Operations

📥 Pulling Thesis Updates from Overleaf

To fetch and merge edits made directly in the Overleaf web editor into the local `thesis/` folder:

```bash
git subtree pull --prefix=thesis overleaf main --squash
```

📤 Pushing Local Thesis Changes to Overleaf

To sync local LaTeX edits or newly generated experiment figures (`thesis/figures/`) up to Overleaf:

```bash
git subtree push --prefix=thesis overleaf main
```

🌐 Pushing Full Project to GitHub

To archive the complete end-to-end project (code, data pipelines, and thesis subfiles):

```bash
git add .
git commit -m "docs/results: update analysis pipeline and corresponding thesis chapters"
git push origin main
```

---