# RSNA Knee Abnormality Detection — Project Handoff

You (Claude Code) are taking over a Kaggle competition project mid-way. Read this file fully before acting. Do NOT assume any prior chat context — this file is the source of truth.

## ⚡ FRESH-MACHINE / REMOTE SETUP — DO THIS FIRST
This repo ships **code + derived labels only**. Raw Kaggle competition data is NOT in the repo (redistribution is against competition rules) and secrets are NOT in the repo.
1. **Provide Kaggle creds:** put `kaggle.json` at `~/.kaggle/kaggle.json` (or set `KAGGLE_USERNAME`/`KAGGLE_KEY`). Accept the competition rules on kaggle.com first.
2. **Regenerate the data:** `uv run --with pandas --with kaggle python setup_data.py`
   → recreates `train.csv`, `*_series.csv`, `reports/<uid>.txt` (4,407), `gold_key.csv` (58), `*_sids.txt`.
3. **(Only if re-labeling reports)** provide a Gemini API key as env var `GEMINI_API_KEY` (Google AI Studio). Not needed for Stage 2 vision work.
4. `train_labels.csv` (the Stage-1 output) IS in the repo — you do not regenerate it unless you want to.
⚠️ Keep this repo **PRIVATE** — `train_labels.csv` is derived from competition data.

## The competition
- **RSNA Knee Abnormality Detection** (Kaggle, Research **code** competition). Personal entry.
- **Task:** for each knee-MRI *study*, predict probability (0–1) of **12 abnormalities**:
  `ACL, MCL, Medial Meniscus, Lateral Meniscus, Medial OA, Lateral OA, PF OA, Effusion, Synovitis, Baker's, Contusion, Fracture`
- **Metric:** macro-averaged **AUC-ROC** across the 12 labels.
- **Submission:** notebook-only, **internet disabled**, ≤9 h runtime, output file **`submission.csv`**. There is also an **Efficiency Track** (accuracy vs runtime).
- **Deadline:** final submission **2026-10-22**. Winners must open-source code + weights + a short video.
- **Test set has NO reports** — only images. So the vision model must predict from DICOM images alone.

## THE KEY DATA FACT (why this project exists)
`train.csv` has **4,407 studies but only 58 carry ground-truth 12-label annotations**. The other 4,349 have only a **free-text radiology Report** (multilingual: Spanish, Turkish, Greek, Bulgarian, Dutch, German, Croatian, English). So this is a **weakly/report-supervised** problem: we mined labels from the reports to create training targets, and reserve the 58 gold for validation.

## STAGE 1 — REPORT LABELING (✅ DONE)
- Approach: agentic radiologist reading of each report (no regex) → 12 calibrated probabilities. Prompt lives in `label_nvidia.py` (`PROMPT_HEAD`).
- Final labeler: **Gemini 2.5 Flash-Lite** via Google AI Studio (OpenAI-compat endpoint), `--thinkoff`, max_tokens 700.
- Result: all **4,407 labeled, 0 errors, macro-AUC 0.8901** on the 58 gold (matches Claude's 0.895; Gemini Flash 0.874; NVIDIA Llama-3.1-8B 0.843). Cost ~$0.62.
- **Deliverable:** `train_labels.csv` = 4,407 studies × 12 probability columns. **This is the training target for Stage 2.**
- Note: the 58-gold set is *enriched for abnormal cases*, so full-population predicted means are lower than gold rates — expected, not drift. AUC ranking is what matters.
- Weakest label across all models = **Synovitis (~0.75)** — candidate for a one-line prompt tweak + cheap re-run (~$0.62) later.

### Files in this directory
- `train.csv` — 4,407 studies: StudyInstanceUID, Report (text), + 12 label cols (only 58 rows filled).
- `train_labels.csv` — **the mined soft labels (Stage-1 output, USE THIS as targets).**
- `reports/<StudyInstanceUID>.txt` — one report per study (all 4,407).
- `out_gemini/<uid>.json` — per-study Gemini label outputs (source of train_labels.csv).
- `gold_key.csv` — the 58 true-labeled studies (StudyInstanceUID + 12 binary labels) → **validation set**.
- `gold_sids.txt`, `all_sids.txt` — UID lists.
- `label_nvidia.py` — the labeler (provider-agnostic; `--base_url`, `--keyenv`, `--thinkoff`). Reusable if re-labeling.
- `score_auc.py` — macro-AUC scorer: `python score_auc.py ~/rsna_knee <out_dir>`.
- `setup_data.py` — regenerates competition data + reports/ + gold_key.csv from Kaggle (run first on a fresh machine).
- Keys are NOT in the repo. Kaggle creds → `~/.kaggle/kaggle.json`. Gemini key → env `GEMINI_API_KEY` (only if re-labeling).
- Python: use `uv run --with pandas --with numpy --with openai python ...` (no global installs).

## STAGE 2 — VISION MODEL (⬜ NEXT, not started)
Goal: train image→12-probabilities on the DICOMs using `train_labels.csv` as targets; validate on the 58 gold; submit an inference-only notebook.

**Compute decision (budget = ~$0):** train on **Kaggle free GPU** (T4×2/P100 16GB, 30h/week, 9h/session). The DICOM data (~hundreds of GB / ~1TB) is **pre-mounted at `/kaggle/input`** — do NOT download it locally. Do not use paid cloud (user is cost-constrained) or company A100s (keep the project personal).

**Submission mechanic:** the notebook does NOT train. Train (on Kaggle GPU, in stages with checkpoints), upload weights as a private Kaggle **Dataset/Model**, and the **submission notebook only loads weights + runs inference → `submission.csv`**. Any pretrained backbone must also be uploaded as a dataset (internet is off at submission).

**Planned 3 notebooks:**
1. **Preprocess/cache** — DICOM → small resized arrays (2.5D: pick key slices per series), save as a private Kaggle dataset so training epochs are fast.
2. **Train** — pretrained backbone (EfficientNet-B0/B3 or ConvNeXt-tiny, weights bundled as dataset) → per-series encoder → study-level attention pooling → 12 sigmoids, BCE against `train_labels.csv`. fp16, checkpoint/resume across 9h sessions. Route series by plane + fluid-sensitive metadata (in `train_series.csv` / `test_series.csv`: columns Fluid_Sensitive, Fat_Suppression, Anatomical_Plane).
3. **Inference** — load weights, predict on `/kaggle/input` test, write `submission.csv`.

**First steps for Stage 2:** (a) confirm exact dataset size via kaggle CLI; (b) start a Kaggle notebook that reads `/kaggle/input`, explores series/slice counts per study; (c) build the preprocessing/caching notebook.

## Suggested first prompt to give me (the new session)
> Read HANDOFF.md in this directory. Stage 1 (report labeling) is done — train_labels.csv is ready. Start Stage 2: scope the Kaggle dataset size and draft the preprocessing/caching notebook for the DICOMs (2.5D key-slice extraction), everything designed to run on Kaggle free GPU with data pre-mounted at /kaggle/input.
