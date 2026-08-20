# Kaggle notebooks (Stage 2)

End-to-end lives on Kaggle so train == submission environment. Data is
pre-mounted at `/kaggle/input/rsna-knee-abnormality-detection` (~927 GB, do NOT
download locally).

## nb1_preprocess.py — build the cache
Turns DICOMs into a compact 2.5D key-slice cache and publishes it as a **private
Kaggle dataset** for fast training epochs.

**Run:**
1. New Kaggle Notebook → add the competition as input data → GPU not needed (CPU).
2. Upload `train_labels.csv` as a private dataset named `rsna-knee-labels`
   (nb1 reads `/kaggle/input/rsna-knee-labels/train_labels.csv`; else falls back
   to the competition `train.csv` gold columns).
3. Paste `nb1_preprocess.py` into a cell (or `%run`), then run twice:
   ```python
   import os; os.environ["SPLIT"]="train"   # then run main()
   os.environ["SPLIT"]="test"               # then run main() again
   ```
4. **Save Version** (commit). Output under `/kaggle/working/cache` becomes a
   dataset — set it **Private**. This is the input to nb2.

**Output** per study `<uid>.npz`: `x` float16 `[3, K, 3, 224, 224]`
(plane-slot Sag/Cor/Ax × key-slice × 2.5D channel × H × W), `planes[3]`,
`fluid[3]`; plus `manifest_<split>.csv` and `meta_<split>.json`.

**Config** (top of nb1): `IMG=224`, `K_SLICES=5`, canonical triplet
(sag→non-fluid, cor/ax→fluid-sensitive). Est. train cache ~19 GB; shrink via
`K_SLICES` if needed. Every study has all 3 planes + a fluid series (see EDA.md),
so no missing-data handling is required.

## nb2_train.py — train
Cache → backbone encodes each 2.5D key-slice → mean over key-slices → **masked
attention pool over the 3 plane-slots** → 12 sigmoids. Weighted-BCE vs soft
labels, validate on the 58 gold (macro-AUC), fp16, **checkpoint/resume** across
the 9h session limit. Logic smoke-tested on a synthetic cache (train loop,
resume, gold AUC all verified).

**Run (GPU notebook):**
1. Add inputs: the **cache** dataset from nb1, the **labels** dataset
   (`train_labels.csv` + `gold_key.csv`), and for DAX the **weights** dataset.
2. Keep internet ON (timm fetches ImageNet weights).
3. Set backbone + hyperparams via env, then run `main()`:
   ```python
   import os; os.environ.update(CACHE_DIR="/kaggle/input/rsna-knee-cache",
       LABELS_CSV="/kaggle/input/rsna-knee-labels/train_labels.csv",
       GOLD_CSV="/kaggle/input/rsna-knee-labels/gold_key.csv",
       BACKBONE="effnet_b0")   # or "convnext_tiny" | "dax_vits16"
   ```
4. Re-run the notebook to resume (checkpoint in `/kaggle/working`). **Save
   Version** → publish `best_<backbone>.pt` as a private dataset for nb3.

The 58 gold studies are **held out of training** and used only for validation.
Compare `effnet_b0` vs `dax_vits16` on gold macro-AUC to settle the backbone.

## nb3_infer.py — TODO
Internet OFF. Load weights + backbone (bundled datasets) → predict
`/kaggle/input` test → `submission.csv`.
