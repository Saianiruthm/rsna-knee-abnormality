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

## nb2_train.py — TODO
Cache → per-series encoder (EfficientNet-B0 default | DAX ViT-S option) →
study-level attention pooling → 12 sigmoids, weighted-BCE vs soft labels,
validate on 58 gold (macro-AUC), fp16, checkpoint/resume. Checkpoints saved as a
private dataset for nb3.

## nb3_infer.py — TODO
Internet OFF. Load weights + backbone (bundled datasets) → predict
`/kaggle/input` test → `submission.csv`.
