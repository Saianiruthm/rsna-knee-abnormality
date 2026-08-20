# Stage 2 — Status & Decisions Log

Companion to `HANDOFF.md`. Captures decisions made after Stage-1 handoff so the
project can be resumed anywhere (clone repo → read `HANDOFF.md` + `STATUS.md` + `EDA.md`).

## Where we are
- **Setup done:** Kaggle auth works; data regenerated (small CSVs only, no DICOM bulk); `train_series.csv`/`test_series.csv` present.
- **Labels verified:** `train_labels.csv` vs 58-gold macro-AUC = **0.8901** (matches HANDOFF). Intact.
- **Data profiled:** see `EDA.md`. Competition data ~927 GB DICOM, layout `{train,test}_series/<StudyUID>/<SeriesUID>/<SOPUID>.dcm`.
- **Next:** build notebook #1 (preprocess/cache) on Kaggle.

## Compute decision
- **Train on Kaggle free GPU (P100 16 GB / T4×2) for now.** Data is pre-mounted at `/kaggle/input` → zero data movement; end-to-end in the submission environment; small models fit easily. Design around 9 h/session + 30 h/week with checkpoint/resume.
- **Vultr ($300 credit, expires 2026-09-07) held in reserve** for a heavier phase (large sweeps that exceed Kaggle's weekly quota, or a big model). If used: **preprocess stays on Kaggle** (data lives there); Vultr only pulls the small cache to train.
  - Preferred box if/when used: single strong x86 GPU (L40S 48 GB if in stock, else A16). Avoid reserved-capacity commitments and GH200/ARM unless needed.
  - **Persistence = Kaggle, not Vultr.** Cache = a private Kaggle dataset; checkpoints → Kaggle dataset or repo (~40 MB fp16). Destroy the Vultr instance when idle; no snapshot/backup fees (Vultr auto-backup ≈ $225/mo — leave OFF).
- **OCI ($300, 30-day trial)** = a later second compute window; OCI GPU shapes need a service-limit-increase request first.

## Model plan (unchanged in spirit from HANDOFF)
- Targets = `train_labels.csv` (soft, BCE). Validate on 58 gold (macro-AUC).
- Per-series encoder → study-level **attention pooling** → 12 sigmoids.
- **Backbones to A/B:** ImageNet EfficientNet-B0/B3 or ConvNeXt-tiny (default) **vs DAX ViT-S/16** (see below). Backbone slot is swappable.
- Class imbalance (Fracture/MCL/Lateral OA rare) → weighted-BCE / focal.

## DAX (DINO Adapted to X-ray) — evaluation outcome
- **Use as a benchmarked, fine-tuned backbone option, not the sole bet.** Apache-2.0, weights on HuggingFace (`joshua-scheuplein/DAX-*`), ViT-S = 21.7M / feat dim 384, 224²/3-ch, plain-torch load — bundles cleanly as an internet-off Kaggle dataset.
- **Pro:** float-based preprocessing preserves our **12-bit dynamic range** (no 8-bit crush).
- **Con:** pretrained on projection **X-ray**, our data is **MRI** soft tissue (ACL/menisci/effusion/synovitis) — real modality gap, no MRI transfer claims. → **fine-tune** it (not frozen linear-probe); let 58-gold macro-AUC decide vs the ImageNet default.

## Cache spec (drives notebook #1)
- DICOM → **2.5D key-slice** arrays, **224²**, **3-channel** (adjacent slices), **float16**, percentile-normalized per volume. Backbone-agnostic (EfficientNet + DAX).
- Carry `Anatomical_Plane` + `FluidSat` per series into the cache for the routing head.
- **100% plane+fluid coverage** (see EDA) → assume fixed complete per-study set; no missing-data handling.
- Output = a **private Kaggle dataset** so training epochs are fast.

## Open decisions (settle when writing notebook #1)
- Key-slice budget: fixed N (e.g. 5 center slices) vs plane-aware (more for sagittal).
- Canonical per-study series selection (which of the 6 plane×fluid buckets to feed).
- Cache dtype confirmed float16 (not uint8) to preserve dynamic range.

## Roadmap
1. **Notebook #1 — preprocess/cache** (`kaggle/nb1_preprocess.py`) ✅ drafted, smoke-tested on sample DICOMs. ← run on Kaggle next
2. **Notebook #2 — train** (`kaggle/nb2_train.py`) ✅ drafted, smoke-tested (train loop, resume, gold AUC).
3. **Notebook #3 — inference** (`kaggle/nb3_infer.py`) ✅ drafted, smoke-tested (load, predict, submission format).

All three are authored + locally validated on synthetic/sample data. Remaining
work is on Kaggle (needs `/kaggle/input`): run nb1 → publish cache dataset →
run nb2 (compare effnet_b0 vs dax_vits16 on gold macro-AUC) → run nb3 → submit.
