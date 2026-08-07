# RSNA Knee Abnormality — Exploratory Data Analysis

Generated for Stage 2 planning. All numbers computed locally from `train.csv`,
`train_labels.csv`, `gold_key.csv`, `train_series.csv`, `test_series.csv`, plus
DICOM headers from an 11-slice sample. **No bulk data downloaded.**

## A. Scale & structure

| Item | Value |
|---|---|
| Train studies | 4,407 |
| Train series | 24,371 |
| Est. slices (mean 37.1/series) | ~0.90M (~927 GB raw DICOM) |
| Gold (validation) studies | 58 |
| Visible test studies / series | 3 / 15 (real test is hidden, rerun at submission) |
| Series per study | min 3 · median 5 · mean 5.5 · p95 9 · max 14 |

## B. Available fields ("params")

- **Study-level** (`train.csv`): `StudyInstanceUID`, `Report` (free text, multilingual), 12 label cols (only 58 gold rows filled).
- **Targets** (`train_labels.csv`): 4,407 × 12 soft probabilities (Stage-1 mined labels).
- **Series-level** (`train_series.csv` / `test_series.csv`): `SeriesInstanceUID`, `Anatomical_Plane`, `Fluid_Sensitive`, `Fat_Suppression`.
- **DICOM-level** (from sample): MR, MONOCHROME2, uint16 **12-bit stored / 16-bit allocated**; matrix 640²–960² (+ a 640×1280 3D patella); in-plane spacing 0.16–0.25 mm; slice thickness 3 mm / gap 3.3 mm (3D: 0.6 mm); `ImageOrientationPatient`, `SeriesDescription`, `InstanceNumber`. Intensity is **uncalibrated** (per-series max ranges ~400–3900).

## C. Per-pathology prevalence

Targets are soft probabilities, so "positive" is reported three ways.

| Pathology | gold+ /58 | mined p≥.5 /4407 | mined % | mean p |
|---|---|---|---|---|
| Medial Meniscus | 26 | 2,079 | 47.2 | .455 |
| PF OA | 21 | 1,568 | 35.6 | .354 |
| Effusion | 35 (60%) | 1,503 | 34.1 | .376 |
| Baker's | 12 | 1,093 | 24.8 | .263 |
| Medial OA | 15 | 1,072 | 24.3 | .275 |
| ACL | 24 | 920 | 20.9 | .226 |
| Lateral Meniscus | 23 | 915 | 20.8 | .220 |
| Contusion | 19 | 897 | 20.4 | .226 |
| Synovitis | 27 | 673 | 15.3 | .215 |
| Lateral OA | 11 | 604 | 13.7 | .183 |
| MCL | 9 | 413 | 9.4 | .130 |
| Fracture | 18 | 350 | 7.9 | .104 |

Notes:
- The 58-gold set is **enriched for abnormal cases** (e.g. Effusion 60% gold vs 34% mined) — expected, documented in HANDOFF. AUC ranking is what matters.
- **Rare classes:** Fracture (7.9%), MCL (9.4%), Lateral OA (13.7%). Gold has only **9 MCL / 11 Lateral OA / 12 Baker's** positives → their gold AUC is noisy; do not over-read single-label swings.

## C2. Multi-label structure

- Mean **2.74** labels/study (p≥.5); median 2; max 12.
- **722 studies (16%)** have no label ≥.5 ("normal" knees).
- Strongest co-occurrences: Medial Meniscus+Effusion (916), Medial Meniscus+PF OA (867), Medial Meniscus+Medial OA (859), Medial OA+PF OA (693), Effusion+Synovitis (526).

## D. Views / planes — key structural finding

| | Sagittal | Coronal | Axial |
|---|---|---|---|
| series count | 9,864 | 8,609 | 5,898 |

Plane × Fluid_Sensitive (all 6 buckets populated):

| | non-fluid (0) | fluid-sensitive (1) |
|---|---|---|
| Axial | 1,179 | 4,719 |
| Coronal | 3,985 | 4,624 |
| Sagittal | 5,197 | 4,667 |

- **`Fluid_Sensitive` ≡ `Fat_Suppression` — 100% identical, zero nulls** (train and test). Collapses to one axis → routing is Plane(3) × FluidSat(2) = **6 buckets, not 12**.
- **Every study (100%) has all 3 planes AND ≥1 fluid-sensitive series.** No missing-plane / missing-fluid cases anywhere.

## E. Reports

All 4,407 reports non-empty. Char length: median 977 · mean 1,098 · p95 2,452 · max 4,743. Multilingual (per HANDOFF: Spanish, Turkish, Greek, Bulgarian, Dutch, German, Croatian, English).

## Design implications for the notebooks

1. **100% plane+fluid coverage → assume a fixed, complete per-study set.** No missing-data logic. Canonical routing set, e.g. sagittal-PD, coronal-fluid-sens, axial-fluid-sens.
2. **Preserve 12-bit dynamic range** in the cache (float16, percentile-normalize per volume) — do **not** 8-bit-quantize; low-contrast findings (effusion, synovitis, marrow edema) live in that range.
3. **Class imbalance** (Fracture/MCL/Lateral OA rare) → weighted-BCE or focal loss.
4. **Effusion/Synovitis are fluid-findings** → bias their key-slice selection toward fluid-sensitive series.
5. **Resize to a common grid** (matrices vary 640²–960²); 224²/3-ch is backbone-agnostic (EfficientNet & DAX ViT-S).
6. Label correlations → attention pooling + BCE baseline; correlation-aware head is a later lever.
