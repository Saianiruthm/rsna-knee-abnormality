#!/usr/bin/env python3
"""
Notebook #1 — Preprocess / cache  (RSNA Knee Abnormality Detection, Stage 2)
============================================================================
Runs INSIDE a Kaggle notebook, against the competition data pre-mounted at
`/kaggle/input/rsna-knee-abnormality-detection`. Emits a compact cache to
`/kaggle/working/cache`; "Save Version" then publishes it as a PRIVATE Kaggle
dataset that notebook #2 (train) mounts for fast epochs.

Design (see EDA.md / STATUS.md):
- Every study has all 3 planes + a fluid-sensitive series (100% coverage), so we
  select a CANONICAL TRIPLET, one series per plane, fluid-status chosen to expose
  each plane's findings:
      sagittal -> non-fluid  (menisci, ACL, cartilage)
      coronal  -> fluid-sens (MCL, marrow edema, effusion)
      axial    -> fluid-sens (PF joint, effusion, synovitis, Baker's)
  Fallback to the other fluid-status if a bucket is empty, else any series in plane.
  Among candidates in a bucket, pick the series with the most slices.
- Per selected series: KEY SLICES from the middle 70%, expanded to 3-channel 2.5D
  (slice i-1, i, i+1). Resize to 224^2, per-volume percentile-normalized, float16.
- Output per study: <uid>.npz  with
      x      float16 [3, K, 3, 224, 224]   (plane-slot, key-slice, channel, H, W)
      planes int8    [3]                    (0=Sag,1=Cor,2=Ax; -1 if missing)
      fluid  int8    [3]                    (0/1; -1 if missing)
  plus manifest.csv (uid, chosen series UIDs, +12 label cols for train).

Tunables are the CONFIG block below. Defaults ~19 GB train cache.
Deps are all preinstalled on Kaggle: pydicom, numpy, pandas, opencv (cv2).
"""
import os, glob, json
import numpy as np
import pandas as pd
import pydicom
import cv2
from concurrent.futures import ProcessPoolExecutor, as_completed

# ----------------------------- CONFIG ---------------------------------------
COMP_DIR   = "/kaggle/input/rsna-knee-abnormality-detection"
OUT_DIR    = "/kaggle/working/cache"
SPLIT      = os.environ.get("SPLIT", "train")   # "train" or "test"
IMG        = 224          # output HxW
K_SLICES   = 5            # key slices per series
MID_FRAC   = 0.70         # keep central fraction of the stack for key-slice picks
CLIP_LO_HI = (1.0, 99.0)  # per-volume percentile window for normalization
N_WORKERS  = max(1, (os.cpu_count() or 4))
PLANES     = ["Sagittal", "Coronal", "Axial"]
PLANE_ID   = {"Sagittal": 0, "Coronal": 1, "Axial": 2}
# desired fluid-sensitivity per plane slot (True=prefer fluid-sensitive series)
PREFER_FLUID = {"Sagittal": False, "Coronal": True, "Axial": True}
LABELS = ["ACL","MCL","Medial Meniscus","Lateral Meniscus","Medial OA","Lateral OA",
          "PF OA","Effusion","Synovitis","Baker's","Contusion","Fracture"]
# ----------------------------------------------------------------------------


def series_dir(study_uid, series_uid):
    return f"{COMP_DIR}/{SPLIT}_series/{study_uid}/{series_uid}"


def order_slices(files):
    """Return dcm paths ordered along the through-plane axis.
    Uses ImagePositionPatient projected on the slice normal (IOP row x col);
    falls back to InstanceNumber."""
    recs = []
    for f in files:
        try:
            d = pydicom.dcmread(f, stop_before_pixels=True)
        except Exception:
            continue
        key = None
        iop = getattr(d, "ImageOrientationPatient", None)
        ipp = getattr(d, "ImagePositionPatient", None)
        if iop is not None and ipp is not None and len(iop) == 6:
            r = np.array(iop[:3], float); c = np.array(iop[3:], float)
            n = np.cross(r, c)
            key = float(np.dot(np.array(ipp, float), n))
        if key is None:
            key = float(getattr(d, "InstanceNumber", 0) or 0)
        recs.append((key, f))
    recs.sort(key=lambda t: t[0])
    return [f for _, f in recs]


def key_indices(n, k=K_SLICES, mid=MID_FRAC):
    if n <= 0:
        return []
    lo = int((1 - mid) / 2 * n)
    hi = max(lo, int((1 + mid) / 2 * n) - 1)
    if hi <= lo:
        lo, hi = 0, n - 1
    if k == 1:
        return [(lo + hi) // 2]
    return [int(round(lo + (hi - lo) * j / (k - 1))) for j in range(k)]


def read_pixels(path):
    d = pydicom.dcmread(path)
    a = d.pixel_array.astype(np.float32)
    slope = float(getattr(d, "RescaleSlope", 1) or 1)
    inter = float(getattr(d, "RescaleIntercept", 0) or 0)
    if slope != 1 or inter != 0:
        a = a * slope + inter
    return a


def resize(a):
    return cv2.resize(a, (IMG, IMG), interpolation=cv2.INTER_AREA)


def build_series_tensor(files):
    """files: ordered dcm paths -> float16 [K, 3, IMG, IMG], per-volume normalized."""
    n = len(files)
    idxs = key_indices(n)
    if not idxs:
        return None
    # read each key slice and its +/-1 neighbors (clamped)
    needed = sorted(set(j for i in idxs for j in (i - 1, i, i + 1) if 0 <= j < n))
    px = {j: resize(read_pixels(files[j])) for j in needed}
    stack = np.stack([px[j] for j in needed]).astype(np.float32)
    lo, hi = np.percentile(stack, CLIP_LO_HI)
    if hi <= lo:
        hi = lo + 1.0
    out = np.zeros((len(idxs), 3, IMG, IMG), np.float16)
    for s, i in enumerate(idxs):
        for ch, j in enumerate((i - 1, i, i + 1)):
            j = min(max(j, 0), n - 1)
            v = (px[j] - lo) / (hi - lo)
            out[s, ch] = np.clip(v, 0, 1).astype(np.float16)
    return out


def select_series(rows):
    """rows: DataFrame of one study's series -> {plane: (series_uid, fluid)}."""
    chosen = {}
    for plane in PLANES:
        cand = rows[rows.Anatomical_Plane == plane]
        if len(cand) == 0:
            continue
        want = 1 if PREFER_FLUID[plane] else 0
        pref = cand[cand.Fluid_Sensitive == want]
        pool = pref if len(pref) else cand  # fallback: any fluid-status in plane
        # among pool, prefer the series with most slices on disk
        best, best_n = None, -1
        for _, r in pool.iterrows():
            n = len(glob.glob(series_dir(r.StudyInstanceUID, r.SeriesInstanceUID) + "/*.dcm"))
            if n > best_n:
                best_n, best = n, r
        chosen[plane] = (best.SeriesInstanceUID, int(best.Fluid_Sensitive))
    return chosen


def process_study(args):
    study_uid, rows = args
    try:
        chosen = select_series(rows)
        x = np.zeros((3, K_SLICES, 3, IMG, IMG), np.float16)
        planes = np.full(3, -1, np.int8)
        fluid = np.full(3, -1, np.int8)
        chosen_uids = {}
        for plane, (suid, fl) in chosen.items():
            files = order_slices(glob.glob(series_dir(study_uid, suid) + "/*.dcm"))
            t = build_series_tensor(files)
            if t is None:
                continue
            slot = PLANE_ID[plane]
            k = min(K_SLICES, t.shape[0])
            x[slot, :k] = t[:k]
            planes[slot] = slot
            fluid[slot] = fl
            chosen_uids[plane] = suid
        np.savez_compressed(f"{OUT_DIR}/{study_uid}.npz", x=x, planes=planes, fluid=fluid)
        return study_uid, chosen_uids, None
    except Exception as e:
        return study_uid, {}, f"{type(e).__name__}: {e}"


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    ser = pd.read_csv(f"{COMP_DIR}/{SPLIT}_series.csv")
    groups = list(ser.groupby("StudyInstanceUID"))
    print(f"[{SPLIT}] {len(groups)} studies, {len(ser)} series, {N_WORKERS} workers")

    labels = None
    if SPLIT == "train":
        # train_labels.csv ships in the repo; upload it as an input dataset on Kaggle,
        # or fall back to the competition train.csv gold columns.
        for p in ["/kaggle/input/rsna-knee-labels/train_labels.csv",
                  f"{COMP_DIR}/train.csv"]:
            if os.path.exists(p):
                labels = pd.read_csv(p).set_index("StudyInstanceUID")
                print("labels from", p)
                break

    manifest, errors = [], []
    with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
        futs = [ex.submit(process_study, g) for g in groups]
        for i, fut in enumerate(as_completed(futs), 1):
            uid, chosen, err = fut.result()
            if err:
                errors.append((uid, err))
            row = {"StudyInstanceUID": uid,
                   **{f"series_{p}": chosen.get(p, "") for p in PLANES}}
            if labels is not None and uid in labels.index:
                for c in LABELS:
                    row[c] = labels.loc[uid].get(c, np.nan)
            manifest.append(row)
            if i % 200 == 0:
                print(f"  {i}/{len(groups)}  errors={len(errors)}")

    pd.DataFrame(manifest).to_csv(f"{OUT_DIR}/manifest_{SPLIT}.csv", index=False)
    json.dump({"split": SPLIT, "img": IMG, "k_slices": K_SLICES,
               "n_studies": len(groups), "n_errors": len(errors),
               "errors": errors[:50]},
              open(f"{OUT_DIR}/meta_{SPLIT}.json", "w"), indent=2)
    print(f"DONE [{SPLIT}] studies={len(groups)} errors={len(errors)} -> {OUT_DIR}")
    if errors:
        print("first errors:", errors[:5])


if __name__ == "__main__":
    main()
