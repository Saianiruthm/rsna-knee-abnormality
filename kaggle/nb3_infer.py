#!/usr/bin/env python3
"""
Notebook #3 — Inference / submission  (RSNA Knee Abnormality Detection, Stage 2)
==============================================================================
The SUBMISSION notebook. **Internet is OFF here.** Loads the fine-tuned
checkpoint from nb2 (which already contains the backbone weights, so no separate
pretrained download is needed), preprocesses the test DICOMs on the fly with the
SAME logic as nb1, predicts 12 probabilities per study, and writes
`/kaggle/working/submission.csv` in the sample_submission format.

Reuses nb1_preprocess.py + nb2_train.py as the single source of truth for
preprocessing and the model — add them as a utility (see kaggle/README.md) so
`import` works; SRC_DIR points at wherever they live.
"""
import os, sys, glob
import numpy as np, pandas as pd, torch

# ----------------------------- CONFIG ---------------------------------------
COMP_DIR = os.environ.get("COMP_DIR", "/kaggle/input/rsna-knee-abnormality-detection")
SRC_DIR  = os.environ.get("SRC_DIR", "/kaggle/input/rsna-knee-code")  # holds nb1/nb2 .py
CKPT     = os.environ.get("CKPT", "/kaggle/input/rsna-knee-ckpt/best_effnet_b0.pt")
OUT      = os.environ.get("OUT", "/kaggle/working/submission.csv")
# ----------------------------------------------------------------------------

sys.path.insert(0, SRC_DIR)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nb1_preprocess as nb1
import nb2_train as nb2

nb1.SPLIT = "test"
nb1.COMP_DIR = COMP_DIR


def build_study_tensor(study_uid, rows):
    """Same selection + 2.5D key-slice build as nb1, but in-memory (no save)."""
    chosen = nb1.select_series(rows)
    x = np.zeros((3, nb1.K_SLICES, 3, nb1.IMG, nb1.IMG), np.float32)
    planes = np.full(3, -1, np.int64)
    for plane, (suid, _fl) in chosen.items():
        files = nb1.order_slices(glob.glob(nb1.series_dir(study_uid, suid) + "/*.dcm"))
        t = nb1.build_series_tensor(files)
        if t is None:
            continue
        slot = nb1.PLANE_ID[plane]
        k = min(nb1.K_SLICES, t.shape[0])
        x[slot, :k] = t[:k].astype(np.float32)
        planes[slot] = slot
    return x, planes


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(CKPT, map_location=dev, weights_only=False)
    backbone = ck.get("backbone", "effnet_b0")
    model = nb2.StudyModel(backbone).to(dev).eval()
    model.load_state_dict(ck["model"])
    print(f"loaded {backbone} from {CKPT} (val auc {ck.get('auc','?')})")

    ser = pd.read_csv(f"{COMP_DIR}/test_series.csv")
    sub_cols = list(pd.read_csv(f"{COMP_DIR}/sample_submission.csv").columns)  # StudyInstanceUID + 12
    rows = []
    groups = list(ser.groupby("StudyInstanceUID"))
    for i, (uid, g) in enumerate(groups, 1):
        x, planes = build_study_tensor(uid, g)
        xt = torch.from_numpy(x)[None].to(dev)
        pt = torch.from_numpy(planes)[None].to(dev)
        with torch.no_grad():
            p = torch.sigmoid(model(xt, pt))[0].cpu().numpy()
        rows.append({"StudyInstanceUID": uid, **dict(zip(nb2.LABEL_COLS, p.tolist()))})
        if i % 50 == 0:
            print(f"  {i}/{len(groups)}")

    sub = pd.DataFrame(rows)
    # exact column order/names from sample_submission
    sub = sub[["StudyInstanceUID"] + [c for c in sub_cols if c != "StudyInstanceUID"]]
    sub.to_csv(OUT, index=False)
    print(f"wrote {OUT}  shape={sub.shape}")
    print(sub.head())


if __name__ == "__main__":
    main()
