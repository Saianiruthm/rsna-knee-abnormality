#!/usr/bin/env python3
"""
Notebook #2 — Train  (RSNA Knee Abnormality Detection, Stage 2)
==============================================================
Runs on a Kaggle GPU notebook (P100 / T4x2). Reads the private cache dataset
built by nb1 (2.5D key-slice .npz), trains image -> 12 sigmoids against the soft
labels in train_labels.csv (BCE), validates on the 58 gold studies (macro-AUC),
fp16 + checkpoint/resume so it survives the 9h session limit. Checkpoints go to
/kaggle/working and are published as a private dataset for nb3 (inference).

Architecture:
  x [B, 3, K, 3, 224, 224]  (plane-slot Sag/Cor/Ax x key-slice x 2.5D-channel)
    -> backbone encodes each of the 3*K images        -> [B, 3, K, D]
    -> mean over K key-slices                          -> [B, 3, D]  (per plane)
    -> masked attention pool over the 3 plane-slots    -> [B, D]
    -> linear head                                     -> [B, 12] logits

Backbones (swap via BACKBONE): "effnet_b0" (timm, ImageNet, default) |
"convnext_tiny" | "dax_vits16" (DINO ViT-S/16 pretrained on X-ray; bundle the
checkpoint as a dataset — see DAX_CKPT). Training notebook may keep internet ON
to fetch timm weights; the SUBMISSION notebook (nb3) must be offline, so weights
get bundled as datasets there.
"""
import os, glob, json, time
import numpy as np, pandas as pd
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ----------------------------- CONFIG ---------------------------------------
CACHE_DIR = os.environ.get("CACHE_DIR", "/kaggle/input/rsna-knee-cache")
LABELS_CSV= os.environ.get("LABELS_CSV","/kaggle/input/rsna-knee-labels/train_labels.csv")
GOLD_CSV  = os.environ.get("GOLD_CSV", "/kaggle/input/rsna-knee-labels/gold_key.csv")
WORK      = os.environ.get("WORK", "/kaggle/working")
BACKBONE  = os.environ.get("BACKBONE", "effnet_b0")
DAX_CKPT  = os.environ.get("DAX_CKPT", "/kaggle/input/dax-weights/dax-vit-s-16-a.pth")
EPOCHS    = int(os.environ.get("EPOCHS", 12))
BS        = int(os.environ.get("BS", 8))
LR        = float(os.environ.get("LR", 3e-4))
WD        = float(os.environ.get("WD", 1e-4))
NUM_WORK  = int(os.environ.get("NUM_WORK", 2))
SEED      = 42
LABEL_COLS= ["ACL","MCL","Medial Meniscus","Lateral Meniscus","Medial OA","Lateral OA",
             "PF OA","Effusion","Synovitis","Baker's","Contusion","Fracture"]
# ----------------------------------------------------------------------------


def rank_auc(y, s):
    """Tie-aware ROC-AUC via mean ranks (matches score_auc.py)."""
    y = np.asarray(y); s = np.asarray(s, float)
    P = (y == 1).sum(); N = (y == 0).sum()
    if P == 0 or N == 0:
        return float("nan")
    _, inv, cnt = np.unique(s, return_inverse=True, return_counts=True)
    csum = np.cumsum(cnt); avg = (csum - cnt + csum + 1) / 2.0
    r = avg[inv]
    return (r[y == 1].sum() - P * (P + 1) / 2) / (P * N)


def macro_auc(Y, S):
    return float(np.nanmean([rank_auc(Y[:, i], S[:, i]) for i in range(Y.shape[1])]))


class CacheDataset(Dataset):
    def __init__(self, uids, cache_dir, y_map):
        self.uids = uids; self.dir = cache_dir; self.y = y_map
    def __len__(self):
        return len(self.uids)
    def __getitem__(self, i):
        uid = self.uids[i]
        d = np.load(f"{self.dir}/{uid}.npz")
        x = torch.from_numpy(d["x"].astype(np.float32))      # [3,K,3,224,224]
        planes = torch.from_numpy(d["planes"].astype(np.int64))
        y = torch.from_numpy(self.y[uid].astype(np.float32))  # [12]
        return x, planes, y


def build_backbone(name):
    import timm
    if name == "effnet_b0":
        net = timm.create_model("efficientnet_b0", pretrained=True, num_classes=0, global_pool="avg")
        return net, net.num_features
    if name == "convnext_tiny":
        net = timm.create_model("convnext_tiny", pretrained=True, num_classes=0, global_pool="avg")
        return net, net.num_features
    if name == "dax_vits16":
        net = timm.create_model("vit_small_patch16_224", pretrained=False, num_classes=0)
        ck = torch.load(DAX_CKPT, map_location="cpu", weights_only=False)
        sd = ck.get("teacher", ck)
        sd = {k.replace("module.", "").replace("backbone.", ""): v for k, v in sd.items()}
        msg = net.load_state_dict(sd, strict=False)
        print("DAX load:", len(msg.missing_keys), "missing,", len(msg.unexpected_keys), "unexpected")
        return net, net.num_features
    raise ValueError(name)


class StudyModel(nn.Module):
    def __init__(self, backbone="effnet_b0", n_labels=12):
        super().__init__()
        self.enc, D = build_backbone(backbone)
        self.D = D
        self.q = nn.Parameter(torch.randn(D) * 0.02)   # attention query over plane-slots
        self.att = nn.Linear(D, D)
        self.head = nn.Sequential(nn.LayerNorm(D), nn.Dropout(0.2), nn.Linear(D, n_labels))
    def forward(self, x, planes):
        B, S, K, C, H, W = x.shape
        f = self.enc(x.reshape(B * S * K, C, H, W))     # [B*S*K, D]
        f = f.reshape(B, S, K, self.D).mean(2)           # [B, S, D] mean over key-slices
        scores = (torch.tanh(self.att(f)) @ self.q)      # [B, S]
        scores = scores.masked_fill(planes < 0, float("-inf"))
        w = torch.softmax(scores, dim=1).unsqueeze(-1)   # [B, S, 1]
        z = (w * f).sum(1)                               # [B, D]
        return self.head(z)                              # [B, 12] logits


def load_labels():
    tl = pd.read_csv(LABELS_CSV).set_index("StudyInstanceUID")
    y_map = {u: tl.loc[u, LABEL_COLS].values.astype(np.float32) for u in tl.index}
    gold = pd.read_csv(GOLD_CSV).set_index("StudyInstanceUID")
    return y_map, gold


def main():
    torch.manual_seed(SEED); np.random.seed(SEED)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    y_map, gold = load_labels()
    have = {os.path.basename(p)[:-4] for p in glob.glob(f"{CACHE_DIR}/*.npz")}
    gold_uids = [u for u in gold.index if u in have]
    train_uids = [u for u in have if u not in set(gold_uids) and u in y_map]
    print(f"cache={len(have)} train={len(train_uids)} gold_val={len(gold_uids)} dev={dev}")

    # pos_weight from soft-label prevalence (mild imbalance correction)
    Ytr = np.stack([y_map[u] for u in train_uids])
    p = Ytr.mean(0).clip(1e-3, 1 - 1e-3)
    pos_w = torch.tensor(((1 - p) / p).clip(1, 10), dtype=torch.float32, device=dev)
    crit = nn.BCEWithLogitsLoss(pos_weight=pos_w)

    dl = DataLoader(CacheDataset(train_uids, CACHE_DIR, y_map), batch_size=BS,
                    shuffle=True, num_workers=NUM_WORK, pin_memory=(dev == "cuda"), drop_last=True)
    Yg = gold.loc[gold_uids, LABEL_COLS].values.astype(int)

    model = StudyModel(BACKBONE).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    scaler = torch.cuda.amp.GradScaler(enabled=(dev == "cuda"))

    ckpt = f"{WORK}/ckpt_{BACKBONE}.pt"
    start, best = 0, -1.0
    if os.path.exists(ckpt):
        s = torch.load(ckpt, map_location=dev)
        model.load_state_dict(s["model"]); opt.load_state_dict(s["opt"])
        start, best = s["epoch"] + 1, s.get("best", -1)
        print(f"resumed from epoch {start} (best {best:.4f})")

    for ep in range(start, EPOCHS):
        model.train(); t0 = time.time(); tot = 0.0
        for x, planes, y in dl:
            x, planes, y = x.to(dev), planes.to(dev), y.to(dev)
            opt.zero_grad()
            with torch.cuda.amp.autocast(enabled=(dev == "cuda")):
                loss = crit(model(x, planes), y)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            tot += loss.item() * x.size(0)
        # validate on gold
        model.eval(); preds = []
        with torch.no_grad():
            for u in gold_uids:
                d = np.load(f"{CACHE_DIR}/{u}.npz")
                x = torch.from_numpy(d["x"].astype(np.float32))[None].to(dev)
                pl = torch.from_numpy(d["planes"].astype(np.int64))[None].to(dev)
                preds.append(torch.sigmoid(model(x, pl))[0].cpu().numpy())
        auc = macro_auc(Yg, np.stack(preds))
        print(f"ep{ep} loss={tot/len(train_uids):.4f} gold_macro_auc={auc:.4f} ({time.time()-t0:.0f}s)")
        state = {"model": model.state_dict(), "opt": opt.state_dict(),
                 "epoch": ep, "best": max(best, auc), "auc": auc, "backbone": BACKBONE}
        torch.save(state, ckpt)
        if auc > best:
            best = auc; torch.save(state, f"{WORK}/best_{BACKBONE}.pt")
    print(f"DONE best gold macro-AUC = {best:.4f}")


if __name__ == "__main__":
    main()
