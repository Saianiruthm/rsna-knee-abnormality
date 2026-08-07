import sys, os, json, glob
import pandas as pd, numpy as np
D = sys.argv[1]
OUTDIR = sys.argv[2] if len(sys.argv) > 2 else f"{D}/out"
LABELS = ["ACL","MCL","Medial Meniscus","Lateral Meniscus","Medial OA","Lateral OA",
          "PF OA","Effusion","Synovitis","Baker's","Contusion","Fracture"]

def auc(y, s):
    y = np.asarray(y); s = np.asarray(s, float)
    P = (y == 1).sum(); N = (y == 0).sum()
    if P == 0 or N == 0: return float('nan')
    order = np.argsort(s, kind='mergesort'); r = np.empty(len(s)); r[order] = np.arange(1, len(s)+1)
    # average ranks for ties
    _, inv, cnt = np.unique(s, return_inverse=True, return_counts=True)
    csum = np.cumsum(cnt); start = csum - cnt
    avg = (start + csum + 1) / 2.0
    r = avg[inv]
    return (r[y == 1].sum() - P*(P+1)/2) / (P*N)

key = pd.read_csv(f"{D}/gold_key.csv").set_index("StudyInstanceUID")
rows = {}
for f in glob.glob(f"{OUTDIR}/*.json"):
    try:
        o = json.load(open(f)); rows[o["uid"]] = o
    except Exception as e:
        print("bad file", f, e)
pred = pd.DataFrame([{**{"StudyInstanceUID": u}, **{c: rows[u].get(c, 0.5) for c in LABELS}} for u in rows]).set_index("StudyInstanceUID")
common = key.index.intersection(pred.index)
print(f"gold={len(key)}  labeled={len(pred)}  scored={len(common)}")
missing = set(key.index) - set(pred.index)
if missing: print(f"MISSING {len(missing)} (not yet labeled)")
key, pred = key.loc[common, LABELS].astype(int), pred.loc[common, LABELS].astype(float)
aucs = []
print(f"\n{'label':16s} {'gold+':>5s} {'AUC':>6s}")
for c in LABELS:
    a = auc(key[c].values, pred[c].values); aucs.append(a)
    print(f"{c:16s} {int(key[c].sum()):5d} {a:6.3f}")
macro = np.nanmean(aucs)
print(f"\nMACRO-AUC (competition metric) = {macro:.4f}")
