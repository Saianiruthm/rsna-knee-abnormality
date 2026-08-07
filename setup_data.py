#!/usr/bin/env python3
"""Regenerate the competition data + derived files on a fresh machine.
Requires Kaggle API creds (~/.kaggle/kaggle.json or KAGGLE_USERNAME/KAGGLE_KEY env)
and that you've accepted the competition rules on kaggle.com.
Run:  uv run --with pandas --with kaggle python setup_data.py
Produces: train.csv, train_series.csv, test_series.csv, sample_submission.csv,
          reports/<uid>.txt, gold_key.csv, gold_sids.txt, all_sids.txt
Does NOT redistribute data — it downloads from Kaggle under your account."""
import os, subprocess, sys
import pandas as pd
COMP = "rsna-knee-abnormality-detection"
D = os.path.dirname(os.path.abspath(__file__))
LABELS = ["ACL","MCL","Medial Meniscus","Lateral Meniscus","Medial OA","Lateral OA",
          "PF OA","Effusion","Synovitis","Baker's","Contusion","Fracture"]

def dl(fn):
    if os.path.exists(f"{D}/{fn}"): return
    print("downloading", fn)
    subprocess.run(["kaggle","competitions","download","-f",fn,"-c",COMP,"-p",D], check=True)
    z = f"{D}/{fn}.zip"
    if os.path.exists(z):
        subprocess.run(["unzip","-o",z,"-d",D], check=True); os.remove(z)

for f in ["train.csv","train_series.csv","test_series.csv","sample_submission.csv","test.csv"]:
    try: dl(f)
    except Exception as e: print("WARN", f, e)

tr = pd.read_csv(f"{D}/train.csv")
os.makedirs(f"{D}/reports", exist_ok=True)
for _, r in tr.iterrows():
    rep = str(r["Report"]) if pd.notna(r["Report"]) else ""
    open(f"{D}/reports/{r['StudyInstanceUID']}.txt","w",encoding="utf-8").write(rep)
gold = tr[tr[LABELS].notna().all(axis=1)].copy()
gold[["StudyInstanceUID"]+LABELS].astype({c:int for c in LABELS}).to_csv(f"{D}/gold_key.csv", index=False)
open(f"{D}/gold_sids.txt","w").write("\n".join(gold["StudyInstanceUID"])+"\n")
open(f"{D}/all_sids.txt","w").write("\n".join(tr["StudyInstanceUID"])+"\n")
print(f"OK: {len(tr)} studies, {len(gold)} gold. reports/, gold_key.csv, *_sids.txt regenerated.")
