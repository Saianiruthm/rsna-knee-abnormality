#!/usr/bin/env python3
"""Resumable, rate-limited report labeler via NVIDIA NIM (OpenAI-compatible).
Usage: label_nvidia.py --dir DIR --uids UIDS.txt --out OUTDIR [--model M] [--rpm 40] [--workers 8] [--limit N]
Reads NVIDIA_API_KEY from env. Writes OUTDIR/<uid>.json per report; skips existing (resume)."""
import os, sys, json, time, re, argparse, threading, glob
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI

LABELS = ["ACL","MCL","Medial Meniscus","Lateral Meniscus","Medial OA","Lateral OA",
          "PF OA","Effusion","Synovitis","Baker's","Contusion","Fracture"]

PROMPT_HEAD = """You are a musculoskeletal (MSK) radiologist. Below is ONE knee-MRI report. It may be in ANY language (Spanish, Turkish, Greek, Bulgarian, Dutch, German, Croatian, English, ...). Read it in its original language; do not translate first.

For each of the 12 findings below, decide the PROBABILITY (0.00-1.00) that the finding is present in THIS knee, based ONLY on what the report states. Reason from clinical meaning, NOT keyword matching. Calibrate:
  explicit complete tear / clearly present            -> 0.90-1.00
  partial tear / grade-2 / 'in favor of' / suspected  -> 0.55-0.85
  mild / grade-1 / 'possible' / 'cannot exclude'      -> 0.25-0.50
  degeneration / signal change WITHOUT a tear         -> low unless it implies the finding
  explicitly normal / intact / 'no ...'               -> 0.00-0.05
  report SILENT on that structure                     -> ~0.10 (a low prior, NOT 0.5)

The 12 findings (clinical meaning):
  ACL              - anterior cruciate ligament tear/rupture/sprain
  MCL              - medial collateral ligament injury/sprain/tear
  Medial Meniscus  - medial meniscus tear
  Lateral Meniscus - lateral meniscus tear
  Medial OA        - osteoarthritis / cartilage loss in the MEDIAL tibiofemoral compartment
  Lateral OA       - osteoarthritis / cartilage loss in the LATERAL tibiofemoral compartment
  PF OA            - patellofemoral OA / chondromalacia / trochlear-or-patellar cartilage loss
  Effusion         - joint effusion (clinically noted joint fluid)
  Synovitis        - synovitis / synovial thickening / reactive synovium / hoffitis
  Baker's          - Baker's / popliteal cyst
  Contusion        - bone contusion / bone bruise / traumatic bone-marrow edema
  Fracture         - any fracture (subchondral, osteochondral, impaction, avulsion, insufficiency, cortical)

Respond with ONLY a JSON object (no markdown, no commentary), keys exactly:
{"language":"<detected language>", %s}
where each finding maps to a number in [0,1].

REPORT:
""" % ", ".join(f'"{l}":0.0' for l in LABELS)

class RateLimiter:
    def __init__(self, rpm):
        self.interval = 60.0 / rpm; self.lock = threading.Lock(); self.next = time.monotonic()
    def wait(self):
        with self.lock:
            now = time.monotonic()
            if now < self.next: time.sleep(self.next - now)
            self.next = max(now, self.next) + self.interval

def parse_json(txt):
    m = re.search(r"\{.*\}", txt, re.S)
    if not m: raise ValueError("no json object in response")
    return json.loads(m.group(0))

USAGE = {"in": 0, "out": 0, "lock": threading.Lock()}

def label_one(client, model, report, uid, retries=6, max_tokens=400, thinkoff=False):
    msg = [{"role": "user", "content": PROMPT_HEAD + report}]
    extra = {"reasoning_effort": "none"} if thinkoff else {}
    last = None
    for a in range(retries):
        try:
            r = client.chat.completions.create(model=model, messages=msg, temperature=0.0,
                                               max_tokens=max_tokens, extra_body=extra)
            u = getattr(r, "usage", None)
            if u:
                with USAGE["lock"]:
                    USAGE["in"] += u.prompt_tokens or 0; USAGE["out"] += u.completion_tokens or 0
            o = parse_json(r.choices[0].message.content or "")
            out = {"uid": uid, "language": str(o.get("language", ""))}
            for l in LABELS:
                v = float(o.get(l, 0.1)); out[l] = min(1.0, max(0.0, v))
            return out
        except Exception as e:
            last = e
            wait = min(30, 3 * (2 ** a)) if "429" in str(e) else 1.5 * (a + 1)
            time.sleep(wait)
    return {"uid": uid, "language": "", "_error": str(last), **{l: 0.5 for l in LABELS}}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True); ap.add_argument("--uids", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--model", default="qwen/qwen2.5-72b-instruct")
    ap.add_argument("--rpm", type=int, default=40); ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--keyenv", default="NVIDIA_API_KEY")
    ap.add_argument("--base_url", default="https://integrate.api.nvidia.com/v1")
    ap.add_argument("--max_tokens", type=int, default=400)
    ap.add_argument("--thinkoff", action="store_true")
    args = ap.parse_args()
    key = os.environ.get(args.keyenv)
    if not key: sys.exit(f"{args.keyenv} not set")
    os.makedirs(args.out, exist_ok=True)
    client = OpenAI(base_url=args.base_url, api_key=key, timeout=90, max_retries=0)
    rl = RateLimiter(args.rpm)
    uids = [l.strip() for l in open(args.uids) if l.strip()]
    todo = [u for u in uids if not os.path.exists(f"{args.out}/{u}.json")]
    if args.limit: todo = todo[:args.limit]
    print(f"model={args.model} total={len(uids)} done={len(uids)-len(todo)} todo={len(todo)} rpm={args.rpm}", flush=True)
    done = err = 0; t0 = time.time()
    def work(uid):
        rl.wait()
        rep = open(f"{args.dir}/reports/{uid}.txt", encoding="utf-8").read()
        res = label_one(client, args.model, rep, uid, max_tokens=args.max_tokens, thinkoff=args.thinkoff)
        json.dump(res, open(f"{args.out}/{uid}.json", "w"))
        return "_error" in res
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(work, u): u for u in todo}
        for i, f in enumerate(as_completed(futs), 1):
            if f.result(): err += 1
            done += 1
            if done % 25 == 0 or done == len(todo):
                el = time.time() - t0; rate = done / el * 60 if el else 0
                print(f"  {done}/{len(todo)}  err={err}  {rate:.0f}/min  eta={(len(todo)-done)/rate if rate else 0:.1f}min", flush=True)
    el = time.time() - t0
    n = max(1, done - err)
    print(f"finished: {done} processed, {err} errors, {el:.0f}s", flush=True)
    print(f"USAGE tokens in={USAGE['in']} out={USAGE['out']} | per-ok-report in={USAGE['in']/n:.0f} out={USAGE['out']/n:.0f}", flush=True)

if __name__ == "__main__":
    main()
