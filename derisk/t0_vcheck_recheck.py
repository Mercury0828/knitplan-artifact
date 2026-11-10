"""Recompute the certified tau_eq (after the the tolerance analysis) for every D3 cell and recheck its stored V-check
differences. Output derisk/results/T0_vcheck_R2.json."""
_RUNS = __import__("os").environ.get("KNITPLAN_RUNS", "D:/runs/knitplan")   # data root
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
import glob, json, pickle, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from knitplan import engine as E
from knitplan.shots import load_tables

rows, cache = [], {}
for f in sorted(glob.glob(str(ROOT / "derisk/results/D3/*.json"))):
    c = json.loads(Path(f).read_text())
    if c["id"] not in cache:
        d = pickle.load(open(f"{_RUNS}/pilot/{c['id']}.pkl", "rb")); cache = {c["id"]: (d["scheme"], E.sub_models(d["scheme"]))}
    sch, models = cache[c["id"]]
    tabs = {p: load_tables(f"{_RUNS}/shots/{c['id']}/{c['noise']}/sub{p}_S{c['S']}.npz")[2] for p in range(sch.r)}
    sel = {int(k): v for k, v in c["sel"].items()} if c["sel"] else None
    tau, B = E.tau_eq(models, tabs, c["S"], set(c["A"]), sel, sch.c)
    md = max(p["vcheck_diff"] for p in c["plans"].values() if "vcheck_diff" in p)
    rows.append({"cell": f"{c['id']}/{c['noise']}/S{c['S']}/{c['query']}", "tau": tau, "B": B, "max_diff": md, "pass": md <= tau})
out = {"date": time.strftime("%Y-%m-%d"), "cells": len(rows), "fails": [r["cell"] for r in rows if not r["pass"]],
       "max_diff_over_tau": max(r["max_diff"] / r["tau"] for r in rows if r["tau"] > 0), "rows": rows}
(ROOT / "derisk/results/T0_vcheck_R2.json").write_text(json.dumps(out, indent=1))
print(out["cells"], out["fails"], out["max_diff_over_tau"])
