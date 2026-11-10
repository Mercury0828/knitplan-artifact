"""D1 — support measurement (the pre-registered protocol §D1). Pilot; nothing frozen.

Per (instance, noise, S): raw support per configuration; merged-factor support (per slice and total) relative to
4^{|delta(j)|} 2^{m_j}; pushed-down factor support for marginals |A| in {10, 20, n}; supports of the default-rule
plan's intermediates (sparse, row cap) and of the flat answer relative to 2^{|A|}.
Checkpointed per unit (one JSON per unit in derisk/results/D1/); a rerun skips finished units.
"""
import json, os, pickle, sys, time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from knitplan import engine as E
from knitplan.shots import load_tables

OUT = ROOT / "derisk" / "results" / "D1"; OUT.mkdir(parents=True, exist_ok=True)
BUDGETS = [1000, 10000, 100000]
SEED = 20261107


def query_sets(n, idx):
    rng = np.random.default_rng(SEED + idx)
    sizes = [10, 20] + ([n] if n > 20 else [])
    return {k: sorted(rng.choice(n, size=k, replace=False).tolist()) for k in sizes}


def unit(args):
    iid, idx, noise, S = args
    path = OUT / f"{iid}_{noise}_S{S}.json"
    if path.exists():
        return iid, noise, S, "skip"
    d = pickle.load(open(f"D:/runs/knitplan/pilot/{iid}.pkl", "rb"))
    sch = d["scheme"]; models = E.sub_models(sch)
    tabs, raw = {}, {}
    for p in range(sch.r):
        _, _, t = load_tables(f"D:/runs/knitplan/shots/{iid}/{noise}/sub{p}_S{S}.npz")
        tabs[p] = t
        w = len(sch.sub_qubits[p])
        ratios = [len(i) / min(S, 2 ** w) for i, _ in t.values()]
        raw[p] = {"width": w, "configs": len(t), "mean": float(np.mean(ratios)), "min": float(np.min(ratios)),
                  "max": float(np.max(ratios)), "mean_distinct": float(np.mean([len(i) for i, _ in t.values()]))}
    rec = {"id": iid, "noise": noise, "S": S, "c": sch.c, "r": sch.r, "n": sch.n, "raw": raw, "merged": {}, "queries": {}}
    for m in models:
        f = E.build_factor(m, tabs[m.p], S, A=None, sel=None, position="after")
        per = [len(i) / 2 ** len(m.out_qubits) for i, _ in f.data.values()]
        rec["merged"][m.p] = {"deg": len(m.deg_cuts), "m": len(m.out_qubits), "nnz": f.nnz(),
                              "ratio": f.nnz() / f.dense_shape(), "slice_min": float(min(per)),
                              "slice_mean": float(np.mean(per)), "slice_max": float(max(per))}
    for k, A in query_sets(sch.n, idx).items():
        q = {"A": A, "phi": {}}
        for m in models:
            f = E.build_factor(m, tabs[m.p], S, A=set(A), sel=None, position="before")
            q["phi"][m.p] = {"kept": len(f.outs), "nnz": f.nnz(), "ratio": f.nnz() / f.dense_shape()}
        try:
            t0 = time.perf_counter()
            plan = E.default_rule_plan(models, tabs, S, set(A), None)
            plan = E.Plan(plan.position, plan.order, ("S",) * len(plan.fmts), plan.gran)  # sparse: measure supports
            st = E.RunStats()
            outs, i, v = E.execute(plan, models, tabs, S, set(A), None, sch.c, stats=st)
            q["intermediates"] = [{"step": s, "nnz": a, "dense": b, "ratio": a / b} for s, a, b in st.inter]
            q["answer"] = {"nnz": int(i.size), "ratio": i.size / 2 ** len(A)}
            q["seconds"] = time.perf_counter() - t0
        except E.MemoryCap as e:
            q["intermediates"] = "DNF " + str(e)
        rec["queries"][k] = q
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(rec, indent=1)); os.replace(tmp, path)
    return iid, noise, S, "done"


def main():
    man = json.loads((ROOT / "bench" / "PILOT_MANIFEST.json").read_text())
    jobs = []
    for idx, inst in enumerate(man["instances"]):
        if inst["status"] != "ok":
            continue
        for noise in inst["noise"]:
            for S in BUDGETS:
                if Path(f"D:/runs/knitplan/shots/{inst['id']}/{noise}/sub{inst['r']-1}_S{S}.npz").exists():
                    jobs.append((inst["id"], idx, noise, S))
    with Pool(int(sys.argv[1]) if len(sys.argv) > 1 else 3) as pool:
        for r in pool.imap_unordered(unit, jobs):
            print(*r, flush=True)


if __name__ == "__main__":
    main()
