"""E1-W — support in the wide-subcircuit stratum (the protocol; H1 / K-SPARSE ladder, the pre-registered protocol Entry 5). No timings.

Per (instance, noise, S): merged-factor support per subcircuit relative to 4^{|delta(j)|} 2^{m_j}; pushed-down factor
support for each E1 query relative to 4^{|delta(j)|} 2^{|A cap O_j|} (with |A cap O_j| recorded); flat answer support of the
sparse left-deep default plan when it fits the row cap. Checkpointed per unit: derisk/results/E1/W/*.json.
Usage: python derisk/e1_support.py [workers]
"""
_RUNS = __import__("os").environ.get("KNITPLAN_RUNS", "D:/runs/knitplan")   # data root
import json, os, pickle, sys, time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from knitplan import engine as E
from knitplan.shots import load_tables

OUT = ROOT / "derisk" / "results" / "E1" / "W"; OUT.mkdir(parents=True, exist_ok=True)
SEED = 20261108


def query_sets(n, idx):
    rng = np.random.default_rng(SEED + idx)
    sizes = sorted(set([10, 20, n // 2, n]))
    return {f"marg{k}": sorted(rng.choice(n, size=k, replace=False).tolist()) for k in sizes}


def unit(args):
    iid, idx, noise, S = args
    path = OUT / f"{iid}_{noise}_S{S}.json"
    if path.exists():
        return iid, noise, S, "skip"
    d = pickle.load(open(f"{_RUNS}/e1/inst/{iid}.pkl", "rb"))
    sch = d["scheme"]; models = E.sub_models(sch)
    tabs = {p: load_tables(f"{_RUNS}/e1/shots/{iid}/{noise}/sub{p}_S{S}.npz")[2] for p in range(sch.r)}
    rec = {"id": iid, "noise": noise, "S": S, "c": sch.c, "r": sch.r, "n": sch.n, "merged": {}, "queries": {}}
    for m in models:
        f = E.build_factor(m, tabs[m.p], S, A=None, sel=None, position="before")
        rec["merged"][m.p] = {"deg": len(m.deg_cuts), "m": len(m.out_qubits), "nnz": f.nnz(),
                              "ratio": f.nnz() / f.dense_shape()}
    for k, A in query_sets(sch.n, idx).items():
        q = {"A": A, "phi": {}}
        for m in models:
            f = E.build_factor(m, tabs[m.p], S, A=set(A), sel=None, position="before")
            q["phi"][m.p] = {"deg": len(m.deg_cuts), "kept": len(f.outs), "nnz": f.nnz(),
                             "ratio": f.nnz() / f.dense_shape()}
        try:
            t0 = time.perf_counter()
            fs = {m.p: E.build_factor(m, tabs[m.p], S, A=set(A), sel=None, position="before") for m in models}
            plan = E.default_rule_from_factors(fs)
            plan = E.Plan("before", plan.order, ("S",) * len(plan.fmts), plan.gran)
            st = E.RunStats()
            outs, i, v = E.fold(plan, fs, sch.c, stats=st)
            q["intermediates"] = [{"step": s, "nnz": a, "dense": b, "ratio": a / b} for s, a, b in st.inter]
            q["answer"] = {"nnz": int(i.size), "ratio": i.size / 2 ** len(A)}
            q["seconds"] = time.perf_counter() - t0
        except (E.MemoryCap, MemoryError) as e:
            q["intermediates"] = "DNF " + str(e)[:200]
        rec["queries"][k] = q
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(rec, indent=1)); os.replace(tmp, path)
    return iid, noise, S, "done"


def main():
    man = json.loads((ROOT / "bench" / "E1_MANIFEST.json").read_text())
    jobs = []
    for idx, inst in enumerate(man["instances"]):
        for noise in inst["noise"]:
            for S in inst["budgets"]:
                if Path(f"{_RUNS}/e1/shots/{inst['id']}/{noise}/sub{inst['r']-1}_S{S}.npz").exists():
                    jobs.append((inst["id"], idx, noise, S))
    with Pool(int(sys.argv[1]) if len(sys.argv) > 1 else 2) as pool:
        for r in pool.imap_unordered(unit, jobs):
            print(*r, flush=True)


if __name__ == "__main__":
    main()
