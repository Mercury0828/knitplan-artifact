"""Untimed equivalence test of bench/systems.py: every system vs KNITPLAN within the certified tau_eq (V1, all entries).
Usage: python tools/test_systems.py   (light, single thread)"""
import json, pickle, sys, time
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "bench")); sys.path.insert(0, str(ROOT / "derisk"))
from knitplan import engine as E, planner as P
from knitplan.shots import load_tables
import systems as SY

CASES = [("pilot", "qaoa20a", "D:/runs/knitplan/pilot/{i}.pkl", "D:/runs/knitplan/shots/{i}/ideal/sub{p}_S{S}.npz", 10000),
         ("e1", "bv30r4w10", "D:/runs/knitplan/e1/inst/{i}.pkl", "D:/runs/knitplan/e1/shots/{i}/ideal/sub{p}_S{S}.npz", 1000),
         ("e1", "adder40r4w14", "D:/runs/knitplan/e1/inst/{i}.pkl", "D:/runs/knitplan/e1/shots/{i}/ideal/sub{p}_S{S}.npz", 1000)]
model = P.CostModel(json.loads((ROOT / "derisk/results/E1/cost_model.json").read_text())["k"])
fails = 0
ONLY = next((a.split("=", 1)[1].split(",") for a in sys.argv if a.startswith("--only=")), None)
ONLYQ = next((a.split("=", 1)[1].split(",") for a in sys.argv if a.startswith("--queries=")), None)
for tag, iid, ip, sp, S in CASES:
    d = pickle.load(open(ip.format(i=iid), "rb")); sch = d["scheme"]; circ = d.get("circ")
    tabs = {p: load_tables(sp.format(i=iid, p=p, S=S))[2] for p in range(sch.r)}
    rng = np.random.default_rng(7)
    A10 = sorted(rng.choice(sch.n, 10, replace=False).tolist())
    rest = [q for q in range(sch.n) if q not in A10]
    sel = {int(q): int(rng.integers(2)) for q in rng.choice(rest, 4, replace=False)}
    from knitplan import points as PT
    Zp, _ = PT.sample_points(E.sub_models(sch), tabs, sch.n, 1000, np.random.default_rng(11))
    for qn, A, s in [("marg10", A10, None), ("groupby10sel4", A10, sel), ("points1000", [], None)]:
        if ONLYQ and qn not in ONLYQ:
            continue
        ctx = SY.make_ctx(circ, sch, tabs, S, A, s, model)
        if qn.startswith("points"):
            ctx["points"] = Zp     # per-entry tolerance: tau_eq of the full answer (A = all qubits)
            tau, _ = E.tau_eq(ctx["models"], tabs, S, set(range(sch.n)), None, sch.c)
        else:
            tau, _ = E.tau_eq(ctx["models"], tabs, S, set(A), s, sch.c)
        ref = SY.knitplan(ctx)
        for name, fn in SY.SYSTEMS.items():
            if ONLY and name not in ONLY:
                continue
            if name in ("cutqc_fd", "cutqc_opt", "tensorqc_dense") and sch.n > 24 and "--full" not in sys.argv:
                continue   # light mode: full-distribution baselines need 2^n memory (n = 30: ~16 GB, ~23 min)
            t = time.perf_counter()
            try:
                ans = fn(ctx); dt = time.perf_counter() - t
                diff = SY.max_diff(ans, ref); ok = diff <= tau
                fails += not ok
                print(f"{iid:14s} {qn:14s} {name:16s} {dt:8.3f}s diff {diff:.2e} tau {tau:.2e} {'OK' if ok else 'FAIL'}", flush=True)
            except (SY.Infeasible, E.MemoryCap) as e:
                print(f"{iid:14s} {qn:14s} {name:16s} infeasible: {e}", flush=True)
print("FAILS", fails)
