"""D3 — triviality screen (the pre-registered protocol §D3 + Entry 2). Pilot timings; nothing frozen.

Usage: python derisk/d3_screen.py [instance ids comma-separated] ; results per cell in derisk/results/D3/*.json
(checkpointed: finished cells are skipped on rerun).
"""
_RUNS = __import__("os").environ.get("KNITPLAN_RUNS", "D:/runs/knitplan")   # data root
import os
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[v] = "1"
import json, pickle, statistics, sys, time
from pathlib import Path

import numpy as np
import psutil

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from knitplan import engine as E
from knitplan.shots import load_tables

OUT = ROOT / "derisk" / "results" / "D3"; OUT.mkdir(parents=True, exist_ok=True)
BUDGETS = [1000, 10000, 100000]
CAP = 120.0
REPS = 5
SEED = 20261107
CONTROLS = {"bv20", "bv20r3"}


def queries(n, idx):
    rng = np.random.default_rng(SEED + 1000 + idx)
    A10 = sorted(rng.choice(n, size=10, replace=False).tolist())
    A20 = sorted(rng.choice(n, size=20, replace=False).tolist())
    Ag = sorted(rng.choice(n, size=10, replace=False).tolist())
    rest = [q for q in range(n) if q not in Ag]
    sel = {int(q): int(rng.integers(2)) for q in rng.choice(rest, size=5, replace=False)}
    return {"marg10": (A10, None), "marg20": (A20, None), "groupby10sel5": (Ag, sel)}


def timed(fn, reps=REPS):
    t = time.perf_counter(); out = fn(); w = time.perf_counter() - t
    ts = []
    for _ in range(reps):
        t = time.perf_counter(); out = fn(); ts.append(time.perf_counter() - t)
    return out, w, statistics.median(ts), ts


def norm(plan):
    o = list(plan.order)
    if len(o) > 1 and o[0] > o[1]:
        o[0], o[1] = o[1], o[0]
    return E.Plan(plan.position, tuple(o), plan.fmts, plan.gran, name=plan.name)


def answer_dict(outs, idx, val, A):
    i, v = E.canonical(outs, idx, val, A)
    return dict(zip(i.tolist(), v.tolist()))


def cell(iid, idx, noise, S, qname, A, sel, models, tabs, c):
    path = OUT / f"{iid}_{noise}_S{S}_{qname}.json"
    if path.exists():
        return json.loads(path.read_text())
    load0 = psutil.cpu_percent(interval=0.5)
    Aset = set(A)
    builds = {}
    for pos in ("before", "after"):
        fs, w, med, ts = timed(lambda: {m.p: E.build_factor(m, tabs[m.p], S, A=Aset, sel=sel, position=pos)
                                        for m in models})
        builds[pos] = (fs, med, ts)
    tau, W = E.tau_eq(models, tabs, S, Aset, sel, c)
    plans = E.enumerate_plans(len(models))
    res, ref = {}, None
    for plan in plans:
        fs = builds[plan.position][0]
        try:
            t = time.perf_counter(); o, i, v = E.fold(plan, fs, c); w = time.perf_counter() - t
            if w > CAP:
                res[plan.key()] = {"dnf": f"warm-up {w:.1f}s > cap", "total": CAP}; continue
            ans = answer_dict(o, i, v, A)
            if ref is None:
                ref = ans
            diff = max([abs(ans.get(z, 0) - ref.get(z, 0)) for z in set(ans) | set(ref)] + [0.0])
            ts = []
            for _ in range(REPS):
                t = time.perf_counter(); E.fold(plan, fs, c); ts.append(time.perf_counter() - t)
            fm = statistics.median(ts)
            res[plan.key()] = {"build": builds[plan.position][1], "fold": fm, "fold_reps": ts,
                               "total": builds[plan.position][1] + fm, "vcheck_diff": diff, "vcheck_pass": diff <= tau}
        except E.MemoryCap as e:
            res[plan.key()] = {"dnf": "memory cap: " + str(e), "total": CAP}
        except MemoryError as e:  # the shared machine could not allocate (counted like the memory cap)
            res[plan.key()] = {"dnf": "system out of memory: " + str(e)[:120], "total": CAP}
    # heuristics
    fs_b = builds["before"][0]
    dplan, _, plan_med, _ = timed(lambda: E.default_rule_from_factors(fs_b))
    dplan = norm(dplan)
    cq = tuple(sorted(range(len(models)), key=lambda p: len(models[p].out_qubits)))
    heur = {
        "a_default_rule": (dplan, plan_med),
        "b_always_dense_pushdown": (norm(E.Plan("before", cq, ("D",) * (len(models) - 1), "joint")), 0.0),
        "c_always_sparse_pushdown_cutqc_order": (norm(E.Plan("before", cq, ("S",) * (len(models) - 1), "joint")), 0.0),
        "d_pushdown_after_merge": (E.Plan("after", dplan.order, dplan.fmts, dplan.gran), plan_med),
    }
    htimes = {}
    for h, (p, extra) in heur.items():
        r = res[p.key()]
        htimes[h] = {"plan": p.key(), "total": r["total"] + extra, "dnf": "dnf" in r, "planning": extra}
    best_key = min(res, key=lambda k: res[k]["total"])
    best = res[best_key]["total"]
    bh = min(htimes, key=lambda h: htimes[h]["total"])
    dense_best = min((res[k]["total"] for k in res if set(k.split("|")[2]) == {"D"}), default=None)
    sparse_best = min((res[k]["total"] for k in res if "S" in k.split("|")[2]), default=None)
    out = {"id": iid, "noise": noise, "S": S, "query": qname, "A": A, "sel": sel, "c": c, "r": len(models),
           "tau_eq": tau, "plans": res, "heuristics": htimes, "best_plan": best_key, "best_total": best,
           "best_heuristic": bh, "best_heuristic_total": htimes[bh]["total"],
           "ratio_best_heuristic_over_best": htimes[bh]["total"] / best,
           "dense_best": dense_best, "sparse_best": sparse_best,
           "all_vcheck_pass": all(r.get("vcheck_pass", True) for r in res.values()),
           "load_before": load0, "load_after": psutil.cpu_percent(interval=0.5),
           "affinity": psutil.Process().cpu_affinity()}
    path.write_text(json.dumps(out, indent=1, default=lambda x: x if not isinstance(x, np.generic) else x.item()))
    return out


def main():
    try:
        psutil.Process().cpu_affinity([int(os.environ.get("D3_CORE", "4"))])  # one P-core logical CPU (P-cores = logical 0-15)
    except Exception:
        pass
    only = set(sys.argv[1].split(",")) if len(sys.argv) > 1 and sys.argv[1] else set()
    man = json.loads((ROOT / "bench" / "PILOT_MANIFEST.json").read_text())
    for idx, inst in enumerate(man["instances"]):
        if inst["status"] != "ok" or inst["id"] in CONTROLS or (only and inst["id"] not in only):
            continue
        d = pickle.load(open(f"{_RUNS}/pilot/{inst['id']}.pkl", "rb"))
        sch = d["scheme"]; models = E.sub_models(sch)
        for noise in inst["noise"]:
            for S in BUDGETS:
                if not Path(f"{_RUNS}/shots/{inst['id']}/{noise}/sub{sch.r-1}_S{S}.npz").exists():
                    print(inst["id"], noise, S, "shots missing, skipped this pass", flush=True); continue
                tabs = {p: load_tables(f"{_RUNS}/shots/{inst['id']}/{noise}/sub{p}_S{S}.npz")[2]
                        for p in range(sch.r)}
                for qname, (A, sel) in queries(sch.n, idx).items():
                    t = time.time()
                    for attempt in range(4):  # the shared machine sometimes runs out of RAM (other projects)
                        try:
                            o = cell(inst["id"], idx, noise, S, qname, A, sel, models, tabs, sch.c); break
                        except MemoryError:
                            print("system out of memory; retry in 180 s", flush=True); time.sleep(180)
                    else:
                        print(inst["id"], noise, S, qname, "SKIPPED: repeated system OOM", flush=True); continue
                    print(inst["id"], noise, S, qname, "best", o["best_plan"], f"{o['best_total']:.4f}",
                          "bestH", o["best_heuristic"], f"ratio {o['ratio_best_heuristic_over_best']:.2f}",
                          "vcheck", o["all_vcheck_pass"], f"{time.time()-t:.0f}s", flush=True)


if __name__ == "__main__":
    main()
