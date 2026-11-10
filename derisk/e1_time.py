"""E1-T — planner headroom on the E1 workload (the pre-registered protocol Entry 5). Timed: run only under E:/Project-git/CPU_LOCK.txt.

Per cell (instance, noise, S, query): factor build per pushdown position (1 warm-up + 5 timed, median); every plan of
the tree plan space (position below the merge x connected binary join trees x format per node x granularity) folded on
the built factors: 1 warm-up, then 5 timed repetitions (median) unless the warm-up exceeds PRUNE x the best median so
far (then the plan is recorded as pruned with its warm-up time; it cannot be the best). Cap 120 s / memory caps -> DNF
at 120 s. Every timed plan's answer is V-checked against the first plan's within the certified tau_eq.
Fixed rules (a)-(f), planner v1 (calibrated cost model, exact cut statistics) and planner v1-generic (same model,
nnz-only statistics) are looked up in the timed table; planning time is measured (median of 5) and added.
Usage: python derisk/e1_time.py calibrate | run [instance ids]   Results: derisk/results/E1/T/*.json
"""
import os
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[v] = "1"
import json, pickle, statistics, sys, time
from pathlib import Path

import numpy as np
import psutil

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from knitplan import engine as E, planner as P
from knitplan.shots import load_tables

OUT = ROOT / "derisk" / "results" / "E1" / "T"; OUT.mkdir(parents=True, exist_ok=True)
CM = ROOT / "derisk" / "results" / "E1" / "cost_model.json"
CAP, REPS, PRUNE = 120.0, 5, 5.0
SEED = 20261108


def queries(n, idx):
    rng = np.random.default_rng(SEED + idx)
    A10 = sorted(rng.choice(n, size=10, replace=False).tolist())
    A20 = sorted(rng.choice(n, size=20, replace=False).tolist())
    Ah = sorted(rng.choice(n, size=n // 2, replace=False).tolist())
    Ag = sorted(rng.choice(n, size=10, replace=False).tolist())
    rest = [q for q in range(n) if q not in Ag]
    sel = {int(q): int(rng.integers(2)) for q in rng.choice(rest, size=5, replace=False)}
    out = {"marg10": (A10, None), "marg20": (A20, None), "groupby10sel5": (Ag, sel)}
    if n // 2 > 20:
        out["marghalf"] = (Ah, None)
    return out


def med_time(fn, reps=REPS):
    ts = []
    for _ in range(reps):
        t = time.perf_counter(); fn(); ts.append(time.perf_counter() - t)
    return statistics.median(ts), ts


def to_tree(plan: E.Plan):
    return P.TreePlan(plan.position, P.left_deep(plan.order), plan.fmts, plan.gran)


def cell(path, iid, noise, S, qname, A, sel, models, tabs, c, model):
    load0 = psutil.cpu_percent(interval=0.5)
    Aset = set(A)
    builds = {}
    for pos in ("before", "after"):
        fn = lambda: {m.p: E.build_factor(m, tabs[m.p], S, A=Aset, sel=sel, position=pos) for m in models}
        fs = fn()
        med, ts = med_time(fn)
        builds[pos] = (fs, med)
    tau, B = E.tau_eq(models, tabs, S, Aset, sel, c)
    fs_b = builds["before"][0]
    plans = P.enumerate_tree_plans(fs_b, positions=("before",))
    # heuristics / planners (computed before timing so their plans are timed even if outside the enumerated set)
    dplan = E.default_rule_from_factors(fs_b)
    plan_med_a, _ = med_time(lambda: E.default_rule_from_factors(fs_b))
    fr = P.fixed_rules(fs_b, models)
    v1, v1_cost, _ = P.choose(fs_b, model)
    v1_t, _ = med_time(lambda: P.choose(fs_b, model))
    vg, vg_cost, _ = P.choose_generic(fs_b, model)
    vg_t, _ = med_time(lambda: P.choose_generic(fs_b, model))
    heur = {"a_default_rule": (to_tree(dplan), plan_med_a),
            "d_pushdown_after_merge": (P.TreePlan("after", P.left_deep(dplan.order), dplan.fmts, dplan.gran), plan_med_a)}
    for k, p in fr.items():
        heur[k] = (p, 0.0)
    planners = {"v1": (v1, v1_t), "v1_generic": (vg, vg_t)}
    extra = [h[0] for h in heur.values()] + [p[0] for p in planners.values()]
    keys = {p.key() for p in plans}
    for p in extra:
        if p.key() not in keys:
            plans.append(p); keys.add(p.key())
    res, ref, best = {}, None, None
    for plan in plans:
        fs = builds[plan.position][0]
        try:
            t = time.perf_counter(); o, i, v = P.fold_tree(plan, fs, c); w = time.perf_counter() - t
            if w > CAP:
                res[plan.key()] = {"dnf": f"warm-up {w:.1f}s > cap", "total": CAP, "fold": CAP}; continue
            ii, vv = E.canonical(o, i, v, A)
            ans = dict(zip(ii.tolist(), vv.tolist()))
            if ref is None:
                ref = ans
            diff = max([abs(ans.get(z, 0) - ref.get(z, 0)) for z in set(ans) | set(ref)] + [0.0])
            if best is not None and w > PRUNE * best:
                res[plan.key()] = {"pruned_warmup": w, "fold": w, "build": builds[plan.position][1],
                                   "total": builds[plan.position][1] + w, "vcheck_diff": diff, "vcheck_pass": diff <= tau}
                continue
            fm, ts = med_time(lambda: P.fold_tree(plan, fs, c))
            best = fm if best is None else min(best, fm)
            res[plan.key()] = {"build": builds[plan.position][1], "fold": fm, "fold_reps": ts, "warmup": w,
                               "total": builds[plan.position][1] + fm, "vcheck_diff": diff, "vcheck_pass": diff <= tau}
        except E.MemoryCap as e:
            res[plan.key()] = {"dnf": "memory cap: " + str(e), "total": CAP, "fold": CAP}
        except MemoryError as e:
            res[plan.key()] = {"dnf": "system out of memory: " + str(e)[:120], "total": CAP, "fold": CAP}
    space = [p.key() for p in P.enumerate_tree_plans(fs_b, positions=("before",))]
    best_key = min(space, key=lambda k: res[k]["total"])
    best_total = res[best_key]["total"]
    htimes = {h: {"plan": p.key(), "total": res[p.key()]["total"] + extra_t, "planning": extra_t,
                  "dnf": "dnf" in res[p.key()]} for h, (p, extra_t) in heur.items()}
    ptimes = {h: {"plan": p.key(), "total": res[p.key()]["total"] + t_, "planning": t_, "dnf": "dnf" in res[p.key()]}
              for h, (p, t_) in planners.items()}
    bh = min(htimes, key=lambda h: htimes[h]["total"])
    out = {"id": iid, "noise": noise, "S": S, "query": qname, "A": A, "sel": sel, "c": c, "r": len(models),
           "tau_eq_certified": tau, "B": B, "plan_space_size": len(space), "plans": res, "heuristics": htimes,
           "planners": ptimes, "v1_predicted_cost": v1_cost, "best_plan": best_key, "best_total": best_total,
           "best_heuristic": bh, "ratio_best_heuristic_over_best": htimes[bh]["total"] / best_total,
           "ratio_v1_over_best": ptimes["v1"]["total"] / best_total,
           "ratio_v1_over_best_heuristic": ptimes["v1"]["total"] / htimes[bh]["total"],
           "all_vcheck_pass": all(r.get("vcheck_pass", True) for r in res.values()),
           "load_before": load0, "load_after": psutil.cpu_percent(interval=0.5),
           "affinity": psutil.Process().cpu_affinity(), "lock_note": os.environ.get("E1_LOCK_NOTE", "")}
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(out, indent=1, default=lambda x: x.item() if isinstance(x, np.generic) else str(x)))
    os.replace(tmp, path)
    return out


def main():
    try:
        psutil.Process().cpu_affinity([int(os.environ.get("E1_CORE", "4"))])  # calibration is pinned too
    except Exception:
        pass
    if sys.argv[1] == "calibrate":
        model, fit = P.calibrate(seconds_budget=float(sys.argv[2]) if len(sys.argv) > 2 else 300.0)
        CM.write_text(json.dumps({"k": model.k, "fit": fit, "date": time.strftime("%Y-%m-%d %H:%M")}, indent=1))
        print(json.dumps({"k": model.k, "fit": fit}, indent=1)); return
    try:
        psutil.Process().cpu_affinity([int(os.environ.get("E1_CORE", "4"))])
    except Exception:
        pass
    model = P.CostModel(json.loads(CM.read_text())["k"])
    only = set(sys.argv[2].split(",")) if len(sys.argv) > 2 else set()
    man = json.loads((ROOT / "bench" / "E1_MANIFEST.json").read_text())
    for idx, inst in enumerate(man["instances"]):
        if inst["stratum"] not in ("T", "TW") or (only and inst["id"] not in only):
            continue
        d = pickle.load(open(f"D:/runs/knitplan/e1/inst/{inst['id']}.pkl", "rb"))
        sch = d["scheme"]; models = E.sub_models(sch)
        for noise in inst["noise"]:
            for S in inst["budgets"]:
                tabs = {p: load_tables(f"D:/runs/knitplan/e1/shots/{inst['id']}/{noise}/sub{p}_S{S}.npz")[2]
                        for p in range(sch.r)}
                for qname, (A, sel) in queries(sch.n, idx).items():
                    path = OUT / f"{inst['id']}_{noise}_S{S}_{qname}.json"
                    if path.exists():
                        continue
                    t = time.time()
                    o = cell(path, inst["id"], noise, S, qname, A, sel, models, tabs, sch.c, model)
                    print(inst["id"], noise, S, qname, "space", o["plan_space_size"], "best", o["best_plan"],
                          f"{o['best_total']:.4f}", "bestH", o["best_heuristic"],
                          f"H/best {o['ratio_best_heuristic_over_best']:.2f} v1/best {o['ratio_v1_over_best']:.2f}",
                          "vcheck", o["all_vcheck_pass"], f"{time.time()-t:.0f}s", flush=True)


if __name__ == "__main__":
    main()
