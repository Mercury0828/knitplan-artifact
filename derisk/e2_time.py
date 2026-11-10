"""E2-T — planner headroom on the held-out E2 workload (the pre-registered protocol Entries 6, 6a, 6b). Timed: run only under
E:/Project-git/CPU_LOCK.txt, pinned to one P-core (E2_CORE, default 4).

Per cell (instance, noise, S, query):
1. factor build per pushdown position (1 warm-up + 5 timed, median); certified tau_eq.
2. Named plans: fixed rules (a)-(f), planner v1 (exact cut statistics), v1-generic (nnz-only statistics) and planner v2
   (budgeted, rho = 0.1); planning time median of 5 calls, added to the plan's total. Each named plan: 1 warm-up, then
   5 timed repetitions (median); never pruned. Per-plan cap CAP s (warm-up) or a memory cap -> DNF, counted at CAP.
3. Triviality (Entry 6b): the cell is NON-TRIVIAL iff rule (b)'s total (build + fold, timed in step 2) >= T_NT or rule (b)
   is DNF. Trivial cells stop here (named plans only; reported, outside the H3 population).
4. Non-trivial cells: the rest of the tree plan space is screened in ascending order of planner v1's predicted cost
   (exact statistics), 1 warm-up each, then 5 timed repetitions unless the warm-up exceeds PRUNE x the best median so
   far (pruned: cannot be the best); screening stops after SCREEN_BUDGET s of wall time, the remaining plans are
   recorded as unscreened. The oracle is the best total over the timed plans of the space (an upper bound on the true
   oracle's time, so H3 measured this way is conservative).
Every answer is V-checked against the first finished plan within the certified tau_eq.
Usage: python derisk/e2_time.py run [instance ids] | smoke <instance> <S> <query>   Results: derisk/results/E2/T/*.json
"""
import os
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[v] = "1"
import json, pickle, statistics, sys, time
from pathlib import Path

import numpy as np
import psutil

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "derisk"))
from knitplan import engine as E, planner as P
E.DENSE_CAP = 2 ** 28    # E1 / Entry 6 dense memory rule (2 GiB per array); three workers share 64 GB under the lock
from knitplan.shots import load_tables
from e1_time import queries as e1_queries, to_tree

OUT = ROOT / "derisk" / "results" / "E2" / "T"
CM = ROOT / "derisk" / "results" / "E1" / "cost_model.json"   # calibrated on the E1 development run, not refit
CAP, REPS, PRUNE = 60.0, 5, 5.0
T_NT = 0.1               # Entry 6b: non-trivial iff rule (b) total >= 0.1 s or DNF
SCREEN_BUDGET = 300.0    # Entry 6b: wall seconds of plan-space screening per non-trivial cell
REFTMP = Path("D:/runs/knitplan/e2_tmp")   # reference answers passed to respawned workers as files
SEED_M30 = 20261109      # marg30 drawn from its own stream so the E1-style queries are unchanged
RULE_B = "b_dense_cutqc_order"


def queries(n, idx):
    out = e1_queries(n, idx)
    rng = np.random.default_rng(SEED_M30 + idx)
    out["marg30"] = (sorted(rng.choice(n, size=30, replace=False).tolist()), None)
    return out


def med_time(fn, reps=REPS):
    ts, last = [], None
    for _ in range(reps):
        t = time.perf_counter(); last = fn(); ts.append(time.perf_counter() - t)
    return statistics.median(ts), ts, last


def vdiff(a, b):
    ka, va = a; kb, vb = b
    keys = np.concatenate([ka, kb]); vals = np.concatenate([va, -vb])
    if keys.size == 0:
        return 0.0
    u, inv = np.unique(keys, return_inverse=True)
    s = np.zeros(u.size); np.add.at(s, inv, vals)
    return float(np.abs(s).max())


def build_all(iid, idx, noise, S, qname):
    man = json.loads((ROOT / "bench" / "E2_MANIFEST.json").read_text())
    sch, models, tabs = load_inst(man["instances"][idx], noise, S)
    A, sel = queries(sch.n, idx)[qname]
    Aset = set(A)
    fs = {pos: {m.p: E.build_factor(m, tabs[m.p], S, A=Aset, sel=sel, position=pos) for m in models}
          for pos in ("before", "after")}
    return sch, models, tabs, A, sel, fs


def worker(conn, iid, idx, noise, S, qname, core):
    """Plan-execution worker (same pinned core). The parent kills it when a warm-up exceeds CAP (hard time cap)."""
    try:
        psutil.Process().cpu_affinity([core])
    except Exception:
        pass
    sch, models, tabs, A, sel, fs = build_all(iid, idx, noise, S, qname)
    ref = None
    conn.send(("ready",))
    while True:
        msg = conn.recv()
        if msg[0] == "stop":
            return
        if msg[0] == "ref":     # a file path (answers above 2 GB cannot cross a Windows pipe in one message)
            z = np.load(msg[1]); ref = (z["i"], z["v"]); continue
        plan, prune_thr = msg[1], msg[2]
        f = fs[plan.position]
        try:
            t = time.perf_counter(); o, i, v = P.fold_tree(plan, f, sch.c); w = time.perf_counter() - t
            conn.send(("warm", w))
            ans = E.canonical(o, i, v, A); del o, i, v
            if ref is None:
                ref = ans
                rp = REFTMP / f"{iid}_{noise}_S{S}_{qname}_ref.npz"; REFTMP.mkdir(parents=True, exist_ok=True)
                np.savez(rp, i=ans[0], v=ans[1]); conn.send(("newref", str(rp)))
            diff = vdiff(ans, ref); del ans
            if prune_thr is not None and w > prune_thr:
                conn.send(("done", {"pruned_warmup": w, "fold": w, "vcheck_diff": diff})); continue
            fm, ts, _ = med_time(lambda: P.fold_tree(plan, f, sch.c))
            conn.send(("done", {"fold": fm, "fold_reps": ts, "warmup": w, "vcheck_diff": diff}))
        except E.MemoryCap as e:
            conn.send(("warm", None)); conn.send(("done", {"dnf": "memory cap: " + str(e)}))
        except MemoryError as e:
            conn.send(("warm", None)); conn.send(("done", {"dnf": "system out of memory: " + str(e)[:120]}))


class Timer:
    def __init__(self, cellargs, builds, tau):
        self.cellargs, self.builds, self.tau = cellargs, builds, tau
        self.ref, self.best, self.res = None, None, {}
        self.proc = None; self.respawns = 0

    def _spawn(self):
        import multiprocessing as mp
        ctx = mp.get_context("spawn")
        self.conn, child = ctx.Pipe()
        self.proc = ctx.Process(target=worker, args=(child, *self.cellargs), daemon=True)
        self.proc.start()
        assert self.conn.recv()[0] == "ready"
        if self.ref is not None:
            self.conn.send(("ref", self.ref))

    def _kill(self):
        try:
            self.proc.kill(); self.proc.join(10)
        except Exception:
            pass
        self.proc = None; self.respawns += 1

    def close(self):
        if self.proc is not None:
            try:
                self.conn.send(("stop",)); self.proc.join(10)
            except Exception:
                pass
            self.proc = None

    def run(self, plan, prune):
        k = plan.key()
        if k in self.res:
            return self.res[k]
        if self.proc is None:
            self._spawn()
        bt = self.builds[plan.position]
        thr = PRUNE * self.best if (prune and self.best is not None) else None
        self.conn.send(("plan", plan, thr))
        if not self.conn.poll(CAP):
            self._kill()
            r = {"dnf": f"warm-up > cap {CAP:.0f}s (worker killed)", "total": CAP, "fold": CAP}
            self.res[k] = r; return r
        self.conn.recv()   # ("warm", w)
        while True:
            if not self.conn.poll(CAP * (REPS + 1) + 60):
                self._kill()
                r = {"dnf": "repetitions exceeded the hard limit (worker killed)", "total": CAP, "fold": CAP}
                self.res[k] = r; return r
            m = self.conn.recv()
            if m[0] == "newref":
                self.ref = m[1]; continue
            if m[0] == "warm":
                continue
            r = m[1]; break
        if "dnf" in r:
            r.update({"total": CAP, "fold": CAP})
        else:
            r.update({"build": bt, "total": bt + r["fold"], "vcheck_pass": r["vcheck_diff"] <= self.tau})
            if "pruned_warmup" not in r:
                self.best = r["total"] if self.best is None else min(self.best, r["total"])
        self.res[k] = r
        return r


def cell(path, iid, idx, noise, S, qname, A, sel, models, tabs, c, model, screen_budget=SCREEN_BUDGET):
    t_cell = time.perf_counter()
    load0 = psutil.cpu_percent(interval=0.5)
    Aset = set(A)
    fs_by_pos, builds = {}, {}
    for pos in ("before", "after"):
        fn = lambda: {m.p: E.build_factor(m, tabs[m.p], S, A=Aset, sel=sel, position=pos) for m in models}
        fs_by_pos[pos] = fn()
        builds[pos], _, _ = med_time(fn)
    tau, B = E.tau_eq(models, tabs, S, Aset, sel, c)
    fs_b = fs_by_pos["before"]
    space = P.enumerate_tree_plans(fs_b, positions=("before",))
    space_keys = [p.key() for p in space]
    # named plans and planning times
    dplan = E.default_rule_from_factors(fs_b)
    plan_a, _, _ = med_time(lambda: E.default_rule_from_factors(fs_b))
    fr = P.fixed_rules(fs_b, models)
    heur = {"a_default_rule": (to_tree(dplan), plan_a),
            "d_pushdown_after_merge": (P.TreePlan("after", P.left_deep(dplan.order), dplan.fmts, dplan.gran), plan_a)}
    for k, p in fr.items():
        heur[k] = (p, 0.0)
    v1, v1_cost, _ = P.choose(fs_b, model)
    v1_t, _, _ = med_time(lambda: P.choose(fs_b, model))
    vg, vg_cost, _ = P.choose_generic(fs_b, model)
    vg_t, _, _ = med_time(lambda: P.choose_generic(fs_b, model))
    v2_runs = [P.choose_v2(fs_b, models, model) for _ in range(REPS)]
    v2, v2_cost, _, v2_info = v2_runs[0]
    v2_t = statistics.median(r[2] for r in v2_runs)
    v2_plans = sorted({r[0].key() for r in v2_runs})
    planners = {"v1": (v1, v1_t), "v1_generic": (vg, vg_t), "v2": (v2, v2_t)}
    tm = Timer((iid, idx, noise, S, qname, int(os.environ.get("E2_CORE", "4"))), builds, tau)
    for p, _ in list(heur.values()) + list(planners.values()):
        tm.run(p, prune=False)
    rb = tm.res[fr[RULE_B].key()]
    nontrivial = ("dnf" in rb) or rb["total"] >= T_NT
    screened, unscreened, screen_s = 0, 0, 0.0
    if nontrivial:
        stats = {p: P.leaf_stat(f) for p, f in fs_b.items()}
        cache = {}
        pred = {p.key(): model.cost(P.plan_features(p, stats, cache)) for p in space}
        rest = sorted((p for p in space if p.key() not in tm.res), key=lambda p: pred[p.key()])
        t0 = time.perf_counter()
        for p in rest:
            if time.perf_counter() - t0 > screen_budget:
                unscreened += 1; continue
            tm.run(p, prune=True); screened += 1
        screen_s = time.perf_counter() - t0
    tm.close()
    timed = [k for k in space_keys if k in tm.res and "dnf" not in tm.res[k] and "pruned_warmup" not in tm.res[k]]
    named_keys = [p.key() for p, _ in heur.values()] + [p.key() for p, _ in planners.values()]
    htimes = {h: {"plan": p.key(), "total": tm.res[p.key()]["total"] + x, "planning": x, "dnf": "dnf" in tm.res[p.key()]}
              for h, (p, x) in heur.items()}
    ptimes = {h: {"plan": p.key(), "total": tm.res[p.key()]["total"] + x, "planning": x, "dnf": "dnf" in tm.res[p.key()]}
              for h, (p, x) in planners.items()}
    bh = min(htimes, key=lambda h: htimes[h]["total"])
    out = {"id": iid, "noise": noise, "S": S, "query": qname, "A": A, "sel": sel, "c": c, "r": len(models),
           "tau_eq_certified": tau, "B": B, "plan_space_size": len(space), "nontrivial": nontrivial,
           "rule_b_total": rb["total"], "rule_b_dnf": "dnf" in rb, "screened": screened, "unscreened": unscreened,
           "screen_seconds": screen_s, "worker_respawns": tm.respawns, "plans": tm.res, "heuristics": htimes, "planners": ptimes,
           "v1_predicted_cost": v1_cost, "v2_predicted_cost": v2_cost, "v2_info": v2_info, "v2_plans_over_reps": v2_plans,
           "best_heuristic": bh, "load_before": load0, "affinity": psutil.Process().cpu_affinity(),
           "lock_note": os.environ.get("E2_LOCK_NOTE", "")}
    if timed:
        best_key = min(timed, key=lambda k: tm.res[k]["total"])
        bt = tm.res[best_key]["total"]
        out.update({"best_plan": best_key, "best_total": bt,
                    "ratio_best_heuristic_over_best": htimes[bh]["total"] / bt,
                    **{f"ratio_{h}_over_best": ptimes[h]["total"] / bt for h in ptimes},
                    **{f"ratio_{h}_over_best_heuristic": ptimes[h]["total"] / htimes[bh]["total"] for h in ptimes}})
    else:
        out.update({"best_plan": None, "no_plan_finishes": True})
    out["all_vcheck_pass"] = all(r.get("vcheck_pass", True) for r in tm.res.values())
    out["load_after"] = psutil.cpu_percent(interval=0.5)
    out["cell_seconds"] = time.perf_counter() - t_cell
    out["named_keys"] = named_keys
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(out, indent=1, default=lambda x: x.item() if isinstance(x, np.generic) else str(x)))
    os.replace(tmp, path)
    return out


def load_inst(inst, noise, S):
    d = pickle.load(open(f"D:/runs/knitplan/e2/inst/{inst['id']}.pkl", "rb"))
    sch = d["scheme"]; models = E.sub_models(sch)
    tabs = {p: load_tables(f"D:/runs/knitplan/e2/shots/{inst['id']}/{noise}/sub{p}_S{S}.npz")[2] for p in range(sch.r)}
    return sch, models, tabs


def main():
    try:
        psutil.Process().cpu_affinity([int(os.environ.get("E2_CORE", "4"))])
    except Exception:
        pass
    model = P.CostModel(json.loads(CM.read_text())["k"])
    man = json.loads((ROOT / "bench" / "E2_MANIFEST.json").read_text())
    if sys.argv[1] == "smoke":   # untimed code check; scratch output; prints only sizes and durations, no ratio
        iid, S, qn = sys.argv[2], int(sys.argv[3]), sys.argv[4]
        idx = [i["id"] for i in man["instances"]].index(iid); inst = man["instances"][idx]
        sch, models, tabs = load_inst(inst, "ideal", S)
        A, sel = queries(sch.n, idx)[qn]
        o = cell(Path(os.environ["E2_SMOKE_OUT"]) / "smoke.json", iid, idx, "ideal", S, qn, A, sel, models, tabs, sch.c,
                 model, screen_budget=float(os.environ.get("E2_SMOKE_BUDGET", "60")))
        print(json.dumps({k: o[k] for k in ("plan_space_size", "nontrivial", "screened", "unscreened", "screen_seconds",
                                            "worker_respawns", "cell_seconds", "all_vcheck_pass")}))
        return
    only = set(sys.argv[2].split(",")) if len(sys.argv) > 2 else set()
    for idx, inst in enumerate(man["instances"]):
        if (only and inst["id"] not in only) or inst["role"] == "control":   # Entry 6b: controls not timed
            continue
        for noise in inst["noise"]:
            for S in inst["budgets"]:
                sch, models, tabs = load_inst(inst, noise, S)
                for qname, (A, sel) in queries(sch.n, idx).items():
                    path = OUT / f"{inst['id']}_{noise}_S{S}_{qname}.json"
                    if path.exists():
                        continue
                    t = time.time()
                    o = cell(path, inst["id"], idx, noise, S, qname, A, sel, models, tabs, sch.c, model)
                    for rp in REFTMP.glob(f"{inst['id']}_{noise}_S{S}_{qname}_ref.npz"):
                        rp.unlink()
                    print(inst["id"], noise, S, qname, "space", o["plan_space_size"], "nontrivial", o["nontrivial"],
                          "screened", o["screened"], "unscreened", o["unscreened"], "vcheck", o["all_vcheck_pass"],
                          f"{time.time()-t:.0f}s", flush=True)


if __name__ == "__main__":
    main()
