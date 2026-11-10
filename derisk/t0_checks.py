"""T0 R1 follow-ups (rounds/T0-semantics_round1_response.md), no timings:
(1) recheck every D3 V-check (stored |plan answer - reference plan answer|) against the certified tau_eq;
(2) support estimates per fold step vs the exact L3 support: predicted_support (Phase-0 upper bound used by the default
    rule), exact_support_nc (T0 R1 (e) formula, no product enumeration), the actual nnz of the product (and the
    enumerated union of candidate keys, which must equal exact_support_nc), and whether the default rule's choices
    change when the exact support replaces the bound.
Usage: python derisk/t0_checks.py  -> derisk/results/T0_checks/<cell>.json + T0_checks_summary.json
"""
import os
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[v] = "1"
import glob, json, pickle, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from knitplan import engine as E
from knitplan.shots import load_tables

ENUM_CAP = 5_000_000


def enum_union(f, g, elim):
    tot = 0
    nf = len(f.outs)
    for rk, pairs in E._consistent_groups(f, g, elim).items():
        codes = [((g.data[kg][0][:, None] << nf) | f.data[kf][0][None, :]).ravel() for kf, kg in pairs
                 if f.data[kf][0].size and g.data[kg][0].size]
        if codes:
            tot += np.unique(np.concatenate(codes)).size
    return tot


def rows_bound(f, g, elim):
    return sum(f.data[kf][0].size * g.data[kg][0].size for ps in E._consistent_groups(f, g, elim).values()
               for kf, kg in ps)


def default_choices(fs, sup_fn, rho=1 / 16):
    r = len(fs)
    if r == 1:
        return None
    best = None
    for a in range(r):
        for b in range(a + 1, r):
            sh = sorted(set(fs[a].vars) & set(fs[b].vars))
            if not sh:
                continue
            s = sup_fn(fs[a], fs[b], sh)
            if best is None or s < best[0]:
                best = (s, a, b)
    order = [best[1], best[2]]
    rest = [p for p in range(r) if p not in order]
    while rest:
        cur = set(v for p in order for v in fs[p].vars)
        cand = sorted(rest, key=lambda p: (not (set(fs[p].vars) & cur), fs[p].nnz()))
        order.append(cand[0]); rest.remove(cand[0])
    fmts, h = [], fs[order[0]]
    for i, p in enumerate(order[1:]):
        g = fs[p]
        later = set(v for q in order[i + 2:] for v in fs[q].vars)
        elim = sorted((set(h.vars) | set(g.vars)) - later)
        dense = (4 ** len(set(h.vars) | set(g.vars) - set(elim))) * 2 ** (len(h.outs) + len(g.outs))
        s = sup_fn(h, g, elim)
        fmts.append("S" if s < dense * rho else "D")
        if i < len(order) - 2:
            if rows_bound(h, g, elim) > ENUM_CAP:
                return tuple(order), tuple(fmts) + ("?",)
            h = E.product_sum(h, g, elim, "S")
    return tuple(order), tuple(fmts)


CELLDIR = ROOT / "derisk/results/T0_checks"


def one_cell(c, sch, models):
    iid, noise, S = c["id"], c["noise"], c["S"]
    tabs = {p: load_tables(f"D:/runs/knitplan/shots/{iid}/{noise}/sub{p}_S{S}.npz")[2] for p in range(sch.r)}
    A = c["A"]; sel = {int(k): v for k, v in c["sel"].items()} if c["sel"] else None
    det = E.tau_eq_details(models, tabs, S, set(A), sel, sch.c)
    tau, B = E.tau_eq(models, tabs, S, set(A), sel, sch.c)
    diffs = [p["vcheck_diff"] for p in c["plans"].values() if "vcheck_diff" in p]
    vrow = {"cell": f"{iid}/{noise}/S{S}/{c['query']}", "tau_cert": tau, "tau_legacy": c["tau_eq"],
            "B": B, "D_max": det["D_max"], "max_diff": max(diffs), "plans_checked": len(diffs),
            "pass": max(diffs) <= tau}
    fs = {m.p: E.build_factor(m, tabs[m.p], S, A=set(A), sel=sel, position="before") for m in models}
    steps = []
    for a in range(sch.r):
        for b in range(a + 1, sch.r):
            sh = sorted(set(fs[a].vars) & set(fs[b].vars))
            if not sh:
                continue
            later = set(v for p in range(sch.r) if p not in (a, b) for v in fs[p].vars)
            elim = sorted((set(fs[a].vars) | set(fs[b].vars)) - later)
            pred, dense = E.predicted_support(fs[a], fs[b], elim)
            ex = E.exact_support_nc(fs[a], fs[b], elim)
            rb = rows_bound(fs[a], fs[b], elim)
            row = {"pair": [a, b], "elim": elim, "predicted": pred, "exact_nc": ex, "dense": dense, "rows": rb}
            if rb <= ENUM_CAP:
                row["enum_union"] = enum_union(fs[a], fs[b], elim)
                row["actual_nnz"] = E.product_sum(fs[a], fs[b], elim, "S").nnz()
            steps.append(row)
    d_pred = default_choices(fs, lambda f, g, e: E.predicted_support(f, g, e)[0])
    d_exact = default_choices(fs, E.exact_support_nc)
    srow = {"cell": vrow["cell"], "steps": steps, "default_with_predicted": d_pred, "default_with_exact": d_exact,
            "choice_changes": d_pred != d_exact}
    return vrow, srow


def main():
    CELLDIR.mkdir(parents=True, exist_ok=True)
    out = {"date": time.strftime("%Y-%m-%d"), "vcheck": [], "support": []}
    files = sorted(glob.glob(str(ROOT / "derisk/results/D3/*.json")))
    cache = {}
    for f in files:
        cp = CELLDIR / Path(f).name
        if not cp.exists():
            c = json.loads(Path(f).read_text())
            if c["id"] not in cache:
                d = pickle.load(open(f"D:/runs/knitplan/pilot/{c['id']}.pkl", "rb"))
                cache = {c["id"]: (d["scheme"], E.sub_models(d["scheme"]))}
            sch, models = cache[c["id"]]
            for attempt in range(6):
                try:
                    vrow, srow = one_cell(c, sch, models); break
                except MemoryError:
                    print("system out of memory; retry in 120 s", flush=True); time.sleep(120)
            else:
                print(c["id"], "SKIPPED after repeated OOM", flush=True); continue
            tmp = cp.with_suffix(".tmp")
            tmp.write_text(json.dumps({"vcheck": vrow, "support": srow},
                                      default=lambda x: x.item() if isinstance(x, np.generic) else str(x)))
            os.replace(tmp, cp)
            print(vrow["cell"], "vcheck", vrow["pass"], "choice_changes", srow["choice_changes"], flush=True)
        rec = json.loads(cp.read_text())
        out["vcheck"].append(rec["vcheck"]); out["support"].append(rec["support"])
    vc = out["vcheck"]; sp = out["support"]
    allsteps = [s for x in sp for s in x["steps"]]
    enum_ok = [s for s in allsteps if "enum_union" in s]
    out["summary"] = {
        "cells": len(vc), "vcheck_fail_cells": [v["cell"] for v in vc if not v["pass"]],
        "max_diff_over_tau_cert": max(v["max_diff"] / v["tau_cert"] for v in vc if v["tau_cert"] > 0),
        "tau_cert_over_legacy_range": [min(v["tau_cert"] / v["tau_legacy"] for v in vc if v["tau_legacy"] > 0),
                                       max(v["tau_cert"] / v["tau_legacy"] for v in vc if v["tau_legacy"] > 0)],
        "cells_with_zero_answer_mass(W=0)": [v["cell"] for v in vc if v["tau_legacy"] == 0],
        "steps": len(allsteps), "steps_enumerated": len(enum_ok),
        "exact_nc_equals_enum_union": all(s["exact_nc"] == s["enum_union"] for s in enum_ok),
        "steps_with_cancellation(actual<exact)": sum(s["actual_nnz"] < s["exact_nc"] for s in enum_ok),
        "predicted_over_exact_gmean": float(np.exp(np.mean([np.log(s["predicted"] / s["exact_nc"]) for s in allsteps if s["exact_nc"]]))),
        "predicted_over_exact_max": max(s["predicted"] / s["exact_nc"] for s in allsteps if s["exact_nc"]),
        "predicted_below_exact_steps": sum(s["predicted"] < s["exact_nc"] for s in allsteps),
        "default_choice_changes": [x["cell"] for x in sp if x["choice_changes"]],
    }
    (ROOT / "derisk/results/T0_checks_summary.json").write_text(json.dumps(out["summary"], indent=1, default=str))
    print(json.dumps(out["summary"], indent=1, default=str))


if __name__ == "__main__":
    main()
