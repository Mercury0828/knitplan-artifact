"""M1-Q1 — equivalence tiers of the main evaluation. Untimed, light.

Per (cell, query), from the timed run's unit files (D:/runs/knitplan/m1/T):
  V1  some baseline with an independent dense code path (dense_pushdown, cutqc_opt, cutqc_fd, cutqc_dd,
      tensorqc_dense) finished and its whole answer matches KNITPLAN's within the certified tau_eq (all entries
      compared; recorded by m1_time as vcheck_pass).
  V2  otherwise: (a) direct point evaluator on V2_SAMPLES entries of KNITPLAN's answer (drawn with a recorded seed from
      its stored entries) plus V2_SAMPLES absent entries: each entry is a full selection, its factors are built by the
      independent naive builder (engine.build_factor_naive) and summed over every cut assignment by one dense einsum;
      absent entries must evaluate to 0 within tau_eq; (b) cross-plan agreement: every other finished system matches
      KNITPLAN within tau_eq (recorded by m1_time).
Output: D:/runs/knitplan/m1/Q1.jsonl (one line per (cell, query); resumable).
Usage: python bench/m1_equiv.py
"""
import json, pickle, sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "bench"))
from knitplan import engine as E
import m1_time as MT
from m1_queries import cell_seed

DENSE_REF = ("dense_pushdown", "cutqc_opt", "cutqc_fd", "cutqc_dd", "tensorqc_dense")
V2_SAMPLES = 10


def direct_value(models, tabs, S, c, sel):
    """Estimator at one fully selected entry (sel fixes every output qubit of the query's support), naive builder."""
    import opt_einsum as oe
    fs = [E.build_factor_naive(m, tabs[m.p], S, A=set(), sel=sel, position="before") for m in models]
    allv = sorted({v for f in fs for v in f.vars}); L = {v: oe.get_symbol(j) for j, v in enumerate(allv)}
    ops, terms = [], []
    for f in fs:
        arr = np.zeros((4,) * len(f.vars))
        for key, (y, v) in f.data.items():
            arr[key] = v.sum() if v.size else 0.0
        ops.append(arr); terms.append("".join(L[v] for v in f.vars))
    return float(oe.contract(",".join(terms) + "->", *ops)) * 2.0 ** -c


def main():
    out = MT.RUNS / "Q1.jsonl"
    done = {json.loads(l)["key"] for l in out.read_text().splitlines() if l.strip()} if out.exists() else set()
    for inst in MT.manifest()["instances"]:
        for noise in inst["noise"]:
            for S in inst["budgets"]:
                cid = MT.cell_id(inst["id"], noise, S)
                qp = MT.QDIR / f"{cid}.pkl"
                if not qp.exists():
                    continue
                qs = pickle.load(open(qp, "rb")); loaded = None
                for q in qs:
                    key = f"{cid}|{q['name']}"
                    if key in done:
                        continue
                    units = {}
                    for s in MT.SYSTEM_ORDER:
                        p = MT.TDIR / f"{cid}_{q['name']}_{s}.json"
                        if p.exists():
                            units[s] = json.loads(p.read_text())
                    if "knitplan" not in units or "dnf" in units["knitplan"]:
                        continue
                    others = {s: u for s, u in units.items() if s != "knitplan" and "dnf" not in u}
                    rec = {"key": key, "cross_agree": all(u.get("vcheck_pass") for u in others.values()),
                           "finished_others": sorted(others)}
                    v1 = [s for s in DENSE_REF if s in others and others[s].get("vcheck_pass")]
                    if v1:
                        rec.update(tier="V1", v1_by=v1)
                    else:
                        if loaded is None:
                            _, sch, models, tabs = MT.load_cell(inst["id"], noise, S); loaded = (sch, models, tabs)
                        sch, models, tabs = loaded
                        keys, vals = pickle.load(open(MT.TDIR / f"{cid}_{q['name']}_ref.pkl", "rb"))
                        rng = np.random.default_rng(cell_seed("Q1", key))
                        pts = q.get("points")
                        present = [(int(keys[t]), float(vals[t])) for t in
                                   (rng.choice(keys.size, min(V2_SAMPLES, keys.size), replace=False) if keys.size else [])]
                        space = pts.shape[0] if pts is not None else 2 ** len(q["A"])
                        have = set(keys.tolist()); absent = []
                        for _ in range(50 * V2_SAMPLES):
                            if len(absent) >= V2_SAMPLES or len(have) >= space:
                                break
                            k = int(rng.integers(space))
                            if k not in have and k not in absent:
                                absent.append(k)
                        worst = 0.0
                        for k, v in present + [(k, 0.0) for k in absent]:
                            if pts is not None:
                                sel = {qq: int(pts[k, qq]) for qq in range(sch.n)}
                            else:
                                sel = dict(q["sel"] or {}); sel.update({qq: (k >> j) & 1 for j, qq in enumerate(q["A"])})
                            worst = max(worst, abs(direct_value(models, tabs, S, sch.c, sel) - v))
                        rec.update(tier="V2", v2_present=len(present), v2_absent=len(absent), v2_max_diff=worst,
                                   v2_pass=worst <= q["tau"], tau=q["tau"])
                    with open(out, "a") as f:
                        f.write(json.dumps(rec) + "\n")
                    print(key, rec["tier"], rec.get("v2_pass", True), rec["cross_agree"], flush=True)


if __name__ == "__main__":
    main()
