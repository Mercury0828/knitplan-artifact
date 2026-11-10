"""D2 — compare Galley's answers and times with KNITPLAN's (reads D:/runs/knitplan/d2). Writes derisk/results/D2.json."""
_RUNS = __import__("os").environ.get("KNITPLAN_RUNS", "D:/runs/knitplan")   # data root
import json, pickle, sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from knitplan import engine as E
from knitplan.shots import load_tables

D = Path(_RUNS + "/d2")


def knit_answer(iid, S, A, sel):
    d = pickle.load(open(f"{_RUNS}/pilot/{iid}.pkl", "rb")); sch = d["scheme"]; models = E.sub_models(sch)
    tabs = {p: load_tables(f"{_RUNS}/shots/{iid}/ideal/sub{p}_S{S}.npz")[2] for p in range(sch.r)}
    o, i, v = E.execute(E.Plan("before", (0, 1), ("D",), "joint"), models, tabs, S, set(A), sel, sch.c)
    if A:
        i, v = E.canonical(o, i, v, A)
    tau, W = E.tau_eq(models, tabs, S, set(A), sel, sch.c)
    return dict(zip(i.tolist(), v.tolist())), tau


def main():
    cases = json.loads((D / "cases.json").read_text())
    rows = []
    for case in cases:
        spec = json.loads((D / f"{case}.json").read_text())
        gj = D / f"{case}_galley.json"
        if not gj.exists():
            rows.append({"case": case, "status": "NOT RUN / DNF"}); continue
        g = json.loads(gj.read_text())
        if "error" in g:
            rows.append({"case": case, "status": "ERROR", "error": g["error"][:500]}); continue
        iid, S, qname = case.split("_")[0], int(case.split("_")[1][1:]), case.split("_")[2]
        A = spec["A"]; sel = {int(k): v for k, v in spec["sel"].items()} or None
        out = np.load(D / f"{case}_galley_out.npy")
        # Galley answer -> canonical index over sorted A
        got = {}
        oq = spec["out_qubits"]
        for ix, val in np.ndenumerate(out):
            if val == 0.0:
                continue
            z = 0
            for j, zj in enumerate(ix):
                for k, q in enumerate(oq[j]):
                    z |= ((zj >> k) & 1) << A.index(q)
            got[z] = got.get(z, 0.0) + float(val)
        ref, tau = knit_answer(iid, S, A, sel)
        if not A:
            ref = {0: sum(ref.values())}
        diff = max([abs(got.get(z, 0) - ref.get(z, 0)) for z in set(got) | set(ref)] + [0.0])
        row = {"case": case, "status": "ok", "max_abs_diff": diff, "tau_eq": tau, "equivalent": diff <= tau,
               "galley_warm_total": g["warm_wall_median"], "galley_warm_opt": g["warm_opt_median"],
               "galley_warm_exec": g["warm_exec_median"], "galley_cold_total": g["cold_wall"]}
        kp = D / f"{iid}_S{S}_{qname}_knitplan.json"
        if kp.exists():
            k = json.loads(kp.read_text())
            row.update({"knitplan_best": k["best_time"], "knitplan_default": k["default_rule_time"],
                        "ratio_galley_warm_over_knitplan_best": g["warm_wall_median"] / k["best_time"],
                        "ratio_galley_exec_only_over_knitplan_best": g["warm_exec_median"] / k["best_time"]})
        rows.append(row)
        print(json.dumps(row), flush=True)
    (ROOT / "derisk" / "results" / "D2.json").write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
