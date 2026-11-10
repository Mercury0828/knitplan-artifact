"""E1-G — compare Galley (D:/runs/knitplan/e1/galley/*_galley.json, run by d2_galley.jl pinned to the E1-T core) with
KNITPLAN at boundary B1 (resident pushed-down factors: KNITPLAN = planner v1 planning + fold of its plan, from the E1-T
cell of the same instance / S / query) and B2 (plan reuse: Galley execution only vs KNITPLAN fold only of the v1 plan).
Answers are checked against KNITPLAN's v1-plan answer within the certified tau_eq (the pre-registered protocol). K-GALLEY condition:
Galley <= 1.10x KNITPLAN on >= 90 % of cases (B1). Writes derisk/results/E1/G.json."""
_RUNS = __import__("os").environ.get("KNITPLAN_RUNS", "D:/runs/knitplan")   # data root
import json, pickle, sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from knitplan import engine as E, planner as P
from knitplan.shots import load_tables

D = Path(_RUNS + "/e1/galley")


def plan_from_key(key):
    pos, tree, fmts, gran = key.split("|")
    toks = tree.replace("(", " ( ").replace(")", " ) ").split()

    def parse(i):
        if toks[i] == "(":
            a, i = parse(i + 1); b, i = parse(i)
            assert toks[i] == ")"
            return (a, b), i + 1
        return int(toks[i]), i + 1
    t, _ = parse(0)
    return P.TreePlan(pos, t, tuple(fmts), gran)


def main():
    rows = []
    for case in json.loads((D / "cases.json").read_text()):
        spec = json.loads((D / f"{case}.json").read_text())
        parts = case.split("_")
        iid, S, qname = parts[0], int(parts[1][1:]), parts[2]
        cell = json.loads((ROOT / f"derisk/results/E1/T/{iid}_ideal_S{S}_{qname}.json").read_text())
        gj = D / f"{case}_galley.json"
        if not gj.exists():
            rows.append({"case": case, "status": "NOT RUN"}); continue
        g = json.loads(gj.read_text())
        if "error" in g:
            rows.append({"case": case, "status": "ERROR", "error": g["error"][:500]}); continue
        A = spec["A"]; sel = {int(k): v for k, v in spec["sel"].items()} or None
        d = pickle.load(open(f"{_RUNS}/e1/inst/{iid}.pkl", "rb")); sch = d["scheme"]; models = E.sub_models(sch)
        tabs = {p: load_tables(f"{_RUNS}/e1/shots/{iid}/ideal/sub{p}_S{S}.npz")[2] for p in range(sch.r)}
        fs = {m.p: E.build_factor(m, tabs[m.p], S, A=set(A), sel=sel, position="before") for m in models}
        v1key = cell["planners"]["v1"]["plan"]
        o, i, v = P.fold_tree(plan_from_key(v1key), fs, sch.c)
        i, v = E.canonical(o, i, v, A)
        ref = dict(zip(i.tolist(), v.tolist()))
        out = np.load(D / f"{case}_galley_out.npy")
        got = {}
        for ix, val in np.ndenumerate(out):
            if val == 0.0:
                continue
            z = 0
            for j, zj in enumerate(ix):
                for k, q in enumerate(spec["out_qubits"][j]):
                    z |= ((zj >> k) & 1) << A.index(q)
            got[z] = got.get(z, 0.0) + float(val)
        diff = max([abs(got.get(z, 0) - ref.get(z, 0)) for z in set(got) | set(ref)] + [0.0])
        k_fold = cell["plans"][v1key]["fold"]; k_plan = cell["planners"]["v1"]["planning"]
        best_fold = min(r["fold"] for k, r in cell["plans"].items() if "dnf" not in r)
        row = {"case": case, "status": "ok", "max_abs_diff": diff, "tau_eq": cell["tau_eq_certified"],
               "equivalent": diff <= cell["tau_eq_certified"],
               "galley_warm_wall": g["warm_wall_median"], "galley_warm_opt": g["warm_opt_median"],
               "galley_warm_exec": g["warm_exec_median"], "galley_cold_wall": g["cold_wall"],
               "knit_v1_fold": k_fold, "knit_v1_planning": k_plan, "knit_best_fold": best_fold,
               "B1_ratio_galley_over_knit": g["warm_wall_median"] / (k_fold + k_plan),
               "B2_ratio_galley_exec_over_knit_fold": g["warm_exec_median"] / k_fold}
        rows.append(row); print(json.dumps(row), flush=True)
    ok = [r for r in rows if r["status"] == "ok"]
    summ = {"cases": len(rows), "ok": len(ok),
            "K_GALLEY_share_galley_within_1.10_B1": (sum(r["B1_ratio_galley_over_knit"] <= 1.10 for r in ok) / len(ok)) if ok else None,
            "all_equivalent": all(r["equivalent"] for r in ok)}
    summ["K_GALLEY_fires_B1"] = summ["K_GALLEY_share_galley_within_1.10_B1"] is not None and summ["K_GALLEY_share_galley_within_1.10_B1"] >= 0.90
    (ROOT / "derisk/results/E1/G.json").write_text(json.dumps({"summary": summ, "rows": rows}, indent=1))
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
