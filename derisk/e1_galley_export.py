"""E1-G — equal-boundary Galley comparison (the pre-registered protocol Entry 5; ladder R-G1-R-G3 for the D2 negative item).

Boundary B1 "resident pushed-down factors": both systems start from the same in-memory pushed-down factors
phi_j(beta_{delta(j)}, z_j) (built below the configuration merge, selection applied) and produce the answer
Q(z) = 2^-c sum_beta prod_j phi_j. KNITPLAN's time = fold time of its plan (planner v1 / best plan; from the E1-T cell
of the same (instance, S, query)); Galley's = galley() wall time warm (planning + execution), and with plan reuse =
its execution time alone (B2). Export: D:/runs/knitplan/e1/galley/<case>.npz/.json, run by derisk/d2_galley.jl with
GALLEY_DIR set. Usage: python derisk/e1_galley_export.py <instance ids comma-separated> [S]
"""
import json, pickle, sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "derisk"))
from knitplan import engine as E
from knitplan.shots import load_tables
from e1_time import queries

OUT = Path("D:/runs/knitplan/e1/galley"); OUT.mkdir(parents=True, exist_ok=True)


def main():
    ids = sys.argv[1].split(","); S = int(sys.argv[2]) if len(sys.argv) > 2 else 10000
    man = json.loads((ROOT / "bench" / "E1_MANIFEST.json").read_text())
    idx_of = {inst["id"]: i for i, inst in enumerate(man["instances"])}
    cases = []
    for iid in ids:
        d = pickle.load(open(f"D:/runs/knitplan/e1/inst/{iid}.pkl", "rb"))
        sch = d["scheme"]; models = E.sub_models(sch)
        tabs = {p: load_tables(f"D:/runs/knitplan/e1/shots/{iid}/ideal/sub{p}_S{S}.npz")[2] for p in range(sch.r)}
        for qname, (A, sel) in queries(sch.n, idx_of[iid]).items():
            arrays, tensors, outs = {}, [], []
            ok = True
            for m in models:
                f = E.build_factor(m, tabs[m.p], S, A=set(A), sel=sel, position="before", fmt="S")
                shape = [4] * len(f.vars) + [2 ** len(f.outs)]
                if np.prod(shape, dtype=float) > 2 ** 28:
                    ok = False; break
                arr = np.zeros(shape)
                for key, (i, v) in f.data.items():
                    arr[key][i] = v
                arrays[f"T{m.p}"] = arr
                tensors.append({"key": f"T{m.p}", "idx": [f"b{k}" for k in f.vars] + [f"z{m.p}"], "fmt": "sparse"})
                outs.append({"idx": f"z{m.p}", "qubits": f.outs})
            if not ok:
                print(iid, qname, "skipped: dense export of a factor exceeds 2^28 entries"); continue
            case = f"{iid}_S{S}_{qname}_pushed"
            np.savez(OUT / f"{case}.npz", **arrays)
            (OUT / f"{case}.json").write_text(json.dumps({
                "case": case, "tensors": tensors, "outputs": [o["idx"] for o in outs], "scale": 2.0 ** -sch.c,
                "out_qubits": [o["qubits"] for o in outs], "A": A, "sel": {str(k): v for k, v in (sel or {}).items()}}))
            cases.append(case); print(case, flush=True)
    (OUT / "cases.json").write_text(json.dumps(sorted(set(cases) | set(json.loads((OUT / "cases.json").read_text())
                                                                       if (OUT / "cases.json").exists() else []))))


if __name__ == "__main__":
    main()
