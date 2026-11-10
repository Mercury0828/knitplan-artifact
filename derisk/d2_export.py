"""D2 — export the same reconstruction query for Galley (the pre-registered protocol §D2), and time the KNITPLAN prototype on the
same in-memory inputs. Writes D:/runs/knitplan/d2/<case>.npz + <case>.json (tensor list, index names, output order).

Forms: 'merged' = psi_j(beta..., z_j, r_j[, sel_j]) per subcircuit; 'raw' = N_j(state_e..., meas_e..., s_e..., z_j,
r_j[, sel_j]) counts/S plus per-cut-end coefficient tensors (Cin[b, state], Cout[b, meas, s]).
A selection is expressed with a one-hot vector over the packed selected bits of each subcircuit.
"""
_RUNS = __import__("os").environ.get("KNITPLAN_RUNS", "D:/runs/knitplan")   # data root
import itertools, json, pickle, statistics, sys, time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from knitplan import engine as E
from knitplan.shots import load_tables, INIT, MEAS

OUT = Path(_RUNS + "/d2"); OUT.mkdir(parents=True, exist_ok=True)
CIN = np.array([[1, 1, 0, 0], [-1, -1, 2, 0], [-1, -1, 0, 2], [1, -1, 0, 0]], float)      # [beta, state]
COUT = np.zeros((4, 3, 2)); COUT[0, 0] = [1, 1]; COUT[1, 1] = [1, -1]; COUT[2, 2] = [1, -1]; COUT[3, 0] = [1, -1]


def split(y, outq, A, selq):
    zq = [q for q in outq if q in A]; sq = [q for q in outq if q in selq]; rq = [q for q in outq if q not in A and q not in selq]
    g = lambda qs: E.gather_bits(y, [outq.index(q) for q in qs])
    return g(zq), g(sq), g(rq), zq, sq, rq


def export(iid, S, qname, A, sel, form):
    d = pickle.load(open(f"{_RUNS}/pilot/{iid}.pkl", "rb"))
    sch = d["scheme"]; models = E.sub_models(sch)
    tabs = {p: load_tables(f"{_RUNS}/shots/{iid}/ideal/sub{p}_S{S}.npz")[2] for p in range(sch.r)}
    sel = sel or {}
    arrays, tensors, outputs = {}, [], []
    for m in models:
        selq = [q for q in m.out_qubits if q in sel]
        zq = [q for q in m.out_qubits if q in A]; rq = [q for q in m.out_qubits if q not in A and q not in sel]
        dims_tail = [2 ** len(zq), 2 ** len(rq)] + ([2 ** len(selq)] if selq else [])
        tail_idx = [f"z{m.p}", f"r{m.p}"] + ([f"q{m.p}"] if selq else [])
        if form == "merged":
            f = E.build_factor(m, tabs[m.p], S, A=None, sel=None, position="after")
            arr = np.zeros([4] * len(m.deg_cuts) + dims_tail)
            for key, (y, v) in f.data.items():
                z, s_, r, *_ = split(y, m.out_qubits, set(A), set(sel))
                ix = tuple(key) + (z, r) + ((s_,) if selq else ())
                np.add.at(arr, ix, v)
            idx = [f"b{k}" for k in m.deg_cuts] + tail_idx
        else:
            nin, nout = len(m.ins), len(m.outs)
            arr = np.zeros([4] * nin + [3] * nout + [2] * nout + dims_tail)
            for conf, (idx_, cnt) in tabs[m.p].items():
                st = tuple(INIT.index(x) for x in conf[0]); me = tuple(MEAS.index(x) for x in conf[1])
                y = E.gather_bits(idx_, m.out_local)
                sbits = [((idx_ >> lp) & 1) for lp in m.outs]
                z, s_, r, *_ = split(y, m.out_qubits, set(A), set(sel))
                ix = tuple(np.full(idx_.size, x) for x in st + me) + tuple(sbits) + (z, r) + ((s_,) if selq else ())
                np.add.at(arr, ix, cnt / S)
            cut_of_in = {m.cut_local[k]: k for k in m.deg_cuts if not m.cut_is_out[k]}
            cut_of_out = {m.cut_local[k]: k for k in m.deg_cuts if m.cut_is_out[k]}
            idx = ([f"st{cut_of_in[l]}" for l in m.ins] + [f"me{cut_of_out[l]}" for l in m.outs]
                   + [f"s{cut_of_out[l]}" for l in m.outs] + tail_idx)
            for k in m.deg_cuts:
                if m.cut_is_out[k]:
                    arrays[f"Cout{k}"] = COUT; tensors.append({"key": f"Cout{k}", "idx": [f"b{k}", f"me{k}", f"s{k}"], "fmt": "dense"})
                else:
                    arrays[f"Cin{k}"] = CIN; tensors.append({"key": f"Cin{k}", "idx": [f"b{k}", f"st{k}"], "fmt": "dense"})
        arrays[f"T{m.p}"] = arr
        tensors.append({"key": f"T{m.p}", "idx": idx, "fmt": "sparse"})
        if selq:
            sv = 0
            for k, q in enumerate(selq):
                sv |= sel[q] << k
            oh = np.zeros(2 ** len(selq)); oh[sv] = 1.0
            arrays[f"H{m.p}"] = oh; tensors.append({"key": f"H{m.p}", "idx": [f"q{m.p}"], "fmt": "sparse"})
        outputs.append({"idx": f"z{m.p}", "qubits": zq})
    case = f"{iid}_S{S}_{qname}_{form}"
    np.savez(OUT / f"{case}.npz", **arrays)
    spec = {"case": case, "tensors": tensors, "outputs": [o["idx"] for o in outputs], "scale": 2.0 ** -sch.c,
            "out_qubits": [o["qubits"] for o in outputs], "A": A, "sel": {str(k): v for k, v in sel.items()}}
    (OUT / f"{case}.json").write_text(json.dumps(spec))
    return case, models, tabs, sch


def knitplan_times(models, tabs, S, A, sel, c):
    """KNITPLAN prototype on the same inputs: best enumerated plan and default rule (build + fold), 1+5 runs."""
    res = {}
    for plan in E.enumerate_plans(len(models)):
        try:
            E.execute(plan, models, tabs, S, set(A), sel, c)
            ts = []
            for _ in range(5):
                t = time.perf_counter(); E.execute(plan, models, tabs, S, set(A), sel, c); ts.append(time.perf_counter() - t)
            res[plan.key()] = statistics.median(ts)
        except E.MemoryCap:
            res[plan.key()] = None
    ok = {k: v for k, v in res.items() if v is not None}
    best = min(ok, key=ok.get)
    ts = []
    for _ in range(6):
        t = time.perf_counter()
        dp = E.default_rule_plan(models, tabs, S, set(A), sel)
        E.execute(dp, models, tabs, S, set(A), sel, c)
        ts.append(time.perf_counter() - t)
    return {"best_plan": best, "best_time": ok[best], "default_rule_time": statistics.median(ts[1:]), "all": res}


def main():
    cases = []
    S = 10000
    for iid in ["adder20", "qaoa20a", "sup20a"]:
        d = pickle.load(open(f"{_RUNS}/pilot/{iid}.pkl", "rb")); n = d["scheme"].n
        rng = np.random.default_rng(42)
        A10 = sorted(rng.choice(n, size=10, replace=False).tolist())
        for qname, A, sel in [("marg10", A10, None), ("full", list(range(n)), None)]:
            for form in ("merged", "raw"):
                case, models, tabs, sch = export(iid, S, qname, A, sel, form)
                cases.append(case)
            kt = knitplan_times(models, tabs, S, A, sel, sch.c)
            (OUT / f"{iid}_S{S}_{qname}_knitplan.json").write_text(json.dumps(kt, indent=1))
            print(iid, qname, "knitplan best", kt["best_plan"], f"{kt['best_time']:.4f}s default {kt['default_rule_time']:.4f}s", flush=True)
    # expressibility tests on qaoa20a: group-by with selection, point lookup, DD bin query
    d = pickle.load(open(_RUNS + "/pilot/qaoa20a.pkl", "rb")); n = d["scheme"].n
    rng = np.random.default_rng(7)
    Ag = sorted(rng.choice(n, size=6, replace=False).tolist()); rest = [q for q in range(n) if q not in Ag]
    sel = {int(q): int(rng.integers(2)) for q in rng.choice(rest, size=4, replace=False)}
    point = {q: int(rng.integers(2)) for q in range(n)}
    act = sorted(rng.choice(n, size=4, replace=False).tolist()); rest2 = [q for q in range(n) if q not in act]
    zoom = {int(q): int(rng.integers(2)) for q in rng.choice(rest2, size=8, replace=False)}
    for qname, A, s in [("groupby6sel4", Ag, sel), ("point", [], point), ("ddbin4zoom8", act, zoom)]:
        case, *_ = export("qaoa20a", S, qname, A, s, "merged"); cases.append(case)
    (OUT / "cases.json").write_text(json.dumps(cases))


if __name__ == "__main__":
    main()
