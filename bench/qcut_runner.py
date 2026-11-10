"""QCut runner for bench/systems.py (main evaluation); run with the knitplan-qcut venv (QCut v2.4.0).
Usage: python qcut_runner.py <dir> <reps>

Equal-semantics route of D5 part 2: the CutQC-convention merged factors (full outputs, built from the same shot tables
by the harness, counted as statistics time) are loaded into QCut's SeparableDistribution, whose groups are the 4^c
global cut-basis assignments and whose modes are each subcircuit's sparse weights per group (stored by complemented
outcome with coefficients -2^-c, so value - constant is the CutQC estimator; validated in D5). The query is answered
with QCut's own operators: marginal(A + B) (QCut sums each mode over the dropped bits, i.e. pushdown inside the
modes), then array() and, for a selection, the slice where the B bits equal sigma (QCut has no selection operator).
Point batches use QCut's value(outcome) per point (the answer key is the point index).
Times per warm run: t_conv (factor COO -> Modes -> SeparableDistribution) and t_exec (marginal + array + slice).
"""
import json, sys, time
from pathlib import Path

import numpy as np
from QCut.execution.distribution import Mode, SeparableDistribution

DENSE_ENTRIES = 2 ** 30


class Infeasible(Exception):
    pass


def build(spec, z):
    allv = spec["allv"]; G = 4 ** len(allv); g = np.arange(G, dtype=np.int64)
    bits, modes = [], []
    for t in spec["factors"]:
        k, vars_, L = t["key"], t["vars"], len(t["outs"])
        lk = np.zeros(G, np.int64)
        for pos, v in enumerate(vars_):
            lk |= ((g >> (2 * allv.index(v))) & 3) << (2 * pos)
        y, key, val = z[f"{k}_y"], z[f"{k}_key"], z[f"{k}_v"]
        uy, inv = np.unique(y, return_inverse=True)
        if uy.size * G > DENSE_ENTRIES:
            raise Infeasible(f"mode of {uy.size} outcomes x {G} groups")
        loc = np.zeros((uy.size, 4 ** len(vars_)))
        loc[inv, key] = val
        idx = (~uy) & ((1 << L) - 1)
        order = np.argsort(idx)
        modes.append(Mode(idx[order], loc[order][:, lk], L)); bits.append(t["outs"])
    return SeparableDistribution(spec["n"], bits, modes, -np.full(G, 2.0 ** -spec["c"]))


def query_points(spec, sd, Zp):
    """Point batch with QCut's own value(outcome); outcome bit q = qubit q (Python int, any width)."""
    w = [1 << q for q in range(Zp.shape[1])]
    out = np.empty(Zp.shape[0])
    for i, row in enumerate(Zp):
        z = sum(wq for wq, b in zip(w, row) if b)
        out[i] = sd.value(z) - sd.constant
    return out


def query(spec, sd):
    if spec.get("points"):
        return query_points(spec, sd, query.Z)
    keep, na, nb = spec["keep"], spec["nA"], len(spec["keep"]) - spec["nA"]
    if not keep:
        raise Infeasible("QCut marginal needs at least one kept bit")
    if 2 ** len(keep) > DENSE_ENTRIES:
        raise Infeasible(f"dense marginal of {len(keep)} bits")
    m = sd.marginal(keep)
    arr = m.array() - m.constant
    if nb:
        s = spec["sigma_code"]
        arr = arr[s << na:(s + 1) << na]
    return arr


def main():
    d = Path(sys.argv[1]); reps = int(sys.argv[2])
    spec = json.loads((d / "spec.json").read_text())
    with np.load(d / "factors.npz") as f:
        z = {k: f[k] for k in f.files}
    query.Z = z.get("points")
    res = {}
    try:
        t0 = time.perf_counter(); arr = query(spec, build(spec, z)); res["cold_total"] = time.perf_counter() - t0
        convs, execs = [], []
        for _ in range(reps):
            t1 = time.perf_counter(); sd = build(spec, z); t2 = time.perf_counter()
            arr = query(spec, sd); t3 = time.perf_counter()
            convs.append(t2 - t1); execs.append(t3 - t2)
        res.update(t_conv=float(np.median(convs)), t_exec=float(np.median(execs)), t_conv_all=convs, t_exec_all=execs)
        i = np.flatnonzero(arr)
        np.savez(d / "answer.npz", keys=i.astype(np.int64), vals=arr[i])
    except (Infeasible, MemoryError) as e:
        res["infeasible"] = f"{type(e).__name__}: {e}"
    (d / "result.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
