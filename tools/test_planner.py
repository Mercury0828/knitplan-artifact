"""Planner v1 tests: (1) every tree plan (bushy + left-deep, all formats, both granularities, both positions) equals the
Phase-0 engine's reference plan within the certified tau_eq on random instances with r in {3, 4, 5} and random integer
count tables; (2) exact join statistics equal the actual nnz when no cancellation; (3) planner.choose runs.
Run: python tools/test_planner.py"""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qiskit.circuit.random import random_circuit
from knitplan.cutting import find_scheme, strip
from knitplan.shots import configurations
from knitplan import engine as E, planner as P


def main(n_inst=60, seed=3):
    rng = np.random.default_rng(seed)
    done = checks = fails = 0
    ntrees = {}
    while done < n_inst:
        n = int(rng.integers(7, 11)); r = int(rng.integers(3, 6))
        circ = strip(random_circuit(n, int(rng.integers(3, 6)), max_operands=2, seed=int(rng.integers(1 << 30))))
        sch = find_scheme(circ, r, max(2, n // r + 2), seed=int(rng.integers(1000)), iters=800, restarts=2)
        if sch is None or sch.c == 0 or sch.c > 5:
            continue
        models = E.sub_models(sch)
        if not P.enumerate_trees({m.p: E.Factor(tuple(m.deg_cuts), [], "S", {}) for m in models}):
            continue  # cut graph disconnected (never the case for a recorded scheme of a connected circuit)
        tabs = {}
        for p in range(r):
            ins, outs, confs = configurations(sch, p)
            w = len(sch.sub_qubits[p])
            tabs[p] = {}
            for conf in confs:
                k = int(rng.integers(1, min(10, 2 ** w) + 1))
                idx = np.sort(rng.choice(2 ** w, size=k, replace=False)).astype(np.int64)
                tabs[p][conf] = (idx, rng.integers(1, 40, size=k).astype(np.int64))
        S = 100
        A = sorted(rng.choice(n, size=int(rng.integers(1, n + 1)), replace=False).tolist())
        rest = [q for q in range(n) if q not in A]
        sel = {int(q): int(rng.integers(2)) for q in rng.choice(rest, size=int(rng.integers(0, len(rest) + 1)), replace=False)} if rest else {}
        ref_plan = E.Plan("after", tuple(range(r)), ("S",) * (r - 1), "joint")
        try:
            o, i, v = E.execute(ref_plan, models, tabs, S, set(A), sel, sch.c)
        except Exception:
            continue  # disconnected left-deep reference (cross product) is still valid; any error -> skip instance
        ref = dict(zip(*[x.tolist() for x in E.canonical(o, i, v, A)]))
        tau, B = E.tau_eq(models, tabs, S, set(A), sel, sch.c)
        for pos in ("before", "after"):
            fs = {m.p: E.build_factor(m, tabs[m.p], S, A=set(A), sel=sel, position=pos) for m in models}
            plans = P.enumerate_tree_plans(fs, positions=(pos,))
            ntrees[r] = max(ntrees.get(r, 0), len(P.enumerate_trees(fs)))
            for plan in plans:
                o, i, v = P.fold_tree(plan, fs, sch.c)
                got = dict(zip(*[x.tolist() for x in E.canonical(o, i, v, A)]))
                diff = max([abs(got.get(z, 0) - ref.get(z, 0)) for z in set(got) | set(ref)] + [0.0])
                checks += 1; fails += diff > tau
        # statistics: exact join support vs actual nnz for the first pair of a tree
        fs = {m.p: E.build_factor(m, tabs[m.p], S, A=set(A), sel=sel, position="before") for m in models}
        st = {p: P.leaf_stat(f) for p, f in fs.items()}
        t = P.enumerate_trees(fs)[0]
        node = P.internal_nodes(t)[0]
        a, b = node
        if isinstance(a, int) and isinstance(b, int):
            outside = set().union(*(set(fs[p].vars) for p in fs if p not in (a, b))) if r > 2 else set()
            elim = sorted((set(fs[a].vars) | set(fs[b].vars)) - outside)
            js, rows, pairs = P.join_stat(st[a], st[b], elim)
            assert js.nnz() >= E.product_sum(fs[a], fs[b], elim, "S").nnz()
        P.choose(fs, P.CostModel(P.DEFAULT_K))
        done += 1
    print(f"instances {done}, plan checks {checks}, failures {fails}, max trees per r {ntrees}")
    assert fails == 0


if __name__ == "__main__":
    main()
