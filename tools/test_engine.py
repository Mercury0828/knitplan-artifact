"""Smoke tests of the Phase-0 prototype.

T1 exactness: with exact configuration distributions in place of counts (S = 1, counts = probabilities), the
   reconstruction must equal the uncut circuit's exact distribution (validates cut coefficients and bit order).
T2 plan agreement: every plan of the enumerated space returns the same answer as the brute-force direct point
   evaluator on random marginals / selections (finite-shot tables).
"""
import itertools, sys
import numpy as np
sys.path.insert(0, "D:/runs/knitplan/ext/CutQC"); sys.path.insert(0, ".")
from qiskit.quantum_info import Statevector
from helper_functions.benchmarks import generate_circ
from knitplan.cutting import find_scheme, subcircuits, strip
from knitplan.shots import configurations, config_circuit, _probs_noiseless, sample_counts
from knitplan import engine as E


def exact_tables(sub, sch, p):
    ins, outs, confs = configurations(sch, p)
    tabs = {}
    for conf in confs:
        pr = _probs_noiseless(config_circuit(sub, ins, outs, conf), 2)
        i = np.flatnonzero(pr > 0)
        tabs[conf] = (i.astype(np.int64), pr[i])
    return tabs


def direct_point(models, tabs, S, A_sorted, sel, c, z):
    """Independent path: Q(z) = 2^-c sum_beta prod_j phi_j(beta, z_j) using merged factors built with A, sel."""
    fs = [E.build_factor(m, tabs[m.p], S, A=set(A_sorted), sel=sel, position="after") for m in models]
    allv = sorted(set(v for f in fs for v in f.vars))
    tot = 0.0
    for beta in itertools.product(range(4), repeat=len(allv)):
        b = dict(zip(allv, beta)); prod = 1.0
        for f in fs:
            key = tuple(b[v] for v in f.vars)
            local = 0
            for k, q in enumerate(f.outs):
                local |= ((z >> A_sorted.index(q)) & 1) << k
            i, v = f.data[key]
            pos = np.searchsorted(i, local)
            prod *= v[pos] if pos < i.size and i[pos] == local else 0.0
            if prod == 0.0:
                break
        tot += prod
    return tot * 2.0 ** -c


def main():
    rng = np.random.default_rng(5)
    for fam, n, r, W in [("bv", 8, 2, 6), ("supremacy", 9, 2, 7), ("adder", 10, 3, 6), ("regular", 8, 2, 6)]:
        circ = strip(generate_circ(num_qubits=n, depth=1, circuit_type=fam, reg_name="q", connected_only=True, seed=1))
        sch = find_scheme(circ, r, W, seed=0, iters=4000, restarts=4)
        subs = subcircuits(circ, sch)
        models = E.sub_models(sch)
        # ---- T1 exactness
        tabs = {p: exact_tables(subs[p], sch, p) for p in range(r)}
        truth = Statevector(circ).probabilities()  # bit q = qubit q
        A = list(range(n))
        outs, idx, val = E.execute(E.enumerate_plans(r)[0], models, tabs, 1, set(A), None, sch.c)
        idx, val = E.canonical(outs, idx, val, A)
        rec = np.zeros(2 ** n); rec[idx] = val
        err1 = np.abs(rec - truth).max()
        # ---- T2 plan agreement on finite shots
        ftabs = {}
        for p in range(r):
            ftabs[p] = {}
            for conf, (i, pr) in tabs[p].items():
                full = np.zeros(2 ** subs[p].num_qubits); full[i] = pr
                ftabs[p][conf] = sample_counts(full / full.sum(), 2000, rng)
        worst = 0.0
        for trial in range(3):
            A = sorted(rng.choice(n, size=min(4, n), replace=False).tolist())
            rest = [q for q in range(n) if q not in A]
            sel = {q: int(rng.integers(2)) for q in rng.choice(rest, size=2, replace=False).tolist()}
            ref = None
            for plan in E.enumerate_plans(r):
                o, i, v = E.execute(plan, models, ftabs, 2000, set(A), sel, sch.c)
                i, v = E.canonical(o, i, v, A)
                d = dict(zip(i.tolist(), v.tolist()))
                if ref is None:
                    ref = d
                    for z in range(2 ** len(A)):
                        dp = direct_point(models, ftabs, 2000, A, sel, sch.c, z)
                        worst = max(worst, abs(dp - ref.get(z, 0.0)))
                keys = set(ref) | set(d)
                worst = max(worst, max(abs(ref.get(k, 0) - d.get(k, 0)) for k in keys) if keys else 0)
        fdiff = 0.0
        for m in models:
            for pos in ("before", "after"):
                A2 = set(rng.choice(n, size=3, replace=False).tolist())
                fa = E.build_factor(m, ftabs[m.p], 2000, A=A2, sel=None, position=pos)
                fb = E.build_factor_naive(m, ftabs[m.p], 2000, A=A2, sel=None, position=pos)
                for kk in fa.data:
                    da = dict(zip(*[x.tolist() for x in fa.data[kk]])); db = dict(zip(*[x.tolist() for x in fb.data[kk]]))
                    for z in set(da) | set(db):
                        fdiff = max(fdiff, abs(da.get(z, 0) - db.get(z, 0)))
        print(f"factor build edge-wise vs naive max|diff|={fdiff:.2e}")
        print(f"{fam} n={n} r={r} c={sch.c} widths={sch.widths()} | T1 exact max|err|={err1:.2e} | "
              f"T2 plans={len(E.enumerate_plans(r))} max|diff|={worst:.2e}", flush=True)


if __name__ == "__main__":
    main()
