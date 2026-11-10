"""D5 — QCut v2.4.0 on one pilot instance (the pre-registered protocol §D5). Run with the knitplan-qcut venv.

Part 1 (native): QCut cuts the pilot circuit at the frozen wire-cut locations, runs its own experiment on Aer at
S shots per circuit, reconstructs with estimate_probabilities (SeparableDistribution), and answers marginal, point
and top-k queries. Times recorded per step. QCut's wire-cut QPD differs from CutQC's, so its estimator differs.
Part 2 (equal semantics): KNITPLAN's CutQC-convention factors from our shot tables are loaded into QCut's
SeparableDistribution (groups = cut-basis assignments); its marginal / point answers are compared with KNITPLAN's.
"""
import json, pickle, sys, time, itertools
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
import QCut as ck
from QCut import cut, CutOptions
from QCut.execution.distribution import SeparableDistribution, Mode
from knitplan import engine as E
from knitplan.cutting import strip, _wire_sequences
from knitplan.shots import load_tables

IID, S = sys.argv[1] if len(sys.argv) > 1 else "qaoa20a", 10000
OUT = ROOT / "derisk" / "results"; OUT.mkdir(parents=True, exist_ok=True)


def with_cuts(circ, sch):
    """Insert QCut wire-cut markers after the last op of every cut-out segment."""
    seq, qubits = _wire_sequences(circ)
    after = {}
    for sid, s in enumerate(sch.segments):
        if s.ends_with_cut:
            after.setdefault(s.ops[-1], []).append(s.qubit)
    out = QuantumCircuit(circ.num_qubits)
    for k, inst in enumerate(circ.data):
        out.append(inst.operation, [circ.find_bit(q).index for q in inst.qubits])
        for q in after.get(k, []):
            out.append(cut(), [q])
    return out


def timeit(fn):
    t = time.perf_counter(); r = fn(); return r, time.perf_counter() - t


def main():
    d = pickle.load(open(f"D:/runs/knitplan/pilot/{IID}.pkl", "rb"))
    circ, sch = strip(d["circ"]), d["scheme"]
    n = circ.num_qubits
    rng = np.random.default_rng(5)
    A10 = sorted(rng.choice(n, size=10, replace=False).tolist())
    pts = rng.integers(0, 2 ** n, size=1000)
    rec = {"instance": IID, "S": S, "qcut_version": ck.__version__ if hasattr(ck, "__version__") else "2.4.0 (tag)"}
    # ---------------- part 1: native QCut
    cc = with_cuts(circ, sch)
    opts = CutOptions(wire_cut_communication="never")
    cut_circuit, t_cut = timeit(lambda: ck.get_locations_and_subcircuits(cc, options=opts))
    exp, t_exp = timeit(lambda: ck.get_experiment_circuits(cut_circuit, qubits=list(range(n))))
    raw, t_run = timeit(lambda: ck.run_experiments(exp, shots=S, backend=AerSimulator(max_parallel_threads=4)))
    qp, t_rec = timeit(lambda: ck.estimate_probabilities(raw))
    marg, t_marg = timeit(lambda: qp.marginal(A10))
    marg_vals, t_marg_vals = timeit(lambda: marg.probabilities())
    width = qp.width
    pvals, t_pts = timeit(lambda: [qp.probability_of(format(int(x), f"0{width}b")) for x in pts])
    top, t_top = timeit(lambda: qp.top(10))
    full, t_full = timeit(lambda: qp.probabilities())
    rec["native"] = {"num_circuits": int(exp.num_circuits), "t_cut": t_cut, "t_experiment_build": t_exp,
                     "t_aer_run": t_run, "t_reconstruct": t_rec, "t_marginal10_build": t_marg,
                     "t_marginal10_values": t_marg_vals, "t_point_batch_1000": t_pts, "t_top10": t_top,
                     "t_full_2^n": t_full, "sum_full": float(np.sum(full)), "min_full": float(np.min(full)),
                     "separable": type(getattr(qp, "_data", None)).__name__}
    # ---------------- part 2: equal semantics via SeparableDistribution
    models = E.sub_models(sch)
    tabs = {p: load_tables(f"D:/runs/knitplan/shots/{IID}/ideal/sub{p}_S{S}.npz")[2] for p in range(sch.r)}
    t0 = time.perf_counter()
    fs = [E.build_factor(m, tabs[m.p], S, A=None, sel=None, position="after") for m in models]
    allv = sorted(set(v for f in fs for v in f.vars))
    groups = list(itertools.product(range(4), repeat=len(allv)))
    bits, modes = [], []
    for f in fs:
        L = len(f.outs)
        dense = np.zeros((2 ** L, len(groups)))
        for g, beta in enumerate(groups):
            b = dict(zip(allv, beta))
            i, v = f.data[tuple(b[x] for x in f.vars)]
            # QCut indexes modes by the complemented outcome and returns constant - sum: store the complement
            dense[(~i) & ((1 << L) - 1), g] = v
        bits.append(list(f.outs)); modes.append(Mode.from_dense(dense, L))
    coef = -np.full(len(groups), 2.0 ** -sch.c)
    sd = SeparableDistribution(n, bits, modes, coef)
    t_adapt = time.perf_counter() - t0
    # marginal on A10: QCut value = constant - sum ; our estimator = value - constant
    sm, t_sm = timeit(lambda: sd.marginal(A10))
    qm, t_qm = timeit(lambda: sm.array() - sm.constant)
    plan = E.Plan("before", tuple(range(sch.r)), ("D",) * (sch.r - 1), "joint")
    (o, i, v), t_kp = timeit(lambda: E.execute(plan, models, tabs, S, set(A10), None, sch.c))
    i, v = E.canonical(o, i, v, A10)
    ours = np.zeros(2 ** 10); ours[i] = v
    tau, W = E.tau_eq(models, tabs, S, set(A10), None, sch.c)
    pv, t_pv = timeit(lambda: np.array([sd.value(int(x)) - sd.constant for x in pts]))
    rec["equal_semantics"] = {"t_adapter_build": t_adapt, "t_marginal10_qcut": t_sm + t_qm,
                              "t_marginal10_knitplan_dense_pushdown": t_kp,
                              "marginal_max_abs_diff": float(np.abs(qm - ours).max()), "tau_eq": tau,
                              "t_point_batch_1000_qcut": t_pv, "groups": len(groups)}
    (OUT / f"D5_{IID}.json").write_text(json.dumps(rec, indent=1))
    print(json.dumps(rec, indent=1))


if __name__ == "__main__":
    main()
