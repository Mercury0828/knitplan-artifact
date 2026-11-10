"""M1-N — noisy-shot method check for the frozen workload (the pre-registered protocol; run before any main-workload shot).

Exact noisy sampling is infeasible for subcircuits wider than ~12 active qubits (density matrix) and too slow by
trajectories (cost probe 2026-11-08: 9.0 s per noise trajectory for one 22-qubit adder configuration on one core, i.e.
~120 h per subcircuit at K = 4096). Candidate production method GD ("global depolarizing + exact readout"):
  p_GD = (1 - eps) * p_ideal + eps * uniform(2^m),  eps = 1 - prod_{ops of the device-transpiled circuit} (1 - e_op),
with e_op the archived ibm_cleveland gate errors (no fitting), p_ideal the noiseless distribution of the logical
circuit, followed by the same exact per-shot readout flips as the exact path.
This script compares GD with the exact noisy distribution (Aer density_matrix, archived model; readout flips applied to
both) on configurations whose transpiled circuits have <= 12 active qubits, from the E1 noisy instances. Reported per
configuration: TVD(GD, exact), the S = 10^4 sampling TVD, and the expected number of distinct outcomes (support) after
readout at S in {10^3, 10^4, 10^5} under both. The adoption rule is fixed in the pre-registered protocol.
Usage: python derisk/m1_noise_check.py   (light: one thread)  -> derisk/results/M1/N.json
"""
import json, pickle, sys, time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "bench"))
from qiskit import transpile
from qiskit_aer import AerSimulator
from knitplan.cutting import subcircuits
from knitplan.shots import configurations, config_circuit, _active_qubits, _probs_noiseless
from e1_shots import archived_noisy_context

INSTANCES = ["adder40r4w14", "supremacy_d430r4w10", "hwea30r4w10", "hwea40r5w12"]
PER_INSTANCE = 12            # configurations per instance, every k-th in configuration order (fixed before running)
BUDGETS = [1000, 10000, 100000]
OUT = ROOT / "derisk/results/M1"; OUT.mkdir(parents=True, exist_ok=True)


def readout_matrix_apply(p, phys, readout, m):
    """Exact readout channel on a distribution over m bits (bit i read on physical qubit phys[i])."""
    q = p.reshape((2,) * m)          # axis 0 = bit m-1 (C order of the packed index)
    for i in range(m):
        p01, p10 = readout[phys[i]]
        R = np.array([[1 - p01, p10], [p01, 1 - p10]])      # R[read, true]
        ax = m - 1 - i
        q = np.moveaxis(np.tensordot(R, q, axes=([1], [ax])), 0, ax)
    return q.reshape(-1)


def expected_distinct(p, S):
    return float(np.sum(-np.expm1(S * np.log1p(-np.clip(p, 0, 1 - 1e-16)))))


def eps_of(tc, props):
    keep = 1.0
    for ins in tc.data:
        name = ins.operation.name
        if name in ("barrier", "save_probabilities", "rz", "delay"):
            continue
        qs = [tc.find_bit(q).index for q in ins.qubits]
        try:
            e = props.gate_error(name, qs)
        except Exception:
            e = 0.0
        keep *= 1.0 - (e or 0.0)
    return 1.0 - keep


def main():
    ctx = archived_noisy_context(); props = ctx.backend.properties()
    rows = []
    for iid in INSTANCES:
        d = pickle.load(open(f"D:/runs/knitplan/e1/inst/{iid}.pkl", "rb"))
        sch = d["scheme"]; subs = subcircuits(d["circ"], sch)
        cand = [(p, ci) for p in range(sch.r) for ci in range(len(configurations(sch, p)[2]))]
        step = max(1, len(cand) // PER_INSTANCE)
        for p, ci in cand[::step][:PER_INSTANCE]:
            ins, outs, confs = configurations(sch, p)
            qc = config_circuit(subs[p], ins, outs, confs[ci]); m = qc.num_qubits
            tc = transpile(qc, backend=ctx.backend, optimization_level=1, seed_transpiler=7)
            phys = tc.layout.final_index_layout(); act = _active_qubits(tc)
            if act > 12:
                rows.append({"inst": iid, "p": p, "config": ci, "active": act, "skip": "active > 12"}); continue
            t = time.perf_counter()
            tcp = tc.copy(); tcp.save_probabilities(qubits=phys, label="p")
            r = AerSimulator(noise_model=ctx.nm, method="density_matrix", max_parallel_threads=1).run(
                tcp, shots=1, seed_simulator=1).result()
            p_ex = np.clip(np.asarray(r.data(0)["p"], float), 0, None); p_ex /= p_ex.sum()
            p_id = _probs_noiseless(qc, 1)
            eps = eps_of(tc, props)
            p_gd = (1 - eps) * p_id + eps / 2 ** m
            p_ex_ro = readout_matrix_apply(p_ex, phys, ctx.readout, m)
            p_gd_ro = readout_matrix_apply(p_gd, phys, ctx.readout, m)
            tvd = 0.5 * float(np.abs(p_gd_ro - p_ex_ro).sum())
            rng = np.random.default_rng(ci)
            samp = rng.multinomial(10000, p_ex_ro) / 10000
            tvd_s = 0.5 * float(np.abs(samp - p_ex_ro).sum())
            row = {"inst": iid, "p": p, "config": ci, "width": m, "active": act, "eps": eps, "tvd_gd_exact": tvd,
                   "tvd_sampling_1e4": tvd_s, "sec": time.perf_counter() - t}
            for S in BUDGETS:
                ex, gd = expected_distinct(p_ex_ro, S), expected_distinct(p_gd_ro, S)
                row[f"distinct_exact_S{S}"] = ex; row[f"distinct_gd_S{S}"] = gd; row[f"distinct_ratio_S{S}"] = gd / ex
            rows.append(row); print(json.dumps(row), flush=True)
    ok = [r for r in rows if "skip" not in r]
    summ = {"configs": len(ok), "skipped": len(rows) - len(ok)}
    for S in BUDGETS:
        rat = np.array([r[f"distinct_ratio_S{S}"] for r in ok])
        summ[f"distinct_ratio_S{S}"] = {"min": float(rat.min()), "median": float(np.median(rat)), "max": float(rat.max())}
    summ["tvd_gd_exact_median"] = float(np.median([r["tvd_gd_exact"] for r in ok]))
    summ["tvd_sampling_1e4_median"] = float(np.median([r["tvd_sampling_1e4"] for r in ok]))
    (OUT / "N.json").write_text(json.dumps({"summary": summ, "rows": rows}, indent=1))
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
