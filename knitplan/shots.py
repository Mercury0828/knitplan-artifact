"""Pilot shot tables.

For every subcircuit configuration (preparation state at each incoming cut end x measurement basis at each outgoing
cut end, CutQC's scheme), Aer computes the configuration's output distribution over all subcircuit qubits and S shots
are drawn from it with a recorded seed:

* noiseless: Aer statevector method, exact probabilities (``save_probabilities``), multinomial sampling of S shots;
  distributionally identical to S Aer shots.
* noisy (ibm_cleveland calibration noise model, 2026-11-07, no QPU time): the configuration circuit is transpiled to
  the backend (optimization level 1, fixed seed); Aer averages the probability vectors of K Monte-Carlo noise
  trajectories (gate and idle noise of the calibration model); S shots are drawn from that average, then each shot's
  bits are flipped with the calibrated per-qubit readout assignment errors. This approximates S noisy Aer shots;
  K is recorded (pilot approximation, stated in the pre-registered protocol ).

Output per unit (instance, noise, subcircuit): one .npz per shot budget S with, per configuration, the sorted distinct
full-register outcomes (bit i = local subcircuit qubit i) and their counts. Atomic writes; a done-manifest lets a rerun
skip finished units.
"""
from __future__ import annotations

import itertools
import json
import os
import time
from pathlib import Path

import numpy as np
from qiskit import QuantumCircuit, transpile

INIT = ("zero", "one", "plus", "plusI")
MEAS = ("comp", "X", "Y")


def configurations(sch, p):
    """Configurations of subcircuit p: (init tuple over cut-in local qubits, meas tuple over cut-out local qubits)."""
    sids = sch.sub_qubits[p]
    ins = [i for i, sid in enumerate(sids) if sch.segments[sid].starts_with_cut]
    outs = [i for i, sid in enumerate(sids) if sch.segments[sid].ends_with_cut]
    confs = [(a, b) for a in itertools.product(INIT, repeat=len(ins)) for b in itertools.product(MEAS, repeat=len(outs))]
    return ins, outs, confs


def config_circuit(sub: QuantumCircuit, ins, outs, conf):
    init, meas = conf
    qc = QuantumCircuit(sub.num_qubits)
    for q, s in zip(ins, init):
        if s == "one":
            qc.x(q)
        elif s == "plus":
            qc.h(q)
        elif s == "plusI":
            qc.h(q); qc.s(q)
    qc.compose(sub, inplace=True)
    for q, m in zip(outs, meas):
        if m == "X":
            qc.h(q)
        elif m == "Y":
            qc.sdg(q); qc.h(q)
    return qc


def _probs_noiseless(qc, threads):
    from qiskit_aer import AerSimulator
    c = qc.copy()
    c.save_probabilities(label="p")
    sim = AerSimulator(method="statevector", max_parallel_threads=threads)
    r = sim.run(c, shots=1).result()
    return np.asarray(r.data(0)["p"], dtype=float)


def _active_qubits(tc):
    return len({tc.find_bit(q).index for ins in tc.data if ins.operation.name not in ("barrier", "save_probabilities")
                for q in ins.qubits})


PROPS_PKL = Path(__file__).resolve().parents[1] / "derisk/data/ibm_cleveland_properties_2026-11-07.pkl"
GD_MAX_EXACT_ACTIVE = 12      # the pre-registered protocol: exact density matrix up to 12 active qubits, GD above


def archived_target(backend, path=PROPS_PKL):
    """Transpilation target from the live backend's (static) configuration and the archived properties snapshot."""
    import pickle
    from qiskit_ibm_runtime.models import BackendProperties
    from qiskit_ibm_runtime.utils.backend_converter import convert_to_target
    return convert_to_target(backend.configuration(), BackendProperties.from_dict(pickle.load(open(path, "rb"))),
                             include_control_flow=False, include_fractional_gates=False)


def archived_gate_errors(path=PROPS_PKL):
    """(gate name, physical qubits) -> gate_error from the archived ibm_cleveland properties snapshot (2026-11-07,
    the same snapshot the archived noise model was built from)."""
    import pickle
    d = pickle.load(open(path, "rb"))
    errs = {}
    for g in d["gates"]:
        for par in g.get("parameters", []):
            if par.get("name") == "gate_error":
                errs[(g["gate"], tuple(g["qubits"]))] = float(par["value"])
    return errs


def gd_eps(tc, errs):
    """eps = 1 - prod over the ops of the transpiled circuit of (1 - e_op); rz / delay / barrier count 0 (Entry 7a)."""
    keep = 1.0
    for ins in tc.data:
        name = ins.operation.name
        if name in ("barrier", "save_probabilities", "rz", "delay"):
            continue
        keep *= 1.0 - errs.get((name, tuple(tc.find_bit(q).index for q in ins.qubits)), 0.0)
    return 1.0 - keep


class NoisyContext:
    K_FALLBACK = 4096

    def __init__(self, backend, noise_model, readout, K=128, threads=4, seed=7):
        self.backend, self.nm, self.readout, self.K, self.threads, self.seed = backend, noise_model, readout, K, threads, seed
        self.methods = []  # per-configuration simulation method of the K == 0 path (the pre-registered protocol)

    def transpile(self, qc):
        """Device transpilation. With self.target set (M1), against the target built from the archived 2026-11-07
        properties snapshot, so the layout does not depend on the day the shots are drawn (the live backend's error
        rates steer VF2Layout)."""
        if getattr(self, "target", None) is not None:
            return transpile(qc, target=self.target, optimization_level=1, seed_transpiler=self.seed)
        return transpile(qc, backend=self.backend, optimization_level=1, seed_transpiler=self.seed)

    def probs(self, qc, sim_seed):
        from qiskit_aer import AerSimulator
        tc = self.transpile(qc)
        phys = tc.layout.final_index_layout()  # logical i -> physical
        if self.K == "gd":   # the pre-registered protocol (M1 frozen workload)
            act = _active_qubits(tc)
            if act > GD_MAX_EXACT_ACTIVE:
                if not hasattr(self, "errs"):
                    self.errs = archived_gate_errors()
                eps = gd_eps(tc, self.errs)
                p = (1.0 - eps) * _probs_noiseless(qc, self.threads) + eps / 2 ** qc.num_qubits
                self.methods.append(f"gd_active{act}")
                return p / p.sum(), phys
        tc.save_probabilities(qubits=phys, label="p")
        if self.K == 0 or self.K == "gd":  # exact noisy distribution (E1, after E1-N showed the K = 64 trajectory average is biased)
            sim = AerSimulator(noise_model=self.nm, method="density_matrix", max_parallel_threads=self.threads)
            r = sim.run(tc, shots=1, seed_simulator=sim_seed).result()
            if not r.success and "Insufficient memory" in str(r.status) + str(getattr(r.results[0], "status", "")):
                # the pre-registered protocol: routing made the transpiled circuit too wide for a density matrix -> unbiased
                # trajectory average with K_FALLBACK = 4096 (E1-N: TVD 0.0019 to the exact distribution at K = 4096)
                sim = AerSimulator(noise_model=self.nm, method="statevector", max_parallel_threads=self.threads)
                r = sim.run(tc, shots=self.K_FALLBACK, seed_simulator=sim_seed).result()
                self.methods.append(f"trajectories_K{self.K_FALLBACK}_active{_active_qubits(tc)}")
            else:
                self.methods.append(f"density_matrix_active{_active_qubits(tc)}")
        else:
            sim = AerSimulator(noise_model=self.nm, method="statevector", max_parallel_threads=self.threads)
            r = sim.run(tc, shots=self.K, seed_simulator=sim_seed).result()
        p = np.asarray(r.data(0)["p"], dtype=float)
        return p / p.sum(), phys


def sample_counts(p, S, rng):
    counts = rng.multinomial(S, p)
    idx = np.flatnonzero(counts)
    return idx.astype(np.int64), counts[idx].astype(np.int64)


def apply_readout(idx, cnt, phys, readout, rng, width):
    """Flip bits of each shot with the calibrated assignment errors of the physical qubit (exact per-shot sampling)."""
    shots = np.repeat(idx, cnt)
    for i, pq in enumerate(phys):
        p01, p10 = readout[pq]  # P(read 1 | 0), P(read 0 | 1)
        bit = (shots >> i) & 1
        u = rng.random(shots.size)
        flip = np.where(bit == 0, u < p01, u < p10)
        shots = shots ^ (flip.astype(np.int64) << i)
    u, c = np.unique(shots, return_counts=True)
    return u.astype(np.int64), c.astype(np.int64)


def _atomic_save(path: Path, **arrays):
    tmp = path.with_name(path.stem + ".tmp.npz")
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, path)


def generate_unit(out_dir: Path, sub, sch, p, budgets, noise, seed, noisy_ctx=None, threads=4):
    """Generate the shot tables of subcircuit p for every S in budgets. Returns per-S stats."""
    ins, outs, confs = configurations(sch, p)
    rng = np.random.default_rng(seed)
    tables = {S: {} for S in budgets}
    t0 = time.time()
    for ci, conf in enumerate(confs):
        qc = config_circuit(sub, ins, outs, conf)
        if noise == "ideal":
            pr, phys = _probs_noiseless(qc, threads), None
        else:
            pr, phys = noisy_ctx.probs(qc, sim_seed=seed * 100003 + ci)
        pr = np.clip(pr, 0, None); pr = pr / pr.sum()
        for S in budgets:
            idx, cnt = sample_counts(pr, S, rng)
            if noise != "ideal":
                idx, cnt = apply_readout(idx, cnt, phys, noisy_ctx.readout, rng, sub.num_qubits)
            tables[S][ci] = (idx, cnt)
    stats = {}
    for S in budgets:
        arrays = {}
        for ci, (idx, cnt) in tables[S].items():
            arrays[f"i{ci}"] = idx
            arrays[f"c{ci}"] = cnt
        arrays["confs"] = np.array(json.dumps([list(map(list, c)) for c in confs]))
        arrays["ins"] = np.array(ins, dtype=np.int64)
        arrays["outs"] = np.array(outs, dtype=np.int64)
        _atomic_save(out_dir / f"sub{p}_S{S}.npz", **arrays)
        stats[S] = {"configs": len(confs), "mean_distinct": float(np.mean([len(v[0]) for v in tables[S].values()]))}
    return stats, time.time() - t0


def load_tables(path):
    z = np.load(path, allow_pickle=False)
    confs = [tuple(tuple(x) for x in c) for c in json.loads(str(z["confs"]))]
    tabs = {conf: (z[f"i{ci}"], z[f"c{ci}"]) for ci, conf in enumerate(confs)}
    return list(z["ins"]), list(z["outs"]), tabs


def generate_unit_mps(out_dir: Path, sub, sch, p, budgets, seed, threads=1, max_bond=None, noisy_ctx=None):
    """Wide subcircuits (width > 24, the wide-subcircuit stratum): Aer matrix_product_state shots. Per configuration one Aer run with
    S_max = max(budgets) shots (seed recorded); the smaller budgets are uniform subsamples without replacement of those
    S_max iid shots (hence iid samples of their size). Output format identical to generate_unit.
    noisy_ctx given (the pre-registered protocol, GD above 12 active qubits): the configuration circuit is transpiled to the backend
    (same options as NoisyContext.probs), eps = gd_eps with the archived gate errors, each ideal MPS shot is replaced by
    a uniform random bitstring with probability eps (an exact sample of (1 - eps) p_ideal + eps uniform), then the exact
    per-shot readout flips are applied on the layout's physical qubits."""
    from qiskit_aer import AerSimulator
    ins, outs, confs = configurations(sch, p)
    rng = np.random.default_rng(seed)
    smax = max(budgets)
    opts = dict(method="matrix_product_state", max_parallel_threads=threads)
    if max_bond:
        opts["matrix_product_state_max_bond_dimension"] = max_bond
    sim = AerSimulator(**opts)
    tables = {S: {} for S in budgets}
    t0 = time.time()
    bond = []
    for ci, conf in enumerate(confs):
        qc = config_circuit(sub, ins, outs, conf); qc.measure_all()
        res = sim.run(qc, shots=smax, seed_simulator=(seed * 100003 + ci) % 2 ** 31).result()
        cnts = res.get_counts()
        idx = np.array([int(k.replace(" ", ""), 2) for k in cnts], dtype=np.int64)
        cnt = np.array(list(cnts.values()), dtype=np.int64)
        o = np.argsort(idx); idx, cnt = idx[o], cnt[o]
        if noisy_ctx is not None:
            qn = config_circuit(sub, ins, outs, conf)
            tc = noisy_ctx.transpile(qn)
            phys = tc.layout.final_index_layout()
            if not hasattr(noisy_ctx, "errs"):
                noisy_ctx.errs = archived_gate_errors()
            eps = gd_eps(tc, noisy_ctx.errs)
            shots = np.repeat(idx, cnt)
            rep = rng.random(shots.size) < eps
            w = sub.num_qubits
            shots[rep] = (rng.integers(0, 2, size=(int(rep.sum()), w)) << np.arange(w)).sum(axis=1)
            u, c = np.unique(shots, return_counts=True)
            idx, cnt = apply_readout(u.astype(np.int64), c.astype(np.int64), phys, noisy_ctx.readout, rng, w)
            noisy_ctx.methods.append(f"gd_mps_active{_active_qubits(tc)}")
        shots = np.repeat(idx, cnt)
        for S in budgets:
            if S == smax:
                tables[S][ci] = (idx, cnt)
            else:
                sub_s = rng.choice(shots, size=S, replace=False)
                u, c = np.unique(sub_s, return_counts=True)
                tables[S][ci] = (u.astype(np.int64), c.astype(np.int64))
        md = res.results[0].metadata
        bond.append(md.get("matrix_product_state_max_bond_dimension", md.get("MPS_max_bond_dimension")))
    stats = {}
    for S in budgets:
        arrays = {}
        for ci, (idx, cnt) in tables[S].items():
            arrays[f"i{ci}"] = idx
            arrays[f"c{ci}"] = cnt
        arrays["confs"] = np.array(json.dumps([list(map(list, c)) for c in confs]))
        arrays["ins"] = np.array(ins, dtype=np.int64)
        arrays["outs"] = np.array(outs, dtype=np.int64)
        _atomic_save(out_dir / f"sub{p}_S{S}.npz", **arrays)
        stats[S] = {"configs": len(confs), "mean_distinct": float(np.mean([len(v[0]) for v in tables[S].values()])),
                    "method": "matrix_product_state", "max_bond_reported": [b for b in bond[:3]]}
    return stats, time.time() - t0
