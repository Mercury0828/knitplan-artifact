"""E1-N — validate the K = 64 noisy-trajectory approximation (orientation check item 4; the pre-registered protocol Entry 5).
For configurations of the narrowest pilot noisy subcircuits (width <= 10): exact noisy output distribution (Aer
density_matrix with the archived ibm_cleveland noise model, same transpilation) vs the average of K Monte-Carlo
trajectories (K in {64, 256, 1024}); reported: total variation distance (TVD), and the TVD expected from finite
sampling alone at S = 10^4 shots (for scale). Output derisk/results/E1/N.json."""
_RUNS = __import__("os").environ.get("KNITPLAN_RUNS", "D:/runs/knitplan")   # data root
import json, pickle, sys, time
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from qiskit import transpile
from qiskit_aer import AerSimulator
from knitplan.cutting import subcircuits
from knitplan.shots import configurations, config_circuit


def main():
    from knitplan.noise_archive import load as load_noise
    nm = load_noise()
    try:
        from qiskit_ibm_runtime import QiskitRuntimeService
        backend = QiskitRuntimeService().backend("ibm_cleveland")
    except Exception:
        backend = None
    out = []
    for iid in ("adder20r3", "bv20r3"):
        d = pickle.load(open(f"{_RUNS}/pilot/{iid}.pkl", "rb"))
        sch = d["scheme"]; subs = subcircuits(d["circ"], sch)
        p = min(range(sch.r), key=lambda q: sch.widths()[q])
        ins, outs, confs = configurations(sch, p)
        for ci in (0, len(confs) - 1):
            qc = config_circuit(subs[p], ins, outs, confs[ci])
            tc = transpile(qc, backend=backend, optimization_level=1, seed_transpiler=7)
            phys = tc.layout.final_index_layout()
            row = {"id": iid, "sub": p, "width": sch.widths()[p], "config": ci}
            dm = tc.copy(); dm.save_probabilities(qubits=phys, label="p")
            t = time.time()
            exact = np.asarray(AerSimulator(noise_model=nm, method="density_matrix", max_parallel_threads=1)
                               .run(dm, shots=1).result().data(0)["p"]); row["exact_s"] = round(time.time() - t, 1)
            exact = exact / exact.sum()
            for K in (64, 256, 1024):
                tr = tc.copy(); tr.save_probabilities(qubits=phys, label="p")
                pk = np.asarray(AerSimulator(noise_model=nm, method="statevector", max_parallel_threads=1)
                                .run(tr, shots=K, seed_simulator=11).result().data(0)["p"])
                pk = pk / pk.sum()
                row[f"tvd_K{K}"] = float(0.5 * np.abs(pk - exact).sum())
            rng = np.random.default_rng(0)
            emp = rng.multinomial(10000, exact) / 10000
            row["tvd_sampling_S1e4"] = float(0.5 * np.abs(emp - exact).sum())
            out.append(row); print(row, flush=True)
    (ROOT / "derisk/results/E1").mkdir(parents=True, exist_ok=True)
    (ROOT / "derisk/results/E1/N.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
