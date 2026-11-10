"""Build the Phase-0 pilot instance manifest (circuits + frozen wire-cut schemes). Run once; the manifest hash is
recorded in the pre-registered protocol before any D-item runs.

Circuits: CutQC benchmark generators (helper_functions/benchmarks.generate_circ at CutQC commit cda68c7,
depth=1, connected_only=True, seed=1). Cut schemes: knitplan.cutting.find_scheme (simulated annealing, seed 0,
iters 15000, restarts 6) at the stated (r, max subcircuit width W).
"""
_RUNS = __import__("os").environ.get("KNITPLAN_RUNS", "D:/runs/knitplan")   # data root
import hashlib, json, pickle, sys
from pathlib import Path

sys.path.insert(0, _RUNS + "/ext/CutQC"); sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from helper_functions.benchmarks import generate_circ
from qiskit import qasm2
from knitplan.cutting import find_scheme, strip, scheme_manifest
from knitplan.shots import configurations

OUT = Path(_RUNS + "/pilot"); OUT.mkdir(parents=True, exist_ok=True)
ROOT = Path(__file__).resolve().parents[1]

# (instance id, family, n, r, W, noise sets, role)
SPEC = [
    ("sup20a", "supremacy", 20, 2, 18, ["ideal"], "main"),
    ("sup20b", "supremacy", 20, 2, 14, ["ideal", "noisy"], "main,noise-compatible"),
    ("sup30", "supremacy", 30, 2, 24, ["ideal"], "main"),
    ("aqft20", "aqft", 20, 2, 14, ["ideal", "noisy"], "main,noise-compatible"),
    ("aqft30", "aqft", 30, 2, 24, ["ideal"], "main"),
    ("qaoa20a", "regular", 20, 2, 18, ["ideal"], "main"),
    ("qaoa20b", "regular", 20, 2, 14, ["ideal", "noisy"], "main,noise-compatible"),
    ("qaoa30", "regular", 30, 2, 24, ["ideal"], "main"),
    ("adder20", "adder", 20, 2, 14, ["ideal", "noisy"], "main,noise-compatible"),
    ("adder30", "adder", 30, 2, 21, ["ideal"], "main"),
    ("adder20r3", "adder", 20, 3, 11, ["ideal", "noisy"], "r=3 (join order)"),
    ("adder30r3", "adder", 30, 3, 15, ["ideal"], "r=3 (join order)"),
    ("bv20", "bv", 20, 2, 14, ["ideal"], "control (trivially sparse)"),
    ("bv20r3", "bv", 20, 3, 11, ["ideal"], "control (trivially sparse), r=3"),
]
MAX_CONFIGS = 1024


def main():
    man = {"generator": "CutQC generate_circ depth=1 connected_only=True seed=1 (commit cda68c7)",
           "cut_finder": "knitplan.cutting.find_scheme seed=0 iters=15000 restarts=6", "instances": []}
    for iid, fam, n, r, W, noises, role in SPEC:
        circ = strip(generate_circ(num_qubits=n, depth=1, circuit_type=fam, reg_name="q", connected_only=True, seed=1))
        sch = find_scheme(circ, r, W, seed=0, iters=15000, restarts=6)
        if sch is None:
            man["instances"].append({"id": iid, "family": fam, "n": n, "r": r, "W": W, "status": "NO SCHEME"})
            print(iid, "NO SCHEME", flush=True); continue
        confs = [len(configurations(sch, p)[2]) for p in range(r)]
        status = "ok" if max(confs) <= MAX_CONFIGS else f"EXCLUDED (configs {max(confs)} > {MAX_CONFIGS})"
        qasm = qasm2.dumps(circ)
        (OUT / f"{iid}.qasm").write_text(qasm)
        pickle.dump({"circ": circ, "scheme": sch}, open(OUT / f"{iid}.pkl", "wb"))
        man["instances"].append({"id": iid, "family": fam, "n": n, "r": r, "W": W, "c": sch.c, "widths": sch.widths(),
                                 "configs": confs, "noise": noises, "role": role, "status": status,
                                 "qasm_sha256": hashlib.sha256(qasm.encode()).hexdigest(),
                                 "scheme": scheme_manifest(sch)})
        print(iid, fam, n, r, "c", sch.c, sch.widths(), "configs", confs, status, flush=True)
    s = json.dumps(man, indent=1)
    (ROOT / "bench" / "PILOT_MANIFEST.json").write_text(s)
    print("MANIFEST_SHA256", hashlib.sha256(s.encode()).hexdigest())


if __name__ == "__main__":
    main()
