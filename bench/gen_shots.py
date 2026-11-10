"""Pilot shot-table sweep with checkpoint-and-resume.

Unit = (instance, noise, subcircuit); output D:/runs/knitplan/shots/<inst>/<noise>/sub<p>_S<S>.npz (atomic).
Manifest D:/runs/knitplan/shots/manifest.jsonl: one JSON line per finished unit (done); pending = not listed.
Seeds are a function of the unit id, so a resumed sweep produces the same files as a clean one.
Usage: python bench/gen_shots.py [--only inst1,inst2] [--noise ideal|noisy|all] [--root DIR]
"""
_RUNS = __import__("os").environ.get("KNITPLAN_RUNS", "D:/runs/knitplan")   # data root
import argparse, hashlib, json, pickle, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from knitplan.cutting import subcircuits
from knitplan.shots import generate_unit, NoisyContext

ROOT = Path(__file__).resolve().parents[1]
BUDGETS = [1000, 10000, 100000]
K_TRAJ = 64


def unit_seed(*parts):
    return int(hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest(), 16) % 2 ** 32


def noisy_context():
    from qiskit_ibm_runtime import QiskitRuntimeService
    from qiskit_aer.noise import NoiseModel
    b = QiskitRuntimeService().backend("ibm_cleveland")
    nm = NoiseModel.from_backend(b)
    props = b.properties()
    readout = {}
    for q in range(b.configuration().n_qubits):
        p01 = props.qubit_property(q).get("prob_meas1_prep0", (None,))[0]
        p10 = props.qubit_property(q).get("prob_meas0_prep1", (None,))[0]
        ro = props.readout_error(q)
        readout[q] = (p01 if p01 is not None else ro, p10 if p10 is not None else ro)
    json.dump({"backend": "ibm_cleveland", "properties_last_update": str(props.last_update_date),
               "readout": readout}, open(ROOT / "derisk" / "noise_model_readout_2026-11-07.json", "w"), indent=0)
    return NoisyContext(b, nm, readout, K=K_TRAJ, threads=4, seed=7)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="")
    ap.add_argument("--noise", default="all")
    ap.add_argument("--root", default=_RUNS + "/shots")
    a = ap.parse_args()
    out_root = Path(a.root); out_root.mkdir(parents=True, exist_ok=True)
    man_path = out_root / "manifest.jsonl"
    done = set()
    if man_path.exists():
        for line in man_path.read_text().splitlines():
            if line.strip():
                done.add(json.loads(line)["unit"])
    man = json.loads((ROOT / "bench" / "PILOT_MANIFEST.json").read_text())
    only = set(x for x in a.only.split(",") if x)
    ctx = None
    for inst in man["instances"]:
        if inst["status"] != "ok" or (only and inst["id"] not in only):
            continue
        d = pickle.load(open(f"{_RUNS}/pilot/{inst['id']}.pkl", "rb"))
        subs = subcircuits(d["circ"], d["scheme"])
        for noise in inst["noise"]:
            if a.noise != "all" and noise != a.noise:
                continue
            for p in range(inst["r"]):
                uid = f"{inst['id']}|{noise}|{p}"
                if uid in done:
                    continue
                if noise == "noisy" and ctx is None:
                    ctx = noisy_context()
                od = out_root / inst["id"] / noise; od.mkdir(parents=True, exist_ok=True)
                stats, dt = generate_unit(od, subs[p], d["scheme"], p, BUDGETS, noise, unit_seed(uid),
                                          noisy_ctx=ctx, threads=4)
                with open(man_path, "a") as f:
                    f.write(json.dumps({"unit": uid, "seconds": round(dt, 2), "stats": stats,
                                        "finished": time.strftime("%Y-%m-%dT%H:%M:%S")}) + "\n")
                print(uid, f"{dt:.1f}s", stats, flush=True)


if __name__ == "__main__":
    main()
