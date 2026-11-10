"""E1 shot tables with checkpoint-and-resume (same unit / manifest / seed scheme as bench/gen_shots.py).

Unit = (instance, noise, subcircuit) -> D:/runs/knitplan/e1/shots/<inst>/<noise>/sub<p>_S<S>.npz (atomic);
manifest D:/runs/knitplan/e1/shots/manifest.jsonl. Width <= 24: Aer statevector probabilities + multinomial sampling
(ideal) or the exact noisy distribution (Aer density_matrix, archived model) + readout flips (noisy), exactly as the pilot; width > 24 (the protocol):
Aer matrix_product_state shots (knitplan.shots.generate_unit_mps).
Usage: python bench/e1_shots.py [--only ids] [--threads 2]
"""
import argparse, json, pickle, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from knitplan.cutting import subcircuits
from knitplan.shots import generate_unit, generate_unit_mps
from bench.gen_shots import unit_seed, K_TRAJ
from knitplan.shots import NoisyContext, archived_target

ROOT = Path(__file__).resolve().parents[1]


def archived_noisy_context():
    """The pilot's archived ibm_cleveland snapshot (2026-11-07): noise model + readout errors from derisk/; the live
    backend object is used only for its (static) coupling map / basis gates in transpilation."""
    from qiskit_ibm_runtime import QiskitRuntimeService
    b = QiskitRuntimeService().backend("ibm_cleveland")
    from knitplan.noise_archive import load as load_noise
    nm = load_noise()
    ro = json.load(open(ROOT / "derisk/noise_model_readout_2026-11-07.json"))["readout"]
    readout = {int(k): tuple(v) for k, v in ro.items()}
    return NoisyContext(b, nm, readout, K=0, threads=2, seed=7)  # K=0: exact density_matrix (E1-N)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--exp", default="e1", help="e1 (bench/E1_MANIFEST.json), e2 (bench/E2_MANIFEST.json) or m1 "
                    "(bench/MANIFEST.json, frozen main workload; noisy = the pre-registered protocol method)")
    a = ap.parse_args()
    out_root = Path(f"D:/runs/knitplan/{a.exp}/shots"); out_root.mkdir(parents=True, exist_ok=True)
    man_path = out_root / "manifest.jsonl"
    done = set()
    if man_path.exists():
        done = {json.loads(l)["unit"] for l in man_path.read_text().splitlines() if l.strip()}
    man = json.loads((ROOT / "bench" / ("MANIFEST.json" if a.exp == "m1" else f"{a.exp.upper()}_MANIFEST.json")).read_text())
    only = set(x for x in a.only.split(",") if x)
    ctx = None
    for inst in man["instances"]:
        if only and inst["id"] not in only:
            continue
        d = pickle.load(open(f"D:/runs/knitplan/{a.exp}/inst/{inst['id']}.pkl", "rb"))
        sch = d["scheme"]; subs = subcircuits(d["circ"], sch)
        for noise in inst["noise"]:
            for p in range(sch.r):
                uid = f"{a.exp.upper()}|{inst['id']}|{noise}|{p}"
                if uid in done:
                    continue
                od = out_root / inst["id"] / noise; od.mkdir(parents=True, exist_ok=True)
                if sch.widths()[p] > 24:
                    nctx = None
                    if noise != "ideal":
                        assert a.exp == "m1"      # wide noisy shots: Entry 7a GD over MPS samples (M1 only)
                        if ctx is None:
                            ctx = archived_noisy_context(); ctx.threads = a.threads; ctx.K = "gd"
                            ctx.target = archived_target(ctx.backend)
                        ctx.methods = []; nctx = ctx
                    stats, dt = generate_unit_mps(od, subs[p], sch, p, inst["budgets"], unit_seed(uid), threads=a.threads,
                                                  noisy_ctx=nctx)
                else:
                    if noise == "noisy" and ctx is None:
                        ctx = archived_noisy_context()
                        ctx.threads = a.threads
                        if a.exp == "m1":
                            ctx.K = "gd"      # Entry 7a: exact density matrix <= 12 active qubits, GD above
                            ctx.target = archived_target(ctx.backend)
                    if ctx is not None:
                        ctx.methods = []
                    stats, dt = generate_unit(od, subs[p], sch, p, inst["budgets"], noise, unit_seed(uid),
                                              noisy_ctx=ctx, threads=a.threads)
                if noise == "noisy":
                    from collections import Counter
                    stats["noisy_methods"] = dict(Counter(ctx.methods))
                with open(man_path, "a") as f:
                    f.write(json.dumps({"unit": uid, "seconds": round(dt, 2), "stats": stats,
                                        "finished": time.strftime("%Y-%m-%dT%H:%M:%S")}) + "\n")
                print(uid, f"{dt:.1f}s", flush=True)


if __name__ == "__main__":
    main()
