"""M1 feasibility probe for the frozen main workload (cost only: it measures NO support or timing statistic of the
reconstruction). For the MQT Bench families listed in.2 (target-independent level) at the frozen widths and
cut shapes: is there a wire-cut scheme with every subcircuit <= W qubits (W <= 24,; W up to 40 in the
wide stratum of the wide-subcircuit stratum) and <= 1024
configurations per subcircuit, and what does one ideal configuration cost to sample (statevector, or MPS above 24 qubits, 10^4 shots)?
The CutQC families (CUTQC_FAMILIES, generators of bench/e1_probe.py) are probed at the same shapes with generator seed 2
(E1/E2 used seed 1), so that the frozen instances are new circuits wherever the generator is random.
Output: D:/runs/knitplan/m1/probe.jsonl (one line per case, resumable), D:/runs/knitplan/m1/inst/<id>.pkl
Usage: python bench/m1_probe.py [family,family,...]   (light: one thread)
"""
_RUNS = __import__("os").environ.get("KNITPLAN_RUNS", "D:/runs/knitplan")   # data root
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
import json, pickle, sys, time
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1])); sys.path.insert(0, str(Path(__file__).resolve().parent))
from knitplan.cutting import find_scheme, strip, subcircuits
from knitplan.shots import configurations
from e1_probe import sample_cost, generate_circ, gen_supremacy, grid_ok, hetero

OUT = Path(_RUNS + "/m1"); (OUT / "inst").mkdir(parents=True, exist_ok=True)
MAX_CONFIGS = 1024
CUTQC_FAMILIES = ["adder", "supremacy_d4", "supremacy", "hwea", "bv", "aqft", "regular", "hetero"]
SEED = 2
FAMILIES = ["ghz", "graphstate", "qaoa", "qft", "qnn", "randomcircuit", "vqe_real_amp", "vqe_su2", "wstate"]
SHAPES = [(20, 2, 12), (30, 2, 18), (30, 3, 12), (40, 2, 24), (40, 3, 16), (40, 4, 12),
          (60, 3, 24), (60, 4, 18), (80, 4, 24), (80, 5, 18)]
# wide-subcircuit stratum (the wide-subcircuit stratum, Aer MPS shots); probed after the
# shapes above, same finder, cap and inclusion rule; sampling cost measured with MPS when the widest subcircuit > 24
WIDE_SHAPES = [(60, 2, 34), (80, 2, 40), (80, 3, 30)]


def circuit(fam, n):
    if fam in CUTQC_FAMILIES:
        import math, random
        random.seed(SEED); np.random.seed(SEED)
        if fam == "hetero":
            # CutQC's adder generator needs an even width: the largest near-square dense block that leaves an even
            # adder (E1's rule allowed an odd remainder, which fails the generator assertion at n = 20, 30)
            nd = max(m for m in range(4, n // 2 + 1) if grid_ok(m) and (n - m) % 2 == 0)
            return hetero(n - nd, nd, 2, seed=SEED)
        if fam == "supremacy_d4":      # shallow supremacy (4 cycles), as in E1
            if not grid_ok(n):
                raise ValueError("n not a near-square grid")
            i = math.ceil(math.sqrt(n))
            while n % i:
                i -= 1
            return strip(gen_supremacy(i, n // i, 4, regname="q"))
        c = generate_circ(num_qubits=n, depth=1, circuit_type=fam, reg_name="q", connected_only=True, seed=SEED)
        if c is None:
            raise ValueError("generator returned no circuit at this width")
        return strip(c)
    from mqt.bench import get_benchmark, BenchmarkLevel
    return strip(get_benchmark(fam, BenchmarkLevel.INDEP, n, random_parameters=True))


def main():
    fams = sys.argv[1].split(",") if len(sys.argv) > 1 else FAMILIES
    pj = OUT / "probe.jsonl"
    done = {json.loads(l)["id"] for l in pj.read_text().splitlines() if l.strip()} if pj.exists() else set()
    for fam in fams:
        for n, r, W in SHAPES + WIDE_SHAPES:
            iid = f"{fam}{n}r{r}w{W}"
            if iid in done:
                continue
            row = {"id": iid, "family": fam, "n": n, "r": r, "W": W}
            t = time.time(); sch = None
            try:
                circ = circuit(fam, n)
                row["two_q_gates"] = sum(1 for i in circ.data if len(i.qubits) > 1)
                sch = find_scheme(circ, r, W, seed=0, iters=15000, restarts=4, max_configs=MAX_CONFIGS)
            except Exception as e:
                row["status"] = "ERROR " + repr(e)[:200]
            row["finder_s"] = round(time.time() - t, 1)
            if sch is not None:
                confs = [len(configurations(sch, p)[2]) for p in range(r)]
                row.update(c=sch.c, widths=sch.widths(), configs=confs)
                if max(confs) > MAX_CONFIGS:
                    row["status"] = f"EXCLUDED configs {max(confs)}"
                else:
                    row["status"] = "ok"
                    pickle.dump({"circ": circ, "scheme": sch}, open(OUT / "inst" / f"{iid}.pkl", "wb"))
                    subs = subcircuits(circ, sch)
                    pw = max(range(r), key=lambda p: sch.widths()[p])
                    try:
                        meth = "statevector" if sch.widths()[pw] <= 24 else "matrix_product_state"
                        dt, _, _ = sample_cost(subs[pw], sch, pw, method=meth)
                        row.update(sample_width=sch.widths()[pw], sample_method=meth, sample_s_1e4=round(dt, 2),
                                   est_unit_hours_S1e5=round(sum(confs) * dt * 3 / 3600, 2))
                    except Exception as e:
                        row["sample_error"] = repr(e)[:300]
            elif "status" not in row:
                row["status"] = "NO SCHEME"
            with open(pj, "a") as f:
                f.write(json.dumps(row) + "\n")
            print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
