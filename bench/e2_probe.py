"""E2 feasibility probe (cost only; NO support / timing statistic of the reconstruction is measured), for ladder attempt
2 on K-TRIV / H3: the regime E1 did not cover — r >= 3 with subcircuits 20-34 qubits wide (statevector shots for
width <= 24, Aer matrix_product_state above). Same procedure and feasibility rule as bench/e1_probe.py (scheme found,
<= 1024 configurations per subcircuit). Output: D:/runs/knitplan/e2/probe.jsonl, D:/runs/knitplan/e2/inst/<id>.pkl.
Usage: python bench/e2_probe.py [families comma-separated]
"""
_RUNS = __import__("os").environ.get("KNITPLAN_RUNS", "D:/runs/knitplan")   # data root
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
import json, pickle, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e1_probe as P1
from knitplan.cutting import find_scheme, subcircuits
from knitplan.shots import configurations

OUT = Path(_RUNS + "/e2"); (OUT / "inst").mkdir(parents=True, exist_ok=True)
FAMILIES = ["adder", "supremacy_d4", "hwea", "hetero", "bv"]
SHAPES = [(60, 3, 24), (63, 3, 24), (72, 3, 28), (81, 3, 30), (80, 4, 24), (90, 4, 26), (100, 4, 28), (100, 5, 24)]
CASE_ORDER = [f"{fam}{n}r{r}w{W}" for fam in FAMILIES for n, r, W in SHAPES]


def main():
    fams = sys.argv[1].split(",") if len(sys.argv) > 1 else FAMILIES
    pj = OUT / "probe.jsonl"
    for fam in fams:
        for n, r, W in SHAPES:
            iid = f"{fam}{n}r{r}w{W}"
            done = {json.loads(l)["id"] for l in pj.read_text().splitlines() if l.strip()} if pj.exists() else set()
            if iid in done:
                continue
            row = {"id": iid, "family": fam, "n": n, "r": r, "W": W}
            t = time.time(); sch = None
            try:
                circ = P1.circuit(fam, n)
                sch = find_scheme(circ, r, W, seed=0, iters=15000, restarts=4, max_configs=P1.MAX_CONFIGS)
            except Exception as e:
                row["status"] = "ERROR " + repr(e)[:200]
            row["finder_s"] = round(time.time() - t, 1)
            if sch is not None:
                confs = [len(configurations(sch, p)[2]) for p in range(r)]
                row.update(c=sch.c, widths=sch.widths(), configs=confs)
                if max(confs) > P1.MAX_CONFIGS:
                    row["status"] = f"EXCLUDED configs {max(confs)}"
                else:
                    row["status"] = "ok"
                    pickle.dump({"circ": circ, "scheme": sch}, open(OUT / "inst" / f"{iid}.pkl", "wb"))
                    subs = subcircuits(circ, sch)
                    pw = max(range(r), key=lambda p: sch.widths()[p])
                    method = "statevector" if sch.widths()[pw] <= 24 else "matrix_product_state"
                    try:
                        dt, _, md = P1.sample_cost(subs[pw], sch, pw, method=method)
                        row.update(sample_method=method, sample_width=sch.widths()[pw], sample_s_1e4=round(dt, 2),
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
