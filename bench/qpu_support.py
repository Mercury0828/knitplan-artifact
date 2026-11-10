"""Support of the ibm_cleveland shot tables vs noisy Aer and ideal (Q5). Per device instance and subcircuit: mean over
configurations of distinct outcomes / shots, for qpu (S = 4096) and for the main-workload ideal / noisy tables at
S = 10^3 and 10^4 (same instance, same subcircuit). Output derisk/results/M1/qpu_support.json
Usage: python bench/qpu_support.py
"""
_RUNS = __import__("os").environ.get("KNITPLAN_RUNS", "D:/runs/knitplan")   # data root
import json, sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from knitplan.shots import load_tables

RUNS = Path(_RUNS + "/m1")


def share(path, S):
    _, _, t = load_tables(path)
    return float(np.mean([i.size / S for i, _ in t.values()])), int(max(i.size for i, _ in t.values()))


def main():
    b = json.loads((RUNS / "qpu" / "batch.json").read_text())
    rows = []
    for inst in b["instances"]:
        for p, w in enumerate(inst["widths"]):
            r = {"id": inst["id"], "stratum": inst["stratum"], "sub": p, "width": w}
            r["qpu_4096"], _ = share(RUNS / "shots" / inst["id"] / "qpu" / f"sub{p}_S4096.npz", 4096)
            for noise in ("ideal", "noisy"):
                for S in (1000, 10000):
                    r[f"{noise}_{S}"], _ = share(RUNS / "shots" / inst["id"] / noise / f"sub{p}_S{S}.npz", S)
            rows.append(r)
            print(r)
    keys = ["qpu_4096", "ideal_1000", "ideal_10000", "noisy_1000", "noisy_10000"]
    summ = {k: {"median": float(np.median([r[k] for r in rows])), "min": float(min(r[k] for r in rows)),
                "max": float(max(r[k] for r in rows))} for k in keys}
    (ROOT / "derisk/results/M1/qpu_support.json").write_text(json.dumps({"summary": summ, "rows": rows}, indent=1))
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
