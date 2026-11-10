"""Freeze the main-workload instance list (the pre-registered protocol) from the M1 probe, by a rule fixed before any main shot:

  candidate = one row of D:/runs/knitplan/m1/probe.jsonl (bench/m1_probe.py: MQT Bench families + CutQC families with
              generator seed 2, shapes (n, r, W) with n in {20, 30, 40, 60, 80}, W <= 24, plus the wide-subcircuit
              stratum of the wide-subcircuit stratum: (60, 2, 34), (80, 2, 40), (80, 3, 30), MPS shots above 24 qubits)
  include iff  status == ok  (a cut scheme exists with every subcircuit <= W qubits and <= 1024 configurations)
          and  c >= 1        (at least one cut; c = 0 means there is nothing to reconstruct)
          and  est_unit_hours_S1e5 <= 2.0   (ideal shot generation of the instance on one core, probe estimate)
          and  the (circuit, scheme) pair is not identical to an E1 / E2 instance with the same id (E1 / E2 are the
               development and screening sets of the planner)
  every included instance: noise in {ideal, noisy (Entry 7a method)}, S in {10^3, 10^4, 10^5}.
Every excluded candidate is listed with its reason in the manifest (nothing is dropped silently).
Output: bench/MANIFEST.json
Usage: python bench/m1_build.py
"""
_RUNS = __import__("os").environ.get("KNITPLAN_RUNS", "D:/runs/knitplan")   # data root
import json, pickle, sys
from pathlib import Path

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parent)]  # pickles hold knitplan objects

ROOT = Path(__file__).resolve().parents[1]
PROBE = Path(_RUNS + "/m1/probe.jsonl")
EST_CAP_H = 2.0


def dev_instance(iid):
    for exp in ("e1", "e2"):
        p = Path(f"{_RUNS}/{exp}/inst/{iid}.pkl")
        if p.exists():
            return exp, pickle.load(open(p, "rb"))
    return None, None


def main():
    rows = [json.loads(l) for l in PROBE.read_text().splitlines() if l.strip()]
    inc, exc = [], []
    for r in rows:
        why = None
        if r["status"] != "ok":
            why = r["status"]
        elif r["c"] < 1:
            why = "no cut (c = 0)"
        elif r.get("est_unit_hours_S1e5") is None:
            why = "no sampling estimate: " + r.get("sample_error", "")[:80]
        elif r["est_unit_hours_S1e5"] > EST_CAP_H:
            why = f"shot generation est. {r['est_unit_hours_S1e5']} h > {EST_CAP_H} h"
        else:
            exp, d = dev_instance(r["id"])
            if d is not None:
                m = pickle.load(open(f"{_RUNS}/m1/inst/{r['id']}.pkl", "rb"))
                if m["circ"] == d["circ"] and m["scheme"].widths() == d["scheme"].widths() and \
                        m["scheme"].c == d["scheme"].c:
                    why = f"identical to {exp.upper()} development instance"
        if why:
            exc.append({"id": r["id"], "reason": why})
        else:
            inc.append({"id": r["id"], "family": r["family"], "n": r["n"], "r": r["r"], "W": r["W"], "c": r["c"],
                        "widths": r["widths"], "configs": r["configs"], "noise": ["ideal", "noisy"],
                        "budgets": [1000, 10000, 100000], "est_unit_hours_S1e5": r["est_unit_hours_S1e5"],
                        "stratum": "wide" if max(r["widths"]) > 24 else "narrow"})
    out = {"rule": __doc__.split("Output:")[0].strip(), "probe": str(PROBE), "instances": inc, "excluded": exc}
    (ROOT / "bench" / "MANIFEST.json").write_text(json.dumps(out, indent=1))
    fams = sorted({i["family"] for i in inc})
    print(len(inc), "instances;", len(exc), "excluded;", "families:", fams)
    for f in fams:
        print(f, [i["id"] for i in inc if i["family"] == f])


if __name__ == "__main__":
    main()
