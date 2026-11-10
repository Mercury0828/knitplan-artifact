"""Build bench/E2_MANIFEST.json from the E2 feasibility probe (D:/runs/knitplan/e2/probe.jsonl) by the rule fixed in
the pre-registered protocol Entry 6 (no support or timing statistic is read). Prints the manifest sha256.

Rule: status ok (scheme found, <= 1024 configurations per subcircuit); at most 2 instances per (family, r) in probe
case order; family bv is a control (reported separately, never pooled); ideal shots only (subcircuits up to 30 qubits:
no exact noisy simulation); budgets S in {1e3, 1e4, 1e5}; dropped if the estimated sampling time exceeds 3 h per
instance (est_unit_hours_S1e5 from the probe).
"""
_RUNS = __import__("os").environ.get("KNITPLAN_RUNS", "D:/runs/knitplan")   # data root
import hashlib, json, sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    sys.path.insert(0, str(ROOT / "bench"))
    from e2_probe import CASE_ORDER
    rows = [json.loads(l) for l in Path(_RUNS + "/e2/probe.jsonl").read_text().splitlines() if l.strip()]
    rows.sort(key=lambda r: CASE_ORDER.index(r["id"]))
    per, insts = Counter(), []
    for r in rows:
        if r["status"] != "ok" or r.get("est_unit_hours_S1e5") is None or r["est_unit_hours_S1e5"] > 3.0:
            continue
        key = (r["family"], r["r"])
        if per[key] >= 2:
            continue
        per[key] += 1
        insts.append({"id": r["id"], "family": r["family"], "n": r["n"], "r": r["r"], "W": r["W"], "c": r["c"],
                      "widths": r["widths"], "configs": r["configs"], "stratum": f"E2-r{r['r']}", "strata": [f"E2-r{r['r']}"],
                      "noise": ["ideal"], "budgets": [1000, 10000, 100000],
                      "role": "control" if r["family"] == "bv" else "main"})
    man = {"source": _RUNS + "/e2/probe.jsonl (bench/e2_probe.py)", "rule": __doc__.strip().splitlines()[2:],
           "instances": insts}
    s = json.dumps(man, indent=1)
    (ROOT / "bench" / "E2_MANIFEST.json").write_text(s)
    for i in insts:
        print(i["id"], i["c"], i["widths"], i["configs"])
    print("E2_MANIFEST_SHA256", hashlib.sha256(s.encode()).hexdigest())


if __name__ == "__main__":
    main()
