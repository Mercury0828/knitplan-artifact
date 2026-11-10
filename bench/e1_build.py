"""Build bench/E1_MANIFEST.json from the feasibility probe (D:/runs/knitplan/e1/probe_v2.jsonl) by the rule fixed in
the pre-registered protocol Entry 5 (no support or timing statistic is read). Prints the manifest sha256.

Rule: status ok (scheme found, <= 1024 configurations per subcircuit); kind 'wide' with a subcircuit wider than 24 -> stratum W (support), plus TW
(timing) when r >= 3; kind r4 / r5 -> stratum T (timing + support); at most 2 instances per (family, kind) in probe
order; family bv is a control (reported separately, never pooled); budgets S in {1e3, 1e4, 1e5}; noise: ideal for all,
plus 'noisy' (archived ibm_cleveland model, exact density_matrix, E1-N) for T instances with every subcircuit width <= 12 and
<= 400 configurations in total, family != bv; W/TW instances: estimated MPS sampling <= 3 h per instance (else dropped).
"""
_RUNS = __import__("os").environ.get("KNITPLAN_RUNS", "D:/runs/knitplan")   # data root
import hashlib, json, sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    sys.path.insert(0, str(ROOT / "bench"))
    from e1_probe import CASE_ORDER
    rows = [json.loads(l) for l in Path(_RUNS + "/e1/probe_v2.jsonl").read_text().splitlines() if l.strip()]
    rows.sort(key=lambda r: CASE_ORDER.index(r["id"]))  # canonical case order, independent of probe interleaving
    per = Counter()
    insts = []
    for r in rows:
        if r["status"] != "ok":
            continue
        if r["kind"] == "wide" and max(r["widths"]) <= 24:
            continue  # not wide after all (the protocol stratum = subcircuits wider than the statevector cap 24)
        key = (r["family"], r["kind"])  # kinds: wide, r4, r5
        if per[key] >= 2:
            continue
        if r["kind"] == "wide":
            if r.get("sample_s_1e4") is None or sum(r["configs"]) * r["sample_s_1e4"] * 3 / 3600 > 3.0:
                continue
            strata = ["W"] + (["TW"] if r["r"] >= 3 else [])
            noise = ["ideal"]
        else:
            strata = ["T"]
            noise = ["ideal"] + (["noisy"] if max(r["widths"]) <= 12 and sum(r["configs"]) <= 400 and r["family"] != "bv" else [])
        per[key] += 1
        insts.append({"id": r["id"], "family": r["family"], "n": r["n"], "r": r["r"], "W": r["W"], "c": r["c"],
                      "widths": r["widths"], "configs": r["configs"], "stratum": strata[-1] if strata != ["W"] else "W",
                      "strata": strata, "noise": noise, "budgets": [1000, 10000, 100000],
                      "role": "control" if r["family"] == "bv" else "main"})
    man = {"source": _RUNS + "/e1/probe_v2.jsonl (bench/e1_probe.py)", "rule": __doc__.strip().splitlines()[2:],
           "instances": insts}
    s = json.dumps(man, indent=1)
    (ROOT / "bench" / "E1_MANIFEST.json").write_text(s)
    for i in insts:
        print(i["id"], i["strata"], i["noise"], i["c"], i["widths"], i["configs"])
    print("E1_MANIFEST_SHA256", hashlib.sha256(s.encode()).hexdigest())


if __name__ == "__main__":
    main()
