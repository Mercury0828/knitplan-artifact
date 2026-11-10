"""Summarize E1-W (derisk/results/E1/W/*.json, wide instances only) against the the pre-registered protocol H1 expectation:
for subcircuits with |A cap O_j| >= 20 and S <= 10^4, phi_j < 10 % of its dense shape on every non-control family; narrow
queries (|A cap O_j| <= 10) dense. Route: progress = holds on >= 1 non-control family; no progress = phi_j >= 50 % on
every non-control family at every S for |A cap O_j| >= 20. Writes derisk/results/E1/W_summary.json."""
import glob, json, re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    man = {i["id"]: i for i in json.loads((ROOT / "bench/E1_MANIFEST.json").read_text())["instances"]}
    rows = []
    for f in sorted(glob.glob(str(ROOT / "derisk/results/E1/W/*.json"))):
        o = json.loads(Path(f).read_text())
        if man[o["id"]]["stratum"] != "W":
            continue
        fam = man[o["id"]]["family"]
        for q, d in o["queries"].items():
            for p, v in d["phi"].items():
                rows.append({"id": o["id"], "family": fam, "S": o["S"], "query": q, "sub": p, "kept": v["kept"],
                             "deg": v["deg"], "nnz": v["nnz"], "ratio": v["ratio"],
                             "merged_ratio": o["merged"][p]["ratio"]})
    fams = sorted({r["family"] for r in rows if r["family"] != "bv"})
    per = {}
    for fam in fams + ["bv"]:
        wide = [r for r in rows if r["family"] == fam and r["kept"] >= 20 and r["S"] <= 10000]
        widest = [r for r in rows if r["family"] == fam and r["kept"] >= 20]
        narrow = [r for r in rows if r["family"] == fam and r["kept"] <= 10]
        per[fam] = {"cells_kept>=20_S<=1e4": len(wide), "max_ratio": max((r["ratio"] for r in wide), default=None),
                    "expectation_holds": bool(wide) and all(r["ratio"] < 0.10 for r in wide),
                    "all_>=50pct_every_S": bool(widest) and all(r["ratio"] >= 0.5 for r in widest),
                    "narrow_cells": len(narrow), "narrow_dense_share(>=50%)": (sum(r["ratio"] >= 0.5 for r in narrow) / len(narrow)) if narrow else None,
                    "mid_cells_10<kept<20_max_ratio": max((r["ratio"] for r in rows if r["family"] == fam and 10 < r["kept"] < 20), default=None),
                    "merged_ratio_max": max((r["merged_ratio"] for r in rows if r["family"] == fam), default=None)}
    out = {"families": per,
           "route_progress(holds on >=1 non-control family)": any(per[f]["expectation_holds"] for f in fams),
           "route_no_progress": all(per[f]["all_>=50pct_every_S"] for f in fams),
           "rows": rows}
    (ROOT / "derisk/results/E1/W_summary.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "rows"}, indent=1))


if __name__ == "__main__":
    main()
