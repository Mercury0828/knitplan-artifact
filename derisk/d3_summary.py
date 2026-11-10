"""Summarize D3 cells (derisk/results/D3/*.json) against H3 / K-TRIV and the K-SPARSE timing condition. Pilot only."""
import glob, json, math
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def gmean(xs):
    return math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else None


def main():
    cells = [json.loads(Path(f).read_text()) for f in sorted(glob.glob(str(ROOT / "derisk/results/D3/*.json")))]
    n = len(cells)
    ratios = [c["ratio_best_heuristic_over_best"] for c in cells]
    ktriv_share = sum(r <= 1.10 for r in ratios) / n
    h3_cells = [r for r in ratios if r >= 1.10]
    # each single fixed rule vs best (not the per-cell oracle over rules)
    single = {}
    for h in ["a_default_rule", "b_always_dense_pushdown", "c_always_sparse_pushdown_cutqc_order", "d_pushdown_after_merge"]:
        rs = [c["heuristics"][h]["total"] / c["best_total"] for c in cells]
        single[h] = {"gmean_ratio": gmean(rs), "share_within_1.10": sum(r <= 1.10 for r in rs) / n,
                     "max_ratio": max(rs), "dnf_cells": sum(c["heuristics"][h]["dnf"] for c in cells)}
    # K-SPARSE timing condition: dense plan <= 1.10x the best sparse plan
    dvs = [c["dense_best"] / c["sparse_best"] for c in cells if c["dense_best"] and c["sparse_best"]]
    dense_wins = sum(x <= 1.10 for x in dvs) / len(dvs)
    sparse_win_cells = [(c["id"], c["noise"], c["S"], c["query"], round(c["dense_best"] / c["sparse_best"], 2))
                        for c in cells if c["dense_best"] and c["sparse_best"] and c["dense_best"] / c["sparse_best"] > 1.10]
    best_plans = Counter(c["best_plan"].split("|")[2] + "|" + c["best_plan"].split("|")[0] for c in cells)
    vfail = [c["id"] + "/" + c["query"] for c in cells if not c["all_vcheck_pass"]]
    out = {"cells": n, "K_TRIV_share_best_heuristic_within_1.10": ktriv_share, "K_TRIV_fires(>=0.90)": ktriv_share >= 0.9,
           "H3_cells_heuristic_>=1.10x_worse": len(h3_cells), "H3_share": len(h3_cells) / n,
           "H3_gmean_gain_on_those": gmean(h3_cells), "per_rule": single,
           "dense_plan_within_1.10x_of_best_sparse_share": dense_wins,
           "cells_where_sparse_beats_dense_by_>10pct": sparse_win_cells,
           "best_plan_format|position_counts": dict(best_plans), "vcheck_failures": vfail,
           "load_before_median": sorted(c["load_before"] for c in cells)[n // 2]}
    (ROOT / "derisk/results/D3_summary.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
