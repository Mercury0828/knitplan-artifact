"""Summarize E1-T cells (derisk/results/E1/T/*.json) against the pre-registered bars (the pre-registered protocol, same definitions
as derisk/d3_summary.py): per cell, best fixed rule = the per-cell best of rules (a)-(f) (the strongest, virtual-best
reading); H3 = that rule >= 1.10x the oracle on >= 10 % of cells with gmean gain >= 1.3x on those; K-TRIV = it is
<= 1.10x the oracle on >= 90 % of cells. Pooled over non-control cells and per stratum (T-r4, T-r5, noisy); controls
(bv) separate. Also: each single fixed rule, v1 / oracle, v1 / best single rule, v1 / per-cell best rule, v1 vs
v1-generic. Writes derisk/results/E1/T_summary.json."""
import glob, json, math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RULES = ["a_default_rule", "b_dense_cutqc_order", "c_sparse_cutqc_order", "d_pushdown_after_merge",
         "e_dense_optimal_tree", "f_sparse_optimal_tree"]


def gmean(xs):
    xs = list(xs)
    return math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else None


def block(cells):
    n = len(cells)
    if not n:
        return {"cells": 0}
    best = [c["best_total"] for c in cells]
    rh = [c["ratio_best_heuristic_over_best"] for c in cells]
    h3 = [r for r in rh if r >= 1.10]
    single = {}
    for h in RULES:
        rs = [c["heuristics"][h]["total"] / c["best_total"] for c in cells]
        single[h] = {"gmean_ratio": gmean(rs), "share_within_1.10": sum(r <= 1.10 for r in rs) / n, "max_ratio": max(rs),
                     "dnf_cells": sum(c["heuristics"][h]["dnf"] for c in cells)}
    bs = min(single, key=lambda h: single[h]["gmean_ratio"])
    v1 = [c["planners"]["v1"]["total"] / c["best_total"] for c in cells]
    v1_fold = [(c["planners"]["v1"]["total"] - c["planners"]["v1"]["planning"]) / c["best_total"] for c in cells]
    vg = [c["planners"]["v1_generic"]["total"] / c["best_total"] for c in cells]
    v1_vs_bs = [c["planners"]["v1"]["total"] / c["heuristics"][bs]["total"] for c in cells]
    v1fold_vs_bs = [(c["planners"]["v1"]["total"] - c["planners"]["v1"]["planning"]) / c["heuristics"][bs]["total"] for c in cells]
    v1_vs_ph = [c["ratio_v1_over_best_heuristic"] for c in cells]
    v1_vs_vg = [c["planners"]["v1"]["total"] / c["planners"]["v1_generic"]["total"] for c in cells]
    return {
        "cells": n, "oracle_total_s_median": sorted(best)[n // 2],
        "K_TRIV_share_best_rule_within_1.10": sum(r <= 1.10 for r in rh) / n,
        "K_TRIV_fires": sum(r <= 1.10 for r in rh) / n >= 0.90,
        "H3_share_best_rule_>=1.10x": len(h3) / n, "H3_gmean_gain_on_those": gmean(h3),
        "H3_holds": len(h3) / n >= 0.10 and (gmean(h3) or 0) >= 1.3,
        "per_rule": single, "best_single_rule": bs,
        "v1_over_oracle": {"gmean": gmean(v1), "share_within_1.10": sum(r <= 1.10 for r in v1) / n},
        "v1_fold_only_over_oracle": {"gmean": gmean(v1_fold), "share_within_1.10": sum(r <= 1.10 for r in v1_fold) / n},
        "v1_generic_over_oracle": {"gmean": gmean(vg), "share_within_1.10": sum(r <= 1.10 for r in vg) / n},
        "v1_over_best_single_rule": {"gmean": gmean(v1_vs_bs), "share_>=10pct_better": sum(r <= 1 / 1.10 for r in v1_vs_bs) / n,
                                     "share_>=10pct_worse": sum(r >= 1.10 for r in v1_vs_bs) / n},
        "v1_fold_only_over_best_single_rule": {"gmean": gmean(v1fold_vs_bs)},
        "v1_over_per_cell_best_rule": {"gmean": gmean(v1_vs_ph)},
        "v1_over_v1_generic": {"gmean": gmean(v1_vs_vg), "share_>=10pct_better": sum(r <= 1 / 1.10 for r in v1_vs_vg) / n,
                               "share_>=10pct_worse": sum(r >= 1.10 for r in v1_vs_vg) / n},
        "v1_planning_s_median": sorted(c["planners"]["v1"]["planning"] for c in cells)[n // 2],
        "vcheck_failures": [f'{c["id"]}/{c["noise"]}/{c["S"]}/{c["query"]}' for c in cells if not c["all_vcheck_pass"]],
        "load_before_median": sorted(c["load_before"] for c in cells)[n // 2],
    }


def main():
    cells = [json.loads(Path(f).read_text()) for f in sorted(glob.glob(str(ROOT / "derisk/results/E1/T/*.json")))]
    ctrl = [c for c in cells if c["id"].startswith("bv")]
    main_ = [c for c in cells if not c["id"].startswith("bv")]
    out = {"pooled_non_control": block(main_),
           "T-r4": block([c for c in main_ if c["r"] == 4]), "T-r5": block([c for c in main_ if c["r"] == 5]),
           "noisy": block([c for c in main_ if c["noise"] == "noisy"]),
           "ideal": block([c for c in main_ if c["noise"] == "ideal"]),
           "per_family": {f: block([c for c in main_ if c["id"].startswith(f)]) for f in ("adder", "supremacy_d4", "hwea")},
           "control_bv": block(ctrl)}
    (ROOT / "derisk/results/E1/T_summary.json").write_text(json.dumps(out, indent=1))
    p = out["pooled_non_control"]
    print(json.dumps({k: p.get(k) for k in ("cells", "K_TRIV_share_best_rule_within_1.10", "K_TRIV_fires",
                                            "H3_share_best_rule_>=1.10x", "H3_gmean_gain_on_those", "H3_holds",
                                            "best_single_rule", "v1_over_oracle", "v1_over_best_single_rule",
                                            "v1_over_v1_generic")}, indent=1))


if __name__ == "__main__":
    main()
