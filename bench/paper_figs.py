"""Paper figures from archived results (no new measurement). Usage: python bench/paper_figs.py [fig1 ...]
Output: paper/figs/<name>.pdf. Fonts >= 7 pt at the final width (IEEEtran column = 88 mm)."""
import glob, json, sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper" / "figs"
BLUE, ORANGE, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#1f1f1e", "#6b6a64", "#e4e3de"
COL_IN = 88 / 25.4
plt.rcParams.update({"font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
                     "legend.fontsize": 7.5, "font.family": "serif", "pdf.fonttype": 42, "axes.edgecolor": MUTED,
                     "axes.linewidth": 0.6, "xtick.color": INK, "ytick.color": INK,
                     "legend.facecolor": "white", "legend.edgecolor": "none", "legend.framealpha": 1.0})
LAB = dict(fontsize=7.5, color=MUTED, bbox=dict(facecolor="white", edgecolor="none", pad=0.6))


def _logticks(ax, axis, ticks):
    labs = [("%g" % t) for t in ticks]
    if axis == "x":
        ax.set_xticks(ticks); ax.set_xticklabels(labs); ax.minorticks_off()
    else:
        ax.set_yticks(ticks); ax.set_yticklabels(labs); ax.minorticks_off()


def fig1():
    """Sparse vs dense execution of the same plan (CutQC order, pushdown below the merge): time ratio per E1 cell."""
    groups = [("marg10", "marginal\n10 qubits"), ("groupb", "group-by\n10 qubits"), ("marg20", "marginal\n20 qubits"),
              ("margha", "marginal\nn/2 qubits")]
    pts = {"ideal": {g: [] for g, _ in groups}, "noisy": {g: [] for g, _ in groups}}
    for f in glob.glob(str(ROOT / "derisk/results/E1/T/*.json")):
        d = json.load(open(f))
        if d["id"].startswith("bv"):
            continue
        h = d["heuristics"]; b, c = h["b_dense_cutqc_order"], h["c_sparse_cutqc_order"]
        if b["dnf"] or c["dnf"]:
            continue
        pts[d["noise"]][d["query"][:6]].append(c["total"] / b["total"])
    fig, ax = plt.subplots(figsize=(COL_IN, 2.1))
    rng = np.random.default_rng(0)
    for k, (g, _) in enumerate(groups):
        for noise, off, mk, col, fill in (("ideal", -0.18, "o", BLUE, "none"), ("noisy", 0.18, "^", ORANGE, ORANGE)):
            y = np.array(pts[noise][g])
            x = k + off + rng.uniform(-0.07, 0.07, y.size)
            ax.scatter(x, y, s=14, marker=mk, facecolors=fill, edgecolors=col, linewidths=0.8, zorder=3,
                       label=f"{noise} shots" if k == 0 else None)
    ax.axhline(1.0, color=MUTED, lw=0.8, zorder=1)
    ax.set_yscale("log")
    ax.set_ylim(0.1, 100)
    ax.set_yticks([0.1, 0.3, 1, 3, 10, 30])
    ax.set_yticklabels(["0.1", "0.3", "1", "3", "10", "30"])
    ax.set_ylabel("sparse time / dense time")
    ax.set_xticks(range(len(groups))); ax.set_xticklabels([l for _, l in groups])
    ax.set_xlim(-0.55, len(groups) - 0.45)
    ax.grid(axis="y", color=GRID, lw=0.5, zorder=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.text(-0.5, 0.13, "sparse is faster", va="bottom", **LAB)
    ax.text(-0.5, 85, "dense is faster", va="top", **LAB)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False, handletextpad=0.3,
              columnspacing=1.0)
    fig.tight_layout(pad=0.3)
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "fig1_flip.pdf", bbox_inches="tight", pad_inches=0.05)
    n = {k: sum(len(v) for v in pts[k].values()) for k in pts}
    print("fig1: cells", n, "sparse faster:", {k: sum(x < 1 for v in pts[k].values() for x in v) for k in pts})


PAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]


def _e1_cells():
    for f in glob.glob(str(ROOT / "derisk/results/E1/T/*.json")):
        d = json.load(open(f))
        if not d["id"].startswith("bv"):
            yield d


def _style(ax):
    ax.grid(color=GRID, lw=0.5, zorder=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def fig_rules():
    """Q3: CDF over the 154 E1 cells of each strategy's time over the per-cell best plan (DNF counted at the cap)."""
    series = [("dense, CutQC order", lambda d: d["heuristics"]["b_dense_cutqc_order"]["total"], "-", None),
              ("sparse, CutQC order", lambda d: d["heuristics"]["c_sparse_cutqc_order"]["total"], (0, (5, 2)), None),
              ("support-based rule", lambda d: d["heuristics"]["a_default_rule"]["total"], (0, (6, 2, 1, 2)), None),
              ("after-merge support rule", lambda d: d["heuristics"]["d_pushdown_after_merge"]["total"], (0, (1, 1.5)), None),
              ("exhaustive planner", lambda d: d["planners"]["v1"]["total"], (0, (6, 1.5, 1, 1.5, 1, 1.5)), None),
              ("its plans, execution only",
               lambda d: d["planners"]["v1"]["total"] - d["planners"]["v1"]["planning"], "-", "o")]
    cells = list(_e1_cells())
    fig, ax = plt.subplots(figsize=(COL_IN, 2.75))
    for k, (lab, fn, ls, mk) in enumerate(series):
        r = np.sort([max(fn(d) / d["best_total"], 1.0) for d in cells])
        y = np.arange(1, r.size + 1) / r.size
        keep = r <= 29.5; r, y = r[keep], y[keep]   # points beyond the axis are not drawn (curve ends below 1)
        ax.step(r, y, where="post", color=PAL[k], ls=ls, lw=1.4 if mk is None else 0.9, label=lab,
                marker=mk, markevery=12, ms=3.5, mfc="none")
    ax.set_xscale("log"); ax.set_xlim(1, 30)
    ax.set_xticks([1, 2, 5, 10, 30]); ax.set_xticklabels(["1", "2", "5", "10", "30"]); ax.axvline(1.1, color=MUTED, lw=1.0, ls=(0, (2, 2)), zorder=1)
    ax.set_ylim(0, 1.0)
    ax.set_xlabel("time / time of the best plan"); ax.set_ylabel("share of queries")
    _style(ax)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False, fontsize=7.5, handlelength=2.6,
              columnspacing=0.8)
    fig.tight_layout(pad=0.5); fig.savefig(OUT / "fig_rules.pdf", bbox_inches="tight", pad_inches=0.05)
    print("fig_rules cells", len(cells))


def fig_overhead():
    """Q3: exact-statistics planning time vs the best plan's whole answer time, per E1 cell."""
    cells = list(_e1_cells())
    x = np.array([d["best_total"] for d in cells])
    y1 = np.array([d["planners"]["v1"]["planning"] for d in cells])
    y2 = np.array([d["planners"]["v1_generic"]["planning"] for d in cells])
    fig, ax = plt.subplots(figsize=(COL_IN, 2.2))
    ax.scatter(x, y1, s=12, marker="o", facecolors="none", edgecolors=PAL[0], linewidths=0.8, zorder=3,
               label="exact cut statistics")
    ax.scatter(x, y2, s=12, marker="^", facecolors=PAL[1], edgecolors=PAL[1], linewidths=0.6, zorder=3,
               label="non-zero counts only")
    lo, hi = 5e-4, 10
    ax.plot([lo, hi], [lo, hi], color=MUTED, lw=0.8, zorder=1)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    _logticks(ax, "x", [0.001, 0.01, 0.1, 1, 10]); _logticks(ax, "y", [0.001, 0.01, 0.1, 1, 10])
    ax.set_xlabel("time of the best plan (s)"); ax.set_ylabel("planning time (s)")
    _style(ax)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False, fontsize=7.5)
    fig.tight_layout(pad=0.3); fig.savefig(OUT / "fig_overhead.pdf", bbox_inches="tight", pad_inches=0.05)
    print("fig_overhead: planning > answer on", int((y1 > x).sum()), "of", len(x), "cells (exact)")


def fig_e2e():
    """Q4: per-unit time of the fastest external system over KNITPLAN's, by query type (pilot_summary.json)."""
    d = json.load(open(ROOT / "derisk/results/M1/pilot_summary.json"))
    order = ["marginal", "full / half", "group-by", "DD bins", "points"]
    labels = {"marginal": "marginal", "full / half": "full or\nhalf", "group-by": "group-by", "DD bins": "DD bins",
              "points": "point\nbatch"}
    fig, ax = plt.subplots(figsize=(COL_IN, 2.15))
    rng = np.random.default_rng(1)
    for k, q in enumerate(order):
        for ctl, off, mk, col, fill, lab in ((False, -0.17, "o", PAL[0], PAL[0], "non-control"),
                                             (True, 0.17, "s", PAL[1], "none", "GHZ control")):
            y = np.array([r["text"] / r["tk"] for r in d["rows"] if r["qtype"] == q and r["control"] == ctl
                          and not r["qpu"]])
            if y.size == 0:
                continue
            ax.scatter(k + off + rng.uniform(-0.07, 0.07, y.size), y, s=13, marker=mk, facecolors=fill,
                       edgecolors=col, linewidths=0.8, zorder=3, label=lab if k == 0 else None)
    ax.axhline(1.0, color=MUTED, lw=0.8, zorder=1)
    ax.set_yscale("log"); ax.set_ylim(0.05, 40)
    ax.set_yticks([0.1, 0.3, 1, 3, 10, 30]); ax.set_yticklabels(["0.1", "0.3", "1", "3", "10", "30"])
    ax.set_xticks(range(len(order))); ax.set_xticklabels([labels[q] for q in order])
    ax.set_xlim(-0.55, len(order) - 0.45)
    ax.set_ylabel("external / KNITPLAN time")
    ax.grid(axis="y", color=GRID, lw=0.5, zorder=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False)
    fig.tight_layout(pad=0.3); fig.savefig(OUT / "fig_e2e.pdf", bbox_inches="tight", pad_inches=0.05)
    print("fig_e2e units", len(d["rows"]))


def fig_density():
    """Q2: density of the merged factors (mean over subcircuits) against shots, ideal vs noisy (D1 measurements)."""
    D = {}
    for f in glob.glob(str(ROOT / "derisk/results/D1/*.json")):
        d = json.load(open(f))
        if d["id"].startswith("bv"):
            continue
        m = list(d["merged"].values())   # pooled: stored entries over dense entries of all merged factors
        D[(d["id"], d["noise"], d["S"])] = sum(x["nnz"] for x in m) / sum(x["nnz"] / x["ratio"] for x in m)
    ids = sorted({k[0] for k in D if (k[0], "noisy", 10000) in D})
    names = {"adder20": "adder, 2 subcircuits", "adder20r3": "adder, 3 subcircuits", "aqft20": "approximate QFT",
             "qaoa20b": "QAOA", "sup20b": "shallow supremacy"}
    fig, ax = plt.subplots(figsize=(COL_IN, 3.2))
    Ss = [1000, 10000, 100000]
    for k, i in enumerate(ids):
        mk = "osD^v"[k]
        same = all(abs(D[(i, "ideal", S)] - D[(i, "noisy", S)]) < 0.02 for S in Ss)
        for noise, ls, fill in (("ideal", "-", "none"), ("noisy", "--", PAL[k])):
            if same and noise == "noisy":
                continue   # ideal and noisy coincide (within 2 points); one line, labelled as such
            y = [D.get((i, noise, S)) for S in Ss]
            xs = [S * (0.82 + 0.09 * k) for S, v in zip(Ss, y) if v is not None]; ys = [v for v in y if v is not None]  # small per-family x offset so coincident series stay visible
            ax.plot(xs, ys, ls=ls, color=PAL[k], lw=1.2, marker=mk, ms=4, mfc=fill, mec=PAL[k],
                    label=(names.get(i, i) + (" (both)" if same else "")) if noise == "ideal" else None)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xticks(Ss); ax.set_xticklabels(["1,000", "10,000", "100,000"]); ax.minorticks_off()
    _logticks(ax, "y", [0.001, 0.01, 0.1, 1])
    ax.set_xlabel("shots per configuration"); ax.set_ylabel("stored / dense entries")
    _style(ax)
    h, l = ax.get_legend_handles_labels()
    from matplotlib.lines import Line2D
    h += [Line2D([], [], color=MUTED, ls="-", marker="o", mfc="none", ms=4), Line2D([], [], color=MUTED, ls="--", marker="o", ms=4)]
    l += ["ideal shots (hollow)", "noisy shots (filled)"]
    ax.legend(h, l, loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=2, frameon=False, fontsize=7.5)
    fig.tight_layout(pad=0.3); fig.savefig(OUT / "fig_density.pdf", bbox_inches="tight", pad_inches=0.05)
    print("fig_density instances", ids)


def fig_costmodel():
    """Planning: predicted cost of the exhaustive planner's plan against its measured fold time (E1 cells)."""
    cells = list(_e1_cells())
    xs, ys = [], []
    for d in cells:
        p = d["plans"].get(d["planners"]["v1"]["plan"])
        if p and p.get("fold") is not None and d.get("v1_predicted_cost"):
            xs.append(d["v1_predicted_cost"]); ys.append(p["fold"])
    xs, ys = np.array(xs), np.array(ys)
    fig, ax = plt.subplots(figsize=(COL_IN, 2.1))
    ax.scatter(xs, ys, s=12, marker="o", facecolors="none", edgecolors=PAL[0], linewidths=0.8, zorder=3)
    lo, hi = min(xs.min(), ys.min()) / 2, max(xs.max(), ys.max()) * 2
    ax.plot([lo, hi], [lo, hi], color=MUTED, lw=0.8, zorder=1)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    _logticks(ax, "x", [0.0001, 0.001, 0.01]); _logticks(ax, "y", [0.0001, 0.001, 0.01])
    ax.set_xlabel("predicted cost (s)"); ax.set_ylabel("measured execution (s)")
    _style(ax)
    fig.tight_layout(pad=0.3); fig.savefig(OUT / "fig_costmodel.pdf", bbox_inches="tight", pad_inches=0.05)
    r = np.abs(xs - ys) / ys
    from scipy.stats import spearmanr
    print("fig_costmodel n", len(xs), "median rel err %.2f" % np.median(r), "spearman %.2f" % spearmanr(xs, ys)[0],
          "median ratio pred/meas %.2f" % np.median(xs / ys))


if __name__ == "__main__":
    for name in sys.argv[1:] or ["fig1", "fig_rules", "fig_overhead", "fig_e2e", "fig_density", "fig_costmodel"]:
        globals()[name]()
