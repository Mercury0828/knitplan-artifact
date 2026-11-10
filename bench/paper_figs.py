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
                     "axes.linewidth": 0.6, "xtick.color": INK, "ytick.color": INK})


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
    ax.text(-0.5, 0.13, "sparse is faster", fontsize=7, color=MUTED, va="bottom")
    ax.text(-0.5, 85, "dense is faster", fontsize=7, color=MUTED, va="top")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False, handletextpad=0.3,
              columnspacing=1.0)
    fig.tight_layout(pad=0.3)
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "fig1_flip.pdf")
    n = {k: sum(len(v) for v in pts[k].values()) for k in pts}
    print("fig1: cells", n, "sparse faster:", {k: sum(x < 1 for v in pts[k].values() for x in v) for k in pts})


if __name__ == "__main__":
    for name in sys.argv[1:] or ["fig1"]:
        globals()[name]()
