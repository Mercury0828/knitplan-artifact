"""Summary of the M1 untimed sizing pilot (D:/runs/knitplan/m1/pilot/, the pre-registered protocol;).

NOT the locked protocol: one warm-up + one run per unit, shared E-core, no CPU_LOCK. Reported in the paper as
preliminary. External comparator per unit = the fastest of the externals run in BOTH pilot passes (cutqc_dd, li_sparse,
duckdb_pushdown, qcut, galley) that finished and passed the V-check. A unit counts only if KNITPLAN finished (KNITPLAN
time-DNF would count at the pass cap; none expected) and at least one comparator finished. Units where no comparator
finished are counted separately (KNITPLAN answered, nobody else did).
Output: derisk/results/M1/pilot_summary.json and paper/sections/tab_pilot.tex
Usage: python bench/m1_pilot_summary.py
"""
_RUNS = __import__("os").environ.get("KNITPLAN_RUNS", "D:/runs/knitplan")   # data root
import collections, glob, json, math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXT = ["cutqc_dd", "li_sparse", "duckdb_pushdown", "qcut", "galley"]
CONTROLS = ("bv", "ghz")
QTYPE = {"marg10": "marginal", "marg20": "marginal", "marg30": "marginal", "margn": "full / half",
         "marghalf": "full / half", "gb_rand": "group-by", "gb_struct": "group-by", "dd0": "DD bins", "dd1": "DD bins",
         "dd2": "DD bins", "pts1e3": "points", "pts1e5": "points"}
NAMES = {"cutqc_dd": "CutQC DD", "li_sparse": "Li et al.", "duckdb_pushdown": "DuckDB", "qcut": "QCut",
         "galley": "Galley"}


def gm(x):
    return math.exp(sum(map(math.log, x)) / len(x)) if x else float("nan")


def main():
    R = [json.load(open(f)) for f in glob.glob(_RUNS + "/m1/pilot/*.json")]
    by = collections.defaultdict(dict)
    for r in R:
        by[(r["cell"], r["query"])][r["system"]] = r
    rows = []
    only_k, k_fail, errors = 0, 0, 0
    for (cell, q), d in by.items():
        errors += sum(1 for r in d.values() if r.get("dnf_kind") == "error")
        k = d.get("knitplan")
        if k is None:
            continue
        if k.get("time") is None or k.get("vcheck_pass") is False:
            k_fail += 1; continue
        ex = [(d[s]["time"], s) for s in EXT if s in d and d[s].get("time") is not None and d[s].get("vcheck_pass")]
        if not ex:
            only_k += 1; continue
        te, s = min(ex)
        allo = [d[x]["time"] for x in d if x != "knitplan" and d[x].get("time") is not None]
        fam = cell.split("_noisy")[0].split("_qpu")[0]
        fam = next((f for f in CONTROLS if fam.startswith(f)), "x")
        rows.append({"cell": cell, "query": q, "qtype": QTYPE.get(q, q), "control": fam in CONTROLS,
                     "qpu": "_qpu_" in cell, "tk": k["time"], "text": te, "best_ext": s,
                     "k_fastest": k["time"] <= min(allo) if allo else True})
    out = {"units": len(rows), "knitplan_only_answered": only_k, "knitplan_failed": k_fail, "error_units": errors,
           "cells": sorted({r["cell"] for r in rows})}
    groups = {}
    for name, sel in [("non-control", lambda r: not r["control"] and not r["qpu"]),
                      ("control (GHZ)", lambda r: r["control"] and not r["qpu"])]:
        g = [r for r in rows if sel(r)]
        groups[name] = g
    tab = []
    for name, g in groups.items():
        if not g:
            continue
        for qt in ["marginal", "full / half", "group-by", "DD bins", "points", "all"]:
            h = g if qt == "all" else [r for r in g if r["qtype"] == qt]
            if not h:
                continue
            ratios = [r["text"] / r["tk"] for r in h]
            win = collections.Counter(r["best_ext"] for r in h).most_common(1)[0][0]
            tab.append({"group": name, "qtype": qt, "units": len(h), "gmean": gm(ratios),
                        "k_faster_share": sum(x > 1 for x in ratios) / len(h), "strongest": NAMES[win]})
    out["table"] = tab
    (ROOT / "derisk/results/M1").mkdir(parents=True, exist_ok=True)
    (ROOT / "derisk/results/M1/pilot_summary.json").write_text(json.dumps({**out, "rows": rows}, indent=1))
    lines = [r"\begin{table}[t]",
             r"\caption{End-to-end comparison (noisy cells, $S=10^4$): geometric mean "
             r"of the strongest external system's time over KNITPLAN's, share of units where KNITPLAN is faster, and the "
             r"system most often strongest.}\label{tab:pilot}",
             r"\centering\small", r"\begin{tabular}{@{}llrrrl@{}}", r"\toprule",
             r"Instances & Query & Units & gmean & K faster & Strongest\\", r"\midrule"]
    last = None
    for t in tab:
        g = t["group"] if t["group"] != last else ""
        if last is not None and g:
            lines.append(r"\midrule")
        last = t["group"]
        lines.append(f"{g} & {t['qtype']} & {t['units']} & ${t['gmean']:.2f}\\times$ & "
                     f"${100 * t['k_faster_share']:.0f}\\%$ & {t['strongest']}\\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (ROOT / "paper/sections").mkdir(parents=True, exist_ok=True)
    (ROOT / "paper/sections/tab_pilot.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(out, indent=1)[:3000])


if __name__ == "__main__":
    main()
