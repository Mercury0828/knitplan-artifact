"""E2 sizing pass (the pre-registered protocol; untimed, NOT a result): runs ONLY rule (b) (dense, CutQC order, pushdown below the
merge) once per (E2 instance, S = 10^5, query), to size the per-plan cap and per-cell budget fixed in Entry 6b. No
other plan, no oracle, no comparison. Records factor-build time, rule-(b) fold time or DNF reason, and the plan-space
size. Output: D:/runs/knitplan/e2/size.jsonl (outside the results tree on purpose)."""
_RUNS = __import__("os").environ.get("KNITPLAN_RUNS", "D:/runs/knitplan")   # data root
import json, pickle, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "derisk"))
from knitplan import engine as E, planner as P
from knitplan.shots import load_tables
from e2_time import queries

OUT = Path(_RUNS + "/e2/size.jsonl")


def main():
    man = json.loads((ROOT / "bench/E2_MANIFEST.json").read_text())
    done = {json.loads(l)["key"] for l in OUT.read_text().splitlines() if l.strip()} if OUT.exists() else set()
    S = 100000
    shot_man = Path(_RUNS + "/e2/shots/manifest.jsonl")
    shot_done = {json.loads(l)["unit"] for l in shot_man.read_text().splitlines() if l.strip()}
    for idx, inst in enumerate(man["instances"]):
        if not all(f"E2|{inst['id']}|ideal|{p}" in shot_done for p in range(inst.get("r", 99))):
            print(f"skip {inst['id']} (shots not complete yet)", flush=True)
            continue
        d = pickle.load(open(f"{_RUNS}/e2/inst/{inst['id']}.pkl", "rb")); sch = d["scheme"]; models = E.sub_models(sch)
        tabs = {p: load_tables(f"{_RUNS}/e2/shots/{inst['id']}/ideal/sub{p}_S{S}.npz")[2] for p in range(sch.r)}
        for qname, (A, sel) in queries(sch.n, idx).items():
            key = f"{inst['id']}|{qname}"
            if key in done:
                continue
            row = {"key": key, "r": sch.r, "c": sch.c}
            t = time.perf_counter()
            fs = {m.p: E.build_factor(m, tabs[m.p], S, A=set(A), sel=sel, position="before") for m in models}
            row["build_s"] = time.perf_counter() - t
            row["plan_space"] = len(P.enumerate_tree_plans(fs, positions=("before",)))
            b = P.fixed_rules(fs, models)["b_dense_cutqc_order"]
            t = time.perf_counter()
            try:
                P.fold_tree(b, fs, sch.c); row["rule_b_fold_s"] = time.perf_counter() - t
            except (E.MemoryCap, MemoryError) as e:
                row["rule_b_dnf"] = str(e)[:120]; row["rule_b_fold_s"] = time.perf_counter() - t
            with open(OUT, "a") as f:
                f.write(json.dumps(row) + "\n")
            print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
