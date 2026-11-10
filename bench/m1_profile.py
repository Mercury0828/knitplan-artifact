"""Phase breakdown of KNITPLAN on one M1 unit (diagnosis only, untimed; not evidence).
Usage: python bench/m1_profile.py <instance_id> <query> [noise] [S]
Prints build / plan / fold / canonical seconds, the chosen plan, planner info and per-node intermediate sizes."""
import json, pickle, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import m1_time as M
import systems as SY
from knitplan import engine as E, planner as P


def main():
    iid, qn = sys.argv[1], sys.argv[2]
    noise = sys.argv[3] if len(sys.argv) > 3 else "noisy"
    S = int(sys.argv[4]) if len(sys.argv) > 4 else 10000
    model = P.CostModel(json.loads(M.CM.read_text())["k"])
    circ, sch, models, tabs = M.load_cell(iid, noise, S)
    q = {x["name"]: x for x in pickle.load(open(M.QDIR / f"{M.cell_id(iid, noise, S)}.pkl", "rb"))}[qn]
    ctx = SY.make_ctx(circ, sch, tabs, S, q["A"], q["sel"], model)
    if len(sys.argv) > 5 and sys.argv[5] == "new":     # large answers: new path only, one run (bounded memory)
        t0 = time.perf_counter()
        fs = {m.p: E.build_factor(m, ctx["tabs"][m.p], S, A=set(ctx["A"]), sel=ctx["sel"], position="before")
              for m in ctx["models"]}
        plan, cost, tp, info = P.choose_v2(fs, ctx["models"], model)
        t1 = time.perf_counter()
        ans = P.fold_tree(plan, fs, ctx["c"], A_sorted=ctx["A"])
        print(f"answer={ans[0].size} build+plan {t1-t0:.3f} fold+canon {time.perf_counter()-t1:.3f}")
        print("plan", P.tree_str(plan.tree), plan.fmts, plan.gran, "pred", cost, "info", info)
        return
    for rep in range(2):
        t0 = time.perf_counter()
        fs = {m.p: E.build_factor(m, ctx["tabs"][m.p], S, A=set(ctx["A"]), sel=ctx["sel"], position="before")
              for m in ctx["models"]}
        t1 = time.perf_counter()
        plan, cost, tp, info = P.choose_v2(fs, ctx["models"], model)
        t2 = time.perf_counter()
        o, i, v = P.fold_tree(plan, fs, ctx["c"])
        t3 = time.perf_counter()
        ans = E.canonical(o, i, v, ctx["A"])
        t4 = time.perf_counter()
        ans2 = P.fold_tree(plan, fs, ctx["c"], A_sorted=ctx["A"])
        t5 = time.perf_counter()
    print(f"r={sch.r} |A|={len(q['A'])} sel={q['sel']} answer={ans[0].size}")
    print(f"canonical fold (new path) {t5-t4:.4f}  max diff vs old {SY.max_diff(ans, ans2):.3g}  tau {q['tau']:.3g}")
    print({p: (len(f.vars), len(f.outs), f.nnz()) for p, f in fs.items()})
    print(f"build {t1-t0:.4f}  plan {t2-t1:.4f}  fold {t3-t2:.4f}  canon {t4-t3:.4f}  total {t4-t0:.4f}")
    print("plan", P.tree_str(plan.tree), plan.fmts, plan.gran, "pred", cost)
    print("info", info)


if __name__ == "__main__":
    main()
