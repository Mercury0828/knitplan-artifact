"""Equivalence test: vectorised planner.join_stat vs the reference join_stat_ref (specification), on real E1 factors.
For every E1 T instance (ideal, S = 1e3 and 1e4, two queries): every internal node of every tree, exact (leaf-leaf),
estimated and generic statistics compared (support per key, rows, pairs); and planner v1 / v1-generic choices with the
fast vs reference statistics. Prints counts; asserts zero mismatches."""
import json, pickle, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "derisk"))
from knitplan import engine as E, planner as P
from knitplan.shots import load_tables
from e1_time import queries


def same(x, y):
    sx, rx, px = x; sy, ry, py = y
    dx = {k: v for k, v in sx.sup_dict().items() if v}; dy = {k: v for k, v in sy.sup_dict().items() if v}
    return sx.vars == sy.vars and dx == dy and abs(rx - ry) <= 1e-9 * max(1, abs(ry)) and px == py


def main():
    man = json.loads((ROOT / "bench" / "E1_MANIFEST.json").read_text())
    ref_js = P.join_stat_ref
    nodes = mism = plans = pmism = 0
    t_fast = t_ref = 0.0
    for idx, inst in enumerate(man["instances"]):
        if inst["stratum"] != "T":
            continue
        d = pickle.load(open(f"D:/runs/knitplan/e1/inst/{inst['id']}.pkl", "rb")); sch = d["scheme"]; models = E.sub_models(sch)
        for S in (1000, 10000):
            tabs = {p: load_tables(f"D:/runs/knitplan/e1/shots/{inst['id']}/ideal/sub{p}_S{S}.npz")[2] for p in range(sch.r)}
            for qname, (A, sel) in list(queries(sch.n, idx).items())[:2]:
                fs = {m.p: E.build_factor(m, tabs[m.p], S, A=set(A), sel=sel, position="before") for m in models}
                for kind, mk in (("exact", P.leaf_stat), ("generic", P.generic_leaf_stat)):
                    st = {p: mk(f) for p, f in fs.items()}
                    allv = {p: set(f.vars) for p, f in fs.items()}
                    for t in P.enumerate_trees(fs):
                        def ev(t, fn):
                            if isinstance(t, int):
                                return st[t]
                            a = ev(t[0], fn); b = ev(t[1], fn)
                            inside = set(P.leaves(t))
                            outside = set().union(*(allv[p] for p in fs if p not in inside)) if len(inside) < len(fs) else set()
                            elim = tuple(sorted((set(a.vars) | set(b.vars)) - outside))
                            return fn(a, b, elim)[0], (a, b, elim)
                        # compare node by node using the reference child stats as inputs to both
                        def walk(t):
                            nonlocal nodes, mism, t_fast, t_ref
                            if isinstance(t, int):
                                return st[t]
                            a = walk(t[0]); b = walk(t[1])
                            inside = set(P.leaves(t))
                            outside = set().union(*(allv[p] for p in fs if p not in inside)) if len(inside) < len(fs) else set()
                            elim = tuple(sorted((set(a.vars) | set(b.vars)) - outside))
                            t0 = time.perf_counter(); y = ref_js(a, b, elim); t_ref += time.perf_counter() - t0
                            t0 = time.perf_counter(); x = P.join_stat(a, b, elim); t_fast += time.perf_counter() - t0
                            nodes += 1
                            if not same(x, y):
                                mism += 1
                                if mism <= 5:
                                    print("MISMATCH", inst["id"], S, qname, kind, P.tree_str(t), x[1:], y[1:])
                            return y[0]
                        walk(t)
                m = P.CostModel(P.DEFAULT_K)
                fast = (P.choose(fs, m)[:2], P.choose_generic(fs, m)[:2])
                P.join_stat, keep = P.join_stat_ref, P.join_stat
                try:
                    ref = (P.choose(fs, m)[:2], P.choose_generic(fs, m)[:2])
                finally:
                    P.join_stat = keep
                plans += 2
                for (pf, cf), (pr, cr) in zip(fast, ref):
                    if pf.key() != pr.key() or abs(cf - cr) > 1e-9 * abs(cr):
                        pmism += 1; print("PLAN MISMATCH", inst["id"], S, qname, pf.key(), pr.key(), cf, cr)
        print(inst["id"], "nodes", nodes, "mismatches", mism, f"fast {t_fast:.1f}s ref {t_ref:.1f}s", flush=True)
    print(f"nodes {nodes}, stat mismatches {mism}, planner choices {plans}, choice mismatches {pmism}")
    assert mism == 0 and pmism == 0


if __name__ == "__main__":
    main()
