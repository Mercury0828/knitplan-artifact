"""D4 — baseline availability and semantics on one pilot instance (the pre-registered protocol §D4).

(1) CutQC dynamic definition, recursion layer 0, run with CutQC's own DynamicDefinition class on our shot tables
    (helper modules that pull IBMQ / simulator code are stubbed; nothing they provide is used by layer 0), compared
    with KNITPLAN's marginal over the same active qubits.
(2) DuckDB SQL formulation over the merged factors (marginal |A| = 10 and full distribution) and over the pushed-down
    factors, compared with KNITPLAN.
"""
import json, pickle, sys, time, types
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, "D:/runs/knitplan/ext/CutQC")
for name, attrs in {"helper_functions.non_ibmq_functions": ["evaluate_circ"],
                    "helper_functions.conversions": ["quasi_to_real"], "helper_functions.metrics": ["MSE"],
                    "cutqc.evaluator": ["get_num_workers"]}.items():
    m = types.ModuleType(name)
    for a in attrs:
        setattr(m, a, lambda *x, **k: (_ for _ in ()).throw(RuntimeError("stub")))
    sys.modules[name] = m
import duckdb
from cutqc.dynamic_definition import DynamicDefinition
from knitplan import engine as E
from knitplan.shots import load_tables
sys.path.insert(0, str(ROOT / "derisk"))
from d6_equivalence import cutqc_graph, cutqc_entry_probs

IID, S = "qaoa20a", 10000
OUT = ROOT / "derisk" / "results"


def dd_layer0(sch, graph, ep, mem_limit):
    dd = DynamicDefinition(compute_graph=graph, num_cuts=sch.c, subcircuit_entry_probs=ep, mem_limit=mem_limit,
                           recursion_depth=1)
    t = time.perf_counter(); dd.build(); dt = time.perf_counter() - t
    layer = dd.dd_bins[0]
    return layer, dt


def main():
    d = pickle.load(open(f"D:/runs/knitplan/pilot/{IID}.pkl", "rb"))
    circ, sch = d["circ"], d["scheme"]; models = E.sub_models(sch)
    tabs = {p: load_tables(f"D:/runs/knitplan/shots/{IID}/ideal/sub{p}_S{S}.npz")[2] for p in range(sch.r)}
    rec = {"instance": IID, "S": S}
    # ---- (1) CutQC DD layer 0
    subs, graph = cutqc_graph(circ, sch)
    ep = cutqc_entry_probs(sch, subs, graph, tabs, S)
    layer, dt = dd_layer0(sch, graph, ep, mem_limit=10)
    bins = np.asarray(layer["bins"], dtype=float)
    # active global qubits in DD bin bit order (first factor of smart_order = most significant)
    order_bits = []
    for p in layer["smart_order"]:
        m = models[p]; states = layer["subcircuit_state"][p]; L = len(m.out_qubits)
        comp = sorted(m.out_local)
        for t, s in enumerate(states):
            if s == "active":
                lq = comp[L - 1 - t]
                order_bits.append(m.out_qubits[m.out_local.index(lq)])
    A = sorted(order_bits)
    plan = E.Plan("before", (0, 1), ("D",), "joint")
    o, i, v = E.execute(plan, models, tabs, S, set(A), None, sch.c)
    i, v = E.canonical(o, i, v, A)
    ours = np.zeros(2 ** len(A)); ours[i] = v
    # map DD bin index -> canonical index
    nb = len(order_bits); idx = np.arange(2 ** nb); can = np.zeros_like(idx)
    for k, q in enumerate(order_bits):           # bit (nb-1-k) of the DD index is order_bits[k]
        can |= ((idx >> (nb - 1 - k)) & 1) << A.index(q)
    dd_can = np.zeros_like(ours); dd_can[can] = bins
    tau, W = E.tau_eq(models, tabs, S, set(A), None, sch.c)
    rec["cutqc_dd_layer0"] = {"active_qubits": A, "seconds": dt, "max_abs_diff_vs_knitplan": float(np.abs(dd_can - ours).max()),
                              "tau_eq": tau, "note": "CutQC DD merges bins in float32 (merge_prob_vector dtype)"}
    print("DD", rec["cutqc_dd_layer0"], flush=True)
    # ---- (2) DuckDB over merged factors
    con = duckdb.connect(); con.execute("SET threads TO 1")
    fs = {m.p: E.build_factor(m, tabs[m.p], S, A=None, sel=None, position="after") for m in models}
    for p, f in fs.items():
        rows = []
        for key, (y, val) in f.data.items():
            for yy, vv in zip(y.tolist(), val.tolist()):
                rows.append(tuple(key) + (yy, vv))
        cols = [f"b{k}" for k in f.vars] + ["y", "v"]
        con.execute(f"CREATE TABLE psi{p} ({', '.join(c + (' BIGINT' if c != 'v' else ' DOUBLE') for c in cols)})")
        con.executemany(f"INSERT INTO psi{p} VALUES ({', '.join('?' for _ in cols)})", rows)
    rng = np.random.default_rng(42)
    for qname, Aq in [("marg10", sorted(rng.choice(sch.n, size=10, replace=False).tolist())), ("full", list(range(sch.n)))]:
        terms = []
        for p, f in fs.items():
            for q in Aq:
                if q in f.outs:
                    terms.append(f"(((psi{p}.y >> {f.outs.index(q)}) & 1) << {Aq.index(q)})")
        zexpr = " | ".join(terms) if terms else "0"
        shared = [k for k in fs[0].vars if k in fs[1].vars]
        on = " AND ".join(f"psi0.b{k} = psi1.b{k}" for k in shared) or "TRUE"
        sql = f"SELECT {zexpr} AS z, SUM(psi0.v * psi1.v) * {2.0 ** -sch.c} AS q FROM psi0 JOIN psi1 ON {on} GROUP BY z"
        con.execute(sql).fetchall()
        ts = []
        for _ in range(5):
            t = time.perf_counter(); res = con.execute(sql).fetchall(); ts.append(time.perf_counter() - t)
        got = np.zeros(2 ** len(Aq))
        for z, q in res:
            got[z] = q
        o, i, v = E.execute(plan, models, tabs, S, set(Aq), None, sch.c)
        i, v = E.canonical(o, i, v, Aq)
        ours = np.zeros(2 ** len(Aq)); ours[i] = v
        tau, W = E.tau_eq(models, tabs, S, set(Aq), None, sch.c)
        rec[f"duckdb_merged_{qname}"] = {"median_s": float(np.median(ts)), "max_abs_diff": float(np.abs(got - ours).max()),
                                         "tau_eq": tau, "sql": sql}
        print("DuckDB", qname, rec[f"duckdb_merged_{qname}"]["median_s"], rec[f"duckdb_merged_{qname}"]["max_abs_diff"], tau, flush=True)
    rec["duckdb_version"] = duckdb.__version__
    (OUT / "D4.json").write_text(json.dumps(rec, indent=1))


if __name__ == "__main__":
    main()
