"""Systems compared in the main evaluation.

Uniform interface: ``SYSTEMS[name](ctx) -> (keys, vals)`` with keys = sorted int64 codes over the query's free qubits A
in ascending global order (bit k of a key = qubit A[k]), vals = the answer Q(z) (absent key = 0). Each call starts
_RUNS = __import__("os").environ.get("KNITPLAN_RUNS", "D:/runs/knitplan")   # data root
from the raw shot tables and does everything the system needs (statistics, planning / path search, format conversion,
execution), so its wall time is the end-to-end time Γ. A system that cannot run within the memory rule raises
``Infeasible`` (recorded as DNF with the reason, never dropped).

ctx: dict with keys circ, sch, models, tabs, S, A (sorted list), sel (dict qubit -> bit or None), c, n, cost_model.
Memory rule (same for every system, KNITPLAN included): no single dense array above DENSE_ENTRIES float64 entries;
sparse operators above engine.ROW_CAP rows are DNF.
"""
import json, math, sys, time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "derisk"))
from knitplan import engine as E, planner as P, points as PT

DENSE_ENTRIES = 2 ** 30      # 8 GiB of float64 per array, every system


class Infeasible(Exception):
    pass


def _select_and_marginalize(outs, idx, val, A, sel):
    """Apply a fixed-bit selection on full-output keys, then aggregate onto A (canonical order)."""
    if sel:
        m = np.ones(idx.size, bool)
        for q, b in sel.items():
            m &= ((idx >> outs.index(q)) & 1) == b
        idx, val = idx[m], val[m]
    return E.canonical(outs, idx, val, A)


def _finish(outs, idx, val, ctx):
    """Answer the query from a full answer (codes over `outs`): marginal / group-by, or point lookup for a batch."""
    Z = ctx.get("points")
    if Z is None:
        return _select_and_marginalize(outs, idx, val, ctx["A"], ctx["sel"])
    if len(outs) > 62:
        raise Infeasible("full-answer codes need more than 62 bits")
    codes = PT.local_codes(Z, list(outs))
    o = np.argsort(idx); idx, val = idx[o], val[o]
    pos = np.searchsorted(idx, codes); pos[pos >= idx.size] = 0
    hit = (idx[pos] == codes) if idx.size else np.zeros(codes.size, bool)
    keys = np.flatnonzero(hit)
    v = val[pos[hit]]
    nz = v != 0.0
    return keys[nz].astype(np.int64), v[nz]


# ------------------------------------------------------------------------------------------------ KNITPLAN
def knitplan(ctx):
    """Full system: pushdown below the merge (L2), planner v2 (budgeted, cut statistics), executor. Point batches:
    semi-join pushdown + batched fold (knitplan/points.py)."""
    if ctx.get("points") is not None:
        return PT.answer_points(ctx["models"], ctx["tabs"], ctx["S"], ctx["points"], ctx["c"])
    fs = {m.p: E.build_factor(m, ctx["tabs"][m.p], ctx["S"], A=set(ctx["A"]), sel=ctx["sel"], position="before")
          for m in ctx["models"]}
    plan, _, _, _ = P.choose_v2(fs, ctx["models"], ctx["cost_model"])
    o, i, v = P.fold_tree(plan, fs, ctx["c"])
    return E.canonical(o, i, v, ctx["A"])


def default_rule(ctx):
    """The theory-derived default rule (a): push everything down, ascending support order, sparse below rho*.
    Point batches: KNITPLAN's point path is rule (a) already (no planner search), so the two coincide there."""
    if ctx.get("points") is not None:
        return PT.answer_points(ctx["models"], ctx["tabs"], ctx["S"], ctx["points"], ctx["c"])
    fs = {m.p: E.build_factor(m, ctx["tabs"][m.p], ctx["S"], A=set(ctx["A"]), sel=ctx["sel"], position="before")
          for m in ctx["models"]}
    dplan = E.default_rule_from_factors(fs)
    o, i, v = P.fold_tree(P.TreePlan(dplan.position, P.left_deep(dplan.order), dplan.fmts, dplan.gran), fs, ctx["c"])
    return E.canonical(o, i, v, ctx["A"])


# ------------------------------------------------------------------------------------------------ dense tensor networks
def _dense_tensor(f: E.Factor):
    """Engine factor -> dense tensor with one axis per cut variable (size 4) and one per output qubit (size 2).
    Axis order: f.vars, then f.outs in descending bit order (C-order reshape of the packed output index)."""
    if (4 ** len(f.vars)) * (2 ** len(f.outs)) > DENSE_ENTRIES:
        raise Infeasible(f"dense factor of {len(f.vars)} cuts x {len(f.outs)} outputs")
    d = E.to_format(f, "D").data if f.fmt != "D" else f.data
    arr = d.reshape((4,) * len(f.vars) + (2,) * len(f.outs))
    return arr, list(f.vars), list(reversed(f.outs))


def _contract_dense(factors, out_qubits, c, optimizer):
    import opt_einsum as oe
    sym = {}

    def s(key):
        if key not in sym:
            sym[key] = oe.get_symbol(len(sym))
        return sym[key]
    ops, terms = [], []
    for f in factors:
        arr, cv, oq = _dense_tensor(f)
        ops.append(arr)
        terms.append("".join(s(("c", v)) for v in cv) + "".join(s(("q", q)) for q in oq))
    out = "".join(s(("q", q)) for q in reversed(out_qubits))      # bit k of the result = out_qubits[k]
    expr = ",".join(terms) + "->" + out
    path, info = oe.contract_path(expr, *ops, optimize=optimizer)
    if info.largest_intermediate > DENSE_ENTRIES or 2 ** len(out_qubits) > DENSE_ENTRIES:
        raise Infeasible(f"largest intermediate {info.largest_intermediate:.3g} entries")
    res = oe.contract(expr, *ops, optimize=path).reshape(-1) * (2.0 ** -c)
    i = np.flatnonzero(res)
    return i.astype(np.int64), res[i]


def _cotengra():
    import cotengra as ctg
    return ctg.HyperOptimizer(methods=["greedy", "labels"], max_repeats=32, max_time=10, progbar=False,
                              parallel=False)   # the protocol: single thread / single process for every system


def tensorqc_dense(ctx):
    """TensorQC-style: dense merged factors over all outputs, cotengra contraction order, full distribution, then the
    query is answered from it (the published system reconstructs the full distribution)."""
    if 2 ** ctx["n"] > DENSE_ENTRIES:
        raise Infeasible(f"full distribution 2^{ctx['n']}")
    fs = [E.build_factor(m, ctx["tabs"][m.p], ctx["S"], A=None, sel=None, position="after") for m in ctx["models"]]
    allq = list(range(ctx["n"]))
    i, v = _contract_dense(fs, allq, ctx["c"], _cotengra())
    return _finish(allq, i, v, ctx)


def dense_pushdown(ctx):
    """Dense + pushdown (L1, inside the merged factors) + cotengra order: the dense baseline given our rewrite.
    Point batches: the same semi-join pushdown as KNITPLAN, dense point factors (4,)*k + (P,), one batched contraction
    with the point index as a shared output index (cotengra path), in point chunks within the memory rule."""
    if ctx.get("points") is not None:
        return _dense_points(ctx)
    fs = [E.build_factor(m, ctx["tabs"][m.p], ctx["S"], A=set(ctx["A"]), sel=ctx["sel"], position="after")
          for m in ctx["models"]]
    return _contract_dense(fs, list(ctx["A"]), ctx["c"], _cotengra())


def _dense_points(ctx):
    import opt_einsum as oe
    Z = ctx["points"]; P = Z.shape[0]
    fs = PT.build_point_factors(ctx["models"], ctx["tabs"], ctx["S"], Z)
    allv = sorted(set(v for f in fs.values() for v in f.vars))
    L = {v: oe.get_symbol(j) for j, v in enumerate(allv)}; bi = oe.get_symbol(len(allv))
    terms = ["".join(L[v] for v in f.vars) + bi for f in fs.values()]
    expr = ",".join(terms) + "->" + bi
    widest = max(4 ** len(f.vars) for f in fs.values())
    chunk = max(1, min(P, DENSE_ENTRIES // widest))
    shapes = [(4,) * len(f.vars) + (chunk,) for f in fs.values()]
    path, info = oe.contract_path(expr, *shapes, shapes=True, optimize=_cotengra())
    if info.largest_intermediate > DENSE_ENTRIES:
        raise Infeasible(f"largest intermediate {info.largest_intermediate:.3g} entries")
    out = np.zeros(P)
    for s0 in range(0, P, chunk):
        s1 = min(P, s0 + chunk); ops = []
        for f in fs.values():
            arr = np.zeros((4,) * len(f.vars) + (s1 - s0,))
            for key, (i, v) in f.data.items():
                m = (i >= s0) & (i < s1)
                arr[key][i[m] - s0] = v[m]
            ops.append(arr)
        out[s0:s1] = oe.contract(expr, *ops, optimize=path)
    out *= 2.0 ** -ctx["c"]
    k = np.flatnonzero(out)
    return k.astype(np.int64), out[k]


# ------------------------------------------------------------------------------------------------ CutQC
def cutqc_fd(ctx):
    """CutQC full definition (commit cda68c7): attribute_shots on dense subcircuit vectors + GraphContractor, then the
    query is answered from the full distribution (CutQC's FD path reconstructs all 2^n entries)."""
    if 2 ** ctx["n"] > DENSE_ENTRIES:
        raise Infeasible(f"full distribution 2^{ctx['n']}")
    if max(s.num_qubits for s in ctx["subs"]) > 30:
        raise Infeasible("dense subcircuit vector > 2^30")
    from d6_equivalence import cutqc_entry_probs, cutqc_full
    ep = cutqc_entry_probs(ctx["sch"], ctx["subs"], ctx["graph"], ctx["tabs"], ctx["S"])
    full = cutqc_full(ctx["sch"], ctx["subs"], ctx["graph"], ep)
    i = np.flatnonzero(full)
    return _finish(list(range(ctx["n"])), i.astype(np.int64), full[i], ctx)


def _vt_measure(idx, val, meas):
    """QCutSim VT on CutQC's measure_prob: signed trace-out as a sign vector + scatter-add (bincount) instead of the
    per-entry Python loop. Same conventions as cutqc/evaluator.measure_state: bit k of the state <-> meas[k]; effective
    bit j <-> the j-th smallest 'comp' position. Runs over the observed entries of the shot table (the absent ones are
    zero), the strongest form of the same computation."""
    comp = [k for k, m in enumerate(meas) if m == "comp"]
    smask = sum(1 << k for k, m in enumerate(meas) if m not in ("I", "comp"))
    w = val.astype(float)
    if smask:
        w = np.where(np.bitwise_count(idx & smask) & 1, -w, w)
    eff = np.zeros_like(idx)
    for j, k in enumerate(comp):
        eff |= ((idx >> k) & 1) << j
    return np.bincount(eff, weights=w, minlength=1 << len(comp))


CUTQC_TOTAL_ENTRIES = 6 * 2 ** 30     # dense entry vectors held at once by the CutQC paths (48 GiB of float64)


def _cutqc_vt_entries(ctx):
    """CutQC's subcircuit entry vectors (generate_subcircuit_entries + attribute_shots, authors' code) with the
    measurement trace-out done by VT. Dense vectors of 2^(effective outputs) entries, as in CutQC."""
    from d6_equivalence import attribute_shots, generate_subcircuit_entries
    from knitplan.shots import configurations
    sch, subs, graph, S = ctx["sch"], ctx["subs"], ctx["graph"], ctx["S"]
    if max(s.num_qubits for s in subs) > 30:
        raise Infeasible("dense subcircuit vector > 2^30")
    entries, instances = generate_subcircuit_entries(graph)
    tot = sum(len(entries[p]) * 2 ** len(m.out_qubits) for p, m in zip(range(sch.r), ctx["models"]))
    if tot > CUTQC_TOTAL_ENTRIES:
        raise Infeasible(f"CutQC entry vectors {tot:.3g} float64 entries > memory rule")
    ep = {}
    for p in range(sch.r):
        ins, outs, _ = configurations(sch, p)
        mp = {}
        for init, meas in instances[p]:
            phys_meas = tuple("comp" if m in ("I", "Z", "comp") else m for m in meas)
            idx, cnt = ctx["tabs"][p][(tuple(init[i] for i in ins), tuple(phys_meas[i] for i in outs))]
            mp[(init, meas)] = _vt_measure(np.asarray(idx, np.int64), np.asarray(cnt) / S, meas)
        ep[p] = attribute_shots(mp, entries[p])
    return ep


def _ec_contract(graph, ep, c):
    """CutQC's GraphContractor (same edge-basis loop, smart_order) with EC: broadcast outer products accumulated in
    place instead of np.kron. Returns (vector, smart_order); first factor of smart_order = most significant bits."""
    import itertools
    from d6_equivalence import GraphContractor

    class _EC(GraphContractor):
        def compute(self):
            g = self.compute_graph
            edges = g.get_edges(from_node=None, to_node=None)
            tot = None
            for bases in itertools.product(["I", "X", "Y", "Z"], repeat=len(edges)):
                g.assign_bases_to_edges(edge_bases=bases, edges=edges)
                term = None
                for p in self.smart_order:
                    v = self.subcircuit_entry_probs[p][g.get_init_meas(subcircuit_idx=p)]
                    term = v if term is None else np.multiply.outer(term, v)
                if tot is None:
                    tot = np.array(term, dtype=float, copy=True)
                else:
                    tot += term
                g.remove_bases_from_edges(edges=g.edges)
            return tot.reshape(-1) * (2.0 ** -self.num_cuts)
    gc = _EC(compute_graph=graph, subcircuit_entry_probs=ep, num_cuts=c)
    return gc


def cutqc_opt(ctx):
    """CutQC FD at its strongest: CutQC's own entry attribution and contraction graph with the two applicable QCutSim
    optimizations (TODAES 2026, no public code; re-implemented): VT for the measurement trace-out and EC (broadcast outer
    products accumulated in place instead of np.kron) for the 4^c-term contraction. Full distribution, then the query."""
    if 2 ** ctx["n"] > DENSE_ENTRIES:
        raise Infeasible(f"full distribution 2^{ctx['n']}")
    gc = _ec_contract(ctx["graph"], _cutqc_vt_entries(ctx), ctx["sch"].c)
    full = _kron_to_global(ctx["sch"], gc)
    i = np.flatnonzero(full)
    return _finish(list(range(ctx["n"])), i.astype(np.int64), full[i], ctx)


def _dd_merge(vec, states):
    """CutQC merge_prob_vector semantics (states[t] <-> bit L-1-t of the entry index; 'active' bits form the bin id,
    first active = most significant; '0'/'1' = zoomed, entries with another value dropped; 'merged' summed), vectorized
    over the non-zero entries and accumulated in float64 (CutQC accumulates in float32)."""
    L = len(states)
    act = [L - 1 - t for t, st in enumerate(states) if st == "active"]
    i = np.flatnonzero(vec); w = vec[i]
    keep = np.ones(i.size, bool)
    for t, st in enumerate(states):
        if st in (0, 1, "0", "1"):
            keep &= ((i >> (L - 1 - t)) & 1) == int(st)
    i, w = i[keep], w[keep]
    b = np.zeros_like(i); na = len(act)
    for j, bit in enumerate(act):
        b |= ((i >> bit) & 1) << (na - 1 - j)
    return np.bincount(b, weights=w, minlength=1 << na)


def cutqc_dd(ctx):
    """CutQC dynamic definition for one bin query (A active, sigma zoomed, the rest merged) at its strongest: CutQC's
    DD semantics (merge_prob_vector, GraphContractor on the merged vectors) with VT entries, a vectorized float64 merge
    (QCutSim-style; CutQC's float32 merge cannot meet tau_eq, BASELINE STATUS D4) and EC contraction. DD answers bins,
    so a marginal / group-by is one DD recursion layer with this schedule. Not defined for point batches."""
    if ctx.get("points") is not None:
        raise Infeasible("CutQC DD answers bin queries, not point batches")
    sch, models, A, sel = ctx["sch"], ctx["models"], ctx["A"], ctx["sel"] or {}
    if 2 ** len(A) > DENSE_ENTRIES:
        raise Infeasible(f"dense bin vector 2^{len(A)}")
    ep = _cutqc_vt_entries(ctx)
    states = {}
    for m in models:
        comp = sorted(m.out_local); L = len(comp); st = []
        for t in range(L):
            q = m.out_qubits[m.out_local.index(comp[L - 1 - t])]
            st.append("active" if q in A else (str(sel[q]) if q in sel else "merged"))
        states[m.p] = st
    merged = {p: {k: _dd_merge(v, states[p]) for k, v in ep[p].items()} for p in ep}
    gc = _ec_contract(ctx["graph"], merged, sch.c)
    order_bits = []
    for p in gc.smart_order:
        m = models[p]; comp = sorted(m.out_local); L = len(comp)
        for t, st in enumerate(states[p]):
            if st == "active":
                order_bits.append(m.out_qubits[m.out_local.index(comp[L - 1 - t])])
    vec = gc.reconstructed_prob; nb = len(order_bits)
    i = np.flatnonzero(vec)
    can = np.zeros(i.size, np.int64)
    for k, q in enumerate(order_bits):
        can |= ((i >> (nb - 1 - k)) & 1) << A.index(q)
    return E.aggregate(can, vec[i])


def _kron_to_global(sch, gc):
    """Same bit relabelling as d6_equivalence.cutqc_full (first factor in smart_order = most significant bits)."""
    vec, order, models = gc.reconstructed_prob, gc.smart_order, E.sub_models(sch)
    lens = [len(models[p].out_qubits) for p in order]
    idx = np.arange(vec.size, dtype=np.int64); gidx = np.zeros_like(idx); shift = sum(lens)
    for p, L in zip(order, lens):
        shift -= L
        local = (idx >> shift) & ((1 << L) - 1)
        for k, lq in enumerate(sorted(models[p].out_local)):
            gidx |= ((local >> k) & 1) << models[p].out_qubits[models[p].out_local.index(lq)]
    full = np.zeros(2 ** sch.n); full[gidx] = vec
    return full


# ------------------------------------------------------------------------------------------------ Li et al. sparse
def li_sparse(ctx):
    """Li et al. (arXiv 2412.17929, §5): sparse index-value contraction with hash matching over the merged factors (no
    pushdown), contraction order from tensor dimensions (size-optimal tree); full sparse distribution, then the query."""
    if ctx["n"] > 62:
        raise Infeasible("full-answer codes need more than 62 bits (int64 index-value lists)")
    fs = {m.p: E.build_factor(m, ctx["tabs"][m.p], ctx["S"], A=None, sel=None, position="after") for m in ctx["models"]}
    tree = P.dense_optimal_tree(fs)
    nodes = len(P.internal_nodes(tree))
    plan = P.TreePlan("after", tree, ("S",) * nodes, "joint")
    o, i, v = P.fold_tree(plan, fs, ctx["c"])
    return _finish(o, i, v, ctx)


# ------------------------------------------------------------------------------------------------ DuckDB
def duckdb_pushdown(ctx):
    """DuckDB 1 thread: one SQL join-aggregate over the pushed-down factors (L1/L2 applied for it); its optimizer picks
    the join order. Table loading counts as conversion time."""
    import duckdb
    import pandas as pd
    Z = ctx.get("points")
    if Z is None:
        fs = {m.p: E.build_factor(m, ctx["tabs"][m.p], ctx["S"], A=set(ctx["A"]), sel=ctx["sel"], position="before")
              for m in ctx["models"]}
    else:   # point batch: the semi-join pushdown applied for it, full-output factors, joined with a points table
        fs = {m.p: E.build_factor(m, PT.restrict_tables(m, ctx["tabs"][m.p], PT.local_codes(Z, m.out_qubits)),
                                  ctx["S"], A=None, sel=None, position="before") for m in ctx["models"]}
    con = duckdb.connect(); con.execute("SET threads TO 1")
    A = ctx["A"]; zterms = []
    for p, f in fs.items():
        cols = {f"b{k}": [] for k in f.vars}; ys, vs = [], []
        for key, (y, val) in f.data.items():
            if y.size == 0:
                continue
            for k, b in zip(f.vars, key):
                cols[f"b{k}"].append(np.full(y.size, b, np.int8))
            ys.append(y); vs.append(val)
        df = pd.DataFrame({**{k: (np.concatenate(v) if v else np.zeros(0, np.int8)) for k, v in cols.items()},
                           "y": np.concatenate(ys) if ys else np.zeros(0, np.int64),
                           "v": np.concatenate(vs) if vs else np.zeros(0)})
        con.register(f"t{p}", df); con.execute(f"CREATE TABLE phi{p} AS SELECT * FROM t{p}")
        for q in (f.outs if Z is None else []):
            zterms.append(f"(((phi{p}.y >> {f.outs.index(q)}) & 1) << {A.index(q)})")
    ps = list(fs)
    if Z is not None:
        pts = pd.DataFrame({"i": np.arange(Z.shape[0], dtype=np.int64),
                            **{f"y{m.p}": PT.local_codes(Z, m.out_qubits) for m in ctx["models"]}})
        con.register("tpts", pts); con.execute("CREATE TABLE pts AS SELECT * FROM tpts")
        joins = "pts"
        for p in ps:
            conds = [f"phi{p}.y = pts.y{p}"] + [f"phi{q}.b{k} = phi{p}.b{k}" for k in fs[p].vars
                                                 for q in ps[:ps.index(p)] if k in fs[q].vars]
            joins += f" JOIN phi{p} ON " + " AND ".join(conds)
        prod = " * ".join(f"phi{p}.v" for p in ps)
        sql = f"SELECT pts.i AS z, SUM({prod}) * {2.0 ** -ctx['c']!r} AS q FROM {joins} GROUP BY pts.i ORDER BY pts.i"
        res = con.execute(sql).fetchnumpy(); con.close()
        z, q = res["z"].astype(np.int64), res["q"].astype(float)
        return z[q != 0], q[q != 0]
    joins = f"phi{ps[0]}"
    seen = set(fs[ps[0]].vars)
    for p in ps[1:]:
        conds = [f"phi{q}.b{k} = phi{p}.b{k}" for k in fs[p].vars for q in ps[:ps.index(p)] if k in fs[q].vars]
        joins += f" JOIN phi{p} ON " + (" AND ".join(conds) if conds else "TRUE")
        seen |= set(fs[p].vars)
    prod = " * ".join(f"phi{p}.v" for p in ps)
    z = " | ".join(zterms) if zterms else "0"
    sql = f"SELECT {z} AS z, SUM({prod}) * {2.0 ** -ctx['c']!r} AS q FROM {joins} GROUP BY z ORDER BY z"
    res = con.execute(sql).fetchnumpy()
    con.close()
    return res["z"].astype(np.int64), res["q"].astype(float)


# ------------------------------------------------------------------------------------------------ external runtimes
# Galley (Julia) and QCut (own venv) run in a subprocess. Their Γ = statistics time measured here (factor build from
# the shot tables, same code as KNITPLAN) + the conversion and query times their runner measures in-process (median
# over ctx["reps"] warm runs). Interpreter start-up, file I/O and the cold (JIT-compiling) run are excluded and kept in
# ctx["timing"]["excluded"]. The subprocess inherits this process's CPU affinity and priority class.
JULIA = r"D:\_caches\julia-bin\julia-1.13.3\bin\julia.exe"
GALLEY_ENV = r"D:\runs\knitplan\galley_env"
QCUT_PY = r"D:\_caches\envs\knitplan-qcut\Scripts\python.exe"
EXT_TMP = Path(_RUNS + "/bench_tmp")
EXT_TIMEOUT = 3600.0          # DNF rule of the main evaluation (> 1 h)
_ONE_THREAD = {"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "JULIA_NUM_THREADS": "1"}


def _run_ext(cmd, d, timeout, extra_env=None):
    import os, subprocess
    env = {k: v for k, v in os.environ.items()
           if k.upper() not in ("__PYVENV_LAUNCHER__", "PYTHONHOME", "PYTHONPATH", "VIRTUAL_ENV")}  # other venv's python
    env.update(_ONE_THREAD); env.update(extra_env or {})
    t = time.perf_counter()
    try:
        p = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        raise Infeasible(f"DNF: external run > {timeout:.0f} s")
    wall = time.perf_counter() - t
    rj = d / "result.json"
    if p.returncode != 0 or not rj.exists():
        e = p.stderr or p.stdout or ""; err = e[:1200] + " ... " + e[-600:]
        if "OutOfMemory" in err or "MemoryError" in err:
            raise Infeasible(f"out of memory: {err[-300:]}")
        raise RuntimeError(f"external run failed (rc {p.returncode}): {err}")
    res = json.loads(rj.read_text())
    if "infeasible" in res:
        raise Infeasible(res["infeasible"])
    return res, wall


def _timed_stat(ctx, build):
    """Statistics time of an external system (factor build from the shot tables, in this process): one call, or the
    median of ctx["stat_reps"] calls after one warm-up when the timed sweep asks for it."""
    k = ctx.get("stat_reps", 0)
    if not k:
        t = time.perf_counter(); fs = build(); return fs, time.perf_counter() - t
    build(); ts = []
    for _ in range(k):
        t = time.perf_counter(); fs = build(); ts.append(time.perf_counter() - t)
    return fs, float(np.median(ts))


def _ext_dir(tag):
    import uuid
    d = EXT_TMP / f"{tag}_{uuid.uuid4().hex[:10]}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _ext_cleanup(d):
    import shutil
    shutil.rmtree(d, ignore_errors=True)


def galley(ctx):
    """Galley (Deeds et al., SIGMOD 2025; commit in D:/runs/knitplan/ext/Galley.jl): its cost-based logical + physical
    optimizer over the pushed-down factors (L1/L2 applied for it, as for DuckDB), SparseList inputs, one query
    Mat(z..., Agg(+, cuts..., MapJoin(*, phi_1..phi_r))). Output index z_p = packed free outputs of factor p."""
    reps = ctx.get("reps", 1)
    Z = ctx.get("points")
    if Z is None:
        fs, t_stat = _timed_stat(ctx, lambda: [E.build_factor(m, ctx["tabs"][m.p], ctx["S"], A=set(ctx["A"]),
                                                              sel=ctx["sel"], position="before", fmt="S")
                                               for m in ctx["models"]])
    else:   # point batch: semi-join pushdown applied for it; point-indicator inputs I_p(i, z_p) join the query
        fs, t_stat = _timed_stat(ctx, lambda: [E.build_factor(
            m, PT.restrict_tables(m, ctx["tabs"][m.p], PT.local_codes(Z, m.out_qubits)), ctx["S"], A=None, sel=None,
            position="before", fmt="S") for m in ctx["models"]])
    d = _ext_dir("galley"); arrays, tensors = {}, []
    try:
        for f, m in zip(fs, ctx["models"]):
            k = f"T{m.p}"; cols = [[] for _ in f.vars]; ys, vs = [], []
            for key, (y, v) in f.data.items():
                for j, b in enumerate(key):
                    cols[j].append(np.full(y.size, b, np.int64))
                ys.append(y.astype(np.int64)); vs.append(v)
            for j in range(len(f.vars)):
                arrays[f"{k}_c{j}"] = np.concatenate(cols[j]) if cols[j] else np.zeros(0, np.int64)
            arrays[f"{k}_c{len(f.vars)}"] = np.concatenate(ys) if ys else np.zeros(0, np.int64)
            arrays[f"{k}_v"] = np.concatenate(vs) if vs else np.zeros(0)
            if arrays[f"{k}_v"].size == 0:      # NPZ.jl cannot read 0-length arrays: one stored explicit zero instead
                for j in range(len(f.vars) + 1):
                    arrays[f"{k}_c{j}"] = np.zeros(1, np.int64)
                arrays[f"{k}_v"] = np.zeros(1)
            tensors.append({"key": k, "idx": [f"b{x}" for x in f.vars] + [f"z{m.p}"],
                            "dims": [4] * len(f.vars) + [2 ** len(f.outs)], "outs": list(f.outs)})
        outputs = [t["idx"][-1] for t in tensors]
        if Z is not None:
            P_ = Z.shape[0]
            for m, t_ in zip(ctx["models"], list(tensors)):
                k = f"I{m.p}"
                arrays[f"{k}_c0"] = np.arange(P_, dtype=np.int64)
                arrays[f"{k}_c1"] = PT.local_codes(Z, m.out_qubits)
                arrays[f"{k}_v"] = np.ones(P_)
                tensors.append({"key": k, "idx": ["i", t_["idx"][-1]], "dims": [P_, t_["dims"][-1]], "outs": []})
            outputs = ["i"]
        np.savez(d / "factors.npz", **arrays)
        (d / "spec.json").write_text(json.dumps({"tensors": tensors, "outputs": outputs}))
        cmd = [JULIA, f"--project={GALLEY_ENV}", "-t", "1", str(ROOT / "bench" / "galley_runner.jl"), str(d), str(reps)]
        res, wall = _run_ext(cmd, d, ctx.get("timeout", EXT_TIMEOUT), {"JULIA_DEPOT_PATH": r"D:\_caches\julia"})
        with np.load(d / "answer.npz") as z:     # closed before the temp dir is removed (Windows file lock)
            ans = {k: z[k] for k in z.files}
        if Z is not None:
            keys, vals = (np.zeros(0, np.int64), np.zeros(0)) if res["scalar"] else (ans["o0"], ans["v"])
        elif res["scalar"]:
            keys, vals = np.zeros(1, np.int64), ans["v"]
        else:
            keys = np.zeros(ans["v"].size, np.int64)
            for j, t_ in enumerate(tensors):
                zj = ans[f"o{j}"]
                for b, q in enumerate(t_["outs"]):
                    keys |= ((zj >> b) & 1) << ctx["A"].index(q)
            vals = ans["v"]
    finally:
        _ext_cleanup(d)
    vals = vals * (2.0 ** -ctx["c"])
    u, inv = np.unique(keys, return_inverse=True)
    s = np.zeros(u.size); np.add.at(s, inv, vals)
    nz = s != 0.0
    ctx["timing"] = {"t_stat": t_stat, "t_conv": res["t_conv"], "t_exec": res["t_galley"],
                     "t_plan_in_exec": res["t_galley_opt"],
                     "excluded": {"cold_total": res["cold_total"], "subprocess_wall": wall}}
    return u[nz], s[nz]


def qcut(ctx):
    """QCut v2.4.0 SeparableDistribution route (see bench/qcut_runner.py): merged factors over all outputs (no pushdown
    given; QCut's marginal() pushes the sum into its modes itself), then marginal / array / slice. Point batches: QCut's
    own value(outcome) per point."""
    reps = ctx.get("reps", 1)
    fs, t_stat = _timed_stat(ctx, lambda: [E.build_factor(m, ctx["tabs"][m.p], ctx["S"], A=None, sel=None,
                                                          position="after", fmt="S") for m in ctx["models"]])
    allv = sorted({v for f in fs for v in f.vars})
    A = list(ctx["A"]); B = sorted(ctx["sel"]) if ctx["sel"] else []
    d = _ext_dir("qcut"); arrays, specf = {}, []
    try:
        for f, m in zip(fs, ctx["models"]):
            k = f"F{m.p}"; ks, ys, vs = [], [], []
            for key, (y, v) in f.data.items():
                ks.append(np.full(y.size, sum(b << (2 * j) for j, b in enumerate(key)), np.int64))
                ys.append(y.astype(np.int64)); vs.append(v)
            arrays[f"{k}_key"] = np.concatenate(ks) if ks else np.zeros(0, np.int64)
            arrays[f"{k}_y"] = np.concatenate(ys) if ys else np.zeros(0, np.int64)
            arrays[f"{k}_v"] = np.concatenate(vs) if vs else np.zeros(0)
            specf.append({"key": k, "vars": list(f.vars), "outs": list(f.outs)})
        if ctx.get("points") is not None:
            arrays["points"] = np.asarray(ctx["points"], np.uint8)
        np.savez(d / "factors.npz", **arrays)
        (d / "spec.json").write_text(json.dumps({
            "points": ctx.get("points") is not None,
            "n": ctx["n"], "c": ctx["c"], "allv": allv, "factors": specf, "keep": A + B, "nA": len(A),
            "sigma_code": sum(ctx["sel"][q] << j for j, q in enumerate(B)) if B else 0}))
        res, wall = _run_ext([QCUT_PY, str(ROOT / "bench" / "qcut_runner.py"), str(d), str(reps)], d,
                             ctx.get("timeout", EXT_TIMEOUT))
        with np.load(d / "answer.npz") as z:
            keys, vals = z["keys"], z["vals"]
    finally:
        _ext_cleanup(d)
    ctx["timing"] = {"t_stat": t_stat, "t_conv": res["t_conv"], "t_exec": res["t_exec"],
                     "excluded": {"cold_total": res["cold_total"], "subprocess_wall": wall}}
    return keys, vals


SYSTEMS = {"knitplan": knitplan, "default_rule": default_rule, "cutqc_fd": cutqc_fd, "cutqc_opt": cutqc_opt, "cutqc_dd": cutqc_dd, "tensorqc_dense": tensorqc_dense,
           "dense_pushdown": dense_pushdown, "li_sparse": li_sparse, "duckdb_pushdown": duckdb_pushdown,
           "galley": galley, "qcut": qcut}


def make_ctx(circ, sch, tabs, S, A, sel, cost_model):
    from d6_equivalence import cutqc_graph
    ctx = {"circ": circ, "sch": sch, "models": E.sub_models(sch), "tabs": tabs, "S": S, "A": sorted(A), "sel": sel,
           "c": sch.c, "n": sch.n, "cost_model": cost_model}
    if circ is not None:
        ctx["subs"], ctx["graph"] = cutqc_graph(circ, sch)
    return ctx


def max_diff(a, b):
    ka, va = a; kb, vb = b
    keys = np.concatenate([ka, kb]); vals = np.concatenate([va, -vb])
    if keys.size == 0:
        return 0.0
    u, inv = np.unique(keys, return_inverse=True)
    s = np.zeros(u.size); np.add.at(s, inv, vals)
    return float(np.abs(s).max())
