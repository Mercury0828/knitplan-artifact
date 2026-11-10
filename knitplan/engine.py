"""KNITPLAN Phase-0 provisional prototype.

Objects follow.1:
  factor psi_j(beta_{delta(j)}, y) = sum_kappa sum_s a_j(beta; kappa, s) N_{j,kappa}(y, s) / S_{j,kappa}
  estimator P(x) = 2^{-c} sum_beta prod_j psi_j(beta_{delta(j)}, x_{O_j})
  query Q = (A, sigma): Q(z) = sum_{x: x_A = z, x_B = sigma} P(x)
Coefficients a_j copied from CutQC (post_process_helper.get_instance_init_meas / convert_to_physical_init and
evaluator.measure_prob / mutate_measurement_basis at commit cda68c7):
  incoming cut, basis I: +|0> +|1>;  X: +2|+> -|0> -|1>;  Y: +2|+i> -|0> -|1>;  Z: +|0> -|1>
  outgoing cut, basis I: computational measurement, sign +1;  Z: computational, sign (-1)^s;
                X: H then measure, sign (-1)^s;  Y: Sdg H then measure, sign (-1)^s
Basis codes: 0=I, 1=X, 2=Y, 3=Z.

Formats: 'S' = sparse (per cut-assignment key: sorted distinct output indices + values), 'D' = dense numpy array of
shape (4,)*len(vars) + (2**len(outs),). Output bit k of a factor's index corresponds to global qubit outs[k].
"""
from __future__ import annotations

import itertools
import math
import time
from dataclasses import dataclass, field

import numpy as np

IN_TERMS = {0: [(1.0, "zero"), (1.0, "one")], 1: [(2.0, "plus"), (-1.0, "zero"), (-1.0, "one")],
            2: [(2.0, "plusI"), (-1.0, "zero"), (-1.0, "one")], 3: [(1.0, "zero"), (-1.0, "one")]}
OUT_MEAS = {0: "comp", 1: "X", 2: "Y", 3: "comp"}


class MemoryCap(Exception):
    pass


ROW_CAP = 60_000_000      # max rows materialized by one sparse operator (pilot DNF cap)
DENSE_CAP = 2 ** 28       # max entries of one dense intermediate (2 GiB of float64)


# ----------------------------------------------------------------------------------------------- bit helpers
def gather_bits(idx, positions):
    """Pack bits idx[positions[k]] into bit k of the result. Runs of consecutive positions move as one masked shift."""
    out = np.zeros_like(idx)
    k, n = 0, len(positions)
    while k < n:
        e = k + 1
        while e < n and positions[e] == positions[e - 1] + 1:
            e += 1
        p, w = positions[k], e - k
        part = (idx >> p) & ((1 << w) - 1)
        out |= part << k if k else part
        k = e
    return out


def aggregate(idx, val):
    if idx.size == 0:
        return idx, val
    u, inv = np.unique(idx, return_inverse=True)
    v = np.bincount(inv, weights=val, minlength=u.size)
    nz = v != 0.0
    return u[nz], v[nz]


# ----------------------------------------------------------------------------------------------- instance model
@dataclass
class SubModel:
    p: int
    deg_cuts: list          # cut ids incident to the subcircuit (sorted)
    cut_local: dict         # cut id -> local qubit position
    cut_is_out: dict        # cut id -> True if the subcircuit is upstream (measures)
    out_qubits: list        # global output qubits, ascending (= O_j)
    out_local: list         # local positions of those outputs
    ins: list               # local positions with cut-in (config order)
    outs: list              # local positions with cut-out (config order)


def sub_models(sch):
    models = []
    seg_local = {}
    for p, sids in enumerate(sch.sub_qubits):
        for i, sid in enumerate(sids):
            seg_local[sid] = (p, i)
    for p, sids in enumerate(sch.sub_qubits):
        deg, cl, co = [], {}, {}
        for k, (u, d) in enumerate(sch.cuts):
            if seg_local[u][0] == p:
                deg.append(k); cl[k] = seg_local[u][1]; co[k] = True
            if seg_local[d][0] == p:
                deg.append(k); cl[k] = seg_local[d][1]; co[k] = False
        outq = sorted(q for q, sid in sch.out_seg.items() if seg_local[sid][0] == p)
        outl = [seg_local[sch.out_seg[q]][1] for q in outq]
        ins = [i for i, sid in enumerate(sids) if sch.segments[sid].starts_with_cut]
        outs = [i for i, sid in enumerate(sids) if sch.segments[sid].ends_with_cut]
        models.append(SubModel(p, sorted(set(deg)), cl, co, outq, outl, ins, outs))
    return models


@dataclass
class Factor:
    vars: tuple
    outs: list
    fmt: str
    data: object
    def nnz(self):
        if self.fmt == "S":
            return int(sum(len(i) for i, _ in self.data.values()))
        return int(np.count_nonzero(self.data))
    def dense_shape(self):
        return (4 ** len(self.vars)) * (2 ** len(self.outs))


# ----------------------------------------------------------------------------------------------- factor build
IN_MAP = {"zero": [(0, 1.0), (1, -1.0), (2, -1.0), (3, 1.0)], "one": [(0, 1.0), (1, -1.0), (2, -1.0), (3, -1.0)],
          "plus": [(1, 2.0)], "plusI": [(2, 2.0)]}   # state -> [(beta, coefficient)]  (transpose of IN_TERMS)


def build_factor(model: SubModel, tables, S, A=None, sel=None, position="after", fmt="S"):
    """Same semantics as build_factor_naive, computed edge by edge (the coefficient tensor a_j is a Kronecker product
    over the incident cut ends, so it is applied one cut end at a time)."""
    outq = model.out_qubits
    keepq = [q for q in outq if (A is None or q in A)]
    selq = [q for q in outq if sel is not None and q in sel]
    ypos = [model.out_local[outq.index(q)] for q in (keepq if position == "before" else outq)]
    ny = len(ypos)
    spos = list(model.outs)  # local positions of cut-measured qubits (config meas order)
    # stage 0: per configuration, code = y | s << ny
    cur = {}
    for conf, (idx, cnt) in tables.items():
        w = cnt.astype(float) / S
        if selq and position == "before":
            m = np.ones(idx.size, bool)
            for q in selq:
                m &= ((idx >> model.out_local[outq.index(q)]) & 1) == sel[q]
            idx, w = idx[m], w[m]
        code = gather_bits(idx, ypos) | (gather_bits(idx, spos) << ny)
        cur[conf[0] + conf[1]] = aggregate(code, w)
    nin = len(model.ins)
    # stage 1: incoming cut ends, state axis -> basis axis
    for slot in range(nin):
        nxt = {}
        for key, (c, v) in cur.items():
            for beta, coef in IN_MAP[key[slot]]:
                nk = key[:slot] + (beta,) + key[slot + 1:]
                nxt.setdefault(nk, []).append((c, coef * v))
        cur = {k: aggregate(np.concatenate([a for a, _ in ps]), np.concatenate([b for _, b in ps]))
               for k, ps in nxt.items()}
    # stage 2: outgoing cut ends, meas axis -> basis axis; drop the s bit
    nout = len(model.outs)
    for j in range(nout):
        slot = nin + j
        bit = ny  # earlier ends were dropped, so this end's s bit is now at position ny
        nxt = {}
        for key, (c, v) in cur.items():
            sb = (c >> bit) & 1
            low = c & ((1 << bit) - 1)
            rest = (c >> (bit + 1)) << bit
            nc = low | rest
            sign = 1.0 - 2.0 * sb
            m = key[slot]
            targets = [(0, v), (3, v * sign)] if m == "comp" else [(1 if m == "X" else 2, v * sign)]
            for beta, vv in targets:
                nk = key[:slot] + (beta,) + key[slot + 1:]
                nxt.setdefault(nk, []).append((nc, vv))
        cur = {k: aggregate(np.concatenate([a for a, _ in ps]), np.concatenate([b for _, b in ps]))
               for k, ps in nxt.items()}
    # re-key: slots (ins..., outs...) -> vars order
    vars_ = tuple(model.deg_cuts)
    slot_of = {}
    for k in vars_:
        if model.cut_is_out[k]:
            slot_of[k] = nin + model.outs.index(model.cut_local[k])
        else:
            slot_of[k] = model.ins.index(model.cut_local[k])
    data = {}
    for beta in itertools.product(range(4), repeat=len(vars_)):
        skey = [None] * (nin + nout)
        for k, b in zip(vars_, beta):
            skey[slot_of[k]] = b
        data[beta] = cur.get(tuple(skey), (np.zeros(0, np.int64), np.zeros(0)))
    if position == "after" and (A is not None or selq):
        local_keep = [outq.index(q) for q in keepq]
        for beta, (y, v) in data.items():
            if selq:
                m = np.ones(y.size, bool)
                for q in selq:
                    m &= ((y >> outq.index(q)) & 1) == sel[q]
                y, v = y[m], v[m]
            data[beta] = aggregate(gather_bits(y, local_keep), v)
    return to_format(Factor(vars_, keepq, "S", data), fmt)


def build_factor_naive(model: SubModel, tables, S, A=None, sel=None, position="after", fmt="S"):
    """Build the (pushed-down) factor phi_j of one subcircuit from its shot tables.

    tables: conf -> (full-register idx, counts). A: global free qubits of the query (None = all outputs kept).
    sel: dict global qubit -> bit for selected qubits. position: 'before' = map rows to the query's output space
    before merging configurations (L2); 'after' = merge into psi_j(beta, y) first, then aggregate / select (L1).
    """
    outq = model.out_qubits
    keepq = [q for q in outq if (A is None or q in A)]
    selq = [q for q in outq if sel is not None and q in sel]
    keep_pos = [model.out_local[outq.index(q)] for q in keepq]
    all_pos = model.out_local
    # per-config row preprocessing
    pre = {}
    for conf, (idx, cnt) in tables.items():
        w = cnt.astype(float) / S
        if selq and position == "before":
            m = np.ones(idx.size, bool)
            for q in selq:
                m &= ((idx >> model.out_local[outq.index(q)]) & 1) == sel[q]
            idx, w = idx[m], w[m]
        y = gather_bits(idx, keep_pos if position == "before" else all_pos)
        pre[conf] = (idx, y, w)
    vars_ = tuple(model.deg_cuts)
    data = {}
    for beta in itertools.product(range(4), repeat=len(vars_)):
        b = dict(zip(vars_, beta))
        in_cuts = [k for k in vars_ if not model.cut_is_out[k]]
        out_cuts = [k for k in vars_ if model.cut_is_out[k]]
        # config order: ins / outs lists are local positions; map cuts to those slots
        in_slot = {k: model.ins.index(model.cut_local[k]) for k in in_cuts}
        out_slot = {k: model.outs.index(model.cut_local[k]) for k in out_cuts}
        meas = [None] * len(model.outs)
        for k in out_cuts:
            meas[out_slot[k]] = OUT_MEAS[b[k]]
        sign_pos = [model.cut_local[k] for k in out_cuts if b[k] != 0]
        ys, ws = [], []
        for terms in itertools.product(*[IN_TERMS[b[k]] for k in in_cuts]):
            coef = 1.0
            init = [None] * len(model.ins)
            for k, (cf, st) in zip(in_cuts, terms):
                coef *= cf; init[in_slot[k]] = st
            conf = (tuple(init), tuple(meas))
            idx, y, w = pre[conf]
            if sign_pos:
                par = np.zeros(idx.size, dtype=np.int64)
                for sp in sign_pos:
                    par ^= (idx >> sp) & 1
                w = w * (1.0 - 2.0 * par)
            ys.append(y); ws.append(coef * w)
        y = np.concatenate(ys) if ys else np.zeros(0, np.int64)
        w = np.concatenate(ws) if ws else np.zeros(0)
        data[beta] = aggregate(y, w)
    if position == "after" and (A is not None or selq):
        # aggregate / select on the merged factor (L1)
        local_keep = [outq.index(q) for q in keepq]
        for beta, (y, v) in data.items():
            if selq:
                m = np.ones(y.size, bool)
                for q in selq:
                    m &= ((y >> outq.index(q)) & 1) == sel[q]
                y, v = y[m], v[m]
            data[beta] = aggregate(gather_bits(y, local_keep), v)
    f = Factor(vars_, keepq, "S", data)
    return to_format(f, fmt)


def to_format(f: Factor, fmt):
    if f.fmt == fmt:
        return f
    nv, no = len(f.vars), len(f.outs)
    if fmt == "D":
        if f.dense_shape() > DENSE_CAP:
            raise MemoryCap(f"dense {f.dense_shape()}")
        arr = np.zeros((4,) * nv + (2 ** no,))
        for key, (i, v) in f.data.items():
            arr[key][i] = v
        return Factor(f.vars, f.outs, "D", arr)
    data = {}
    for key in itertools.product(range(4), repeat=nv):
        row = f.data[key]
        i = np.flatnonzero(row)
        data[key] = (i.astype(np.int64), row[i])
    return Factor(f.vars, f.outs, "S", data)


# ----------------------------------------------------------------------------------------------- operators
LET = "abcdefghijklmnopqrstuvw"


def product_sum(f: Factor, g: Factor, elim, fmt):
    """h(vars, outs_f+outs_g) = sum_{elim} f * g. Output bits: f's outs low, g's outs high."""
    rvars = tuple(sorted((set(f.vars) | set(g.vars)) - set(elim)))
    f, g = to_format(f, fmt), to_format(g, fmt)
    nf = len(f.outs)
    if fmt == "D":
        shape = (4 ** len(rvars)) * (2 ** (len(f.outs) + len(g.outs)))
        if shape > DENSE_CAP:
            raise MemoryCap(f"dense {shape}")
        allv = sorted(set(f.vars) | set(g.vars))
        L = {v: LET[i] for i, v in enumerate(allv)}
        sf = "".join(L[v] for v in f.vars) + "Y"
        sg = "".join(L[v] for v in g.vars) + "Z"
        sr = "".join(L[v] for v in rvars) + "ZY"
        arr = np.einsum(f"{sf},{sg}->{sr}", f.data, g.data, optimize=True)
        arr = arr.reshape((4,) * len(rvars) + (2 ** (len(f.outs) + len(g.outs)),))
        return Factor(rvars, f.outs + g.outs, "D", arr)
    shared = [v for v in f.vars if v in g.vars]
    fpos = {v: i for i, v in enumerate(f.vars)}
    gpos = {v: i for i, v in enumerate(g.vars)}
    gb = {}
    for kg in g.data:
        gb.setdefault(tuple(kg[gpos[v]] for v in shared), []).append(kg)
    rows = 0
    for kf, (fi, _) in f.data.items():
        for kg in gb.get(tuple(kf[fpos[v]] for v in shared), []):
            rows += fi.size * g.data[kg][0].size
    if rows > ROW_CAP:
        raise MemoryCap(f"sparse rows {rows}")
    acc = {}
    for kf, (fi, fv) in f.data.items():
        if fi.size == 0:
            continue
        for kg in gb.get(tuple(kf[fpos[v]] for v in shared), []):
            gi, gv = g.data[kg]
            if gi.size == 0:
                continue
            assign = {v: kf[fpos[v]] for v in f.vars}
            assign.update({v: kg[gpos[v]] for v in g.vars})
            rk = tuple(assign[v] for v in rvars)
            idx = ((gi[:, None] << nf) | fi[None, :]).ravel()
            val = (gv[:, None] * fv[None, :]).ravel()
            acc.setdefault(rk, []).append((idx, val))
    data = {}
    for rk in itertools.product(range(4), repeat=len(rvars)):
        parts = acc.get(rk, [])
        if parts:
            data[rk] = aggregate(np.concatenate([p[0] for p in parts]), np.concatenate([p[1] for p in parts]))
        else:
            data[rk] = (np.zeros(0, np.int64), np.zeros(0))
    return Factor(rvars, f.outs + g.outs, "S", data)


def sum_out(f: Factor, v, fmt):
    f = to_format(f, fmt)
    ax = f.vars.index(v)
    rvars = tuple(x for x in f.vars if x != v)
    if fmt == "D":
        return Factor(rvars, f.outs, "D", f.data.sum(axis=ax))
    acc = {}
    for k, (i, val) in f.data.items():
        acc.setdefault(k[:ax] + k[ax + 1:], []).append((i, val))
    data = {rk: aggregate(np.concatenate([p[0] for p in ps]), np.concatenate([p[1] for p in ps]))
            for rk, ps in acc.items()}
    return Factor(rvars, f.outs, "S", data)


# ----------------------------------------------------------------------------------------------- plans
@dataclass
class Plan:
    position: str            # 'before' | 'after'  (pushdown below / above the configuration merge)
    order: tuple             # fold order of subcircuit factors (left-deep join tree)
    fmts: tuple              # format per fold step (len r-1), 'S' | 'D'
    gran: str                # 'joint' (sum out all newly internal cuts in the product) | 'seq' (one cut at a time)
    leaf_fmt: str = "S"
    name: str = ""
    def key(self):
        return f"{self.position}|{'-'.join(map(str, self.order))}|{''.join(self.fmts)}|{self.gran}"


def enumerate_plans(r):
    plans = []
    for pos in ("before", "after"):
        for order in itertools.permutations(range(r)):
            if r > 1 and order[0] > order[1]:
                continue  # the first two factors commute
            for fmts in itertools.product("SD", repeat=max(r - 1, 0)):
                for gran in ("joint", "seq"):
                    plans.append(Plan(pos, order, fmts, gran))
    return plans


@dataclass
class RunStats:
    t_build: float = 0.0
    t_exec: float = 0.0
    inter: list = field(default_factory=list)   # (step, nnz, dense_shape)


def execute(plan: Plan, models, tables_by_sub, S, A, sel, c, stats: RunStats | None = None):
    """Run a plan; return (answer outs order, sparse idx, val) scaled by 2^-c."""
    t0 = time.perf_counter()
    factors = {m.p: build_factor(m, tables_by_sub[m.p], S, A=A, sel=sel, position=plan.position, fmt="S")
               for m in models}
    t1 = time.perf_counter()
    outs, idx, val = fold(plan, factors, c, stats)
    if stats is not None:
        stats.t_build, stats.t_exec = t1 - t0, time.perf_counter() - t1
    return outs, idx, val


def fold(plan: Plan, factors, c, stats: RunStats | None = None):
    """Execute the elimination part of a plan on already built (pushed-down) factors."""
    order = list(plan.order)
    h = factors[order[0]]
    remaining = [factors[p] for p in order[1:]]
    for step, g in enumerate(remaining):
        fmt = plan.fmts[step]
        later = set(v for x in remaining[step + 1:] for v in x.vars)
        elim = sorted((set(h.vars) | set(g.vars)) - later)
        if plan.gran == "joint":
            h = product_sum(h, g, elim, fmt)
        else:
            first = [v for v in elim if v in h.vars and v in g.vars][:1]
            h = product_sum(h, g, first, fmt)
            for v in elim:
                if v in h.vars:
                    h = sum_out(h, v, fmt)
        if stats is not None:
            stats.inter.append((step, h.nnz(), h.dense_shape()))
    # leftover vars (single-factor instance) are summed out
    for v in list(h.vars):
        h = sum_out(h, v, h.fmt)
    if h.fmt == "D":
        arr = h.data.reshape(-1)
        i = np.flatnonzero(arr)
        idx, val = i.astype(np.int64), arr[i]
    else:
        idx, val = h.data[()]
    val = val * (2.0 ** -c)
    return h.outs, idx, val


def canonical(outs, idx, val, A_sorted):
    """Re-map answer bits from factor output order to ascending global qubit order."""
    pos = [outs.index(q) for q in A_sorted]
    if len(pos) == len(outs) and len(set(pos)) == len(pos):
        # a bit permutation maps distinct keys to distinct keys: sort instead of unique + bincount
        g = gather_bits(idx, pos) if pos != list(range(len(pos))) else idx
        o = np.argsort(g, kind="stable"); g, v = g[o], val[o]
        nz = v != 0.0
        if g.size > 1 and not (g[1:] > g[:-1]).all():   # duplicate keys in the input: fall back
            return aggregate(g, v)
        return g[nz], v[nz]
    return aggregate(gather_bits(idx, pos), val)


U64 = 2.0 ** -53
IN_ABS = {"zero": {0: 1, 1: 1, 2: 1, 3: 1}, "one": {0: 1, 1: 1, 2: 1, 3: 1}, "plus": {1: 2}, "plusI": {2: 2}}
MEAS_OF_BETA = {0: "comp", 1: "X", 2: "Y", 3: "comp"}


def gamma(d, u=U64):
    if d * u >= 1.0:
        return math.inf
    return d * u / (1.0 - d * u)


def round_up(x):
    return math.nextafter(x * (1.0 + 4 * U64), math.inf)


def tau_eq_details(models, tables_by_sub, S, A, sel, c, u=U64, exact_beta_max_c=9):
    """Certified-tolerance ingredients (the analysis item (f).

    B = min{6^c prod_j M_j, 2^-c prod_j lambda_j, 2^-c sum_beta prod_j C_j(beta)} (M_j = max_kappa selected counts of
    kappa / S; the Phase-0 draft used 6^c, valid only for tables normalised per configuration) bounds sum_z H(z), H(z) = the absolute sum of the
    fully expanded terms of Q(z); C_j(beta) = sum_kappa |a_j(beta; kappa)| * (selected counts of kappa) / S, computed in
    exact rationals (|a_j| does not depend on s or y). D_j = rounding budget of one entry of the built factor phi_j:
    1 (count normalization) + (L_j - 1) additions, L_j <= sum_kappa max_z (#rows of kappa mapping to z) bounds the number of
    raw terms of one entry (coefficients +-1, +-2 and signs are exact). Fold steps add D_f + D_g + 4^|elim| (one product
    rounding, <= 4^|elim| - 1 additions in any summation tree, also for the 'seq' granularity); the final leftover
    sum-out adds <= 4^c - 1; the 2^-c scaling is exact. Every cut is eliminated once and there are r - 1 binary nodes,
    so every plan of the family (left-deep or bushy, joint or seq) has D_P <= D_max = sum_j D_j + (r - 1) + (4^c - 1). Arithmetic contract: binary64 round-to-nearest; counts and S convert exactly; every relevant inexact
    operation satisfies the relative-error model; coefficient multiplications and final power-of-two scaling are exact.
    Absence of harmful overflow/underflow is a separate assumption, not a consequence of a lower bound S**(-r).
    """
    from fractions import Fraction
    C, lam, Dj = [], [], []
    selected_max_bound = Fraction(6) ** c
    for m in models:
        outq = m.out_qubits
        selq = [q for q in outq if sel and q in sel]
        keepq = [q for q in outq if A is None or q in A]
        keep_pos = [m.out_local[outq.index(q)] for q in keepq]
        mass, L = {}, 0
        for conf, (idx, cnt) in tables_by_sub[m.p].items():
            msk = np.ones(idx.size, bool)
            for q in selq:
                msk &= ((idx >> m.out_local[outq.index(q)]) & 1) == sel[q]
            mass[conf] = Fraction(sum(map(int, cnt[msk])), S)  # Python ints: no silent int64 overflow
            if msk.any():
                _, mult = np.unique(gather_bits(idx[msk], keep_pos), return_counts=True)
                L += int(mult.max())
        selected_max_bound *= max(mass.values(), default=Fraction(0))
        Dj.append(max(L, 1))
        Cj = {}
        for beta in itertools.product(range(4), repeat=len(m.deg_cuts)):
            b = dict(zip(m.deg_cuts, beta))
            tot = Fraction(0)
            for conf, ms in mass.items():
                if ms == 0:
                    continue
                w = 1
                for k in m.deg_cuts:
                    if m.cut_is_out[k]:
                        if conf[1][m.outs.index(m.cut_local[k])] != MEAS_OF_BETA[b[k]]:
                            w = 0; break
                    else:
                        w *= IN_ABS[conf[0][m.ins.index(m.cut_local[k])]].get(b[k], 0)
                        if w == 0:
                            break
                if w:
                    tot += w * ms
            Cj[beta] = tot
        C.append(Cj); lam.append(sum(Cj.values(), Fraction(0)))
    two_c = Fraction(1, 2 ** c)
    cands = {"selected_mass_max": selected_max_bound}
    prod = two_c
    for x in lam:
        prod *= x
    cands["prod_lambda"] = prod
    if c <= exact_beta_max_c:
        allv = sorted(set(v for m in models for v in m.deg_cuts))
        tot = Fraction(0)
        for beta in itertools.product(range(4), repeat=len(allv)):
            b = dict(zip(allv, beta))
            t = Fraction(1)
            for m, Cj in zip(models, C):
                t *= Cj[tuple(b[k] for k in m.deg_cuts)]
                if t == 0:
                    break
            tot += t
        cands["sum_beta"] = two_c * tot
    B = min(cands.values())
    D_max = sum(Dj) + (len(models) - 1) + (4 ** c - 1)
    return {"B": round_up(float(B)), "B_candidates": {k: float(v) for k, v in cands.items()}, "D_j": Dj,
            "D_max": D_max, "gamma_D_max": gamma(D_max, u), "u": u}


def tau_eq(models, tables_by_sub, S, A, sel, c, D_other=None, D_other_extra=0, u=U64):
    """Certified equivalence tolerance (replaces the unconditional 1e-12 * W, which is not a bound):
    tau = roundUp((gamma_{D_max} + gamma_{D_other}) * B); D_other = rounding budget of the other implementation compared
    (default: another plan of the same family, D_max; D_other_extra adds to D_max, e.g. a brute-force reference that sums
    2^(n-|A|-|B|) full-distribution entries per answer key). Returns (tau, B)."""
    d = tau_eq_details(models, tables_by_sub, S, A, sel, c, u)
    D2 = D_other if D_other is not None else d["D_max"] + D_other_extra
    g2 = gamma(D2, u)
    return round_up((d["gamma_D_max"] + g2) * d["B"]), d["B"]


def tau_eq_legacy(models, tables_by_sub, S, A, sel, c, rel=1e-12):
    """Phase-0 tolerance (kept for comparison only): tau_eq = rel * W, W = 2^-c sum_beta prod_j sum_z |phi_j(beta, z)|.
    NOT certified (the analysis (f) counterexample)."""
    fs = [build_factor(m, tables_by_sub[m.p], S, A=A, sel=sel, position="after") for m in models]
    allv = sorted(set(v for f in fs for v in f.vars))
    L = {v: LET[i] for i, v in enumerate(allv)}
    arrs, subs = [], []
    for f in fs:
        a = np.zeros((4,) * len(f.vars))
        for k, (_, v) in f.data.items():
            a[k] = np.abs(v).sum()
        arrs.append(a); subs.append("".join(L[v] for v in f.vars))
    W = float(np.einsum(",".join(subs) + "->", *arrs)) * 2.0 ** -c
    return rel * W, W


# ----------------------------------------------------------------------------------------------- default rule
def predicted_support(f: Factor, g: Factor, elim):
    """Upper bound on the support of sum_elim f*g: sum over result keys of min(dense, sum over consistent pairs |f||g|)."""
    fpos = {v: i for i, v in enumerate(f.vars)}; gpos = {v: i for i, v in enumerate(g.vars)}
    shared = [v for v in f.vars if v in g.vars]
    rvars = tuple(sorted((set(f.vars) | set(g.vars)) - set(elim)))
    per = {}
    for kf, (fi, _) in f.data.items():
        for kg, (gi, _) in g.data.items():
            if any(kf[fpos[v]] != kg[gpos[v]] for v in shared):
                continue
            a = {v: kf[fpos[v]] for v in f.vars}; a.update({v: kg[gpos[v]] for v in g.vars})
            rk = tuple(a[v] for v in rvars)
            per[rk] = per.get(rk, 0) + fi.size * gi.size
    cap = 2 ** (len(f.outs) + len(g.outs))
    return sum(min(cap, x) for x in per.values()), (4 ** len(rvars)) * cap


def _consistent_groups(f: Factor, g: Factor, elim):
    """Result key -> list of (f key, g key) pairs that join on every common cut index (the analysis (e): join on all common
    coordinates, then project the eliminated ones away)."""
    fpos = {v: i for i, v in enumerate(f.vars)}; gpos = {v: i for i, v in enumerate(g.vars)}
    shared = [v for v in f.vars if v in g.vars]
    rvars = tuple(sorted((set(f.vars) | set(g.vars)) - set(elim)))
    groups = {}
    for kf in f.data:
        for kg in g.data:
            if any(kf[fpos[v]] != kg[gpos[v]] for v in shared):
                continue
            a = {v: kf[fpos[v]] for v in f.vars}; a.update({v: kg[gpos[v]] for v in g.vars})
            groups.setdefault(tuple(a[v] for v in rvars), []).append((kf, kg))
    return groups


def exact_support_nc(f: Factor, g: Factor, elim):
    """Exact support of sum_elim f*g under no-cancellation: sum over result keys of |U_t x V_t union|,
    computed without enumerating products: group each u by its signature {t : u in U_t}; then
    |union_t U_t x V_t| = sum_signatures (#u with that signature) * |union_{t in signature} V_t|. f, g sparse."""
    total = 0
    for rk, pairs in _consistent_groups(f, g, elim).items():
        us = [(t, f.data[kf][0]) for t, (kf, _) in enumerate(pairs) if f.data[kf][0].size and g.data[kg_of(pairs, t)][0].size]
        if not us:
            continue
        allu = np.concatenate([u for _, u in us]); tid = np.concatenate([np.full(u.size, t) for t, u in us])
        order = np.lexsort((tid, allu)); allu, tid = allu[order], tid[order]
        bounds = np.flatnonzero(np.diff(allu)) + 1
        sigs = {}
        for chunk in np.split(tid, bounds):
            key = tuple(chunk.tolist()); sigs[key] = sigs.get(key, 0) + 1
        for sig, cnt in sigs.items():
            vs = [g.data[pairs[t][1]][0] for t in sig]
            total += cnt * (vs[0].size if len(vs) == 1 else np.unique(np.concatenate(vs)).size)
    return total


def kg_of(pairs, t):
    return pairs[t][1]


def default_rule_plan(models, tables_by_sub, S, A, sel, rho_star=1 / 16):
    """Push all aggregation down (L1 and L2: position 'before'); fold factors greedily, smallest predicted output
    support first; sparse if predicted support < dense shape * rho_star."""
    fs = {m.p: build_factor(m, tables_by_sub[m.p], S, A=A, sel=sel, position="before") for m in models}
    return default_rule_from_factors(fs, rho_star)


def default_rule_from_factors(fs, rho_star=1 / 16):
    """The default rule's planning step on already built (pushed-down below the merge) factors."""
    r = len(fs)
    if r == 1:
        return Plan("before", (0,), (), "joint", name="default")
    best = None
    for a, b in itertools.combinations(range(r), 2):
        if not set(fs[a].vars) & set(fs[b].vars):
            continue
        sup, _ = predicted_support(fs[a], fs[b], sorted(set(fs[a].vars) & set(fs[b].vars)))
        if best is None or sup < best[0]:
            best = (sup, a, b)
    order = [best[1], best[2]]
    rest = [p for p in range(r) if p not in order]
    while rest:  # next factor: the one sharing a cut with the current set and smallest in nnz
        cur = set(v for p in order for v in fs[p].vars)
        cand = sorted(rest, key=lambda p: (not (set(fs[p].vars) & cur), fs[p].nnz()))
        order.append(cand[0]); rest.remove(cand[0])
    # formats by predicted support of each fold step (computed on the sparse factors, pilot approximation:
    # the accumulated intermediate is approximated by the product bound)
    fmts = []
    h = fs[order[0]]
    for i, p in enumerate(order[1:]):
        g = fs[p]
        later = set(v for q in order[i + 2:] for v in fs[q].vars)
        elim = sorted((set(h.vars) | set(g.vars)) - later)
        sup, dense = predicted_support(h, g, elim)
        fmts.append("S" if sup < dense * rho_star else "D")
        if i < len(order) - 2:
            h = product_sum(h, g, elim, "S") if sup < 5e6 else h  # refine stats while cheap
    return Plan("before", tuple(order), tuple(fmts), "seq", name="default")


def fixed_heuristics(models, tables_by_sub, S, A, sel):
    r = len(models)
    # CutQC's smart_order: ascending factor vector length (2^{m_j})
    cq = tuple(sorted(range(r), key=lambda p: len(models[p].out_qubits)))
    return {
        "always_dense_pushdown": Plan("before", cq, ("D",) * (r - 1), "joint", name="always_dense_pushdown"),
        "always_sparse_pushdown_cutqc_order": Plan("before", cq, ("S",) * (r - 1), "joint",
                                                   name="always_sparse_pushdown_cutqc_order"),
        "pushdown_after_merge_default": None,  # filled by caller from the default rule with position='after'
    }
