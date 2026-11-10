"""Point-batch queries: the estimator at P full bitstrings z_1..z_P in one call.

A point batch is the union of P selections, each fixing every output qubit (A = {}, sigma = z_i). Answering it point
by point with the L2 pushdown would rebuild every factor P times. KNITPLAN executes it as one sum-product with the point
index i as a shared (batch) variable:
  1. semi-join pushdown below the configuration merge: each subcircuit's shot table is restricted to the outcomes whose
     output bits equal some point's restriction z_i|O_j, then the factor is built from the restricted table (L2 applied
     to the union of the selections: a shot is kept iff it can contribute to some point);
  2. point factors psi_j(beta_delta(j), i) = phi_j(beta_delta(j), z_i|O_j) (a lookup per cut-basis key);
  3. fold over the cut variables with the point index shared by every factor: sparse (sorted point-id lists, merge by
     intersection) or dense ((4,)*k + (P,) arrays, batched einsum), chosen per fold step by predicted density.
Each point's value is the same sum of products as the single-selection query, so L2 gives the same estimator.

Representation of a batch: Z = uint8 array (P, n), Z[i, q] = bit of qubit q in point i. Answer: (keys, vals) with keys
= point indices with a non-zero value (absent = 0).
"""
import itertools

import numpy as np

from . import engine as E

RHO_STAR = 1 / 16


def local_codes(Z, qubits):
    """Pack Z[:, qubits[k]] into bit k (one int64 per point)."""
    out = np.zeros(Z.shape[0], np.int64)
    for k, q in enumerate(qubits):
        out |= Z[:, q].astype(np.int64) << k
    return out


def global_codes(Z):
    if Z.shape[1] > 62:
        raise ValueError("global point codes need n <= 62")
    return local_codes(Z, list(range(Z.shape[1])))


def restrict_tables(model: E.SubModel, tables, ycodes):
    """Semi-join pushdown below the merge: keep the shots whose output bits match some point."""
    want = np.unique(ycodes)
    pos = [model.out_local[k] for k in range(len(model.out_qubits))]
    out = {}
    for conf, (idx, cnt) in tables.items():
        m = np.isin(E.gather_bits(idx, pos), want, assume_unique=False)
        out[conf] = (idx[m], cnt[m])
    return out


def point_factor(f: E.Factor, ycodes):
    """phi_j(beta, y) (sparse, full outputs) -> psi_j(beta, i) = phi_j(beta, ycodes[i]); sorted point ids per key."""
    data = {}
    for key, (y, v) in f.data.items():
        if y.size == 0:
            data[key] = (np.zeros(0, np.int64), np.zeros(0)); continue
        pos = np.searchsorted(y, ycodes)
        pos[pos >= y.size] = 0
        hit = y[pos] == ycodes
        i = np.flatnonzero(hit)
        data[key] = (i.astype(np.int64), v[pos[hit]])
    return E.Factor(f.vars, ["#point"], "S", data)


def build_point_factors(models, tabs, S, Z):
    fs = {}
    for m in models:
        y = local_codes(Z, m.out_qubits)
        f = E.build_factor(m, restrict_tables(m, tabs[m.p], y), S, A=None, sel=None, position="before")
        fs[m.p] = point_factor(f, y)
    return fs


def _dense(f, P):
    if f.fmt == "D":
        return f
    if (4 ** len(f.vars)) * P > E.DENSE_CAP:
        raise E.MemoryCap(f"dense point factor {(4 ** len(f.vars)) * P}")
    arr = np.zeros((4,) * len(f.vars) + (P,))
    for key, (i, v) in f.data.items():
        arr[key][i] = v
    return E.Factor(f.vars, f.outs, "D", arr)


def _sparse(f):
    if f.fmt == "S":
        return f
    data = {}
    for key in itertools.product(range(4), repeat=len(f.vars)):
        row = f.data[key]; i = np.flatnonzero(row)
        data[key] = (i.astype(np.int64), row[i])
    return E.Factor(f.vars, f.outs, "S", data)


def predicted_nnz(f, g):
    """Upper bound on the output entries of the batched product: sum over consistent key pairs of min(|f_k|, |g_k|)."""
    shared = [v for v in f.vars if v in g.vars]
    fpos = {v: j for j, v in enumerate(f.vars)}; gpos = {v: j for j, v in enumerate(g.vars)}
    fn = {k: (i.size if f.fmt == "S" else int(np.count_nonzero(i))) for k, i in _keyed_sizes(f)}
    gn = {k: (i.size if g.fmt == "S" else int(np.count_nonzero(i))) for k, i in _keyed_sizes(g)}
    gb = {}
    for kg, n in gn.items():
        gb.setdefault(tuple(kg[gpos[v]] for v in shared), []).append(n)
    tot = 0
    for kf, n in fn.items():
        for m in gb.get(tuple(kf[fpos[v]] for v in shared), []):
            tot += min(n, m)
    return tot


def _keyed_sizes(f):
    if f.fmt == "S":
        return [(k, i) for k, (i, _) in f.data.items()]
    return [(k, f.data[k]) for k in itertools.product(range(4), repeat=len(f.vars))]


def batched_product_sum(f, g, elim, fmt, P):
    """h(rvars, i) = sum_{elim} f(.., i) * g(.., i)  (point index shared)."""
    rvars = tuple(sorted((set(f.vars) | set(g.vars)) - set(elim)))
    if fmt == "D":
        if (4 ** len(rvars)) * P > E.DENSE_CAP:
            raise E.MemoryCap(f"dense {(4 ** len(rvars)) * P}")
        f, g = _dense(f, P), _dense(g, P)
        allv = sorted(set(f.vars) | set(g.vars)); L = {v: E.LET[j] for j, v in enumerate(allv)}
        sf = "".join(L[v] for v in f.vars) + "Y"; sg = "".join(L[v] for v in g.vars) + "Y"
        sr = "".join(L[v] for v in rvars) + "Y"
        return E.Factor(rvars, ["#point"], "D", np.einsum(f"{sf},{sg}->{sr}", f.data, g.data, optimize=True))
    f, g = _sparse(f), _sparse(g)
    shared = [v for v in f.vars if v in g.vars]
    fpos = {v: j for j, v in enumerate(f.vars)}; gpos = {v: j for j, v in enumerate(g.vars)}
    gb = {}
    for kg in g.data:
        gb.setdefault(tuple(kg[gpos[v]] for v in shared), []).append(kg)
    acc = {}
    for kf, (fi, fv) in f.data.items():
        if fi.size == 0:
            continue
        for kg in gb.get(tuple(kf[fpos[v]] for v in shared), []):
            gi, gv = g.data[kg]
            if gi.size == 0:
                continue
            common, a, b = np.intersect1d(fi, gi, assume_unique=True, return_indices=True)
            if common.size == 0:
                continue
            assign = {v: kf[fpos[v]] for v in f.vars}; assign.update({v: kg[gpos[v]] for v in g.vars})
            acc.setdefault(tuple(assign[v] for v in rvars), []).append((common, fv[a] * gv[b]))
    data = {}
    for rk in itertools.product(range(4), repeat=len(rvars)):
        parts = acc.get(rk, [])
        if len(parts) == 1:
            data[rk] = parts[0]
        elif parts:
            data[rk] = E.aggregate(np.concatenate([p[0] for p in parts]), np.concatenate([p[1] for p in parts]))
        else:
            data[rk] = (np.zeros(0, np.int64), np.zeros(0))
    return E.Factor(rvars, ["#point"], "S", data)


def order_rule(fs):
    """Rule (a)'s order on point factors: start from the smallest factor (nnz), then repeatedly the smallest factor
    sharing a cut with those already folded (an unconnected one only when none is left)."""
    rest = sorted(fs, key=lambda p: fs[p].nnz())
    order = [rest.pop(0)]
    while rest:
        cur = set(v for p in order for v in fs[p].vars)
        nxt = min(rest, key=lambda p: (not (set(fs[p].vars) & cur), fs[p].nnz()))
        order.append(nxt); rest.remove(nxt)
    return order


def answer_points(models, tabs, S, Z, c, rho=RHO_STAR, trace=None):
    """KNITPLAN point-batch execution: semi-join pushdown, rule-(a) order, per-step format by predicted density."""
    P = Z.shape[0]
    fs = build_point_factors(models, tabs, S, Z)
    order = order_rule(fs)
    h = fs[order[0]]
    for step, p in enumerate(order[1:]):
        g = fs[p]
        later = set(v for q in order[step + 2:] for v in fs[q].vars)
        elim = sorted((set(h.vars) | set(g.vars)) - later)
        rvars = (set(h.vars) | set(g.vars)) - set(elim)
        dense_size = (4 ** len(rvars)) * P
        est = min(predicted_nnz(h, g), dense_size)
        fmt = "S" if est < dense_size * rho or dense_size > E.DENSE_CAP else "D"
        h = batched_product_sum(h, g, elim, fmt, P)
        if trace is not None:
            trace.append((p, fmt, est, dense_size))
    for v in list(h.vars):            # single-factor instance: sum the leftover cut variables
        h = _sparse(h)
        acc = {}
        ax = h.vars.index(v)
        for k, (i, val) in h.data.items():
            acc.setdefault(k[:ax] + k[ax + 1:], []).append((i, val))
        h = E.Factor(tuple(x for x in h.vars if x != v), h.outs, "S",
                     {rk: E.aggregate(np.concatenate([a for a, _ in ps]), np.concatenate([b for _, b in ps]))
                      for rk, ps in acc.items()})
    if h.fmt == "D":
        arr = h.data.reshape(-1); i = np.flatnonzero(arr)
        keys, vals = i.astype(np.int64), arr[i]
    else:
        keys, vals = h.data[()]
    nz = vals != 0.0
    return keys[nz], vals[nz] * (2.0 ** -c)


def sample_points(models, tabs, n, P, rng, frac_in=0.5):
    """Point batch for the workload: round(frac_in * P) points drawn from the product of the subcircuits' observed
    output supports (each subcircuit's output bits drawn uniformly from the distinct output codes it observed in any
    configuration; most such points have a non-zero estimate), the rest uniformly from {0,1}^n. Duplicates removed.
    Returns (Z, is_in)."""
    nin = int(round(frac_in * P))
    Zin = np.zeros((nin, n), np.uint8)
    for m in models:
        pos = [m.out_local[k] for k in range(len(m.out_qubits))]
        obs = np.unique(np.concatenate([E.gather_bits(idx, pos) for idx, _ in tabs[m.p].values()]))
        y = obs[rng.integers(obs.size, size=nin)]
        for k, q in enumerate(m.out_qubits):
            Zin[:, q] = (y >> k) & 1
    Zout = rng.integers(0, 2, size=(P - nin, n), dtype=np.uint8)
    Z = np.concatenate([Zin, Zout])
    is_in = np.r_[np.ones(nin, bool), np.zeros(P - nin, bool)]
    _, first = np.unique(Z, axis=0, return_index=True)
    first.sort()
    return Z[first], is_in[first]
