"""KNITPLAN planner v1 (M1, first repair attempt for K-TRIV; the pre-registered protocol Entry 5).

Plan space (extends the Phase-0 prototype): pushdown position x binary join tree over the subcircuit factors
(bushy and left-deep; children of every internal node share at least one cut index — no cross products) x format per
internal node {sparse, dense} x elimination granularity {joint, seq}. Elimination at a node: every cut index that occurs
only inside the node's subtree (the analysis (e): a variable is eliminated only when all factors containing it are joined).

Planner v1 = exhaustive enumeration of that space scored by a calibrated cost model:
  dense node:  k_dense * (#multiply-adds of the einsum) + k_dconv * (dense size of each sparse child converted)
  sparse node: k_rows * rows + k_pairs * (#consistent key pairs) + k_sconv * (dense size of each dense child converted)
  + k_node per node; constants fitted by non-negative least squares on operator microbenchmarks (calibrate()).
Supports: leaves exact (per cut-basis key); a join's per-key support is exact (exact_support_nc, no cancellation) for
two leaves and estimated by the random-placement union estimator cap * (1 - prod_t (1 - |U_t||V_t| / cap)) above.
"""
from __future__ import annotations

import itertools
import math
import time
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from . import engine as E


# ----------------------------------------------------------------------------------------------- trees
def _vars_of(fs, leaves):
    return set(v for p in leaves for v in fs[p].vars)


def enumerate_trees(fs):
    """All unordered binary trees over the factor ids whose internal nodes join children sharing a cut index (when the
    cut graph of a node's factor set is disconnected, the node is a cross product between whole components)."""
    ids = tuple(sorted(fs))
    var = {p: set(fs[p].vars) for p in ids}

    def components(sub):
        comp, seen = [], set()
        for s in sub:
            if s in seen:
                continue
            stack, c = [s], set()
            while stack:
                x = stack.pop()
                if x in c:
                    continue
                c.add(x)
                stack.extend(y for y in sub if y not in c and var[x] & var[y])
            seen |= c; comp.append(frozenset(c))
        return comp

    @lru_cache(maxsize=None)
    def trees(sub):
        sub = tuple(sorted(sub))
        if len(sub) == 1:
            return [sub[0]]
        out = []
        comps = components(sub)
        first, rest = sub[0], sub[1:]
        for k in range(0, len(rest)):
            for comb in itertools.combinations(rest, k):
                L = (first,) + comb
                R = tuple(x for x in rest if x not in comb)
                if not R:
                    continue
                if len(comps) > 1:
                    # disconnected cut graph (a subcircuit set with no shared cut, e.g. an isolated subcircuit): the
                    # cross product is unavoidable; split only between whole connected components
                    if any(not (cp <= set(L) or cp <= set(R)) for cp in comps):
                        continue
                elif not (set().union(*(var[p] for p in L)) & set().union(*(var[p] for p in R))):
                    continue
                for tl in trees(L):
                    for tr in trees(R):
                        out.append((tl, tr))
        return out

    return trees(ids)


def leaves(t):
    return (t,) if isinstance(t, int) else leaves(t[0]) + leaves(t[1])


def internal_nodes(t):
    """Post-order list of internal nodes."""
    if isinstance(t, int):
        return []
    return internal_nodes(t[0]) + internal_nodes(t[1]) + [t]


def tree_str(t):
    return str(t) if isinstance(t, int) else f"({tree_str(t[0])} {tree_str(t[1])})"


@dataclass
class TreePlan:
    position: str
    tree: object
    fmts: tuple          # one per internal node, post-order
    gran: str = "joint"
    name: str = ""

    def key(self):
        return f"{self.position}|{tree_str(self.tree)}|{''.join(self.fmts)}|{self.gran}"


def enumerate_tree_plans(fs, positions=("before",), grans=("joint", "seq")):
    out = []
    for t in enumerate_trees(fs):
        k = len(internal_nodes(t))
        for pos in positions:
            for fmts in itertools.product("SD", repeat=k):
                for g in grans:
                    out.append(TreePlan(pos, t, fmts, g))
    return out


def left_deep(order):
    t = order[0]
    for p in order[1:]:
        t = (t, p)
    return t


# ----------------------------------------------------------------------------------------------- execution
def fold_tree(plan: TreePlan, factors, c):
    all_vars = {p: set(factors[p].vars) for p in factors}
    fmt_list = list(plan.fmts)

    def ev(t, idx_box):
        if isinstance(t, int):
            return factors[t]
        h = ev(t[0], idx_box)
        g = ev(t[1], idx_box)
        fmt = fmt_list[idx_box[0]]; idx_box[0] += 1
        inside = set(leaves(t))
        outside = set().union(*(all_vars[p] for p in factors if p not in inside)) if len(inside) < len(factors) else set()
        elim = sorted((set(h.vars) | set(g.vars)) - outside)
        if plan.gran == "joint":
            return E.product_sum(h, g, elim, fmt)
        first = [v for v in elim if v in h.vars and v in g.vars][:1]
        out = E.product_sum(h, g, first, fmt)
        for v in elim:
            if v in out.vars:
                out = E.sum_out(out, v, fmt)
        return out

    h = ev(plan.tree, [0])
    for v in list(h.vars):
        h = E.sum_out(h, v, h.fmt)
    if h.fmt == "D":
        arr = h.data.reshape(-1)
        i = np.flatnonzero(arr)
        idx, val = i.astype(np.int64), arr[i]
    else:
        idx, val = h.data[()]
    return h.outs, idx, val * (2.0 ** -c)


# ----------------------------------------------------------------------------------------------- statistics
@dataclass
class Stat:
    """Support statistics of a (real or estimated) factor: vars, #outs, per-key support counts (dict `sup` or dense array
    `arr` over (4,)*len(vars); either may be None and is derived on demand; absent keys = 0), per-key support sets
    (leaves only; None above)."""
    vars: tuple
    nouts: int
    sup: dict | None
    sets: dict | None = None
    arr: np.ndarray | None = None

    def array(self):
        if self.arr is None:
            arr = np.zeros((4,) * len(self.vars))
            for k, v in self.sup.items():
                arr[k] = v
            self.arr = arr
        return self.arr

    def sup_dict(self):
        if self.sup is None:
            arr = self.arr
            self.sup = {tuple(int(x) for x in k): (int(arr[tuple(k)]) if float(arr[tuple(k)]).is_integer() else float(arr[tuple(k)]))
                        for k in np.argwhere(arr != 0)} if arr.ndim else ({(): arr.item()} if arr.item() else {})
        return self.sup

    def nnz(self):
        return float(self.array().sum())


def leaf_stat(f: E.Factor):
    f = E.to_format(f, "S")
    return Stat(f.vars, len(f.outs), {k: int(i.size) for k, (i, _) in f.data.items()}, {k: i for k, (i, _) in f.data.items()})


def _expand(st: Stat, allv):
    """st's dense support array with axes reordered to allv (size-1 axes for vars st does not have)."""
    arr = st.array()
    order = sorted(range(len(st.vars)), key=lambda i: allv.index(st.vars[i]))
    arr = np.transpose(arr, order) if len(order) > 1 else arr
    shape = [4 if v in st.vars else 1 for v in allv]
    return arr.reshape(shape)


def join_stat(a: Stat, b: Stat, elim):
    """Vectorised join statistics (same values as join_stat_ref, which is kept as the specification and test oracle)."""
    allv = tuple(sorted(set(a.vars) | set(b.vars)))
    rvars = tuple(v for v in allv if v not in set(elim))
    eax = tuple(i for i, v in enumerate(allv) if v in set(elim))
    cap = 2 ** (a.nouts + b.nouts)
    A = _expand(a, allv); B = _expand(b, allv)
    nz = (A > 0) & (B > 0)
    nz = np.broadcast_to(nz, tuple(4 for _ in allv))
    J = np.broadcast_to(A * B, nz.shape)
    pairs = int(np.count_nonzero(nz))
    rows = J[nz].sum()
    rows = int(rows) if float(rows).is_integer() else float(rows)
    exact = a.sets is not None and b.sets is not None
    if exact:
        out = np.zeros((4,) * len(rvars))
        coords = np.argwhere(nz)
        if len(coords):
            rpos = [allv.index(v) for v in rvars]
            apos = [allv.index(v) for v in a.vars]; bpos = [allv.index(v) for v in b.vars]
            lin = lambda pos: np.ravel_multi_index(tuple(coords[:, i] for i in pos), (4,) * len(pos)) if pos else                 np.zeros(len(coords), dtype=np.int64)
            g_lin, a_lin, b_lin = lin(rpos), lin(apos), lin(bpos)
            for g, n in _union_exact_vec(g_lin, a_lin, b_lin, _csr(a, len(a.vars)), _csr(b, len(b.vars)),
                                         a.nouts, b.nouts, a.sets, b.sets, a.vars, b.vars).items():
                out[np.unravel_index(g, (4,) * len(rvars)) if rvars else ()] = n
    else:
        q = np.prod(np.where(nz, np.maximum(0.0, 1.0 - J / cap), 1.0), axis=eax) if eax else             np.where(nz, np.maximum(0.0, 1.0 - J / cap), 1.0)
        out = np.rint(cap * (1.0 - q))
    return Stat(rvars, a.nouts + b.nouts, None, None, out), rows, pairs


_BATCH = 1 << 22


def _union_exact_batch(groups, asets, bsets, na, nb):
    """{rk: |union_{(ka,kb) in group} U_ka x V_kb|} for all groups. Small groups are counted together with one np.unique
    over (group id, u, v) codes; groups with many product rows use the signature method (_union_exact)."""
    res = {}
    keys = list(groups)
    gid_bits = max(1, (len(keys) - 1).bit_length())
    shift = na + nb
    batched = gid_bits + shift <= 62
    buf, size = [], 0

    def flush():
        nonlocal buf, size
        if buf:
            u = np.unique(np.concatenate(buf))
            cnt = np.bincount((u >> shift).astype(np.int64), minlength=len(keys))
            for g in np.flatnonzero(cnt):
                res[keys[g]] = int(cnt[g])
        buf, size = [], 0

    for g, rk in enumerate(keys):
        ps = groups[rk]
        grows = sum(asets[ka].size * bsets[kb].size for ka, kb in ps)
        if not batched or grows > _BATCH:
            res[rk] = _union_exact([(ka, kb, 0, 0) for ka, kb in ps], asets, bsets)
            continue
        if size + grows > _BATCH:
            flush()
        base = np.int64(g) << np.int64(shift)
        for ka, kb in ps:
            u = asets[ka].astype(np.int64); v = bsets[kb].astype(np.int64)
            buf.append((base | (u[:, None] << np.int64(nb)) | v[None, :]).ravel())
        size += grows
    flush()
    return res


def _csr(st: Stat, nv):
    """Concatenated support sets of a leaf by linear key: (offsets, lengths, values)."""
    if getattr(st, "_csr", None) is not None:
        return st._csr
    K = 4 ** nv
    lens = np.zeros(K, dtype=np.int64); parts = [None] * K
    for k, u in st.sets.items():
        li = int(np.ravel_multi_index(k, (4,) * nv)) if nv else 0
        lens[li] = u.size; parts[li] = u.astype(np.int64)
    offs = np.zeros(K, dtype=np.int64); offs[1:] = np.cumsum(lens)[:-1]
    vals = np.concatenate([x for x in parts if x is not None] or [np.zeros(0, dtype=np.int64)])
    # parts must be laid out in linear-key order for offs to be right
    vals = np.concatenate([parts[i] if parts[i] is not None else np.zeros(0, dtype=np.int64) for i in range(K)])
    st._csr = (offs, lens, vals)
    return st._csr


def _union_exact_vec(g, pa, pb, ca, cb, na, nb, asets, bsets, avars, bvars):
    """{group: |union over its pairs of U_pa x V_pb|}, fully vectorised: ragged cross products of the pairs, encoded as
    (group, u, v) int64 codes, counted by np.unique in chunks cut at group boundaries; a group whose product rows exceed
    the chunk size falls back to the signature method."""
    aoff, alen, aval = ca; boff, blen, bval = cb
    order = np.argsort(g, kind="stable"); g, pa, pb = g[order], pa[order], pb[order]
    la, lb = alen[pa], blen[pb]; n = la * lb
    ug, gstart = np.unique(g, return_index=True)
    gend = np.append(gstart[1:], len(g))
    grows = np.add.reduceat(n, gstart) if len(g) else np.zeros(0, dtype=np.int64)
    res = {}
    shift = na + nb
    gbits = max(1, int(ug.max()).bit_length()) if len(ug) else 1
    vec_ok = gbits + shift <= 62
    i = 0
    while i < len(ug):
        if not vec_ok or grows[i] > _BATCH:  # big group: signature method
            s0, s1 = gstart[i], gend[i]
            ka = [tuple(np.unravel_index(int(x), (4,) * len(avars))) if avars else () for x in pa[s0:s1]]
            kb = [tuple(np.unravel_index(int(x), (4,) * len(bvars))) if bvars else () for x in pb[s0:s1]]
            ka = [tuple(int(y) for y in k) for k in ka]; kb = [tuple(int(y) for y in k) for k in kb]
            res[int(ug[i])] = _union_exact([(x, y, 0, 0) for x, y in zip(ka, kb)], asets, bsets)
            i += 1; continue
        j, tot = i, 0
        while j < len(ug) and grows[j] <= _BATCH and tot + grows[j] <= _BATCH:
            tot += grows[j]; j += 1
        s0, s1 = gstart[i], gend[j - 1]
        nn = n[s0:s1]
        if tot:
            pidx = np.repeat(np.arange(s0, s1), nn)
            starts = np.repeat(np.cumsum(nn) - nn, nn)
            local = np.arange(tot, dtype=np.int64) - starts
            lbp = lb[pidx]
            u = aval[aoff[pa[pidx]] + local // lbp]; v = bval[boff[pb[pidx]] + local % lbp]
            codes = np.unique((g[pidx].astype(np.int64) << np.int64(shift)) | (u << np.int64(nb)) | v)
            gg, cnt = np.unique(codes >> np.int64(shift), return_counts=True)
            for x, c in zip(gg.tolist(), cnt.tolist()):
                res[x] = c
        i = j
    return res


def join_stat_ref(a: Stat, b: Stat, elim):
    asup, bsup = a.sup_dict(), b.sup_dict()
    pa = {v: i for i, v in enumerate(a.vars)}; pb = {v: i for i, v in enumerate(b.vars)}
    shared = [v for v in a.vars if v in b.vars]
    rvars = tuple(sorted((set(a.vars) | set(b.vars)) - set(elim)))
    cap = 2 ** (a.nouts + b.nouts)
    groups = {}
    bidx = {}
    for kb in bsup:
        bidx.setdefault(tuple(kb[pb[v]] for v in shared), []).append(kb)
    pairs = rows = 0
    for ka, sa in asup.items():
        if sa == 0:
            continue
        for kb in bidx.get(tuple(ka[pa[v]] for v in shared), []):
            sb = bsup[kb]
            if sb == 0:
                continue
            asg = {v: ka[pa[v]] for v in a.vars}; asg.update({v: kb[pb[v]] for v in b.vars})
            groups.setdefault(tuple(asg[v] for v in rvars), []).append((ka, kb, sa, sb))
            pairs += 1; rows += sa * sb
    sup = {}
    exact = a.sets is not None and b.sets is not None
    for rk, ps in groups.items():
        if exact:
            sup[rk] = _union_exact(ps, a.sets, b.sets)
        else:
            q = 1.0
            for _, _, sa, sb in ps:
                q *= max(0.0, 1.0 - sa * sb / cap)
            sup[rk] = int(round(cap * (1.0 - q)))
    return Stat(rvars, a.nouts + b.nouts, {k: v for k, v in sup.items() if v}, None), rows, pairs


def _union_exact(ps, asets, bsets):
    """|union_t U_t x V_t| by signatures, no product enumeration."""
    us = [(t, asets[ka]) for t, (ka, kb, _, _) in enumerate(ps)]
    allu = np.concatenate([u for _, u in us]); tid = np.concatenate([np.full(u.size, t) for t, u in us])
    order = np.lexsort((tid, allu)); allu, tid = allu[order], tid[order]
    bounds = np.flatnonzero(np.diff(allu)) + 1
    sigs = {}
    for chunk in np.split(tid, bounds):
        key = tuple(chunk.tolist()); sigs[key] = sigs.get(key, 0) + 1
    tot = 0
    for sig, cnt in sigs.items():
        vs = [bsets[ps[t][1]] for t in sig]
        tot += cnt * (vs[0].size if len(vs) == 1 else np.unique(np.concatenate(vs)).size)
    return tot


# ----------------------------------------------------------------------------------------------- cost model
FEATURES = ("node", "dense_madds", "dense_conv", "rows", "pairs", "sparse_conv", "dense_out")


@dataclass
class CostModel:
    k: dict

    def cost(self, feats):
        return sum(self.k.get(f, 0.0) * x for f, x in feats.items())


DEFAULT_K = {"node": 2e-4, "dense_madds": 1.5e-9, "dense_conv": 3e-9, "rows": 1.5e-7, "pairs": 3e-6,
             "sparse_conv": 4e-9, "dense_out": 1e-9}


def node_features(sa: Stat, sb: Stat, elim, fmt, child_fmts, rows, pairs, gran):
    allv = set(sa.vars) | set(sb.vars)
    rv = allv - set(elim)
    f = dict.fromkeys(FEATURES, 0.0)
    f["node"] = 1.0 if gran == "joint" else 1.0 + len(elim)
    dsize = lambda s: (4 ** len(s.vars)) * (2 ** s.nouts)
    if fmt == "D":
        f["dense_madds"] = (4 ** len(allv)) * (2 ** (sa.nouts + sb.nouts))
        f["dense_conv"] = sum(dsize(s) for s, cf in ((sa, child_fmts[0]), (sb, child_fmts[1])) if cf == "S")
        f["dense_out"] = (4 ** len(rv)) * (2 ** (sa.nouts + sb.nouts))
    else:
        f["rows"] = rows
        f["pairs"] = pairs
        f["sparse_conv"] = sum(dsize(s) for s, cf in ((sa, child_fmts[0]), (sb, child_fmts[1])) if cf == "D")
    return f


def plan_features(plan: TreePlan, leaf_stats, cache=None, join=None):
    """Total feature vector of a plan, from statistics only (no execution). `join` overrides join_stat (planner v2)."""
    cache = {} if cache is None else cache
    join = join_stat if join is None else join
    all_vars = {p: set(s.vars) for p, s in leaf_stats.items()}
    tot = dict.fromkeys(FEATURES, 0.0)
    fl = list(plan.fmts)
    box = [0]

    def ev(t):
        if isinstance(t, int):
            return leaf_stats[t], "S"
        a, fa = ev(t[0]); b, fb = ev(t[1])
        fmt = fl[box[0]]; box[0] += 1
        inside = set(leaves(t))
        outside = set().union(*(all_vars[p] for p in leaf_stats if p not in inside)) if len(inside) < len(leaf_stats) else set()
        elim = tuple(sorted((set(a.vars) | set(b.vars)) - outside))
        ck = (tree_str(t), elim)
        if ck not in cache:
            cache[ck] = join(a, b, elim)
        st, rows, pairs = cache[ck]
        for f, x in node_features(a, b, elim, fmt, (fa, fb), rows, pairs, plan.gran).items():
            tot[f] += x
        # memory peaks of this node (engine caps: DENSE_CAP entries per dense array, ROW_CAP products per sparse op);
        # keys start with "_" so the cost model ignores them
        if fmt == "D":
            dsz = lambda s: (4 ** len(s.vars)) * (2 ** s.nouts)
            rv = (set(a.vars) | set(b.vars)) - set(elim)
            peak = max((4 ** len(rv)) * (2 ** (a.nouts + b.nouts)), dsz(a), dsz(b))
            tot["_peak_dense"] = max(tot.get("_peak_dense", 0), peak)
        else:
            tot["_peak_rows"] = max(tot.get("_peak_rows", 0), rows)
        return st, fmt

    ev(plan.tree)
    return tot


def feasible(feats):
    """A plan is feasible if no node exceeds the engine's memory caps (predicted from the same statistics)."""
    return feats.get("_peak_dense", 0) <= E.DENSE_CAP and feats.get("_peak_rows", 0) <= E.ROW_CAP


def choose(fs, model: CostModel, positions=("before",), grans=("joint", "seq")):
    """Planner v1: exhaustive enumeration scored by the cost model. Returns (plan, predicted cost, planning seconds)."""
    t0 = time.perf_counter()
    stats = {p: leaf_stat(f) for p, f in fs.items()}
    cache = {}
    best = None
    for plan in enumerate_tree_plans(fs, positions, grans):
        c = model.cost(plan_features(plan, stats, cache))
        if best is None or c < best[1]:
            best = (plan, c)
    return best[0], best[1], time.perf_counter() - t0


# ----------------------------------------------------------------------------------------------- tuned fixed rules
def dense_optimal_tree(fs):
    """Size-only optimal tree for dense execution (the tensor-network contraction order on dense shapes, what an
    einsum path optimizer would pick): exhaustive over trees, minimising total dense multiply-adds."""
    best = None
    for t in enumerate_trees(fs):
        cost = 0
        all_vars = {p: set(fs[p].vars) for p in fs}

        def ev(t):
            nonlocal cost
            if isinstance(t, int):
                return set(fs[t].vars), len(fs[t].outs)
            va, oa = ev(t[0]); vb, ob = ev(t[1])
            inside = set(leaves(t))
            outside = set().union(*(all_vars[p] for p in fs if p not in inside)) if len(inside) < len(fs) else set()
            cost += (4 ** len(va | vb)) * (2 ** (oa + ob))
            return (va | vb) - outside, oa + ob

        ev(t)
        if best is None or cost < best[0]:
            best = (cost, t)
    return best[1]


def fixed_rules(fs, models):
    """Fixed (statistics-free) rules, all with pushdown below the merge."""
    r = len(fs)
    cq = tuple(sorted(range(r), key=lambda p: len(models[p].out_qubits)))
    # CutQC smart order must be connected for a left-deep tree without cross products; repair by greedy connection
    order = [cq[0]]
    rest = list(cq[1:])
    while rest:
        cur = _vars_of(fs, order)
        nxt = next((p for p in rest if set(fs[p].vars) & cur), rest[0])
        order.append(nxt); rest.remove(nxt)
    ld = left_deep(order)
    dt = dense_optimal_tree(fs)
    k = r - 1
    return {
        "b_dense_cutqc_order": TreePlan("before", ld, ("D",) * k, "joint"),
        "c_sparse_cutqc_order": TreePlan("before", ld, ("S",) * k, "joint"),
        "e_dense_optimal_tree": TreePlan("before", dt, ("D",) * k, "joint"),
        "f_sparse_optimal_tree": TreePlan("before", dt, ("S",) * k, "joint"),
    }


# ----------------------------------------------------------------------------------------------- calibration
def calibrate(seconds_budget=120.0, seed=0, reps=3):
    """Fit the cost constants by NNLS on random operator microbenchmarks (sparse / dense, mixed child formats,
    1-2 shared cut indices, 4-14 outputs per child, densities 1e-3..1). Independent of every workload instance."""
    from scipy.optimize import nnls
    rng = np.random.default_rng(seed)
    X, y = [], []
    t_end = time.time() + seconds_budget
    while time.time() < t_end:
        nv = int(rng.integers(1, 3)); extra_a = int(rng.integers(0, 2)); extra_b = int(rng.integers(0, 2))
        va = tuple(range(nv)) + tuple(10 + i for i in range(extra_a))
        vb = tuple(range(nv)) + tuple(20 + i for i in range(extra_b))
        oa, ob = int(rng.integers(4, 13)), int(rng.integers(4, 13))
        if (4 ** len(set(va) | set(vb))) * 2 ** (oa + ob) > 2 ** 27:
            continue
        da, db = 10 ** rng.uniform(-3, 0), 10 ** rng.uniform(-3, 0)
        fa = _rand_factor(rng, va, oa, da); fb = _rand_factor(rng, vb, ob, db)
        elim = list(range(nv))
        fmt = "S" if rng.random() < 0.5 else "D"
        cfa = "S" if rng.random() < 0.5 else "D"; cfb = "S" if rng.random() < 0.5 else "D"
        A = E.to_format(fa, cfa); B = E.to_format(fb, cfb)
        sa, sb = leaf_stat(fa), leaf_stat(fb)
        _, rows, pairs = join_stat(Stat(sa.vars, sa.nouts, sa.sup), Stat(sb.vars, sb.nouts, sb.sup), elim)
        try:
            E.product_sum(A, B, elim, fmt)
            ts = []
            for _ in range(reps):
                t = time.perf_counter(); E.product_sum(A, B, elim, fmt); ts.append(time.perf_counter() - t)
        except E.MemoryCap:
            continue
        feats = node_features(sa, sb, elim, fmt, (cfa, cfb), rows, pairs, "joint")
        X.append([feats[f] for f in FEATURES]); y.append(float(np.median(ts)))
    X = np.array(X); y = np.array(y)
    # relative-error weighting: divide each row by its time
    w = 1.0 / np.maximum(y, 1e-6)
    k, _ = nnls(X * w[:, None], y * w)
    pred = X @ k
    return CostModel(dict(zip(FEATURES, k.tolist()))), {"n": int(len(y)),
                                                        "median_rel_err": float(np.median(np.abs(pred - y) / y)),
                                                        "p90_rel_err": float(np.quantile(np.abs(pred - y) / y, 0.9))}


def _rand_factor(rng, vars_, nouts, density):
    data = {}
    for k in itertools.product(range(4), repeat=len(vars_)):
        m = int(rng.binomial(2 ** nouts, density))
        i = np.sort(rng.choice(2 ** nouts, size=m, replace=False)).astype(np.int64) if m else np.zeros(0, np.int64)
        data[k] = (i, rng.normal(size=i.size))
    return E.Factor(tuple(vars_), list(range(nouts)), "S", data)


# ----------------------------------------------------------------------------------------------- generic statistics
def generic_leaf_stat(f: E.Factor):
    """What a general sparse-tensor optimizer knows (R-G1 ablation): only the factor's total nnz, spread uniformly over
    the cut-basis keys; no per-key support sets, so every join uses the random-placement estimator."""
    f = E.to_format(f, "S")
    keys = list(f.data)
    tot = sum(int(i.size) for i, _ in f.data.values())
    return Stat(f.vars, len(f.outs), {k: tot / max(len(keys), 1) for k in keys}, None)


def choose_generic(fs, model: CostModel, positions=("before",), grans=("joint", "seq")):
    t0 = time.perf_counter()
    stats = {p: generic_leaf_stat(f) for p, f in fs.items()}
    cache, best = {}, None
    for plan in enumerate_tree_plans(fs, positions, grans):
        c = model.cost(plan_features(plan, stats, cache))
        if best is None or c < best[1]:
            best = (plan, c)
    return best[0], best[1], time.perf_counter() - t0


# ----------------------------------------------------------------------------------------------- planner v2 (ladder attempt 2)
UNION_S_PER_ROW = 6e-8  # rough cost of the exact two-leaf union per product row (np.unique on int64 codes)


def choose_v2(fs, models, model: CostModel, rho=0.1, positions=("before",), grans=("joint", "seq")):
    """Planner v2 = planner v1 with a planning budget proportional to the predicted execution cost (the pre-registered protocol).

    Planning can save at most the execution time of the fallback plan, so v2 spends at most rho x (predicted cost of the
    fallback) on planning: fallback = rule (b) (dense, CutQC order), its cost predicted from nnz-only statistics (cheap).
    Within the budget it scores candidate plans in a fixed order — the fixed-rule plans (b), (e), (c), (f) first, then
    the rest of the space in enumeration order — with the same cost model and cut statistics as v1; an exact two-leaf
    support union whose estimated cost exceeds the remaining budget is replaced by the random-placement estimator. It
    stops when the budget is spent or the space is exhausted, and returns the best-scored plan (with an unbounded budget
    it is exactly v1). Returns (plan, predicted cost, planning seconds, info)."""
    t0 = time.perf_counter()
    fr = fixed_rules(fs, models)
    gst = {p: generic_leaf_stat(f) for p, f in fs.items()}
    # memory budget (problem statement: the answer must stay within a memory budget). Dense peaks depend on shapes
    # only, so they are checked on the cheap nnz-only statistics before any join statistic is computed. When the dense
    # fallback (b) cannot run under the engine caps, the fallback -- and the planning budget's reference -- is the
    # sparse rule (c); only feasible plans can become `best`.
    fallback = fr["b_dense_cutqc_order"]
    fb_feats = plan_features(fallback, gst)
    if fb_feats.get("_peak_dense", 0) > E.DENSE_CAP:
        fallback = fr["c_sparse_cutqc_order"]; fb_feats = plan_features(fallback, gst)
    pred_fb = model.cost(fb_feats)
    budget = rho * pred_fb
    info = {"budget_s": budget, "pred_fallback": pred_fb, "scored": 0, "estimated_unions": 0, "exhausted": False,
            "infeasible_skipped": 0, "fallback": "b" if fallback is fr["b_dense_cutqc_order"] else "c"}
    best = (fallback, None)

    def left():
        return budget - (time.perf_counter() - t0)

    if left() <= 0:
        return best[0], pred_fb, time.perf_counter() - t0, info
    stats = {p: leaf_stat(f) for p, f in fs.items()}
    cache = {}

    def budgeted_join(a, b, elim):
        if a.sets is not None and b.sets is not None:
            rows = float((_expand(a, tuple(sorted(set(a.vars) | set(b.vars)))) *
                          _expand(b, tuple(sorted(set(a.vars) | set(b.vars))))).sum())
            if rows * UNION_S_PER_ROW > left():
                info["estimated_unions"] += 1
                return join_stat(Stat(a.vars, a.nouts, a.sup, None, a.arr), Stat(b.vars, b.nouts, b.sup, None, b.arr), elim)
        return join_stat(a, b, elim)

    order = [fr[k] for k in ("b_dense_cutqc_order", "e_dense_optimal_tree", "c_sparse_cutqc_order", "f_sparse_optimal_tree")]
    seen = {p.key() for p in order}
    rest = [p for p in enumerate_tree_plans(fs, positions, grans) if p.key() not in seen]
    for plan in order + rest:
        if left() <= 0:
            break
        if any(f == "D" for f in plan.fmts) and plan_features(plan, gst).get("_peak_dense", 0) > E.DENSE_CAP:
            info["infeasible_skipped"] += 1      # dense node over the cap: skipped before any join statistic
            continue
        feats = plan_features(plan, stats, cache, join=budgeted_join)
        info["scored"] += 1
        if not feasible(feats):
            info["infeasible_skipped"] += 1
            continue
        c = model.cost(feats)
        if best[1] is None or c < best[1]:
            best = (plan, c)
    else:
        info["exhausted"] = True
    return best[0], (best[1] if best[1] is not None else pred_fb), time.perf_counter() - t0, info
