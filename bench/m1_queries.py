"""Query set of the frozen main workload (the pre-registered protocol).

Per (instance, S, noise) cell, with the seed fixed from the cell id:
  marg10, marg20, marg30   marginal over |A| random qubits (only when |A| < n)
  margn                    full distribution (A = all qubits), only when n <= MARGN_MAX
  marghalf                 |A| = n/2 random qubits, when n > MARGN_MAX (wide queries, the wide-subcircuit stratum caveat (i))
  gb_rand                  group-by: |A| = 10 random qubits, 5 other random qubits selected; sigma = their bits in
                           one point drawn from the subcircuits' observed outputs (so the selection is not empty by
                           construction of a random sigma)
  gb_struct                group-by over a structured subset: A = qubits 0..9, selection on qubits 10..14, sigma as
                           above
  dd0, dd1, dd2            CutQC's dynamic-definition bin sequence (mem_limit DD_MEM, recursion_depth DD_DEPTH):
                           layer 0 = CutQC's distribute_load schedule; later layers zoom into the largest bins found so
                           far (CutQC's rule: top DD_DEPTH bins above 1/2^n/10, largest first). The bin values come
                           from the reference answer (KNITPLAN), so every system runs the same fixed sequence. n > DD_MEM.
  pts1e3, pts1e5           point batches of 10^3 / 10^5 bitstrings, half drawn from the product of the subcircuits'
                           observed outputs, half uniform (knitplan.points.sample_points), duplicates removed.
A query is a dict {name, A (sorted list), sel (dict or None), points (array or None)}.
"""
import hashlib

import numpy as np

from knitplan import points as PT

MARGN_MAX = 40
DD_MEM, DD_DEPTH = 20, 3


def cell_seed(*parts):
    return int(hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest(), 16) % 2 ** 32


def _sigma(models, tabs, n, qubits, rng):
    Z, _ = PT.sample_points(models, tabs, n, 1, rng, frac_in=1.0)
    return {int(q): int(Z[0, q]) for q in qubits}


def static_queries(models, tabs, n, seed):
    rng = np.random.default_rng(seed)
    qs = []
    for k in (10, 20, 30):
        if k < n:
            qs.append({"name": f"marg{k}", "A": sorted(rng.choice(n, k, replace=False).tolist()), "sel": None})
    if n <= MARGN_MAX:
        qs.append({"name": "margn", "A": list(range(n)), "sel": None})
    else:   # wide queries where the full marginal is not run (the protocol caveat (i): |A ∩ O_j| must reach the wide regime)
        qs.append({"name": "marghalf", "A": sorted(rng.choice(n, n // 2, replace=False).tolist()), "sel": None})
    if n >= 15:
        A = sorted(rng.choice(n, 10, replace=False).tolist())
        rest = [q for q in range(n) if q not in A]
        B = sorted(rng.choice(rest, 5, replace=False).tolist())
        qs.append({"name": "gb_rand", "A": A, "sel": _sigma(models, tabs, n, B, rng)})
        qs.append({"name": "gb_struct", "A": list(range(10)), "sel": _sigma(models, tabs, n, range(10, 15), rng)})
    for P, name in ((1000, "pts1e3"), (100000, "pts1e5")):
        Z, is_in = PT.sample_points(models, tabs, n, P, rng)
        qs.append({"name": name, "A": [], "sel": None, "points": Z, "points_in": int(is_in.sum())})
    return qs


def _distribute(caps, total):
    """CutQC DynamicDefinition.distribute_load (same arithmetic, dict order = subcircuit index)."""
    load = min(sum(caps.values()), total); cap_tot = sum(caps.values())
    loads = {p: int(caps[p] / cap_tot * load) if cap_tot else 0 for p in caps}
    left = load - sum(loads.values())
    for p in loads:
        while left > 0 and loads[p] < caps[p]:
            loads[p] += 1; left -= 1
    return loads


def dd_sequence(models, n, answer, mem=DD_MEM, depth=DD_DEPTH):
    """CutQC's DD bin sequence as (A, sel) queries. answer(A, sel) -> (keys, vals) canonical over A (reference)."""
    order = {}
    for m in models:                       # CutQC state position t <-> local comp position sorted descending
        comp = sorted(m.out_local); L = len(comp)
        order[m.p] = [m.out_qubits[m.out_local.index(comp[L - 1 - t])] for t in range(L)]
    seq, queue = [], []
    merged = {p: list(order[p]) for p in order}
    fixed = {}
    for layer in range(depth):
        if layer > 0:
            if not queue:
                break
            _, fixed, merged = queue.pop(0)
        loads = _distribute({p: len(merged[p]) for p in merged}, mem)
        active = {p: merged[p][:loads[p]] for p in merged}
        rest = {p: merged[p][loads[p]:] for p in merged}
        A = sorted(q for p in active for q in active[p])
        seq.append({"name": f"dd{layer}", "A": A, "sel": dict(fixed) or None})
        if layer < depth - 1 and any(rest.values()):
            keys, vals = answer(A, dict(fixed) or None)
            if keys.size:
                top = np.argsort(-vals, kind="stable")[:depth]
                for t in top:
                    if vals[t] > 1 / 2 ** n / 10:
                        f2 = dict(fixed); f2.update({q: int((keys[t] >> j) & 1) for j, q in enumerate(A)})
                        queue.append((float(vals[t]), f2, rest))
                queue = sorted(queue, key=lambda x: -x[0])[:depth]
    return seq
