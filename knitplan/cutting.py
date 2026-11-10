"""Wire-cut schemes (fixed inputs). A small recorded cut finder: local search over gate partitions.

KNITPLAN never searches cuts at query time. This module exists only to produce, once per pilot instance, a frozen
wire-cut scheme that is then written into the instance manifest.

Data model
----------
* ``ops``: the circuit's instructions (no measurements / barriers), in order.
* A partition assigns each multi-qubit op to a part 0..r-1; single-qubit ops follow the nearest preceding
  multi-qubit op on their wire (or the following one if none precedes).
* For each qubit, consecutive ops in different parts create a cut edge (upstream part -> downstream part).
* A subcircuit's qubits are the wire segments (maximal runs of one qubit inside one part).
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass, field

from qiskit import QuantumCircuit


@dataclass
class Segment:
    qubit: int            # original qubit
    part: int             # subcircuit index
    ops: list             # op indices (in order)
    starts_with_cut: bool # prepared from an incoming cut
    ends_with_cut: bool   # measured into an outgoing cut


@dataclass
class CutScheme:
    n: int
    r: int
    parts: list                      # part index per op
    segments: list = field(default_factory=list)   # list[Segment]
    sub_qubits: list = field(default_factory=list) # per part: list of segment ids (subcircuit qubit order)
    cuts: list = field(default_factory=list)       # (upstream_seg, downstream_seg)
    out_seg: dict = field(default_factory=dict)    # original qubit -> segment id holding its output

    @property
    def c(self):
        return len(self.cuts)

    def widths(self):
        return [len(q) for q in self.sub_qubits]

    def to_json(self):
        return {"n": self.n, "r": self.r, "parts": self.parts, "c": self.c, "widths": self.widths()}


def strip(circ: QuantumCircuit) -> QuantumCircuit:
    out = QuantumCircuit(circ.num_qubits)
    for inst in circ.data:
        if inst.operation.name in ("barrier", "measure"):
            continue
        out.append(inst.operation, [circ.find_bit(q).index for q in inst.qubits])
    return out


def _wire_sequences(circ):
    seq = [[] for _ in range(circ.num_qubits)]
    qubits = []
    for k, inst in enumerate(circ.data):
        qs = tuple(circ.find_bit(q).index for q in inst.qubits)
        qubits.append(qs)
        for q in qs:
            seq[q].append(k)
    return seq, qubits


def assign_single_qubit_ops(circ, multi_part):
    """Extend a partition of the multi-qubit ops to all ops."""
    seq, qubits = _wire_sequences(circ)
    part = dict(multi_part)
    for q, ops in enumerate(seq):
        multi = [k for k in ops if len(qubits[k]) > 1]
        for i, k in enumerate(ops):
            if len(qubits[k]) > 1:
                continue
            prev = [m for m in ops[:i] if len(qubits[m]) > 1]
            nxt = [m for m in ops[i + 1:] if len(qubits[m]) > 1]
            if prev:
                part[k] = part[prev[-1]]
            elif nxt:
                part[k] = part[nxt[0]]
            else:
                part[k] = None  # idle-wire op; resolved later
        if not multi:
            for k in ops:
                part[k] = None
    return part


def evaluate(circ, part_of_multi, r):
    """Return (cuts, widths) for a partition of the multi-qubit ops."""
    seq, qubits = _wire_sequences(circ)
    width = [0] * r
    cuts = 0
    for q, ops in enumerate(seq):
        multi = [part_of_multi[k] for k in ops if len(qubits[k]) > 1]
        if not multi:
            continue
        runs = 1
        width[multi[0]] += 1
        for a, b in zip(multi, multi[1:]):
            if a != b:
                runs += 1
                cuts += 1
                width[b] += 1
    # idle wires (no multi-qubit op) are added to the narrowest part later; count them now
    idle = sum(1 for q, ops in enumerate(seq) if not any(len(qubits[k]) > 1 for k in ops))
    for _ in range(idle):
        width[width.index(min(width))] += 1
    return cuts, width


def config_counts(circ, part_of_multi, r):
    """Configurations per part, 4^{#incoming cuts} * 3^{#outgoing cuts} (CutQC's subcircuit variants)."""
    seq, qubits = _wire_sequences(circ)
    ins, outs = [0] * r, [0] * r
    for q, ops in enumerate(seq):
        multi = [part_of_multi[k] for k in ops if len(qubits[k]) > 1]
        for a, b in zip(multi, multi[1:]):
            if a != b:
                outs[a] += 1; ins[b] += 1
    return [4 ** i * 3 ** o for i, o in zip(ins, outs)]


def find_scheme(circ: QuantumCircuit, r: int, max_width: int, seed: int = 0, iters: int = 20000,
                restarts: int = 8, max_configs=None):
    """Local search (simulated annealing) for a partition with few cuts and every width <= max_width.
    max_configs (E1 probe v2): also penalise parts with more than max_configs configurations (100 per factor of 2
    above the cap), so the cuts are spread over the parts; Phase-0 schemes used max_configs=None."""
    circ = strip(circ)
    seq, qubits = _wire_sequences(circ)
    multi_ops = [k for k, qs in enumerate(qubits) if len(qs) > 1]
    if not multi_ops:
        return None
    rng = random.Random(seed)
    n = circ.num_qubits

    def cost(p):
        cuts, width = evaluate(circ, p, r)
        excess = sum(max(0, w - max_width) for w in width)
        empty = sum(1 for w in width if w == 0)
        pen = 0.0
        if max_configs:
            import math
            pen = sum(max(0.0, math.log2(x / max_configs)) for x in config_counts(circ, p, r))
        return cuts + 100 * excess + 1000 * empty + 100 * pen, cuts, width

    best = None
    for rs in range(restarts):
        if rs % 2 == 0:   # spatial init: by lowest qubit
            p = {k: min(min(qubits[k]) * r // n, r - 1) for k in multi_ops}
        else:             # temporal init: by op position
            p = {k: min(i * r // max(1, len(multi_ops)), r - 1) for i, k in enumerate(multi_ops)}
        cur, _, _ = cost(p)
        T = 2.0
        for it in range(iters):
            k = rng.choice(multi_ops)
            old = p[k]
            new = rng.randrange(r)
            if new == old:
                continue
            p[k] = new
            c2, _, _ = cost(p)
            if c2 <= cur or rng.random() < pow(2.718, (cur - c2) / T):
                cur = c2
            else:
                p[k] = old
            T = max(0.05, T * 0.9995)
        total, cuts, width = cost(p)
        if max(width) <= max_width and min(width) > 0:
            if best is None or (cuts, max(width)) < (best[0], max(best[1])):
                best = (cuts, width, dict(p))
    if best is None:
        return None
    return build_scheme(circ, best[2], r)


def build_scheme(circ: QuantumCircuit, multi_part, r) -> CutScheme:
    circ = strip(circ)
    seq, qubits = _wire_sequences(circ)
    part = assign_single_qubit_ops(circ, multi_part)
    sch = CutScheme(n=circ.num_qubits, r=r, parts=[None] * len(qubits))
    # idle wires go to the narrowest part
    base_width = [0] * r
    for q, ops in enumerate(seq):
        prev = None
        for k in ops:
            if part[k] is not None and part[k] != prev:
                base_width[part[k]] += 1
                prev = part[k]
    for q, ops in enumerate(seq):
        if all(part[k] is None for k in ops):
            tgt = base_width.index(min(base_width))
            base_width[tgt] += 1
            for k in ops:
                part[k] = tgt
            if not ops:
                part[("idle", q)] = tgt
    for k in range(len(qubits)):
        sch.parts[k] = part[k]
    # segments
    sch.sub_qubits = [[] for _ in range(r)]
    for q, ops in enumerate(seq):
        if not ops:
            p = part[("idle", q)]
            sid = len(sch.segments)
            sch.segments.append(Segment(q, p, [], False, False))
            sch.sub_qubits[p].append(sid)
            sch.out_seg[q] = sid
            continue
        runs = []
        for k in ops:
            if runs and runs[-1][0] == part[k]:
                runs[-1][1].append(k)
            else:
                runs.append((part[k], [k]))
        prev_sid = None
        for i, (p, run) in enumerate(runs):
            sid = len(sch.segments)
            sch.segments.append(Segment(q, p, run, i > 0, i < len(runs) - 1))
            sch.sub_qubits[p].append(sid)
            if prev_sid is not None:
                sch.cuts.append((prev_sid, sid))
            prev_sid = sid
        sch.out_seg[q] = prev_sid
    return sch


def subcircuits(circ: QuantumCircuit, sch: CutScheme):
    """Bare subcircuits (QuantumCircuit per part); subcircuit qubit i = segment sch.sub_qubits[p][i]."""
    circ = strip(circ)
    seq, qubits = _wire_sequences(circ)
    seg_of_op_qubit = {}
    for sid, s in enumerate(sch.segments):
        for k in s.ops:
            seg_of_op_qubit[(k, s.qubit)] = sid
    local = {}
    for p, sids in enumerate(sch.sub_qubits):
        for i, sid in enumerate(sids):
            local[sid] = i
    subs = [QuantumCircuit(len(sids), name=f"sub{p}") for p, sids in enumerate(sch.sub_qubits)]
    for k, inst in enumerate(circ.data):
        p = sch.parts[k]
        qs = [local[seg_of_op_qubit[(k, q)]] for q in qubits[k]]
        subs[p].append(inst.operation, qs)
    return subs


def scheme_manifest(sch: CutScheme):
    return {
        "n": sch.n, "r": sch.r, "c": sch.c, "widths": sch.widths(),
        "segments": [{"qubit": s.qubit, "part": s.part, "n_ops": len(s.ops), "cut_in": s.starts_with_cut,
                      "cut_out": s.ends_with_cut} for s in sch.segments],
        "sub_qubits": sch.sub_qubits, "cuts": sch.cuts, "out_seg": {str(k): v for k, v in sch.out_seg.items()},
        "parts": sch.parts,
    }
