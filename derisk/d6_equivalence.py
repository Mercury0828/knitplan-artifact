"""D6 — factor construction and equivalence (the pre-registered protocol §D6).

(a) L1 / L2 on 10,000 random small instances (random circuits, random wire-cut schemes, random integer count tables,
    random queries): every plan vs brute force from the full estimator.
(b) KNITPLAN merged factors vs CutQC's own attribute_shots entries (CutQC commit cda68c7; functions measure_state,
    measure_prob, attribute_shots are executed from their source file; post_process_helper and graph_contraction are
    imported as-is) on every pilot instance that fits in memory; relative L1 <= 1e-12.
(c) full-distribution reconstruction on the n = 20 instances: every KNITPLAN plan vs CutQC GraphContractor
    (+ output reordering), max abs diff vs tau_eq.
tau_eq: the certified tolerance E.tau_eq = roundUp((gamma_D1 + gamma_D2) * B); the Phase-0
value 1e-12 * W (E.tau_eq_legacy) is recorded alongside for comparison. Output: derisk/results/D6.json (Phase-0 run with
the legacy tau) and derisk/results/D6_certified.json (rerun with the certified tau).
"""
_RUNS = __import__("os").environ.get("KNITPLAN_RUNS", "D:/runs/knitplan")   # data root
import ast, itertools, json, math, pickle, sys, time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, _RUNS + "/ext/CutQC")
from qiskit.circuit.random import random_circuit
from knitplan.cutting import find_scheme, strip, subcircuits
from knitplan.shots import configurations, load_tables
from knitplan import engine as E
from cutqc.post_process_helper import generate_compute_graph, generate_subcircuit_entries
from cutqc.graph_contraction import GraphContractor

# --- CutQC evaluator functions, executed from their source (module import pulls in IBMQ helpers we do not need)
_src = Path(_RUNS + "/ext/CutQC/cutqc/evaluator.py").read_text()
_tree = ast.parse(_src)
_ns = {"np": np, "itertools": itertools}
for node in _tree.body:
    if isinstance(node, ast.FunctionDef) and node.name in ("measure_prob", "measure_state", "attribute_shots",
                                                           "mutate_measurement_basis"):
        exec(compile(ast.Module([node], []), "cutqc/evaluator.py", "exec"), _ns)
measure_prob, attribute_shots = _ns["measure_prob"], _ns["attribute_shots"]

OUT = ROOT / "derisk" / "results"; OUT.mkdir(parents=True, exist_ok=True)
REL = 1e-12
OUTF = "D6_certified.json"


# ------------------------------------------------------------------------------------------------- (a) L1 / L2
def brute_answer(models, tabs, S, c, n, A, sel):
    plan = E.Plan("after", tuple(range(len(models))), ("S",) * (len(models) - 1), "joint")
    outs, idx, val = E.execute(plan, models, tabs, S, None, None, c)
    idx, val = E.canonical(outs, idx, val, list(range(n)))
    full = np.zeros(2 ** n); full[idx] = val
    ans = {}
    for x in range(2 ** n):
        if any(((x >> q) & 1) != b for q, b in sel.items()):
            continue
        z = 0
        for k, q in enumerate(A):
            z |= ((x >> q) & 1) << k
        ans[z] = ans.get(z, 0.0) + full[x]
    return ans


def part_a(n_target=10000, seed=11):
    rng = np.random.default_rng(seed)
    done = fails = attempts = 0
    worst_ratio = worst_legacy = max_tau_over_legacy = 0.0
    min_tau_over_legacy = math.inf
    t0 = time.time()
    while done < n_target:
        attempts += 1
        n = int(rng.integers(4, 9)); r = int(rng.integers(2, 4)) if n >= 6 else 2
        circ = strip(random_circuit(n, int(rng.integers(3, 7)), max_operands=2, seed=int(rng.integers(1 << 30))))
        sch = find_scheme(circ, r, max(2, n // r + 2), seed=int(rng.integers(1000)), iters=600, restarts=2)
        if sch is None or sch.c == 0 or sch.c > 4:
            continue
        models = E.sub_models(sch)
        for _ in range(20):
            if done >= n_target:
                break
            tabs = {}
            for p in range(r):
                ins, outs, confs = configurations(sch, p)
                w = len(sch.sub_qubits[p])
                tabs[p] = {}
                for conf in confs:
                    k = int(rng.integers(1, min(12, 2 ** w) + 1))
                    idx = np.sort(rng.choice(2 ** w, size=k, replace=False)).astype(np.int64)
                    tabs[p][conf] = (idx, rng.integers(1, 50, size=k).astype(np.int64))
            S = 100
            A = sorted(rng.choice(n, size=int(rng.integers(0, n + 1)), replace=False).tolist())
            rest = [q for q in range(n) if q not in A]
            selq = rng.choice(rest, size=int(rng.integers(0, len(rest) + 1)), replace=False).tolist() if rest else []
            sel = {int(q): int(rng.integers(2)) for q in selq}
            ref = brute_answer(models, tabs, S, sch.c, n, A, sel)
            tau, B = E.tau_eq(models, tabs, S, set(A), sel, sch.c, D_other_extra=2 ** (n - len(A) - len(sel)))
            tau_l, W = E.tau_eq_legacy(models, tabs, S, set(A), sel, sch.c)
            for plan in E.enumerate_plans(r):
                outs, idx, val = E.execute(plan, models, tabs, S, set(A), sel, sch.c)
                idx, val = E.canonical(outs, idx, val, A) if A else (idx, val)
                got = dict(zip(idx.tolist(), val.tolist()))
                diff = max([abs(got.get(z, 0.0) - v) for z, v in ref.items()] + [abs(v) for z, v in got.items()
                                                                                  if z not in ref] + [0.0])
                worst_ratio = max(worst_ratio, diff / tau if tau > 0 else 0.0)
                worst_legacy = max(worst_legacy, diff / tau_l if tau_l > 0 else 0.0)
                min_tau_over_legacy = min(min_tau_over_legacy, tau / tau_l if tau_l > 0 else math.inf)
                max_tau_over_legacy = max(max_tau_over_legacy, tau / tau_l if tau_l > 0 else 0.0)
                if diff > tau:
                    fails += 1
            done += 1
    return {"instances": done, "circuits_tried": attempts, "plan_checks_failed": fails,
            "worst_diff_over_tau": worst_ratio, "worst_diff_over_legacy_tau": worst_legacy,
            "tau_over_legacy_range": [min_tau_over_legacy, max_tau_over_legacy], "seconds": round(time.time() - t0, 1)}


# ------------------------------------------------------------------------------------------------- (b), (c)
def cutqc_graph(circ, sch):
    subs = subcircuits(circ, sch)
    path = {}
    for q in range(circ.num_qubits):
        segs = [sid for sid, s in enumerate(sch.segments) if s.qubit == q]
        path[circ.qubits[q]] = [{"subcircuit_idx": sch.segments[sid].part,
                                 "subcircuit_qubit": subs[sch.segments[sid].part].qubits[sch.sub_qubits[sch.segments[sid].part].index(sid)]}
                                for sid in segs]
    counter = {p: {"effective": sum(1 for q, sid in sch.out_seg.items() if sch.segments[sid].part == p)}
               for p in range(sch.r)}
    return subs, generate_compute_graph(counter, subs, path)


def cutqc_entry_probs(sch, subs, graph, tables_by_sub, S):
    entries, instances = generate_subcircuit_entries(graph)
    out = {}
    for p in range(sch.r):
        w = subs[p].num_qubits
        ins, outs, _ = configurations(sch, p)
        meas_probs = {}
        for init, meas in instances[p]:
            phys_meas = tuple("comp" if m in ("I", "Z", "comp") else m for m in meas)
            conf = (tuple(init[i] for i in ins), tuple(phys_meas[i] for i in outs))
            idx, cnt = tables_by_sub[p][conf]
            vec = np.zeros(2 ** w); vec[idx] = cnt / S
            meas_probs[(init, meas)] = measure_prob(unmeasured_prob=vec, meas=meas)
        out[p] = attribute_shots(meas_probs, entries[p])
    return out


def basis_of_label(lbl):
    return {"I": 0, "X": 1, "Y": 2, "Z": 3}[lbl]


def compare_factors(sch, subs, models, entry_probs, tables_by_sub, S):
    worst = 0.0
    for m in models:
        f = E.build_factor(m, tables_by_sub[m.p], S, A=None, sel=None, position="after")
        sub = subs[m.p]
        comp_local = sorted(m.out_local)  # CutQC effective bit k = k-th smallest local comp qubit
        perm = [comp_local.index(m.out_local[k]) for k in range(len(m.out_qubits))]  # our bit k -> cutqc bit
        for (einit, emeas), vec in entry_probs[m.p].items():
            key = []
            for k in m.deg_cuts:
                lq = m.cut_local[k]
                key.append(basis_of_label(emeas[lq] if m.cut_is_out[k] else einit[lq]))
            ours = np.zeros_like(vec)
            i, v = f.data[tuple(key)]
            ours[_permute(i, perm)] = v
            denom = np.abs(vec).sum()
            rel = np.abs(ours - vec).sum() / denom if denom > 0 else np.abs(ours).sum()
            worst = max(worst, rel)
    return worst


def _permute(idx, perm):
    out = np.zeros_like(idx)
    for k, t in enumerate(perm):
        out |= ((idx >> k) & 1) << t
    return out


def cutqc_full(sch, subs, graph, entry_probs):
    gc = GraphContractor(compute_graph=graph, subcircuit_entry_probs=entry_probs, num_cuts=sch.c)
    vec = gc.reconstructed_prob
    order = gc.smart_order
    models = E.sub_models(sch)
    # bit layout of kron: first factor in smart_order = most significant
    lens = [len(models[p].out_qubits) for p in order]
    full = np.zeros(2 ** sch.n)
    idx = np.arange(vec.size, dtype=np.int64)
    gidx = np.zeros_like(idx)
    shift = sum(lens)
    for p, L in zip(order, lens):
        shift -= L
        local = (idx >> shift) & ((1 << L) - 1)
        comp_local = sorted(models[p].out_local)
        for k, lq in enumerate(comp_local):
            q = models[p].out_qubits[models[p].out_local.index(lq)]
            gidx |= ((local >> k) & 1) << q
    full[gidx] = vec
    return full


def main():
    global OUTF
    res = {"date": time.strftime("%Y-%m-%d"), "tau": "certified", "part_a": None, "part_b": [], "part_c": []}
    if len(sys.argv) > 1 and sys.argv[1] == "a":
        res = json.loads((OUT / OUTF).read_text())
        res["part_a"] = part_a()
        print("D6a", res["part_a"], flush=True)
        (OUT / OUTF).write_text(json.dumps(res, indent=1))
        return
    man = json.loads((ROOT / "bench" / "PILOT_MANIFEST.json").read_text())
    S = 10000
    for inst in man["instances"]:
        if inst["status"] != "ok":
            continue
        d = pickle.load(open(f"{_RUNS}/pilot/{inst['id']}.pkl", "rb"))
        circ, sch = d["circ"], d["scheme"]
        models = E.sub_models(sch)
        tabs = {p: load_tables(f"{_RUNS}/shots/{inst['id']}/ideal/sub{p}_S{S}.npz")[2] for p in range(sch.r)}
        fits = all(len(configurations(sch, p)[2]) * 2 ** len(sch.sub_qubits[p]) <= 2 ** 28 and
                   4 ** len(m.deg_cuts) * 2 ** len(m.out_qubits) <= 2 ** 27 for p, m in enumerate(models))
        if not fits:
            res["part_b"].append({"id": inst["id"], "status": "SKIPPED (CutQC dense entries do not fit)"}); continue
        t = time.time()
        subs, graph = cutqc_graph(circ, sch)
        ep = cutqc_entry_probs(sch, subs, graph, tabs, S)
        worst = compare_factors(sch, subs, models, ep, tabs, S)
        res["part_b"].append({"id": inst["id"], "S": S, "worst_rel_L1": float(worst), "pass": bool(worst <= REL),
                              "seconds": round(time.time() - t, 1)})
        print("D6b", inst["id"], worst, flush=True)
        if inst["n"] <= 24:
            full = cutqc_full(sch, subs, graph, ep)
            A = list(range(sch.n))
            tau, B = E.tau_eq(models, tabs, S, set(A), None, sch.c)
            tau_l, W = E.tau_eq_legacy(models, tabs, S, set(A), None, sch.c)
            worst_c = 0.0
            for plan in E.enumerate_plans(sch.r):
                try:
                    outs, idx, val = E.execute(plan, models, tabs, S, set(A), None, sch.c)
                except E.MemoryCap as e:
                    continue
                idx, val = E.canonical(outs, idx, val, A)
                ours = np.zeros(2 ** sch.n); ours[idx] = val
                worst_c = max(worst_c, float(np.abs(ours - full).max()))
            res["part_c"].append({"id": inst["id"], "S": S, "plans": len(E.enumerate_plans(sch.r)),
                                  "max_abs_diff": worst_c, "tau_eq": tau, "B": B, "tau_eq_legacy": tau_l, "W": W,
                                  "pass": bool(worst_c <= tau)})
            print("D6c", inst["id"], worst_c, tau, flush=True)
        (OUT / OUTF).write_text(json.dumps(res, indent=1))
    res["part_a"] = part_a()
    print("D6a", res["part_a"], flush=True)
    (OUT / OUTF).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
