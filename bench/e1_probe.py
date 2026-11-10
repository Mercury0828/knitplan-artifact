"""E1 feasibility probe (cost only; measures NO support / timing statistic of the reconstruction): which instances of the
E1 workload exist (cut scheme found, <= 1024 configurations per subcircuit) and what one subcircuit configuration costs to
sample (Aer statevector for width <= 24, Aer matrix_product_state for wider). Run BEFORE the pre-registered protocol Entry 5; its
output only fixes the instance list by the pre-set feasibility rule stated there.
Output: D:/runs/knitplan/e1/probe.jsonl (one line per case), D:/runs/knitplan/e1/inst/<id>.pkl
"""
_RUNS = __import__("os").environ.get("KNITPLAN_RUNS", "D:/runs/knitplan")   # data root
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
import json, math, pickle, sys, time
from pathlib import Path

sys.path.insert(0, _RUNS + "/ext/CutQC"); sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from helper_functions.benchmarks import generate_circ
from qiskit import QuantumCircuit
QuantumCircuit.u3 = lambda self, theta, phi, lam, q: self.u(theta, phi, lam, q)  # qiskit 2 removed u3; U == u3 exactly
from qcg.generators import gen_supremacy
from knitplan.cutting import find_scheme, strip, subcircuits
from knitplan.shots import configurations, config_circuit

OUT = Path(_RUNS + "/e1"); (OUT / "inst").mkdir(parents=True, exist_ok=True)
MAX_CONFIGS = 1024


def grid_ok(m):
    i = math.ceil(math.sqrt(m))
    while m % i:
        i -= 1
    return abs(i - m // i) <= 2


def hetero(n_sparse, n_dense, bridges, seed=1):
    """Heterogeneous composite: adder block (sparse output) on qubits [0, n_sparse) and a supremacy block (dense output)
    on [n_sparse, n_sparse + n_dense), joined by `bridges` CNOTs from the adder's top qubits into the supremacy block,
    placed after both blocks (so some subcircuits are sparse and some dense)."""
    a = strip(generate_circ(num_qubits=n_sparse, depth=1, circuit_type="adder", reg_name="q", connected_only=True, seed=seed))
    s = strip(generate_circ(num_qubits=n_dense, depth=1, circuit_type="supremacy", reg_name="q", connected_only=True, seed=seed))
    qc = QuantumCircuit(n_sparse + n_dense)
    qc.compose(a, qubits=list(range(n_sparse)), inplace=True)
    qc.compose(s, qubits=list(range(n_sparse, n_sparse + n_dense)), inplace=True)
    for b in range(bridges):
        qc.cx(n_sparse - 1 - b, n_sparse + b)
    return qc


def circuit(fam, n):
    if fam.startswith("hetero"):
        nd = max(m for m in range(4, n // 2 + 1) if grid_ok(m))
        return hetero(n - nd, nd, 2)
    if fam == "supremacy_d4":  # shallow supremacy (4 cycles instead of CutQC's 8), the protocol low-entanglement family
        if not grid_ok(n):
            raise ValueError("n not a near-square grid")
        i = math.ceil(math.sqrt(n))
        while n % i:
            i -= 1
        return strip(gen_supremacy(i, n // i, 4, regname="q"))
    return strip(generate_circ(num_qubits=n, depth=1, circuit_type=fam, reg_name="q", connected_only=True, seed=1))


def sample_cost(sub, sch, p, S=10000, method="statevector"):
    from qiskit_aer import AerSimulator
    ins, outs, confs = configurations(sch, p)
    conf = confs[-1]  # last configuration: |+i> preparations and Y measurements where present (most rotated)
    qc = config_circuit(sub, ins, outs, conf); qc.measure_all()
    sim = AerSimulator(method=method, max_parallel_threads=1)
    t = time.time(); r = sim.run(qc, shots=S, seed_simulator=1).result(); dt = time.time() - t
    md = r.results[0].metadata
    return dt, None, {k: md[k] for k in md if "bond" in k.lower() or "mps" in k.lower()}


# probe v2 (after v1 showed generator / cutter failures): grid-friendly sizes for supremacy, shallow supremacy, hwea
# via the u3 shim, the configuration cap inside the cut finder's objective.
FAMILIES = ["regular", "aqft", "adder", "supremacy", "supremacy_d4", "hwea", "bv", "hetero"]
SHAPES = [(48, 2, 28, "wide"), (56, 2, 34, "wide"), (64, 2, 40, "wide"), (60, 3, 24, "wide"), (63, 3, 24, "wide"),
          (30, 4, 10, "r4"), (36, 4, 14, "r4"), (40, 4, 14, "r4"), (42, 4, 14, "r4"), (40, 5, 12, "r5"), (42, 5, 12, "r5")]
CASE_ORDER = [f"{fam}{n}r{r}w{W}" for fam in FAMILIES for n, r, W, _ in SHAPES]


def main():
    fams = sys.argv[1].split(",") if len(sys.argv) > 1 else FAMILIES   # parallel probes split by family
    cases = [(fam, n, r, W, kind) for fam in fams for n, r, W, kind in SHAPES]
    done = set()
    pj = OUT / "probe_v2.jsonl"
    if pj.exists():
        done = {json.loads(l)["id"] for l in pj.read_text().splitlines() if l.strip()}
    for fam, n, r, W, kind in cases:
        iid = f"{fam}{n}r{r}w{W}"
        if pj.exists():  # re-read: parallel probes append to the same file
            done = {json.loads(l)["id"] for l in pj.read_text().splitlines() if l.strip()}
        if iid in done:
            continue
        row = {"id": iid, "family": fam, "n": n, "r": r, "W": W, "kind": kind}
        t = time.time()
        try:
            circ = circuit(fam, n)
            sch = find_scheme(circ, r, W, seed=0, iters=15000, restarts=4, max_configs=MAX_CONFIGS)
        except Exception as e:
            row.update(status="ERROR " + repr(e)[:200]); sch = None
        row["finder_s"] = round(time.time() - t, 1)
        if sch is not None:
            confs = [len(configurations(sch, p)[2]) for p in range(r)]
            row.update(c=sch.c, widths=sch.widths(), configs=confs, two_q_gates=sum(1 for i in circ.data if len(i.qubits) > 1))
            if max(confs) > MAX_CONFIGS:
                row["status"] = f"EXCLUDED configs {max(confs)}"
            else:
                row["status"] = "ok"
                pickle.dump({"circ": circ, "scheme": sch}, open(OUT / "inst" / f"{iid}.pkl", "wb"))
                subs = subcircuits(circ, sch)
                pw = max(range(r), key=lambda p: sch.widths()[p])
                method = "statevector" if sch.widths()[pw] <= 24 else "matrix_product_state"
                try:
                    dt, distinct, md = sample_cost(subs[pw], sch, pw, method=method)
                    row.update(sample_method=method, sample_width=sch.widths()[pw], sample_s_1e4=round(dt, 2),
                               mps_meta=md,
                               est_unit_hours_S1e5=round(sum(confs) * dt * 3 / 3600, 2))
                except Exception as e:
                    row["sample_error"] = repr(e)[:300]
        elif "status" not in row:
            row["status"] = "NO SCHEME"
        with open(pj, "a") as f:
            f.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
