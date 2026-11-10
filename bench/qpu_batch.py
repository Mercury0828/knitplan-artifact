"""M1 real-device batch on ibm_cleveland.

Selection rule (fixed in derisk/QPU_PLAN.md before submission, from bench/MANIFEST.json only):
  c <= 2, non-control family (bv and ghz are controls, as in Entry 7);
  narrow part: per family the narrow-stratum instance with the largest n <= 40 (ties: fewer subcircuits, then fewer
               configurations);
  wide part:   per family the wide-stratum instance (a subcircuit > 24 qubits) with the largest n (same ties).
Every configuration of every subcircuit of a selected instance runs once with S = 4096 shots.

  plan     print the batch and the QPU-time estimate; writes D:/runs/knitplan/m1/qpu/batch.json (no network)
  submit   re-reads service.usage(), refuses if the estimate x 2 exceeds 600 s or the remaining allocation; transpiles
           every configuration circuit for the live backend (optimization level 1, seed_transpiler 11, layout recorded),
           one SamplerV2 job per instance, job ids in D:/runs/knitplan/m1/qpu/jobs.json
  fetch    downloads finished jobs; counts -> D:/runs/knitplan/m1/shots/<id>/qpu/sub<p>_S4096.npz (same format as
           knitplan.shots.generate_unit: bit i = local subcircuit qubit i); usage per job recorded
"""
_RUNS = __import__("os").environ.get("KNITPLAN_RUNS", "D:/runs/knitplan")   # data root
import json, pickle, sys, time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "bench")]
from knitplan.cutting import subcircuits
from knitplan.shots import configurations, config_circuit, _atomic_save

RUNS = Path(_RUNS + "/m1")
QDIR = RUNS / "qpu"
S_DEV, ENVELOPE_S, CONTROLS = 4096, 600.0, {"bv", "ghz"}
SEC_PER_SHOT = 0.26e-3          # rep_delay 250 us + circuit + measurement (derisk/QPU_PLAN.md)
SEC_PER_JOB = 3.0


def select():
    man = json.loads((ROOT / "bench" / "MANIFEST.json").read_text())
    pool = [i for i in man["instances"] if i["c"] <= 2 and i["family"] not in CONTROLS]
    out = []
    for stratum, ok in (("narrow", lambda i: i["stratum"] == "narrow" and i["n"] <= 40),
                        ("wide", lambda i: i["stratum"] == "wide")):
        for fam in sorted({i["family"] for i in pool if ok(i)}):
            c = [i for i in pool if ok(i) and i["family"] == fam]
            best = max(c, key=lambda i: (i["n"], -i["r"], -sum(i["configs"])))
            out.append({"id": best["id"], "family": fam, "stratum": stratum, "n": best["n"], "c": best["c"],
                        "widths": best["widths"], "configs": best["configs"], "circuits": sum(best["configs"])})
    return out


def estimate(batch):
    circ = sum(b["circuits"] for b in batch)
    return circ, circ * S_DEV * SEC_PER_SHOT + SEC_PER_JOB * len(batch)


def plan():
    batch = select(); circ, est = estimate(batch)
    QDIR.mkdir(parents=True, exist_ok=True)
    (QDIR / "batch.json").write_text(json.dumps({"S": S_DEV, "instances": batch, "circuits": circ,
                                                 "estimate_s": round(est, 1)}, indent=1))
    for b in batch:
        print(b["stratum"], b["id"], "c", b["c"], "widths", b["widths"], "circuits", b["circuits"])
    print(f"{len(batch)} instances, {circ} circuits, {circ * S_DEV} shots, estimate {est:.0f} s (x2 = {2 * est:.0f} s)")


def _circuits(iid):
    d = pickle.load(open(RUNS / "inst" / f"{iid}.pkl", "rb"))
    sch = d["scheme"]; subs = subcircuits(d["circ"], sch)
    out = []
    for p in range(sch.r):
        ins, outs, confs = configurations(sch, p)
        for ci, conf in enumerate(confs):
            qc = config_circuit(subs[p], ins, outs, conf); qc.measure_all()
            out.append((p, ci, qc))
    return sch, out


def submit():
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
    from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
    batch = select(); circ, est = estimate(batch)
    svc = QiskitRuntimeService(); use = svc.usage()
    print("usage:", {k: use[k] for k in ("usage_consumed_seconds", "usage_remaining_seconds")})
    if 2 * est > ENVELOPE_S or 2 * est > use["usage_remaining_seconds"]:
        sys.exit(f"refused: estimate x 2 = {2 * est:.0f} s exceeds the envelope or the remaining allocation")
    be = svc.backend("ibm_cleveland")
    pm = generate_preset_pass_manager(optimization_level=1, backend=be, seed_transpiler=11)
    jf = QDIR / "jobs.json"
    jobs = json.loads(jf.read_text()) if jf.exists() else {}
    for b in batch:
        if b["id"] in jobs:
            continue
        sch, cs = _circuits(b["id"])
        isa = [pm.run(qc) for _, _, qc in cs]
        layouts = [list(t.layout.final_index_layout()) for t in isa]
        job = SamplerV2(mode=be).run(isa, shots=S_DEV)
        jobs[b["id"]] = {"job_id": job.job_id(), "pubs": [[p, ci] for p, ci, _ in cs], "layouts": layouts,
                         "submitted": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        jf.write_text(json.dumps(jobs, indent=1))
        print(b["id"], job.job_id(), len(isa), "pubs", flush=True)


def fetch():
    from qiskit_ibm_runtime import QiskitRuntimeService
    svc = QiskitRuntimeService()
    jf = QDIR / "jobs.json"; jobs = json.loads(jf.read_text())
    for iid, j in jobs.items():
        if j.get("fetched"):
            continue
        job = svc.job(j["job_id"]); st = str(job.status())
        print(iid, st, flush=True)
        if st not in ("DONE", "JobStatus.DONE"):
            continue
        res = job.result()
        sch = pickle.load(open(RUNS / "inst" / f"{iid}.pkl", "rb"))["scheme"]
        per = {}
        for k, (p, ci) in enumerate(j["pubs"]):
            cnt = res[k].data.meas.get_counts()
            keys = np.array([int(b, 2) for b in cnt], dtype=np.int64); vals = np.array(list(cnt.values()), np.int64)
            o = np.argsort(keys); per.setdefault(p, {})[ci] = (keys[o], vals[o])
        out = RUNS / "shots" / iid / "qpu"; out.mkdir(parents=True, exist_ok=True)
        for p in range(sch.r):
            ins, outs, confs = configurations(sch, p)
            arr = {}
            for ci in range(len(confs)):
                arr[f"i{ci}"], arr[f"c{ci}"] = per[p][ci]
            arr["confs"] = np.array(json.dumps([list(map(list, c)) for c in confs]))
            arr["ins"] = np.array(ins, dtype=np.int64); arr["outs"] = np.array(outs, dtype=np.int64)
            _atomic_save(out / f"sub{p}_S{S_DEV}.npz", **arr)
        j["fetched"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        try:
            j["usage"] = job.usage()
        except Exception as e:
            j["usage"] = repr(e)[:120]
        jf.write_text(json.dumps(jobs, indent=1))


if __name__ == "__main__":
    {"plan": plan, "submit": submit, "fetch": fetch}[sys.argv[1]]()
