"""M1-T — main timed evaluation (Tab-main / Tab-scale; the pre-registered protocol). Timed: run only under
E:/Project-git/CPU_LOCK.txt, worker pinned to one P-core (M1_CORE, default 4), every system single-threaded.

  prep  [ids]   untimed: per cell (instance, noise, S) the query set (bench/m1_queries.py; the DD bin sequence uses
                KNITPLAN's answers to pick the zoomed bins) and the certified tau_eq of every query, written to
                D:/runs/knitplan/m1/queries/<cell>.pkl
  run   [ids]   timed: unit = (cell, query, system), in the fixed order of SYSTEM_ORDER. Per unit: one warm-up run
                (excluded from timing), then REPS timed runs; the unit's time is their median (all times kept).
                External systems (Galley, QCut) do their warm-up and REPS runs inside their runner; their time is
                t_stat (median of REPS builds after a warm-up, in this process) + median t_conv + median t_exec.
                A run longer than CAP s or a worker above MEM_CAP bytes is killed -> DNF with the time / memory at stop.
                A system's own memory rule (bench/systems.py: no array > 2^30 entries; sparse operator output > OUT_CAP entries;
                CutQC entry vectors > 48 GiB) -> DNF "infeasible" with the reason, at zero cost.
                Every answer is V-checked against KNITPLAN's answer of the same query within the certified tau_eq
                (KNITPLAN's own answer is checked against every finished baseline the same way).
  smoke <id> <noise> <S> <query>   untimed code check of one cell's query on every system (scratch output)
Peak memory: the parent samples the worker's working set every 50 ms during each unit (sampled peak, reported).
Checkpoint: one JSON per unit in D:/runs/knitplan/m1/T/ (atomic); a rerun skips finished units.
"""
import os
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[v] = "1"
import json, pickle, statistics, sys, time
from pathlib import Path

import numpy as np
import psutil

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "bench")); sys.path.insert(0, str(ROOT / "derisk"))

RUNS = Path("D:/runs/knitplan/m1")
# M1_PILOT=1: untimed sizing pass (not evidence; no lock, E-core): warm-up + 1 run, CAP 600 s, noisy S = 10^4 cells
# only, output D:/runs/knitplan/m1/pilot/ -- used only to size the lock booking and to catch harness errors early
PILOT = os.environ.get("M1_PILOT") == "1"
QDIR, TDIR = RUNS / "queries", RUNS / (os.environ.get("M1_TDIR") or ("pilot" if PILOT else "T"))
CM = ROOT / "derisk" / "results" / "E1" / "cost_model.json"   # calibrated on the E1 development run, not refit
CAP, REPS = (float(os.environ.get("M1_PILOT_CAP", "600")), 1) if PILOT else (3600.0, 5)
MEM_CAP_FULL = 60 * 2 ** 30
# Parallel mode (up to 3 workers on distinct P-cores): each worker is capped at M1_MEMCAP_GIB (default 60); a unit that
# hits a cap below 60 GiB is DEFERRED (not DNF) and rerun later alone with the full cap (`run ... ` with M1_MEMCAP_GIB=60).
MEM_CAP = int(float(os.environ.get("M1_MEMCAP_GIB", "60")) * 2 ** 30)
SYSTEM_ORDER = ["knitplan", "default_rule", "dense_pushdown", "cutqc_dd", "cutqc_opt", "cutqc_fd", "tensorqc_dense",
                "li_sparse", "duckdb_pushdown", "qcut", "galley"]
EXTERNAL = {"galley", "qcut"}


def manifest():
    """The frozen main workload, or with M1_SET=qpu the device cells of derisk/QPU_PLAN.md (noise "qpu", S = 4096;
    Tab-device, outside the bar population)."""
    if os.environ.get("M1_SET") == "qpu":
        b = json.loads((RUNS / "qpu" / "batch.json").read_text())
        return {"instances": [{"id": i["id"], "noise": ["qpu"], "budgets": [b["S"]]} for i in b["instances"]]}
    return json.loads((ROOT / "bench" / "MANIFEST.json").read_text())


def load_cell(iid, noise, S):
    from knitplan import engine as E
    from knitplan.shots import load_tables
    d = pickle.load(open(RUNS / "inst" / f"{iid}.pkl", "rb"))
    sch = d["scheme"]
    tabs = {p: load_tables(RUNS / "shots" / iid / noise / f"sub{p}_S{S}.npz")[2] for p in range(sch.r)}
    return d["circ"], sch, E.sub_models(sch), tabs


def cell_id(iid, noise, S):
    return f"{iid}_{noise}_S{S}"


def _atomic_json(path, obj):
    tmp = path.with_suffix(".tmp"); tmp.write_text(json.dumps(obj, indent=1)); os.replace(tmp, path)


# ------------------------------------------------------------------------------------------------ prep (untimed)
def prep(ids):
    from knitplan import engine as E, planner as P
    import systems as SY, m1_queries as MQ
    QDIR.mkdir(parents=True, exist_ok=True)
    model = P.CostModel(json.loads(CM.read_text())["k"])
    for inst in manifest()["instances"]:
        if ids and inst["id"] not in ids:
            continue
        for noise in inst["noise"]:
            for S in inst["budgets"]:
                cid = cell_id(inst["id"], noise, S); path = QDIR / f"{cid}.pkl"
                if path.exists():
                    continue
                circ, sch, models, tabs = load_cell(inst["id"], noise, S)
                qs = MQ.static_queries(models, tabs, sch.n, MQ.cell_seed("M1", cid))

                def answer(A, sel):
                    return SY.knitplan(SY.make_ctx(None, sch, tabs, S, A, sel, model))
                if sch.n > MQ.DD_MEM:
                    qs += MQ.dd_sequence(models, sch.n, answer)
                for q in qs:
                    Aq = set(range(sch.n)) if q.get("points") is not None else set(q["A"])
                    q["tau"], _ = E.tau_eq(models, tabs, S, Aq, None if q.get("points") is not None else q["sel"], sch.c)
                tmp = path.with_suffix(".tmp"); pickle.dump(qs, open(tmp, "wb")); os.replace(tmp, path)
                print(cid, [q["name"] for q in qs], flush=True)


# ------------------------------------------------------------------------------------------------ timed worker
def worker(conn, iid, noise, S, core):
    try:
        psutil.Process().cpu_affinity([core])
    except Exception:
        pass
    from knitplan import engine as E, planner as P
    import systems as SY
    model = P.CostModel(json.loads(CM.read_text())["k"])
    circ, sch, models, tabs = load_cell(iid, noise, S)
    qs = {q["name"]: q for q in pickle.load(open(QDIR / f"{cell_id(iid, noise, S)}.pkl", "rb"))}
    refs = {}
    conn.send(("ready",))
    while True:
        msg = conn.recv()
        if msg[0] == "stop":
            return
        if msg[0] == "ref":
            refs[msg[1]] = msg[2]; continue
        qn, sysname = msg[1], msg[2]
        q = qs[qn]
        ctx = SY.make_ctx(circ, sch, tabs, S, q["A"], q["sel"], model)
        if q.get("points") is not None:
            ctx["points"] = q["points"]
        fn = SY.SYSTEMS[sysname]
        try:
            if sysname in EXTERNAL:
                ctx["reps"] = REPS; ctx["stat_reps"] = REPS
                conn.send(("run", 0))
                t = time.perf_counter(); ans = fn(ctx); wall = time.perf_counter() - t
                tm = ctx["timing"]
                total = tm["t_stat"] + tm["t_conv"] + tm["t_exec"]
                res = {"time": total, "timing": tm, "call_wall": wall}
            else:
                conn.send(("run", 0))
                t = time.perf_counter(); ans = fn(ctx); warm = time.perf_counter() - t
                ts = []
                for k in range(1, REPS + 1):
                    conn.send(("run", k))
                    t = time.perf_counter(); fn(ctx); ts.append(time.perf_counter() - t)
                res = {"time": statistics.median(ts), "reps": ts, "warmup": warm}
            if sysname == "knitplan":
                refs[qn] = ans; conn.send(("newref", qn, ans))
            ref = refs.get(qn)
            res["answer_entries"] = int(ans[0].size)
            if ref is not None:
                res["vcheck_diff"] = SY.max_diff(ans, ref); res["vcheck_pass"] = res["vcheck_diff"] <= q["tau"]
            res["tau"] = q["tau"]
            conn.send(("done", res))
        except (SY.Infeasible, E.MemoryCap) as e:
            conn.send(("done", {"dnf": f"infeasible: {e}", "dnf_kind": "memory rule"}))
        except MemoryError as e:
            conn.send(("done", {"dnf": f"system out of memory: {str(e)[:150]}", "dnf_kind": "oom"}))
        except Exception as e:     # a crash of one system is recorded for that unit (never silently skipped)
            import traceback
            conn.send(("done", {"dnf": f"error: {e!r}"[:300], "dnf_kind": "error",
                                "traceback": traceback.format_exc()[-2000:]}))


class CellRunner:
    def __init__(self, iid, noise, S, core):
        self.args = (iid, noise, S, core); self.proc = None; self.refs = {}; self.respawns = 0

    def _spawn(self):
        import multiprocessing as mp
        c = mp.get_context("spawn")
        self.conn, child = c.Pipe()
        self.proc = c.Process(target=worker, args=(child, *self.args), daemon=True); self.proc.start()
        assert self.conn.recv()[0] == "ready"
        self.ps = psutil.Process(self.proc.pid)
        for qn, a in self.refs.items():
            self.conn.send(("ref", qn, a))

    def _kill(self):
        try:
            self.proc.kill(); self.proc.join(10)
        except Exception:
            pass
        self.proc = None; self.respawns += 1

    def close(self):
        if self.proc is not None:
            try:
                self.conn.send(("stop",)); self.proc.join(10)
            except Exception:
                pass
            self.proc = None

    def unit(self, qn, sysname):
        if self.proc is None:
            self._spawn()
        self.conn.send(("unit", qn, sysname))
        peak, run_k, t_run = 0, None, time.perf_counter(); t0 = t_run
        while True:
            if self.conn.poll(0.05):
                try:
                    m = self.conn.recv()
                except (EOFError, OSError):      # worker died without a message (native crash)
                    code = self.proc.exitcode if self.proc is not None else None
                    self._kill()
                    res = {"dnf": f"worker died (exit code {code})", "dnf_kind": "error", "at_run": run_k}
                    break
                if m[0] == "run":
                    run_k, t_run = m[1], time.perf_counter(); continue
                if m[0] == "newref":
                    self.refs[m[1]] = m[2]; continue
                res = m[1]; break
            try:
                rss = self.ps.memory_info().rss + sum(c.memory_info().rss for c in self.ps.children(recursive=True)); peak = max(peak, rss)  # incl. runner subprocesses (QCut, Galley)
            except psutil.Error:
                rss = 0
            el = time.perf_counter() - t_run
            if rss > MEM_CAP or el > (CAP if sysname not in EXTERNAL else CAP * (REPS + 2)):
                self._kill()
                res = {"dnf": (f"memory > {MEM_CAP / 2**30:.0f} GiB" if rss > MEM_CAP else f"run > {CAP:.0f} s"),
                       "dnf_kind": "memory" if rss > MEM_CAP else "time", "at_run": run_k,
                       "time_at_stop": el, "rss_at_stop": rss}
                break
        res["peak_rss_sampled"] = peak; res["unit_wall"] = time.perf_counter() - t0
        return res


def lock_holder():
    try:
        return Path("E:/Project-git/CPU_LOCK.txt").read_text(encoding="utf-8-sig").splitlines()[0][:120]
    except Exception as e:
        return f"unreadable: {e}"


def run(ids, core):
    TDIR.mkdir(parents=True, exist_ok=True)
    for inst in manifest()["instances"]:
        if ids and inst["id"] not in ids:
            continue
        for noise in inst["noise"]:
            for S in inst["budgets"]:
                if PILOT and not (noise in ("noisy", "qpu") and S in (10000, 4096)):
                    continue
                cid = cell_id(inst["id"], noise, S)
                qs = [q["name"] for q in pickle.load(open(QDIR / f"{cid}.pkl", "rb"))]
                sysset = os.environ.get("M1_SYSTEMS", "").split(",") if PILOT and os.environ.get("M1_SYSTEMS") else SYSTEM_ORDER
                todo = [(qn, s) for qn in qs for s in SYSTEM_ORDER if s in sysset and not (TDIR / f"{cid}_{qn}_{s}.json").exists()]
                if not todo:
                    continue
                cr = CellRunner(inst["id"], noise, S, core)
                for qn in qs:     # references of finished units (respawned workers need them)
                    p = TDIR / f"{cid}_{qn}_ref.pkl"
                    if p.exists():
                        cr.refs[qn] = pickle.load(open(p, "rb"))
                for qn, s in todo:
                    h0, l0 = lock_holder(), psutil.cpu_percent(interval=None)
                    res = cr.unit(qn, s)
                    if res.get("dnf_kind") == "memory" and MEM_CAP < MEM_CAP_FULL:
                        _atomic_json(TDIR / f"{cid}_{qn}_{s}.deferred", res)
                        print(cid, qn, s, "DEFERRED (memory cap of parallel mode)", flush=True)
                        continue
                    res.update({"cell": cid, "query": qn, "system": s, "lock_start": h0, "lock_end": lock_holder(),
                                "core": core, "finished": time.strftime("%Y-%m-%dT%H:%M:%S")})
                    if qn in cr.refs and not (TDIR / f"{cid}_{qn}_ref.pkl").exists():
                        pickle.dump(cr.refs[qn], open(TDIR / f"{cid}_{qn}_ref.pkl", "wb"))
                    _atomic_json(TDIR / f"{cid}_{qn}_{s}.json", res)
                    print(cid, qn, s, res.get("dnf") or f"{res['time']:.4g}s v={res.get('vcheck_pass')}", flush=True)
                cr.close()


def main():
    cmd = sys.argv[1]
    if cmd == "prep":
        prep(set(sys.argv[2].split(",")) if len(sys.argv) > 2 else set())
    elif cmd == "run":
        core = int(os.environ.get("M1_CORE", "4"))
        if PILOT:   # light process rules: BelowNormal, inherited by the spawned workers
            psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
        try:
            psutil.Process().cpu_affinity([core])
        except Exception:
            pass
        run(set(sys.argv[2].split(",")) if len(sys.argv) > 2 else set(), core)


if __name__ == "__main__":
    main()
