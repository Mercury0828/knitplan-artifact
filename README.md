# KNITPLAN

Code, workloads and result data for the paper **"KNITPLAN: Support-Aware Query Planning for Circuit-Cutting
Reconstruction"** (Jiachen Shen, Hui Zhong).

KNITPLAN answers marginal, group-by, bin and point queries over the shot tables of a cut quantum circuit. It treats
reconstruction as a sum-product (join-aggregate) query: it pushes aggregation and selection below the configuration
merge, computes the support of each cut elimination from cut-indexed statistics, and chooses join order and per-node
sparse/dense format under a planning-time and memory budget. It returns CutQC's finite-shot estimator within a
tolerance certified from the inputs.

## Layout

| Path | Contents |
|---|---|
| `knitplan/` | the system: cut schemes (`cutting.py`), shot tables (`shots.py`), factor build, operators, executor and certified tolerance (`engine.py`), statistics, cost model and planner (`planner.py`), point batches (`points.py`) |
| `bench/` | workload generation (`*_build.py`, `*_probe.py`, `m1_queries.py`), shot generation (`e1_shots.py`, `gen_shots.py`), the compared systems (`systems.py`, `galley_runner.jl`, `qcut_runner.py`), the end-to-end harness (`m1_time.py`, `m1_equiv.py`), device runs (`qpu_batch.py`, `qpu_support.py`), paper figures (`paper_figs.py`), and the frozen instance manifests (`*MANIFEST.json`) |
| `derisk/` | the planner study (`e1_time.py`, `e2_time.py`), sparsity measurements (`d1_*`, `e1_support.py`), equivalence checks (`d6_equivalence.py`, `t0_*`), Galley comparison (`e1_galley_*`), and their result files under `derisk/results/` |
| `derisk/data/` | the archived `ibm_cleveland` noise model and device properties (2026-11-07) used for noisy shots |
| `tools/test_*.py` | unit and equivalence tests |
| `env/ENVIRONMENT.md`, `requirements.txt` | the exact software environment |

## Setup

Python 3.13 with the packages in `requirements.txt` (`pip install -r requirements.txt`). Galley runs under Julia 1.13
with Galley.jl 0.2.2 and Finch 2.1.1; QCut 2.4.0 runs in a separate Python environment; CutQC is used at commit
`cda68c7`. Exact versions and commits are listed in `env/ENVIRONMENT.md`. Scripts write large intermediate files
(shot tables, query files, timing records) under `D:/runs/knitplan/`; change the `RUNS` / output constants at the top
of each script to use another location.

## Reproducing

1. Tests: `python tools/test_engine.py`, `python tools/test_planner.py`, `python tools/test_join_stat.py`,
   `python tools/test_tau_cert.py`, `python tools/test_systems.py`.
2. Shots: `python bench/e1_shots.py --exp e1` (development workload) and `--exp m1` (main workload).
3. Planner study: `python derisk/e1_time.py run` and `python derisk/e2_time.py run` (timed; run pinned to one core
   on an otherwise idle machine).
4. End-to-end comparison: `python bench/m1_time.py prep`, then `python bench/m1_time.py run [instance ids]`.
5. Figures: `python bench/paper_figs.py`.

The result files used in the paper are in `derisk/results/`.

## License

MIT (see `LICENSE`). Third-party systems (CutQC, QCut, Galley, DuckDB, cotengra) are used through their public
releases and keep their own licenses.
