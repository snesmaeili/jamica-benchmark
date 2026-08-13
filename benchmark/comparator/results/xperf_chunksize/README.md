# Chunk-size cross-implementation study (ds004505 / synthetic)

Reproducible artifacts for the "chunk size is the hidden variable" investigation: fit-time and
memory as a function of each AMICA implementation's batching knob, on GPU and CPU, at **release**
and **main**, plus a **steady-state (compile-excluded)** re-measurement.

## What this answers
1. Each implementation exposes a batching knob that moves fit time up to ~17× and peak memory up to ~30×; defaults are footguns
   (pAMICA `block_size=512` is the worst; its nearest measured point, 1024, is already ~15× off its GPU
   optimum — 512 itself was not re-measured on the 25-subject sweep, so we don't quote an exact 512 ratio).
2. The optimum **flips by device** (small/mid chunks win on CPU, large/full on GPU — e.g. amica-python
   (scott-huberty) is fastest at a 65K chunk on GPU, a ~4K chunk on CPU, and OOMs at full-batch on GPU).
3. jamica's *apparent* flat GPU curve is **~entirely a JIT-compile artifact** of short (20-iter)
   runs. In steady-state ms/iter the curve is a ~19× monotonic spread like everyone else. What jamica
   actually has is a large fixed compile/setup cost + the fastest steady-state per-iteration, so it
   trails on tiny runs and leads once iterations amortize the compile (the 20-iter-synthetic-loses →
   100-iter-real-wins crossover). See the panel review below.

## Versions (pinned commits)
| impl | package | release | main |
|---|---|---|---|
| jamica (formerly amica-python) | `snesmaeili/jamica` | `92003b4` | `df18b5e` (incl. `2cd81e4` CPU E-step rework) |
| scott-huberty | `scott-huberty/amica-python` (imports as `amica`) | — | `e15e1588` |
| pyamica | `DerAndereJohannes/pyamica` | — | `a8a4d7e0` |
| pAMICA | `sccn/pAMICA` (a.k.a. neuromechanist; imports as `pamica`) | — | `0c4da39e` |

Release↔main for amica differs by 7 commits; only `2cd81e4` ("Make CPU fits faster and much
smaller") is perf-relevant and it touches the **CPU** E-step only — the GPU path is unchanged, so
GPU release≈main. **CPU numbers from a `92003b4` build are pre-`2cd81e4`.**

## Environments
- **GPU:** NVIDIA H100 80GB, SciNet **Trillium** (`def-kjerbi`, `--gpus-per-node=1`). Modules:
  `StdEnv/2023 python/3.11 scipy-stack/2026a cuda/12.6 cudnn`. JAX `XLA_PYTHON_CLIENT_PREALLOCATE=false`.
- **CPU:** 8 cores, Alliance **fir** (`rrg-kjerbi_cpu`, 48G). Modules `StdEnv/2023 python/3.11
  scipy-stack/2026a` (pAMICA needs `python/3.12`).
- **venvs** under `/scratch/yorguin/amica-benchmark-repro/`: `.venv_fir_gpu` (amica, JAX),
  `.venv_competitors[_main]` (scott+pyamica, torch), `.venv_pamica[_main]` (pAMICA, torch py3.12).
  amica *main* is imported via `PYTHONPATH=/scratch/yorguin/amica_main_src:$PYTHONPATH` over
  `.venv_fir_gpu` (append, not replace — replacing drops scipy-stack's numpy).

## Synthetic workload
64 channels, 1 AMICA model, 3 mixtures, `do_sphere=False do_mean=False`, random f64 data.
GPU: 64×200000, 20 iters (30 for steady). CPU: 64×100000, 10 iters. Timed with
`jax.block_until_ready` / `torch.cuda.synchronize`. Memory: JAX `peak_bytes_in_use` (VRAM) /
torch `max_memory_allocated` (VRAM) / `getrusage` `ru_maxrss` (CPU RSS). **Fresh process per point**
for clean peak-memory and cold JAX compile cache (`JAX_COMPILATION_CACHE_DIR` set to an empty dir
per point — otherwise `backend.py` warms `~/.cache/amica/jax_cache`).

## Runners (`sweeps/`)
| script | cluster | produces |
|---|---|---|
| `gpu_release_amica.sbatch` | Trillium | amica release GPU time+VRAM vs chunk |
| `gpu_release_scott_pyamica.sbatch` | Trillium | scott release GPU (pyamica in this one used a bad `n_comps` kwarg — see fixed) |
| `gpu_release_pyamica_fixed.sbatch` | Trillium | pyamica release GPU (`n_components`, corrected) |
| `gpu_release_pamica.sbatch` | Trillium | pAMICA release GPU `block_size` sweep |
| `cpu_release_all4.sbatch` | fir | all 4 release CPU time+RSS vs chunk |
| `cpu_release_amica.sbatch` | fir | amica release CPU (standalone) |
| `gpu_main_scott_pyamica_pamica.sbatch` | Trillium | scott+pyamica+pAMICA **main** GPU |
| `gpu_main_amica.sbatch` | Trillium | amica **main** GPU |
| `cpu_main_all4.sbatch` | fir | scott+pyamica+pAMICA **main** CPU (amica portion needs the append fix) |
| `cpu_main_amica.sbatch` | fir | amica **main** CPU (`2cd81e4` rework) |
| `gpu_steady_amica_release.sbatch` | Trillium | amica release GPU **first-iter vs steady ms/iter**, +512/1024, from `iteration_times` |
| `gpu_steady_all4_main.sbatch` | Trillium | all 4 **steady ms/iter + fixed overhead**, 2-point differencing `(T40−T10)/30` |

Submit: pipe a script to the cluster and `sbatch` it (GPU jobs to Trillium, CPU to fir). Each writes
a labelled table to its `.out`. No arguments; workload is baked in for exact reproduction.

## Data & report
- `chunk_sweep_data.csv` — consolidated tidy dataset (dataset, impl, knob, chunk, time_s, mem_gb, notes).
- `gen_report.py` — renders `report.html` from the numbers (self-contained, theme-aware).

## Panel review (why the curve looked flat)
Independent codex(GPT-5.6)+grok(4.6) methodology panel on amica `92003b4`, in
[`panel/`](panel/) (`REVIEW_REQUEST.md`, `review_gpt-5-6.md`, `review_grok-4-6.md`, `SYNTHESIS.md`).
Run via `agent-utilities/reviews/run_reviewers.sh` against a `git archive` snapshot of the amica repo
at `92003b4`, so each model reviewed the exact code that produced these numbers, blind to the other.
Verdict: fused-scan mechanism is genuine (host dispatch is chunk-invariant) but "flat = robust" is a
compile-masking artifact of the 20-iter window; both prescribed the steady-state re-measurement done
in `gpu_steady_*`.

## Real-workload each-at-optimum (ds004505, 25-subj median, 100 iters, H100)
**Per the current 25-subject main sweep** (`gen_report.py` / `chunk_sweep_data.csv`), each at its best measured chunk: jamica 5.2s (full) · pAMICA 9.2s (full) · pyamica 9.7s (full) · scott-huberty 10.1s (65536, ~1.4 GB). *(This supersedes an earlier release-build opt-campaign that reported 5.5 / 9.2 / 12.8 / 45.7s — different build + different per-impl optima; the sweep above is the authoritative one.)*
Run through the orchestrator (`implementation_perf.py`) with each impl's best chunk set via env
(`AMICA_CHUNK_SIZE`, `AMICA_PAMICA_BLOCK_SIZE`, `AMICA_PYAMICA_CHUNK`, `AMICA_SCOTT_BATCH`). The
env-override for the three competitor runners is upstreamed in `../../runners/` (see git log);
`run_amica_python.py` already honored `AMICA_CHUNK_SIZE`.

## Naming: jamica (formerly amica-python)

Sina's JAX AMICA was renamed **amica-python → amica → jamica** (`snesmaeili/jamica`, PyPI `jamica`,
import `jamica`). This study's **reports and tidy data (`chunk_sweep_data.csv`) display the current
name `jamica`**. The shared benchmark harness (runner impl keys `amica_python_jax*`, the orchestrator,
and the committed legacy measures) is intentionally left on its **run-time keys** — renaming those
would rewrite historical run-time provenance plus the vendored `amica_python/` package and the
paper-reproduction bundle, which is a repo-wide migration best owned upstream. So: reports say
`jamica`; the machine keys still reflect exactly what the runs stamped (`amica_python_jax*`).
