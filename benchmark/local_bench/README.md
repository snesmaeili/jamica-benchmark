# Local-machine benchmarks

The counterpart to `benchmark/cc_benchmark/` (Slurm). Everything here runs on a
workstation or laptop, which is the environment the "usable without a cluster"
claim is actually about.

## Setup

```bash
# 1. An editable install of the package under test
cd ../../../amica-python && python -m venv .venv-dev
.venv-dev/Scripts/pip install -e ".[dev]"

# 2. A competitors environment (pAMICA needs Python >= 3.12)
python3.12 -m venv C:/amica-venvs/comp
C:/amica-venvs/comp/Scripts/pip install pyamica pamica jamica torch psutil scipy

# 3. Point the scripts at both, if your layout differs from the defaults
export AMICA_PYTHON_VENV=/path/to/amica-python/.venv-dev/bin/python
export COMPETITORS_VENV=/path/to/comp/bin/python
export MNE_DATASETS_SAMPLE_PATH=/path/to/mne_data
```

`env.sh` holds the defaults and every one is overridable from the environment,
so nobody has to edit a script to run these.

## Scripts

| script | question it answers | cost |
|---|---|---|
| `run_seed_comparison.sh` | Which implementation is fastest here, and does the gap clear the run-to-run spread? | ~45 min |
| `run_iter_curve.sh` | How does fit time scale with iteration count, per implementation? | ~3.5 h |
| `canary.py` | Did the machine stay as fast as it started? | seconds |
| `analyse_seeds.py` | Median, spread, and whether a ranking is real | instant |

```bash
bash run_seed_comparison.sh                 # 5 seeds, 100 iterations
bash run_iter_curve.sh                      # 4 iteration caps, 1 seed
SEEDS=0,1,2 MAX_ITER=200 bash run_seed_comparison.sh my_tag
```

## Two things that will bite you

**Run nothing else on the machine.** The first iteration-curve campaign was
discarded because other heavy fits ran alongside it; the result had 1000
iterations finishing faster than 700 for three implementations, with every
implementation's per-iteration cost rising and falling together. That shared
pattern is the tell — contention moves all lines at once, a real difference does
not. `run_iter_curve.sh` now times a canary fit before each block so this shows
up in the output instead of having to be reasoned out afterwards.

**Do not compare absolute times across campaigns.** On the reference laptop the
same configuration has produced 66.9 s and 47.0 s in separate clean campaigns.
Orderings and ratios *within* a single back-to-back campaign have been stable
every time. Every script here therefore measures all implementations in one
invocation, and the reported comparisons are within-run.

## Figures

```bash
python ../comparator/plot_iter_curve.py \
    --panel "Local CPU=results/comparator/itercurve_local_cpu" \
    --canary results/comparator/itercurve_local_cpu/canary.jsonl \
    --out results/figures/fig_iter_curve_local_cpu.pdf
```

The same plotter builds the cluster panels; pass several `--panel LABEL=ROOT`
arguments to put local CPU, cluster CPU and cluster GPU side by side.

## Profiling the package itself

These live in the package, not here, because they profile it rather than
compare it:

```bash
python -m jamica.benchmark.profile_cpu        # where an iteration's time goes
python -m jamica.benchmark.profile_memory     # where a fit's peak memory goes
python -m jamica.benchmark.compare_precision  # what float32 costs and buys
```
