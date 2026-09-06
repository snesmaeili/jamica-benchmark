#!/bin/bash
# Peak host RSS across the six paired ds004505 recordings, full batch vs blocked.
#
# Re-measures the campaign behind the manuscript's memory paragraph, which
# predates the E-step blocking change and now understates it badly: on sub-01
# the same comparison is 11.39 GiB full batch against 2.43 GiB blocked, where
# the archived campaign reported 11.32 against 6.63 and the paragraph quotes a
# median reduction of 54% (range 42-63%).
#
# One recording is not six. The paragraph states a range and a median across
# recordings, so rewriting it from a single re-measured point would be inventing
# the other five.
#
# Memory is iteration-independent -- the peak is set by the array sizes, not the
# loop count -- so 60 iterations suffices and matches what five of the six
# archived runs used. Only the two jamica arms run: the paragraph compares this
# package's full-batch and automatic-blocking paths with each other, not with
# other implementations.
#
#SBATCH --job-name=amica_mem_recheck
#SBATCH --account=def-kjerbi_cpu
#SBATCH --time=01:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --array=1-6
#SBATCH --output=%x-%A_%a.out
#SBATCH --error=%x-%A_%a.err
set -o pipefail

cd "$SLURM_SUBMIT_DIR"
source fir_env.sh || exit 1

# --- which jamica is measured -------------------------------------------------
# Default: the `jamica` release installed in this checkout's .venv_fir by
# fir_env.sh (pinned in pins.toml, [[venv]] fir). Set AMICA_SRC to a source
# checkout to measure that instead: implementation_perf.py puts it on PYTHONPATH
# for OUR runner only. It must not go on PYTHONPATH globally -- scott-huberty's
# package is a different project (imported as `amica`), and a global path would
# shadow whatever shares a module name with the checkout.
REPO_ROOT="${REPO_ROOT:-$(cd "$SLURM_SUBMIT_DIR/../.." && pwd)}"
export AMICA_PYTHON_VENV="${AMICA_PYTHON_VENV:-$REPO_ROOT/.venv_fir/bin/python}"

# The competitor venvs were built by setup_competitors.sh / setup_pamica.sh under
# the older clones on fir; override per site. Left unset, every competitor run
# dies instantly with "venv python missing" and the task still exits 0, so the
# array looks like it succeeded while producing nothing.
export COMPETITORS_VENV="${COMPETITORS_VENV:-/scratch/$USER/jamica/.venv_competitors/bin/python}"
export PAMICA_VENV="${PAMICA_VENV:-/scratch/$USER/jamica-benchmark/.venv_pamica/bin/python}"
for _v in "$AMICA_PYTHON_VENV" "$COMPETITORS_VENV" "$PAMICA_VENV"; do
    [ -x "$_v" ] || { echo "FATAL: no interpreter at $_v" >&2; exit 1; }
done

# installed == intended: assert each competitor venv holds the pinned commit
# from pins.toml before the first fit. Catches silent upstream HEAD drift and a
# clobbered install (e.g. the pyamica/pyAMICA name collision).
"$COMPETITORS_VENV" "$SLURM_SUBMIT_DIR/check_env.py" verify --venv competitors || exit 1
"$PAMICA_VENV"      "$SLURM_SUBMIT_DIR/check_env.py" verify --venv pamica      || exit 1

# The measured jamica: the pinned release imported from the venv, or AMICA_SRC at
# the pinned commit (assert_jamica.sh). Fails fast rather than benchmark the
# wrong code, and prints the identity into the job log.
source "$SLURM_SUBMIT_DIR/assert_jamica.sh" || exit 1
# ---------------------------------------------------------------------------

SUBJECT="$SLURM_ARRAY_TASK_ID"
export AMICA_COMPARATOR_RESULTS="${AMICA_MEM_RESULTS:-${AMICA_RESULTS_DIR:-/scratch/$USER/amica_mem}}/mem_recheck"
mkdir -p "$AMICA_COMPARATOR_RESULTS"

# Walltime: the archived runs were 60 iterations at 64 components on recordings
# of 0.8-1.4 M samples. At the measured per-iteration cost (1.45 s blocked,
# 1.99 s full batch at 785k samples, scaling with sample count) the pair comes
# to roughly 12 minutes plus preprocessing. 90 minutes is generous.
echo "=== memory re-check: ds004505 sub-0${SUBJECT}, 64 components, 60 iterations ==="
python ../comparator/implementation_perf.py     --dataset ds004505     --subject "$SUBJECT"     --input-level "${AMICA_INPUT_LEVEL:-bids}"     --n-components 64     --max-iter 60     --amica-device cpu --competitor-device cpu     --amica-chunk-size auto     --out-tag "mem_recheck/sub-0${SUBJECT}"     --skip amica_python_numpy pyamica_torch scott_huberty_torch pamica_torch

echo "=== DONE sub-0${SUBJECT}. Results under $AMICA_COMPARATOR_RESULTS/ ==="
