#!/usr/bin/env bash
# Cross-implementation comparison on the local machine, repeated over seeds.
#
#   bash run_seed_comparison.sh [out-tag]
#
# Answers "which implementation is fastest here, and is the gap bigger than the
# run-to-run spread". One run per implementation cannot answer the second half,
# and on a laptop the spread is easily large enough to invert a ranking.
#
# Every implementation is measured in the same invocation, back to back. That
# matters more than it sounds: absolute timings on a laptop have repeatedly
# failed to reproduce across separate campaigns here, while orderings and ratios
# within one campaign have been stable every time. Compare within a run; do not
# compare a number from this run against one from last week.
#
# analyse_seeds.py reports the median, the spread, and -- the point of the
# exercise -- whether the fastest implementation's range actually clears the
# runner-up's, rather than assuming a median gap is a ranking.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/env.sh"

TAG="${1:-local_seeds}"
cd "$BENCH_REPO"

"$PY" benchmark/comparator/implementation_perf.py \
    --dataset "$DATASET" \
    --n-components "$N_COMPONENTS" \
    --max-iter "${MAX_ITER:-100}" \
    --seeds "${SEEDS:-0,1,2,3,4}" \
    --skip amica_python_numpy \
    --out-tag "$TAG"

echo
"$PY" "$HERE/analyse_seeds.py" "$BENCH_REPO/results/comparator/$TAG"
