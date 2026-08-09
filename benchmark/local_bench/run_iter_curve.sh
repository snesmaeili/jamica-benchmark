#!/usr/bin/env bash
# Local-CPU runtime-vs-iterations curve for every implementation.
#
#   bash run_iter_curve.sh [out-tag]
#
# Each point is a complete fit run to one iteration cap and timed end to end. No
# implementation reports per-iteration times, and instrumenting four third-party
# training loops would make every line a different measurement, so this is the
# one construction that is uniform across all of them.
#
# Two design choices exist because the first version of this campaign produced
# an unusable curve, in which 1000 iterations came out faster than 700 for three
# implementations:
#
#   1. The caps are visited OUT OF ORDER. A measurement spread over hours has
#      the clock advancing with the x axis, so any monotonic machine drift
#      becomes apparent curvature. Shuffled, drift adds scatter instead of shape.
#   2. A fixed canary fit runs before each block. Machine drift then gets
#      recorded rather than inferred afterwards from the shape of the answer.
#
# Nothing else may run on the machine while this is going. That is not advice --
# it is what invalidated the first attempt. Expect ~3.5 h on a 6-core laptop.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/env.sh"

TAG="${1:-itercurve_local_cpu}"
ITERS=${ITERS:-"1000 100 700 400"}      # deliberately not ascending; see above
CANARY_LOG="$BENCH_REPO/results/comparator/$TAG/canary.jsonl"
mkdir -p "$(dirname "$CANARY_LOG")"
: > "$CANARY_LOG"

cd "$BENCH_REPO"
for IT in $ITERS; do
    "$PY" "$HERE/canary.py" "before_iter${IT}" "$CANARY_LOG"
    echo "=== max_iter=$IT ==="
    "$PY" benchmark/comparator/implementation_perf.py \
        --dataset "$DATASET" \
        --n-components "$N_COMPONENTS" \
        --max-iter "$IT" \
        --seeds "${SEEDS:-0}" \
        --skip amica_python_numpy \
        --out-tag "$TAG/iter${IT}"
done
"$PY" "$HERE/canary.py" "after_all" "$CANARY_LOG"

echo
echo "=== done. Build the figure with: ==="
echo "  $PY benchmark/comparator/plot_iter_curve.py \\"
echo "      --panel \"Local CPU=results/comparator/$TAG\" \\"
echo "      --canary \"$CANARY_LOG\" \\"
echo "      --out results/figures/fig_iter_curve_local_cpu.pdf"
