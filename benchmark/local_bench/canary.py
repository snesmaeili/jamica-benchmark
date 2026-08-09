"""A fixed, tiny fit timed between campaign blocks, to detect machine drift.

A runtime-vs-iterations curve measured over hours is only meaningful if the
machine is as fast at the end as at the start. The first attempt at this curve
was contaminated -- other heavy work ran alongside it -- and the tell was that
every implementation's per-iteration cost rose and fell together. Nothing in the
result files recorded that, so it had to be inferred afterwards.

This makes it observable instead: the same fit, same size, same seed, run before
each block. If its time moves, the machine moved, and the block's numbers are
suspect regardless of what they say.
"""

import json
import sys
import time

import numpy as np


def main() -> int:
    tag = sys.argv[1] if len(sys.argv) > 1 else "canary"
    out = sys.argv[2] if len(sys.argv) > 2 else None

    from amica import Amica, AmicaConfig

    rng = np.random.default_rng(12345)
    X = rng.laplace(size=(24, 40000))

    cfg = AmicaConfig(max_iter=25, num_mix_comps=3, do_sphere=False, do_mean=False,
                      fix_init=True)
    Amica(cfg).fit(X[:, :2000])          # absorb compilation, not measured

    t0 = time.perf_counter()
    Amica(cfg).fit(X)
    elapsed = time.perf_counter() - t0

    rec = {"tag": tag, "canary_s": elapsed, "wall": time.time()}
    print(f"[canary] {tag}: {elapsed:.2f} s")
    if out:
        with open(out, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
