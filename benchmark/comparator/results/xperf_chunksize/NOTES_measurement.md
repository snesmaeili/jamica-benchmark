# Measurement note — node contention in the atomic CPU sweep

**Status:** known limitation of the real-data CPU curves. The *ordering* and *optima* are
trustworthy; the *absolute* CPU times carry a contention bias (see estimate). A clean rerun with
`--exclusive` is future work (below).

## The design tradeoff that causes it
The real-data sweep fans out **atomic `(impl, chunk)` cells** (one job fits one implementation at
one chunk size, loading a cached PCA projection). This buys **failure isolation** (a hang/OOM/diverge
wastes only that cell — e.g. Fortran diverging at `block=1024` over 100 iters) and **wall-clock
parallelism** (the slow scott@1024 cell no longer blocks amica@1024).

The cost: with 5 subjects × 5 impls × 5 chunks there are ~125 cells eligible to run at once, versus
~25 for a bundled (subject×chunk, impls-sequential) design. ~5× more concurrent jobs → the scheduler
packs more per node → contention. So the noise is **amplified by the atomic split**, though the root
cause (below) is present in any non-exclusive design.

## Root cause: shared DRAM bandwidth, not file I/O
AMICA fits are **memory-bandwidth-bound**: each iteration streams the whole ~400 MB data matrix
(64 × 785k × f64) through the cores several times with low arithmetic intensity (few FLOPs per byte).
Throughput is gated by RAM→CPU bandwidth, not FLOPs. Co-located cells hammer the node's shared memory
controllers and starve each other; L3-cache eviction compounds it (ironic, since `chunk_size` is
about keeping a block *in* cache). It is **not** filesystem contention — the cached `.npz` is read
once at startup (~seconds); the multi-minute fit is pure in-RAM compute. Two other node-sharing
effects can contribute the same "loaded node is slower" symptom and are not separated here: **core
oversubscription** and **turbo/frequency throttling**.

## Noise estimate (from the GPU 25-subject spread)
Per-cell fit time is **heavy-tailed**. Median vs. the least-contended run (min) and the worst outlier
(max), representative cells:

| cell            | median | min  | max  | median vs min | max vs median |
|-----------------|-------:|-----:|-----:|--------------:|--------------:|
| amica @16384    |  6.8s  |  6.0 | 43.0 |        +13%   |      6.3×      |
| amica @full     |  5.2s  |  4.1 | 30.5 |        +27%   |      5.9×      |
| amica @1024     | 29.2s  | 20.8 | 34.7 |        +40%   |      1.2×      |
| scott @16384    | 15.4s  | 13.3 | 85.7 |        +16%   |      5.6×      |
| pAMICA @1024    |139.6s  | 99.2 |170.4 |        +41%   |      1.2×      |

- The reported **median carries ≈ +15–40%** over the cleanest observation.
- **Individual cells spike 2–6×** (right tail = a cell that shared a node).
- The **median is robust** (the tail doesn't move it much at n=25) and the **relative ordering is
  unaffected** — every cell contends equally, so who-beats-whom and each impl's optimum are reliable.
- **CPU is expected to be worse than the GPU numbers above**, because CPU AMICA is *directly*
  DRAM-bandwidth-bound whereas the GPU's compute is not (the GPU tail here is mostly XLA-compile
  under CPU contention at small chunks). To be quantified from the CPU cell spread.

## Fix (future work)
Keep the atomic split (isolation) and add **`--exclusive` per cell** — each cell owns a whole node,
so it gets the full memory bandwidth with no neighbours. This gives clean absolute timings *and*
retains failure isolation. Cost: concurrency becomes node-limited (fewer cells run at once → slower
campaign), which is the correct price for a benchmark. A controlled-concurrency job array (`%N`
limit) is a cheaper middle ground. The GPU side was already close to clean because each GPU cell
owned its GPU (`--gpus-per-node=1`).

## Bottom line for readers of the curves
Trust the **shapes, the ordering, and each implementation's optimum**. Treat **absolute CPU
fit-times as upper bounds** carrying a ~tens-of-percent contention inflation until an `--exclusive`
rerun replaces them.

## Outcome of the throttled rerun (job array %4, 5 subjects)

Throttling helped only **modestly**: the per-cell **median stayed non-monotonic** (e.g. pyamica@4096
= 2345 s next to @16384 = 970 s), because (a) `%4` still co-locates some cells on `bycore` nodes, and
(b) the 5 subjects differ in length, so a cross-subject median mixes data sizes. The **min across
subjects** (best observed ≈ least-contended) *does* recover a clean chunk trend and is what the report
plots. Clean finding it exposes: **CPU optima are at small/mid chunks** (amica 1024, scott/Fortran
~4096, pyamica 16384) — the opposite of the GPU (full-batch), a cache effect. amica is fastest on CPU
too (~155 s best). pyamica@1024 exceeds the 1 h runner timeout; scott full-batch OOMs.

For truly clean CPU *absolutes* (not just the trend) the remaining lever is `--exclusive`/`bynode`
allocation — rejected here as fair-share-hostile (reserves a 192-core node for an ~8-core job that only
uses ~4). The best-of-5 lower bound is the honest compromise.
