# Grok 4.6 (xAI) — methodology critique of amica's flat GPU chunk-size curve

**Model:** Grok 4.6, released by xAI.
**Commit under review:** the tree as provided (claimed 92003b4). CPU conclusions are for this tree only; `2cd81e4` on main reworks the CPU E-step.
**Independence:** I did not read any other file under `reviews/` except `REVIEW_REQUEST.md`. Judgment is from `amica/config.py`, `amica/solver.py`, `amica/accumulators.py`, `amica/backend.py`, `amica/benchmark/profile_scaling.py`, and the supporting tests/docs those paths pull in.

---

## Concrete dispatch trace (what the benchmark actually ran)

With `chunk_size=int` and `estep="auto"` (the sweep), `Amica.fit` does **not** take the eager Python loop.

1. `config.py` L182, L191: `chunk_size` accepts `"auto"` / `int` / `None`; `estep` is documented as affecting **only** the full-batch path.
2. `solver.py` L1565–1590: an explicit int is stored verbatim in `_eff_chunk_size`. No floor, no auto-rederive, no `min(chunk, n_samples)` rewrite.
3. `solver.py` L1601–1621: single-model + `_eff_chunk_size is not None` + `estep != "classic"` binds

   ```python
   _amica_step_fused(*args, block_size=_cs, sample_weight=sample_weight)
   ```

4. `_amica_step_fused` (`solver.py` L599–616, L617–688) is `@jax.jit` with **`block_size` in `static_argnames`**. It calls `accumulate_stats(..., block_size)`.
5. `accumulate_stats` (`accumulators.py` L337–437):
   - `block_size is None or block_size >= n_samples` → one `compute_chunk_stats` on the whole recording (L389–392). This is the `"full"` / GPU-`auto` graph.
   - otherwise a **static** `n_full = n_samples // block_size` and `jax.lax.fori_loop` over `dynamic_slice` blocks of that size, plus a static-sized tail (L413–436). The loop is inside the caller's traced graph on purpose (L357–360): *"one compiled program and one dispatch per iteration, instead of one per block."*

The eager path `_amica_step_chunked` (`solver.py` L332, Python `for start in range(...)` at L473) is **not** on the benchmark's `estep="auto"` path. It is the `estep="classic"` escape hatch (L1609–1613) and the multi-model chunked path. Config's "chunked path is always fused" (L191) means "always the fused *accumulator*", not "always one XLA program."

**`auto` on this synthetic GPU row is full-batch, not a clever tile.**
`_choose_chunk_size` (`solver.py` L1208–1307) sizes hot buffers at `(1 + 2C + 11 C K) * itemsize` bytes/sample. For the GPU table (C=64, K=3, f64) that is 17 928 B/sample × 200 000 ≈ 3.34 GiB. The GPU budget is `min(0.25 * free VRAM, 4 GiB)` (L1258–1262). On an idle H100 that cap is 4 GiB > 3.34 GiB, so `chunk >= n_samples` and `fit` sets `_eff_chunk_size = None` (L1583–1584). That matches the table: **`auto` → 2.5 s / 1.21 GB, identical to `full`.** `prefer_blocking` is passed (L1581) but **explicitly does nothing on GPU** (L1280: `if prefer_blocking and not on_gpu`). The authors even wrote why (L1236–1239): *"never applies on GPU, where the block size is a VRAM question and small blocks would cost kernel launches."* That sentence is an own-goal for the "chunking is free" story.

**`auto` on the CPU table is `_CPU_TARGET_BLOCK = 4096`** (L1206, L1280–1296), which matches 2.3 s / 0.54 GB ≈ the 4096 row (2.4 s / 0.55 GB). On CPU the same fused graph is *not* flat.

---

## Hypothesis verdicts

### 1. Genuine (fused scan) — **IN, with a narrowed claim**

The architecture is real. The GPU sweep is one donated XLA program per iteration; changing the int only changes a **static tile** inside `fori_loop`, not the host dispatch count. That is exactly what the docstring at `solver.py` L660–667 advertises, and `accumulate_stats` L357–360 is written as a deliberate contrast to `_amica_step_chunked`.

VRAM **does** move (0.57 → 0.68 → 1.21 GB), so a different graph / live-range is selected at full-batch vs blocked. Tests lock the semantics: `test_accumulate_stats_unblocked_matches_direct_call` and `test_accumulate_stats_blocked_is_exact` (`tests/test_amica.py` L408–461).

What is **not** supported:

- **"VRAM tracks the block."** 4096 and 16384 are both 0.57 GB. 65536 is only +0.11 GB for a 4× tile. The authors' own 11·C·K estimate predicts ~69 / 277 / 1 107 / 3 379 MB of E-step temps at 4096 / 16384 / 65536 / 200k; the measured device peak does not follow that line. The 0.57 GB floor is data + executable + whatever XLA keeps alive; only the full-batch jump is clearly the unblocked temps. Blocking is applied; it is not a linear memory dial in this range.
- **"Time is nearly block-invariant *because* tiling is free."** Same fused loop on CPU is 2.4 → 7.1 s (~3×) as the block grows (`_CPU_TARGET_BLOCK` comment at L1190–1204 says this is cache, and they measured it). Intra-graph tiling is not time-invariant in general; it is weakly variant **on this GPU, in this C/T/K window, under this clock.**
- **Host dispatch ≠ device kernels.** `fori_loop` is one JAX dispatch. The body can still be a sequence of GPU kernels executed once per trip (48 trips at 4096 vs 1 at full). The authors' own "small blocks would cost kernel launches" comment is about that inner cost, not Python.

So: fused-scan is the right *mechanism* for why amica does not inherit a Python-loop U-curve. It is not a complete explanation of the published 1.3× number.

### 2. Compile-masking — **IN (live). The table cannot rule it out, and their own tooling says it will happen.**

Every cell is a **different compiled program**. `block_size` is a JIT static argument (`solver.py` L614). 4096 / 16384 / 65536 / `None` are four signatures. Fresh process per cell ⇒ first call of `_amica_step_fused` pays XLA compile. `iteration` is traced, so `lax.cond` compiles both Newton branches even though `newt_start=50` and the synthetic run is only 20 iters — the 20-iter clock never *executes* Newton, but it still *compiles* it.

`fit` records `iteration_times` with `time.perf_counter()` around the step and then `float(ll_curr)` / `bool(is_good)` (`solver.py` L1659–1748, L1881). The first entry **includes compile**. The published number is the **whole 20-iter wall clock**, not those per-iter times.

The in-tree scaler already treats this as a known contamination (`amica/benchmark/profile_scaling.py` L19–22, L38–72):

> the first pays JIT compilation, which is a fixed cost and would otherwise be smeared across the per-iteration figure

`steady_ms_per_iter` drops the first ~3 iters and reports `first_iter_ms` separately. The comparison table did the opposite.

`backend.py` L48–62 installs a persistent compile cache at `~/.cache/amica/jax_cache` with `jax_persistent_cache_min_compile_time_secs = 1.0`. Fresh process is **not** a fresh compile if that cache is warm. The brief does not say they cleared it. So each cell is some unknown mix of (cold compile + 20 iters) or (cache load + 20 iters). Either way, 20 iters is the wrong horizon: a 1.5–2.5 s compile sitting under a 2.5–3.2 s total makes a 1.3× ratio almost automatic.

The synthetic-vs-real **rank flip** is the signature of amortization, not of a permanently fat kernel:

| setting | amica | best competitor |
|---|---|---|
| GPU synthetic, 20 iters | 2.5 s (slowest of the fast set; pyamica 0.5 s) | pyamica / pAMICA |
| real ds004505, 100 iters | **5.5 s (fastest)** | pAMICA 9.2 s |

A fixed compile + a competitive (or better) ms/iter produces exactly this: lose the short synthetic, win the long real. Hypothesis 3's "amica is just slower" does not.

`jax.block_until_ready` on a fresh process folds compile into **every** point. That part of H2 is not speculative.

**I did not re-time this tree on an H100.** I am not claiming compile *is* 2 s. I am claiming the published GPU curve is not a steady-state measurement, and until it is, "1.3× flat" is not an architectural number.

### 3. Baseline-slower-so-flatter — **PARTIAL, and mostly a restatement of H2 on the synthetic GPU clock**

On the **synthetic GPU wall clock**, the floor is higher: amica full 2.5 s vs pyamica 0.5 s (5×) and pAMICA 1.1 s (2.3×). A fixed per-block overhead is then a smaller *fraction* of a 2.5 s bar than of a 0.5 s bar. That arithmetic is correct and it makes the *ratio* look flattering.

It is the wrong physical story if most of the 2.5 s is compile (H2). It is also the wrong story for the workload they actually care about: on ds004505 / 100 iters amica is **fastest**, so the "higher floor" does not survive contact with a longer run.

A weaker, still-true residue: even in steady state, a fatter fused body (one pass that materialises score + LL + Newton stats + M-step sums, `solver.py` L649–656) can hide a small tile-overhead *fractionally*. That is ordinary constant-term arithmetic, not a virtue. Do not sell it as robustness.

### 4. Knob-not-applied / silently-clamped — **OUT**

- Explicit ints pass through (`solver.py` L1587–1588) with no `min_chunk` clamp. The 8192-sample "recommended minimum" at L1298 is a **warning on `auto` only**.
- `auto` on this GPU synthetic **does** override — to `None` / full-batch — but that is the documented memory-budget path, and the table's identical 2.5 s / 1.21 GB pair confirms it.
- VRAM 0.57 → 1.21 GB between blocked and full is incompatible with "the int was ignored."
- `accumulate_stats` L389 only collapses to the unblocked graph when `block_size >= n_samples`. 65536 < 200000, so 65536 is a real 3-trip loop + 3392 tail.

The knob is applied. `auto` is not "the same knob" as an explicit int: on GPU it is a VRAM fit-or-full switch, on CPU it is a hard cap at 4096.

### 5. Competitor U-shape is ordinary; amica never entered the launch-overhead regime — **PARTIAL**

Ordinary U-shape is visible and real for **scott** and **pyamica**:

- pyamica: 3.0 s @ 1024 → 0.5 s @ 16k–65k → 3.1 s @ full (and 5.09 GB). Classic launch-overhead left, memory-traffic right.
- scott: 6.3 → 1.9 → 5.2 s, same shape.
- **pAMICA is not U-shaped on the right.** Full is its *best* (1.1 s). Its pain is only the left tail (13.1 s @ 512).

amica's GPU curve is monotone (faster toward full), like pAMICA, unlike pyamica/scott. That right-hand difference is architectural: amica full-batch is 1.21 GB vs pyamica 5.09 GB / pAMICA 3.47 GB, and it is a single fused pass rather than a recording-length temporary storm. **"Didn't test 1024" does not explain why amica does not fall over at full.**

The left-hand part of H5 **is** a sweep hole. amica's smallest GPU point is 4096. At 4096 the competitors are already near their good regime (pyamica 0.8 vs 0.5; scott 2.9 vs 1.9; pAMICA 2.0 vs 1.1). The 3–13× competitor ranges are earned at **1024/512 and at full**, two places amica was either not measured or is actually strong. Their own GPU `min_chunk` heuristic is 8192 (`solver.py` L1298). They published one point below it and called the curve flat.

If they had taken `estep="classic"` with an int chunk, they *would* have been in the launch-overhead regime (`_amica_step_chunked`'s Python loop, one `compute_chunk_stats` dispatch per block). The flatness is a property of the **fused static-`block_size` path**, not of "amica the package" under every legal config.

---

## The actual explanation of the GPU curve

On this commit, a single-model `estep="auto"` fit with an explicit `chunk_size` is `_amica_step_fused(block_size=N)`: one donated XLA program whose E-step is either a full-batch `compute_chunk_stats` or a `fori_loop` of the same accumulator over static tiles. Host launches therefore do not scale with `T / N`, which is why 4096–65536 barely move a 20-iteration H100 clock while the unblocked graph is both faster (2.5 s) and fatter (1.21 GB). That is a real implementation choice, and it is why amica does not reproduce pyamica/scott's full-batch collapse. It is **not** why the published ratio is 1.3×. That ratio is a 20-iter fresh-process wall clock of four differently-specialized JIT signatures, i.e. it is allowed to be (and on the evidence of the 20-vs-100-iter rank flip, likely is) compile-dominated. The same fused loop on CPU is a 3× *anti*-flat curve that `auto` papers over by pinning to 4096. `auto` on this GPU synthetic papers over the other end by selecting full-batch. The honest object is not "a flat curve"; it is "an intra-graph tile whose host-dispatch cost is ~constant, whose device time still prefers large tiles on GPU and small tiles on CPU, and whose published GPU flatness has not been measured in steady state."

**Is "most robust to the chunk-size knob" defensible?** Only as a carefully scoped sentence about *host dispatch count on the fused GPU path over 4096–full*. As a general robustness claim it is an overstatement.

**Strongest sentence against the robustness framing:** The flattest curve is also the only one never tested at 1024/512, the only one whose 20-iter cells each compile a different `static_argnames=["block_size"]` program, and the one whose authors already refuse to prefer small GPU blocks because "small blocks would cost kernel launches."

---

## Measurement fix (H2 and H3 are live)

Do not publish another whole-fit second-count.

1. Use the instrument you already wrote. `python -m amica.benchmark.profile_scaling` / `steady_ms_per_iter`: take `result.iteration_times`, drop ≥3 warmup iters (first = compile), report **median ms/iter** and **first-iter ms** as two columns. Same for competitors (compile/init separated).
2. Re-run the GPU sweep at **100 iterations** (match ds004505) *and* at 20, so the amortization claim is a number, not a vibe. Optionally `lower(...).compile()` before the timed loop for a clean compile-excluded series.
3. Add GPU points at **512 and 1024**. If fused-scan is the story, those stay flat; if inner-kernel trip count matters, they will not. This is the only way to make H5 earn its keep.
4. Control the JAX cache: one sweep with `JAX_COMPILATION_CACHE_DIR` empty / `jax_compilation_cache_dir` unset (cold), one with it warm. Today `backend.py` quietly warms `~/.cache/amica/jax_cache`.
5. Apples-to-apples per-iteration metric: same `max_iter`, exclude preprocessing, `block_until_ready` on a **warmed** step output, device peak from `memory_stats()["peak_bytes_in_use"]` reset per process (keep that part). Report ms/iter × T-normalized ns/sample, not a 20-iter blob.
6. On CPU, label every 92003b4 number as pre-`2cd81e4`. Do not let a GPU-flatness paragraph launder CPU results that main already rewrote.

Until (1)+(3) exist, the 1.3× figure is a measurement artifact candidate, not a finding.

---

## How the report should phrase it

**Do not write:** "amica-python is the most robust to the chunk-size knob; its GPU curve is flat."

**Write:** "amica's GPU E-step blocking is an intra-XLA `fori_loop` (`_amica_step_fused` / `accumulate_stats`), so *host* dispatch count does not grow as the block shrinks. Over 4096–full on a 20-iteration synthetic H100 run the wall-clock range is 1.3×, versus 3–13× for implementations that loop or over-allocate per block; amica also does not reproduce pyamica/scott's full-batch slowdown, consistent with a 1.21 GB fused peak versus pyamica's 5.09 GB. This is a statement about dispatch structure, not about tile-invariant kernels: the same path is ~3× *slower* at full batch on CPU (and `auto` therefore pins to 4096), VRAM still grows when the unblocked graph is selected, the 20-iter GPU ratio has not been shown to survive compile-exclusion or a 512/1024 left tail, and on the 100-iter real set amica is fastest rather than flattest. `auto` is two different policies (GPU: full if the 4 GiB hot-buffer cap fits; CPU: always 4096) that happen to land on the fast end of each device's curve."

---

## Prioritized bottom line

1. **Mechanism is real, slogan is not.** Fused intra-graph blocking is why amica avoids a Python-launch U-curve and why full-batch does not explode. That is worth reporting. "Flattest ⇒ most robust to the knob" is a self-serving reading of a short, compile-contaminated, left-truncated GPU sweep.
2. **H2 is the load-bearing threat.** Re-publish median ms/iter with compile split out, 100-iter and 512/1024 cells, cache-cold and cache-hot. If that curve is still ~1.3×, *then* you have an architectural flatness result. Today's 2.5–3.2 s table is not that result.
3. **H4 is dead; H3 is a synthetic-clock illusion; H5 explains the missing left tail, not the missing right-hand collapse.** Phrase GPU `auto` as "chose full-batch," CPU `auto` as "chose 4096 because small blocks are faster on CPU in this commit," and do not average those into one robustness trophy.
