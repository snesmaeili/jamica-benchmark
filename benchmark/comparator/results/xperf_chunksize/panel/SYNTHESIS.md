# Synthesis — why is amica's chunk-size curve the flattest? (GPT-5.6 + Grok-4.6, commit 92003b4)

Two independent panelists (blind to each other) reached the **same** verdict: the *mechanism* is real,
the *slogan* ("most robust to the knob / flat curve") is an overstatement of a short, compile-inclusive,
left-truncated sweep.

## Agreed: the mechanism is genuine (H1 in, H4 out)
With `chunk_size=int`, `estep="auto"`, single model, `fit` binds **`_amica_step_fused(block_size=N)`** —
one `@jax.jit` program with `block_size` a **static** arg. Blocking happens *inside* the compiled graph
via `jax.lax.fori_loop` over `dynamic_slice` tiles in `accumulate_stats` (accumulators.py L389–436). The
eager Python-loop path `_amica_step_chunked` is **only** reached with `estep="classic"` — not the sweep.
So **host dispatch count does not grow as the block shrinks**, which is why amica avoids the Python-launch
U-curve *and* why it doesn't blow up at full-batch (1.21 GB fused peak vs pyamica 5.09 GB / pAMICA 3.47 GB).
The knob **is** applied — VRAM moves 0.57→0.68→1.21 GB — so "silently ignored" (H4) is dead. `auto` is two
policies: **GPU** → full-batch when the ~3.34 GB hot-buffer fits the 4 GiB cap (→ identical to `full`);
**CPU** → hard-pinned `_CPU_TARGET_BLOCK=4096`.

## Agreed: the "1.3× flat = robust" claim is over-stated (H2 load-bearing, H3/H5 partial)
1. **Compile-masking (H2) — cannot be ruled out by this data, and is the load-bearing threat.** Each cell
   is a *different compiled program* (block_size is a JIT static arg → 4–6 signatures); fresh-process +
   `block_until_ready` folds a cold XLA compile (or a warm `~/.cache/amica/jax_cache` load — `backend.py`
   L48–62, not cleared) into **every** 20-iter wall clock. A ~1.5–2.5 s compile under a 2.5–3.2 s total
   makes 1.3× "almost automatic." The in-tree `profile_scaling.steady_ms_per_iter` already splits first-iter
   compile from steady ms/iter — **the benchmark did the opposite.** The **20-iter-synthetic (amica slowest
   of the fast set) → 100-iter-real (amica fastest)** rank flip is the *signature* of compile amortization.
2. **Higher-floor (H3) — partial, mostly H2 on the synthetic clock.** amica's 2.5 s floor is 5× pyamica's
   0.5 s, so a fixed overhead is a smaller *fraction* → flatter ratio. Doesn't survive the 100-iter real run
   (amica fastest), so "amica is just slower" is *not* the story.
3. **Left-truncation (H5) — partial.** amica's smallest GPU point is 4096; competitors were pushed to
   512/1024 where they're slow (and amica's own `auto` warns min 8192 on GPU). So part of the "3–13× vs
   1.3×" gap is earned in a regime amica never entered. **But** Grok's counter stands: amica's *right* side
   (full-batch, monotone-fast, 1.21 GB) genuinely doesn't collapse like pyamica/scott — that half is
   architectural, not a sweep hole.

## The defensible finding (what the report must say)
> amica's GPU E-step blocking is an intra-XLA `fori_loop` (`_amica_step_fused`/`accumulate_stats`), so
> **host dispatch count does not grow as the block shrinks**. Over 4096–full on a 20-iter synthetic H100
> run the wall-clock range is 1.3× vs 3–13× for implementations that loop or over-allocate per block, and
> amica does not reproduce pyamica/scott's full-batch collapse (1.21 GB vs 5.09 GB). This is a statement
> about **dispatch structure, not tile-invariant kernels**: the same path is ~3× *slower* at full-batch on
> CPU (so `auto` pins 4096), the 20-iter GPU ratio is not shown to survive compile-exclusion or a 512/1024
> left tail, and on the 100-iter real workload amica is fastest rather than merely flat.

Do **not** write "most robust to the knob" / "flat curve" unqualified. Do **not** say "chunking is free" or
"dispatch count unchanged" without "host".

## Measurement fix (both, converging) — run before finalizing the flatness claim
1. **Steady-state ms/iter:** from `result.iteration_times` drop ≥3 warmup iters → report **median ms/iter**
   + **first-iter ms** as separate columns (both compilers already do this in-tree). Same for competitors.
2. **512 & 1024 GPU points** for amica — locates the knee, makes H5 testable.
3. **100 iters** (match ds004505) as well as 20.
4. **Cache-cold vs cache-warm** controlled (empty `JAX_COMPILATION_CACHE_DIR` vs warm).
5. Label every CPU number **pre-`2cd81e4`** (main reworked the CPU E-step).

Until (1)+(2) exist, treat "1.3× flat" as a measurement-artifact candidate, not a finding.
