# Independent methodology and code critique

**Model/version:** Codex, powered by GPT-5.6.

## Verdict

The flat GPU curve is primarily a genuine consequence of amica's single-model fused implementation over the tested 4,096-to-full range, not a dead or silently clamped knob. An explicit `chunk_size=int` with `estep="auto"` is passed unchanged as a static `block_size` to one outer-jitted `_amica_step_fused`; `accumulate_stats` then performs dynamic slices and a `lax.fori_loop` inside that compiled program. That removes the eager Python/per-block call pattern and makes the observed modest time penalty plausible while the VRAM increase proves that blocking changes allocation. But the benchmark does **not** establish that chunking is free, that steady-state throughput is equally flat, or that amica is robust over the same small-block regime as its competitors: every timing includes first-call compilation, amica's smallest point is 4,096 rather than 512/1,024, and its 2.5 s full-batch floor is five times pyamica's best 0.5 s. The defensible finding is therefore narrow: **amica's fused GPU path was least sensitive among the tested points in this compile-inclusive benchmark.**

## Concrete code-path trace

1. `AmicaConfig` accepts an integer of any size at least one; only the literal `"auto"` invokes a chooser (`amica/config.py:173-191, 211-218`). The comment at `amica/config.py:184-190` is slightly easy to misread: `estep` selects among implementations for full batch, but a chunked single-model non-classic fit is unconditionally fused.
2. In `fit`, `"auto"` calls `_choose_chunk_size`, but an explicit integer takes the direct assignment `_eff_chunk_size = _cfg_cs` (`amica/solver.py:1561-1594`). There is no minimum clamp in that branch.
3. With one model, a non-`None` effective chunk, and `estep="auto"`, the closure at `amica/solver.py:1614-1621` calls `_amica_step_fused(..., block_size=_cs)`. It does **not** call the eager `_amica_step_chunked`; that reference path is selected only by `estep="classic"` (`amica/solver.py:1609-1613`).
4. `_amica_step_fused` is itself `jax.jit`-decorated and declares `block_size` static (`amica/solver.py:599-617`). It passes that value to `accumulate_stats` (`amica/solver.py:683-688`), then performs the natural-gradient, Newton, PDF, center, and scaling updates from the accumulated statistics in the same compiled step (`amica/solver.py:698-792`).
5. `accumulate_stats` uses the unblocked call only when `block_size is None` or is at least the sample count (`amica/accumulators.py:386-392`). Otherwise it slices fixed-width blocks with `jax.lax.dynamic_slice` and accumulates them with `jax.lax.fori_loop` (`amica/accumulators.py:394-436`). `compute_chunk_stats` really creates block-shaped sources, responsibilities/scores, and reductions (`amica/accumulators.py:181-277`). Thus the block is part of the compiled graph and bounds the hot block temporaries; it is not merely metadata.

One qualification matters: “one XLA program” means one Python-to-executable dispatch per AMICA iteration, not one invariant amount of GPU work. The loop trip count is `n_samples // block_size` (`amica/accumulators.py:413-428`), so smaller blocks still execute more loop bodies and may cause more device kernel work/launches depending on XLA lowering. The architecture removes eager per-block dispatch overhead; it does not prove block-size-independent throughput.

## Hypotheses

### 1. Genuine fused scan — **in, with a wording correction**

The core mechanism is real. `_amica_step_fused` is one outer-jitted graph, `block_size` is static, and blocking occurs through `lax.fori_loop` inside `accumulate_stats` (`amica/solver.py:599-617, 683-688`; `amica/accumulators.py:337-360, 413-436`). This is architecturally different from `_amica_step_chunked`, whose Python loop calls jitted `compute_chunk_stats` once per slice (`amica/solver.py:369-370, 469-480`). The 0.57 GB at 4,096/16,384, 0.68 GB at 65,536, and 1.21 GB at full batch are strong external evidence that the block changes the live allocation, even though the 0.57 GB floor hides the difference between the two smallest tested sizes.

The data match the mechanism: going from 4,096 to full changes 20-iteration time only from 3.2 s to 2.5 s while nearly doubling measured VRAM. But “shrinking the block changes tiling, not dispatch count” should be limited to **host/executable dispatch count**. It increases the compiled loop's trip count from roughly 3 blocks at 65,536 to roughly 49 at 4,096 for 200,000 samples, so it can still increase internal GPU work or kernel launches.

### 2. Compile masking — **live/partial; the current experiment cannot rule it out**

`_amica_step_fused` specializes on static `block_size`, so every distinct point requires a distinct compiled executable absent a persistent compilation cache (`amica/solver.py:599-617`). The first `_step_fn` invocation occurs inside the timed fit loop (`amica/solver.py:1659-1728`), and converting `is_good`, likelihood, and Newton status to Python scalars synchronizes each iteration (`amica/solver.py:1745-1748`). The stated external `block_until_ready` also ensures pending device execution completes. Therefore, with a fresh process per point, the wall time includes tracing/compilation plus first execution at every point (subject only to an explicitly enabled persistent JAX compilation cache, which the methodology does not report).

Twenty iterations do not by themselves show that compilation *dominates*, and the table contains no compile-only or warm steady-state measurement from which to estimate its share. Hence compile masking is not established, but it is a material confound: if compile/init contributes an additive second or two, ratios over totals of 2.5-3.2 s will be strongly compressed. Compile cost may also differ between the full graph and the blocked-loop graph, so subtracting one common constant post hoc would be invalid. The code supports a reason for steady-state flatness, but the reported measurement does not demonstrate it.

### 3. Baseline-slower-so-flatter — **partial and unresolved, not an alternative to fusion**

The numerical concern is valid. Amica's best/full point is 2.5 s, versus pyamica's 0.5 s best and pAMICA's 1.1 s full point. Any fixed compile, preprocessing, iteration-update, or synchronization cost compresses a multiplicative chunk-size ratio around that higher floor. The absolute 4,096-to-full difference is still only 0.7 s over the entire 20-iteration fit, so the curve is not flat solely by redefining the ratio; however, the aggregate table cannot say how much of the 2.5 s is block-independent work.

This hypothesis and fused scan can both be true: fusion may make blocking cheap while a larger implementation-specific floor makes the remaining penalty look even smaller fractionally. The real-workload reversal at 100 iterations (amica 5.5 s versus 9.2/12.8/45.7 s) is consistent with short-run startup and workload mix mattering, but it is not proof because the data/workload also changed. Calling the 1.3x ratio an architectural throughput result without separating compile and per-iteration costs would therefore be overclaiming.

### 4. Knob not applied / silently clamped — **out for explicit integer values**

The integer branch passes the requested value through unchanged (`amica/solver.py:1587-1589, 1601-1621`). The only full-batch conversion is in the separate `"auto"` branch when the chosen value is at least `n_samples` (`amica/solver.py:1567-1586`), and `accumulate_stats` uses the exact static block unless it is `None` or at least the recording length (`amica/accumulators.py:386-436`). No floor or rounding clamp exists. The rise from 0.57 to 0.68 to 1.21 GB independently falsifies “ignored knob.” Equal 0.57 GB readings for 4,096 and 16,384 are readily explained by non-block baseline allocations and coarse GB reporting; they do not outweigh the larger-size movement.

`auto` is intentionally different. On this GPU it resolved to full batch, while on single-model CPU it deliberately caps the memory-derived result at `_CPU_TARGET_BLOCK = 4096` when `estep` is non-classic (`amica/solver.py:1190-1205, 1280-1296, 1576-1584`). The CPU auto result of 2.3 s/0.54 GB closely matching explicit 4,096 at 2.4 s/0.55 GB is exactly what that code predicts. This is a hard-coded, measured heuristic at commit 92003b4, not evidence that auto generally optimizes arbitrary machines or current `main`.

### 5. Ordinary competitor U-shapes / amica never reached launch-overhead regime — **partial**

The ordinary explanation is credible for the competitors, and the experiment does not expose amica to their smallest regime. Scott and pyamica improve from 4,096 to 16,384 and then degrade sharply at full (2.9→1.9→5.2 s and 0.8→0.5→3.1 s); pAMICA is especially bad at 512 (13.1 s) but is actually fastest at full (1.1 s), so “all competitor curves are U-shaped because full is too large” is too sweeping. Amica has no 512 or 1,024 result. Its own source explicitly says small GPU blocks cost kernel launches (`amica/solver.py:1236-1239`) and warns below a recommended minimum of at least 8,192 in the auto chooser (`amica/solver.py:1298-1306`). The internal loop count also grows inversely with block size.

Thus the missing 512/1,024 amica points prevent a general robustness conclusion. Nevertheless, “amica is not special at all” is contradicted within the common tested range: at 4,096 versus 16,384, amica changes only 3.2→3.0 s, whereas Scott changes 2.9→1.9, pyamica 0.8→0.5, and pAMICA 2.0→1.3. The fused path demonstrably removes the eager Python-per-block mechanism, even if an unmeasured smaller-block knee almost certainly exists.

## True explanation and robustness framing

For this single-model GPU benchmark, explicit chunk sizes are static tiling parameters of a single compiled AMICA iteration: JAX/XLA loops over block slices internally, `compute_chunk_stats` derives all sufficient statistics from one responsibility pass, and the rest of the update consumes only totals. That makes block changes much cheaper than implementations that repeatedly cross Python/JIT call boundaries, while the growing VRAM confirms real blocking. The remaining 0.7 s spread can reflect extra internal loop work, and its ratio is likely compressed by compilation and other block-independent costs. On CPU, the story is different and more prosaic: `auto` deliberately selects the calibrated 4,096 target, which happened to coincide with the fast/small end of this commit's curve. “Most robust to the chunk-size knob” is defensible only as a descriptive statement about the four implementations and points actually tested; as an architectural or general tuning claim, it is an overstatement.

**Strongest sentence against the robustness framing:** A 1.3x compile-inclusive range around a 2.5-second floor does not establish robust chunking when amica was never tested at 512/1,024 and another implementation finished the same synthetic workload in 0.5 seconds.

## Required measurement fix

Re-run each block size in two explicitly reported phases, using identical data, dtype, update flags, component/mixture counts, and iteration count for all implementations:

1. Report **compile + first execution** separately. For JAX, lower/compile the exact step specialization for each `block_size` (or time its first synchronized call) and record that cost. State whether the persistent compilation cache is disabled.
2. After one synchronized warm-up, reset to identical fresh parameter state and time at least 200 iterations (or enough to make the confidence interval stable), synchronizing once after the loop and also reporting a repeated-run median. Exclude preprocessing, initialization, host-to-device transfer, and compilation from this number. Because state arguments are donated, rebuild/copy the reset state rather than reusing donated buffers.
3. Report steady-state **milliseconds per completed EM iteration** and samples processed per second, plus compile-inclusive end-to-end time as a separate user-facing metric. Ensure convergence/early stopping does not change the iteration count. Apply the same warm-up and scope to competitors; if their “iteration” performs different work, additionally time the common E-step/responsibility-plus-statistics kernel.
4. Extend amica to 2,048, 1,024, and 512. This locates its small-block knee and makes hypothesis 5 testable. Retain fresh processes for independent peak-VRAM measurements, but do not use those fresh-process totals as the sole throughput metric.

## Prioritized bottom line

1. **Phrase the report this way:** “On H100, amica's single-model fused E-step was least sensitive to block size over the tested 4,096-to-full range (2.5-3.2 s, compile-inclusive), while peak VRAM rose from 0.57 to 1.21 GB. Code inspection confirms that the requested block is applied inside one compiled iteration rather than through a Python per-block loop.”
2. **Immediately qualify it:** “These timings include first-call compilation, amica was not tested below 4,096, and it was not the fastest implementation on this synthetic workload; steady-state and smaller-block measurements are needed before claiming general knob robustness.”
3. **Describe CPU separately:** “At commit 92003b4, CPU `auto` deliberately selected the calibrated 4,096-sample fused block and landed near the measured fast/small endpoint.” Do not call the CPU curve flat, and do not generalize this commit-specific heuristic to current `main`.
4. **Do not say:** “Chunking is free,” “dispatch count is unchanged” without the word *host*, or “amica-python is intrinsically the most robust implementation.”
