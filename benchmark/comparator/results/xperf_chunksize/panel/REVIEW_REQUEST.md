# Methodology + code critique: *why* is amica's chunk-size↔fit-time curve the flattest?

**Task (not a general code review).** Adjudicate a single empirical claim with the source in front of
you. We benchmarked four AMICA implementations by sweeping each one's batching knob and measuring fit
time + peak memory. amica-python's curve is conspicuously **flat on GPU** (and its `auto` lands on the
fast regime on CPU) while every competitor swings 3–13×. **Is that flatness a genuine architectural
property, a measurement artifact, or an unflattering-fact-in-disguise?** Read the actual code paths and
give an adversarial, evidence-cited verdict on the hypotheses below. State your model + version first.

You are reviewing amica at commit **92003b4** — the exact build that produced these numbers (installed
from `snesmaeili/amica.git@92003b4`). Not `main`, not any local feature branch. **Note:** main is only 7
commits ahead, but one of them (`2cd81e4`, "Make CPU fits faster and much smaller") reworks the CPU
E-step in `accumulators.py`/`solver.py`. So treat your **CPU** conclusions as specific to 92003b4 — they
may not transfer to current main. The **GPU** path is unchanged release→main, so GPU conclusions carry.

## The data

**GPU synthetic** — H100, one AMICA model, 3 mixtures, data 64×200000, **20 iterations**, timed with
`jax.block_until_ready`; VRAM = XLA `peak_bytes_in_use`, prealloc off, **fresh process per chunk**:

| chunk/block | amica time/VRAM | scott (batch_size) | pyamica (chunk_t) | pAMICA (block_size) |
|---|---|---|---|---|
| 1024        | —               | 6.3s / 0.1 GB | 3.0s / 0.32 GB | — |
| 512         | —               | —             | —              | 13.1s / 0.14 GB |
| 4096        | 3.2s / 0.57 GB  | 2.9s / 0.2 GB | 0.8s / 0.35 GB | 2.0s / 0.20 GB |
| 16384       | 3.0s / 0.57 GB  | 1.9s / 0.3 GB | 0.5s / 0.72 GB | 1.3s / 0.42 GB |
| 65536       | 2.8s / 0.68 GB  | 1.9s / 0.8 GB | 0.5s / 2.19 GB | 1.4s / 1.32 GB |
| full        | 2.5s / 1.21 GB  | 5.2s / 1.9 GB | 3.1s / 5.09 GB | 1.1s / 3.47 GB |
| **auto**    | **2.5s / 1.21 GB** (→full) | — | — | — |

amica GPU range: **2.5–3.2s (1.3×)**. Competitors: 3–13×. But note amica's *full-batch* 2.5s is
**slower** than pyamica's 0.5s and pAMICA's 1.1s best — amica is the flattest **and** not the fastest
on this synthetic GPU workload.

**CPU synthetic** — 8 cores, 64×100000, **10 iterations**, peak RSS via `getrusage`, fresh process/chunk:

| chunk | amica time/RSS | scott | pyamica | pAMICA |
|---|---|---|---|---|
| 4096  | 2.4s / 0.55 GB | 80.7s / 0.67 GB | 18.5s / 0.79 GB | 21.1s / 0.82 GB |
| 16384 | 3.0s / 0.83 GB | 71.5s / 0.86 GB | 8.8s / 1.67 GB | 11.0s / 1.30 GB |
| 65536 | 3.7s / 2.41 GB | 33.4s / 1.18 GB | 10.1s / 2.30 GB | 12.3s / 1.84 GB |
| full  | 7.1s / 2.32 GB | 27.4s / 1.43 GB | 10.2s / 1.75 GB | 12.5s / 2.35 GB |
| **auto** | **2.3s / 0.54 GB** (→chunked) | — | — | — |

On CPU amica is **not** flat (2.3–7.1s, ~3×) — but `auto` picks the fast small-chunk end. So the honest
claim is: *flat on GPU; on CPU `auto` selects well.*

**Real workload** (ds004505, 64 components, 100 iterations, per-subject median, n=25, H100), each impl at
its own optimum: amica 5.5s · pAMICA 9.2s · pyamica 12.8s · scott 45.7s. (Here amica is fastest — the
synthetic-vs-real flip is itself a clue.)

## Read these (in this commit)

- `amica/config.py` L178–217 — `chunk_size` (`"auto"`/int/None) and `estep` (`"auto"/"fused"/"classic"`);
  note the comment *"Only affects the full-batch path; the chunked path is always fused."*
- `amica/solver.py`:
  - `_amica_step_fused` (L617) — the docstring (L647–667): *"one fused XLA program… costs no extra
    dispatches… `block_size` bounds the E-step `(n_comp, n_mix, block)` temporaries… `block_size=None`
    keeps the original full-batch graph exactly."*  **← the flatness candidate.**
  - `_amica_step_chunked` (L332) — the *eager* Python-loop path (`for start in range(...)`, L473).
  - `_amica_step` (L108) and how the fit loop dispatches among these given `chunk_size`/`estep`.
- `amica/accumulators.py` — `compute_chunk_stats` (the blocked E-step accumulator).
- Trace concretely: with `chunk_size=int` and `estep="auto"` (what the benchmark used), **which** step
  function runs, and how does `block_size`/chunk actually enter the compiled graph?

## Hypotheses to adjudicate (rule each in or out, with code/data evidence)

1. **Genuine (fused scan).** The blocked E-step lives inside one XLA graph, so shrinking the block only
   changes intra-graph tiling, not dispatch count → time is nearly block-invariant while VRAM tracks the
   block. *(VRAM does move 0.57→1.21 GB, proving the block is applied.)*
2. **Compile-masking.** 20 iterations is too short: one-time JIT compile (~seconds) dominates the wall
   clock, hiding real per-iteration throughput differences. Would steady-state ms/iter (compile excluded,
   or ≫20 iters) still be flat? Does `block_until_ready` + fresh-process-per-chunk actually fold compile
   into every point?
3. **Baseline-slower-so-flatter.** amica's per-iteration cost is simply higher on this synthetic GPU
   workload (2.5s full vs pyamica 0.5s), so a fixed per-block overhead is a smaller *fraction* → the curve
   looks flat only relative to a higher floor, not because chunking is free.
4. **Knob-not-applied / silently-clamped.** `chunk_size` is accepted but overridden (e.g. `auto` re-derives
   a block ignoring the request, or a floor clamps small blocks). Verify against the VRAM movement.
5. **Competitor curves are U-shaped for the ordinary reason** (too-small = kernel-launch/Python-loop
   overhead; too-large = memory-traffic), i.e. amica isn't special — it just never entered the
   launch-overhead regime because its smallest tested block (4096) is already large enough.

## Deliverable

- A verdict on **each** hypothesis (in/out/partial) citing specific functions/lines and specific data
  rows.
- The **true one-paragraph explanation** of the flat GPU curve, and whether "amica-python is the most
  robust to the chunk-size knob" is a **defensible** claim or an **overstatement** — give the single
  strongest sentence *against* the robustness framing.
- A concrete **measurement fix** if hypothesis 2 or 3 is live (what to re-run: iteration count, compile
  separation, steady-state ms/iter, an apples-to-apples per-iteration metric).
- Bottom line: how should the report phrase the flatness finding so it is correct and not self-serving?

**Independence:** other panelists review the same commit blind to your output. Your entire value is an
independent judgment from the actual code — do not read other files under `reviews/`.
