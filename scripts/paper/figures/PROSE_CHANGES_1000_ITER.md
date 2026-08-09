# Manuscript changes from the 1000-iteration campaigns

Draft for review. Nothing here has been applied to Overleaf.

Source: `results/comparator/cluster/` — six implementations on CPU and four on
GPU, ds004505 sub-01 (64 components x 785,328 samples), at 100/400/700/1000
iterations, one campaign per device on one machine.

---

## 1. `results_final.tex` — fixed-workload runtimes and agreement

**Current (lines ~129-139):**

> On the fixed sub-01 workload, the archived 100-iteration times were 33.8~s
> for chunked JAX-GPU \texttt{amica}, 218.7~s for full-batch JAX-CPU
> \texttt{amica}, 196.3~s for Scott--Huberty \texttt{amica-python}, 670.2~s
> for Fortran AMICA~1.7, and 771.9~s for PyAMICA. [...] individual worst-row
> correlations for the external implementations ranged from 0.939 to 0.976.

**Proposed:**

> On the fixed sub-01 workload at a converged budget of 1000 iterations, fit
> times were 15.5~s for blocked JAX-GPU \texttt{amica}, 1455.0~s for blocked
> JAX-CPU \texttt{amica}, 2130.1~s for full-batch JAX-CPU \texttt{amica},
> 4144.9~s for AMICA-Python (PyTorch), 4947.5~s for \texttt{pAMICA}, 6910.4~s
> for Fortran AMICA~1.7, and 7677.9~s for \texttt{pyamica}
> (Table~\ref{tab:cross-implementation}). Relative to the JAX-CPU result,
> absolute final normalised-likelihood differences were at most
> \(5.7\times10^{-3}\), and worst-row Hungarian-matched unmixing correlations
> ranged from 0.896 to 0.9998.

**Why it changes.** The iteration budget, not the numbers. AMICA requires on the
order of a thousand iterations to converge, so a 100-iteration comparison
measures start-up and compilation alongside the algorithm.

---

## 2. `results_final.tex` — the Fortran agreement figure

This is the substantive change, and it strengthens the paper's central claim.

Worst matched unmixing-row correlation against this implementation:

| implementation | at 100 iterations | at 1000 iterations |
|---|---|---|
| Fortran AMICA 1.7 | 0.9384 | **0.9998** |
| \texttt{pyamica} | 0.9539 | 0.9850 |
| AMICA-Python (PyTorch) | 0.9752 | 0.9816 |
| \texttt{pAMICA} | 0.9524 | **0.8964** |

The manuscript currently reports agreement with the Fortran reference bottoming
out at 0.939. At a converged budget it is **0.9998** — the 0.939 was describing a
transient, since two implementations that will converge to the same decomposition
can still differ appreciably 100 iterations in.

\texttt{pAMICA} moves the other way. That is a real difference in fixed point
rather than measurement noise, and should be reported rather than smoothed:
\texttt{pAMICA} is run with its own algorithm constants, which is the
configuration its authors specify.

**Suggested added sentence:**

> Agreement improves markedly with optimisation budget: against Fortran
> AMICA~1.7 the worst matched unmixing-row correlation rises from 0.938 after
> 100 iterations to 0.9998 after 1000, so agreement measured at short budgets
> reflects the transient rather than the converged decomposition.

---

## 3. `results_final.tex` — memory. **Re-measured; ready to apply.**

**Current (lines ~142-147):**

> Across the six paired recordings, full-batch peak process RSS increased from
> 11.4 to 19.4~GiB over the observed sample-count range, whereas automatic
> chunking remained between 6.6 and 7.2~GiB. The median within-recording
> reduction in total process peak was 54\% (range 42--63\%).

**Re-measured, all six recordings, same protocol (64 components, 60 iterations):**

| recording | samples | full batch | blocked | reduction |
|---|---|---|---|---|
| sub-01 | 785,328 | 11.38 GiB | 2.43 GiB | 79% |
| sub-02 | 1,364,633 | 19.41 GiB | 4.05 GiB | 79% |
| sub-03 | 1,099,796 | 15.74 GiB | 3.28 GiB | 79% |
| sub-04 | 1,057,036 | 15.16 GiB | 3.18 GiB | 79% |
| sub-05 | 1,038,926 | 14.91 GiB | 3.12 GiB | 79% |
| sub-06 | 1,042,909 | 14.95 GiB | 3.15 GiB | 79% |

The full-batch column reproduces the archived 11.4-19.4 GiB, which is what
licenses treating this campaign as a replacement for that one rather than as a
separate measurement.

**Proposed:**

> Across the six paired recordings, full-batch peak process RSS increased from
> 11.4 to 19.4~GiB over the observed sample-count range, whereas the blocked
> expectation step remained between 2.4 and 4.1~GiB
> (Figure~ef{fig:runtime-memory}C). The within-recording reduction in total
> process peak was 79\% at every recording. These measurements support memory
> control over the tested recordings; they are descriptive engineering
> observations rather than a formal complexity estimate.

**Why the reduction is now constant where it used to vary.** The previous
chunking sized its block from a memory budget, so how much it saved depended on
how much memory happened to be free; the reduction ranged 42-63%. Blocking sizes
the block from the measured cost of the E-step's temporaries instead, so those
are bounded the same way on every recording and what remains scales only with the
recording itself. That is why blocked memory now grows with sample count
(2.43 to 4.05 GiB) where the old chunked figure was flat (6.6 to 7.2 GiB) --
the flat part was the budget, not the data.

## 4. `tab_cross_implementation.tex` — regenerated

Already produced by `make_tab_cross_implementation.py`. Three caption passages
are deleted rather than rewritten, because the measurement that forced them is
gone:

- **The split estimator.** "Per-iteration cost is not the same estimator in every
  row" and the accompanying \(^\dagger\)/\(^\ddagger\) footnote. Every row is now
  one least-squares slope through four points.
- **The node-to-node caveat.** "the two runs entering that form were separate
  jobs and, on CPU, were scheduled onto different nodes [...] shifts its
  per-iteration figure by around 10%." Each campaign is now one job per
  implementation on one node.
- **The GPU-ordering claim.** "the JAX implementation is slower on GPU than the
  two PyTorch implementations [...] that ordering reverses once the iteration
  budget is large enough." It does not reverse — this implementation is fastest
  at every budget measured, including 100 (2.7~s against 7.0 and 7.2). The
  original observation came from a GPU campaign whose timings were not
  self-consistent: it recorded 39.9~s at 100 iterations and 14.4~s at 600, a
  negative per-iteration cost.

---

## 5. New figure

`results/figures/fig_iter_curve_grid_log.{pdf,png}` — four panels (local CPU,
local GPU placeholder, cluster CPU, cluster GPU), log-log so a 131x spread stays
legible and each implementation is a straight line whose vertical offset is its
speed ratio.

Suggested caption opening:

> \textbf{Runtime as a function of iteration count.} Each point is a complete
> fit run to that iteration cap and timed end to end. Component count, sample
> count and hardware are identical within a panel, so no implementation is
> advantaged by problem size; they differ between panels, which are therefore
> not directly comparable to one another. Axes are logarithmic: cost is linear
> in iterations, so each implementation is a straight line and the vertical gaps
> are speed ratios.
