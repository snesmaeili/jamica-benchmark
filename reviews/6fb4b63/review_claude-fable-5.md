# Review: amica-python → jamica rename audit (HEAD = 6fb4b63)

**Reviewer:** Claude Fable 5 (Anthropic, model ID `claude-fable-5`).

**Method note:** the working tree provided for review has no `.git` directory, so I could not run
`git diff 9dbf9b5..HEAD` myself. I audited the *state* of the HEAD tree instead: repo-wide greps
(case-insensitive, filename and content), reading the changed files in full, actually executing both
generators in a scratch directory and byte-comparing their output to the committed HTML, and
cross-checking CSV values against independent in-repo prose. Where a checklist item strictly requires
history (item 3, item 5's before/after), I say exactly what state-based evidence does and does not
establish.

## Verdict up front

**The rename is correct, correctly scoped, and safe to keep. No defect found.** Every checklist item
passes on the evidence available; the two things I could not prove byte-for-byte (revert cleanliness
and CSV value-identity vs the pre-rename blob) are strongly corroborated by independent state evidence.
Remaining observations are cosmetic or pre-existing.

## 1. Scope — PASS

`grep -rIl jamica` over the whole tree (excluding `reviews/`) hits **exactly the 7 files** in the net
stat, all under `benchmark/comparator/results/xperf_chunksize/`:
README.md, chunk_sweep_data.csv, gen_report.py, gen_showcase.py, realchunk_gpu_showcase.html,
xperf_chunk_report.html, xperf_chunk_report_standalone.html. A case-insensitive sweep and a
`find -iname '*jamica*'` both confirm: no `jamica` content or filename anywhere else.

The harness is untouched:
- `benchmark/comparator/implementation_perf.py` contains zero `jamica` and still registers the run-time
  keys — `("amica_python_jax", …)` and `("amica_python_jax_chunked", …)` at
  `implementation_perf.py:447-448`, plus help-text references at lines 270, 298, 300.
- `benchmark/comparator/runners/run_amica_python.py` **exists** (was not renamed) and still stamps
  `impl = "amica_python_jax_chunked"` / `"amica_python_jax"` at `run_amica_python.py:44,46`, zero `jamica`.

## 2. No look-alike corruption — PASS

Greps for every corruption pattern (`pyjamica`, `pjamica`, `jamica17`, `jamica-benchmark`,
`scott*jamica`, `jAMICA`, case-insensitive) return **nothing** repo-wide. The look-alikes are intact:

- **scott**: `scott-huberty` / `scott_huberty_torch` present and unmangled (e.g. top-level `README.md`,
  `scripts/paper/figures/make_tab_cross_implementation.py`, `main_figure_stats.json`). The study README
  even still correctly identifies scott's repo as `scott-huberty/amica-python` (imports as `amica`) —
  the *other* "amica-python" package the brief worries about — untouched.
- **Fortran**: `amica17` intact across `slurm/parity/` (e.g. `submit_fortran17_sub01.sh`,
  `parity_campaign.sbatch`); `gen_report.py:262` still says "Fortran amica17 reference".
- **pyamica / pamica / pAMICA**: intact everywhere, including all CSV rows (see §5).
- **Vendored package**: `amica_python/` directory present at repo root with its modules
  (`accumulators.py`, `backend.py`, `likelihood.py`, …), zero `jamica` inside.
- **Cluster paths**: `amica-benchmark` intact in `pyproject.toml`, `README.md`, `slurm/**/submit_*.sh`.

## 3. Revert cleanliness — PASS on state evidence (history not independently replayable here)

Without `.git` I cannot re-derive `0d7a23e = revert(0cd9e05)`. What the HEAD state proves: every class
of file the mistaken repo-wide rename would have touched is now on the original names — harness keys
(§1), the runner filename (§1), committed measures (`results/xperf_ds004505/xperf_summary.json` is keyed
`amica_python_jax` ×1 / `amica_python_jax_chunked` ×2, including `paired_tests_gpu_vs_amica_chunked` /
`paired_tests_cpu_vs_amica_chunked` top-level keys, and contains zero `jamica`), the vendored package,
and every non-study doc/script. If the revert had left residue, it would show up as a `jamica` string or
a renamed file outside the study dir; there is none. I found no anomaly consistent with revert residue.

## 4. Internal consistency of the display rename — PASS (strongest evidence in the review)

Read both generators in full. In `gen_report.py`, every structure that keys the JAX impl uses
`"jamica"` and only `"jamica"`: `IMPLS` (:13), `LABEL` (:14), `KNOB` (:15), `COMMIT` (:16), `COLOR`
(:17), `GPU` (:21), `GPU_BAND` (:29), `CPU_RSS` (:36), `CPU_FIT` (:46), `CPU_FIT_MED` (:53), `REAL_OPT`
(:63), `CPU_IMPLS` (:125), the highlight-row test `im=="jamica"` in `optrows()` (:136), and the CSS
class `.stat.jamica` (:198) which matches the HTML `class="stat jamica"` usage (:225). Same for
`gen_showcase.py` (`IMPLS`/`LABEL`/`KNOB`/`COMMIT`/`COLOR`/`T`/`V`, lines 6-23). Greps for leftover
`"amica"` / `'amica'` keys in both generators: **zero hits** — no KeyError is possible.

**Executed both generators** (Python 3, scratch copy). Both run cleanly and their three outputs are
**byte-identical** to the committed `xperf_chunk_report.html`, `xperf_chunk_report_standalone.html`,
and `realchunk_gpu_showcase.html`. So the committed HTML is exactly what the renamed source produces —
no hand-edit drift, no stale artifact.

## 5. CSV — PASS (well-formedness proven; value-identity corroborated, not byte-proven)

`chunk_sweep_data.csv` at HEAD: 104 lines, `file` reports plain ASCII CSV, **zero CR characters**
(pure LF). A 208-line diff on a 104-line file = every line rewritten (104 del + 104 add), exactly what
CRLF↔LF normalization of the whole file produces; nothing about the diff size implies a data change.

Impl column: `jamica` 23, `pamica` 30, `pyamica` 25, `scott` 25 rows — no bare `amica` row remains, and
no `pyamica`/`pamica` row was mangled (grep for any other `amica` substring outside those three impl
values: zero hits, including the free-text `note` column).

Without the pre-rename blob I cannot byte-compare values, but there is strong independent corroboration
that only the label changed: the CSV's `real_each_optimum` rows (jamica 5.5 s · pamica 9.2 s ·
pyamica 12.8 s @16384 · scott 45.7 s @16384) **exactly match** the study README's pre-existing prose at
`README.md:81` ("amica 5.5s · pAMICA 9.2s · pyamica 12.8s (16384) · scott 45.7s (16384)") — text that
was not touched by this rename. Fabricated or corrupted values would not line up with untouched prose.

**Pre-existing staleness (not rename-caused), confirmed:** the CSV is the older tidy data — its GPU
rows (e.g. jamica 3.2 s / 0.57 GB @4096, plus `gpu_steady_main` / `gpu_fixed_overhead` datasets from the
synthetic sweep) do not match the report's newer real-data numbers (11.0 s / 2.19 GB @4096; optima
5.2/9.2/9.7/10.1 s). Same for `real_pamica_sensitivity` (275 s @512 vs the report's 139.6 s). The
rename touched only the impl labels; refreshing the tidy data to the real-workload round is a separate,
pre-existing chore.

## 6. Naming claims in the README — ACCURATE (external claims unverifiable offline)

The appended note (`README.md:87-95`) says exactly what the tree shows: reports + tidy CSV display
`jamica`; harness/orchestrator/committed measures stay on `amica_python_jax*`; the vendored
`amica_python/` package and paper-reproduction bundle are untouched. All four caveat classes check out
in-tree: scott's separate `scott-huberty/amica-python` package (still correctly described at
`README.md:22`), Fortran `amica17` (intact in `slurm/parity/`), cluster paths (`amica-benchmark` intact),
and raw run JSONs (the only committed measure, `xperf_ds004505/xperf_summary.json`, keeps run-time keys;
no `jamica` in any results JSON). The external chain (`snesmaeili/jamica`, PyPI `jamica`, import
`jamica`) I cannot verify from this offline tree — flagging as unverified, not as wrong.

## Minor observations (none block keeping the rename)

1. **README body still displays the old names** — `README.md:12` ("amica-python's apparent flat GPU
   curve"), the impl table at `README.md:21` (`| amica-python | snesmaeili/amica |`), and `README.md:81`
   ("amica 5.5s"). The commit only *appended* the naming note (+10 lines), so the README's own earlier
   sections show all three historical names while the note explains the rename. Defensible as historical
   provenance (the table records the repo as it was cloned), and the note resolves the ambiguity — but
   if you want the study dir fully self-consistent, a one-line "(now jamica)" in the table would do it.
2. **Cosmetic indentation residue**: `"jamica":   {` keeps the column alignment that fit the shorter
   `"amica"` key (e.g. `gen_report.py:21,29,36,46`; `gen_showcase.py:13,20`) — harmless.
3. **LABEL dropped the "(JAX)" qualifier** (`"amica-python (JAX)"` → `"jamica"`). Intentional-looking
   and fine, but the report legend no longer says the impl is JAX-based anywhere except the methodology
   prose.
4. Untouched-by-design and correctly so: `NOTES_measurement.md`, `panel/`, and `sweeps/*.sbatch` inside
   the study dir still use `amica`/`amica_python` naming — these are historical measurement artifacts /
   job scripts stamped at run time, consistent with the stated "machine keys unchanged" policy.

## Bottom line (prioritized)

1. **Keep it — no defect.** Scope is exactly the 7 intended files; the harness, runner filename,
   committed measures, vendored package, and every look-alike (`scott*`, `pyamica`, `pamica`,
   `amica17`, `amica-benchmark`) are byte-clean of the rename; both generators are internally
   consistent on the `"jamica"` key and reproduce the committed HTML **byte-identically** when run.
2. **CSV is a label-only change as claimed** (LF-normalized whole-file diff; values corroborated by
   untouched README prose) — but it remains the **stale synthetic-round tidy data**; refreshing it to
   the real-data round is the one real follow-up, and it predates this rename.
3. **Optional polish:** add "(now jamica)" to the README's impl table (`README.md:21`) so the study
   README's own body doesn't display three generations of the name, and note that the external
   `snesmaeili/jamica` / PyPI claims were not verifiable from this offline tree.
