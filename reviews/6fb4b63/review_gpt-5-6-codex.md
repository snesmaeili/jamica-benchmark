**Model: GPT-5.6 Codex**

# Audit of the scoped `amica-python` → `jamica` display rename

## Verdict

**Safe to keep. I found no defect caused by `6fb4b63`.** The net change is exactly the intended seven-file, display-only change under `benchmark/comparator/results/xperf_chunksize/`; the mistaken repo-wide rename was completely reverted; the generators and rendered artifacts are internally consistent; and the CSV contains no data mutation beyond the intended implementation label plus line-ending normalization.

The workspace snapshot itself has no `.git` directory. I therefore fetched the exact public commit objects from `snesmaeili/jamica-benchmark`, ran the history comparisons there, and confirmed that the relevant workspace files hash to the blobs at `6fb4b63c761ab0304cfe74b7c786d921c4823530`. I did not use the supplied diff/stat as evidence.

## Findings

No P0/P1/P2 correctness findings.

### 1. Scope and revert integrity: pass

`git diff --name-status 9dbf9b5..6fb4b63` contains exactly these seven modified files:

- `benchmark/comparator/results/xperf_chunksize/README.md`
- `benchmark/comparator/results/xperf_chunksize/chunk_sweep_data.csv`
- `benchmark/comparator/results/xperf_chunksize/gen_report.py`
- `benchmark/comparator/results/xperf_chunksize/gen_showcase.py`
- `benchmark/comparator/results/xperf_chunksize/realchunk_gpu_showcase.html`
- `benchmark/comparator/results/xperf_chunksize/xperf_chunk_report.html`
- `benchmark/comparator/results/xperf_chunksize/xperf_chunk_report_standalone.html`

There is no diff outside that directory. More strongly, `9dbf9b5^{tree}` and `0d7a23e^{tree}` are the same object, `eb38cda58b00d2a4330b28e1cab59739d2852dd9`. Thus the revert restored the entire pre-campaign tree byte-for-byte, not merely the files sampled in the checklist. The net diff also passes `git diff --check`.

The live machinery is deliberately old-named: the orchestrator still registers `amica_python_jax`, `amica_python_jax_chunked`, and `amica_python_numpy` against `run_amica_python.py` (`benchmark/comparator/implementation_perf.py:447-449`), and the runner still emits those keys (`benchmark/comparator/runners/run_amica_python.py:40-46`). Neither file contains `jamica`; `run_amica_python.py` exists; and there is no `run_jamica.py`.

Committed JSONs and legacy measures are necessarily unchanged because the pre-rename tree and revert tree are identical and the final net diff does not include them. Direct tree grep also finds the old keys throughout the committed results, including `results/comparison/*.json`, `results/cross_recording/**`, `results/mem_compare/**`, and `paper_reproduction/results/comparison/**`, with no `jamica` in any JSON.

### 2. No look-alike corruption: pass

A case-insensitive full-tree search excluding `reviews/` finds `jamica` in only the seven intended files. It finds no path named with `jamica` and no instances of `pyjamica`, `jamica17`, `jamica-benchmark`, `scott-huberty/jamica`, or analogous Scott-key corruption.

The distinct names remain intact:

- Scott remains `scott-huberty` in the study metadata (`benchmark/comparator/results/xperf_chunksize/README.md:22`) and `scott_huberty_torch` in the harness/results.
- `pyamica` and `pamica` remain distinct keys and labels (`benchmark/comparator/results/xperf_chunksize/gen_report.py:13-17`, `benchmark/comparator/results/xperf_chunksize/gen_showcase.py:6-10`).
- The CPU-only reference remains Fortran `amica17` (`benchmark/comparator/results/xperf_chunksize/gen_report.py:125-130`; rendered at `benchmark/comparator/results/xperf_chunksize/xperf_chunk_report.html:106-108`).
- The vendored `amica_python/` directory remains present (35 files), and the old `amica-benchmark` cluster paths remain widespread and unchanged. Those are paths/provenance, not display names for Sina's current package.

### 3. Generator consistency and rendered reports: pass

In `gen_report.py`, the renamed implementation is consistently keyed as `jamica` in `IMPLS`, `LABEL`, `KNOB`, `COMMIT`, and `COLOR` (`benchmark/comparator/results/xperf_chunksize/gen_report.py:13-17`); `GPU`, `GPU_BAND`, `CPU_RSS`, `CPU_FIT`, and `CPU_FIT_MED` (`:20-58`); `REAL_OPT` (`:63-64`); `CPU_IMPLS` (`:125`); the highlight predicate (`:133-137`); and the CSS/text (`:198`, `:211-225`). There is no exact `"amica"` key literal left in the file. The same is true of `IMPLS`, `LABEL`, `KNOB`, `COMMIT`, `COLOR`, `T`, and `V` in `gen_showcase.py` (`benchmark/comparator/results/xperf_chunksize/gen_showcase.py:6-21`).

I executed both generators from a detached checkout of `6fb4b63`. They completed without exceptions or `KeyError`; their three outputs were byte-identical to the committed HTML (`git diff --exit-code` returned 0). All three artifacts also parse successfully with Python's `HTMLParser` and pass `xmllint --html --noout`. The rendered labels show `jamica` while preserving `pAMICA`, `pyamica`, `scott-huberty`, and Fortran `amica17` (for example, `realchunk_gpu_showcase.html:33-44` and `xperf_chunk_report.html:56-108`). No rendered artifact contains `amica-python` or a corruption pattern.

### 4. CSV: rename-only data semantics, with pre-existing staleness

The old and new CSVs each contain 103 data records plus the header. The old blob has 104 CRLF endings; the new blob has 104 LF endings. Exactly 23 records differ at the parsed-row level, matching all 23 old `impl=amica` records. After mapping only that column value from `amica` to `jamica`, every field of every row is equal and row order is unchanged. Counts are preserved: Scott 25, pyamica 25, pAMICA 30, and Sina's implementation 23. The pyamica and pAMICA row subsets compare exactly. The intended new rows are visible at `benchmark/comparator/results/xperf_chunksize/chunk_sweep_data.csv:32-39`, `:66-73`, `:89-94`, and `:98`.

The CSV is indeed stale relative to the real-data report, but this predates the rename. For example, its `real_each_optimum` row says 5.5 s (`chunk_sweep_data.csv:98`), while the current report generator says 5.2 s (`gen_report.py:63`); the study README likewise retains the older 5.5/12.8/45.7 summary (`README.md:80-81`). Canonical old-vs-new CSV equality proves `6fb4b63` did not introduce that discrepancy.

### 5. Naming claims and caveats: accurate, with one explicit scope consequence

The naming note accurately describes the display/machine-key split (`benchmark/comparator/results/xperf_chunksize/README.md:87-95`). Current primary metadata corroborates the new identity: the canonical repository is `snesmaeili/jamica`, PyPI publishes `jamica` 0.2.0, the source contains `jamica/__init__.py`, and the upstream changelog states that installation/import changed from `amica` to `jamica`. The older study metadata records Sina's earlier `snesmaeili/amica` identity and separately identifies Scott's independent `scott-huberty/amica-python` package (`README.md:18-24`).

The caveats are correct. Scott's package/import is a separate name collision, Fortran `amica17` is unrelated, cluster strings such as `/scratch/yorguin/amica-benchmark-repro/` are historical filesystem paths (`README.md:30-38`), and generated raw JSON names/keys come from the unchanged runner identifiers. None should be mechanically rewritten by this display patch.

One operational consequence is worth making explicit but is not a regression in this commit: the shared runner still imports `amica` and falls back to `amica_python`, not `jamica` (`benchmark/comparator/runners/run_amica_python.py:48-57`). Therefore this patch does **not** make the shared harness runnable in a fresh `jamica`-only environment. That is consistent with the mandated display-only scope and the exact pre-rename restoration; it belongs in a future deliberate harness migration, not in `6fb4b63`.

## Prioritized bottom line

1. **Keep `6fb4b63` as-is.** The scoped rename is correct, internally complete, and isolated; no rename-induced defect was found.
2. **Separate follow-up, low priority:** refresh the stale real-optimum rows/text (`chunk_sweep_data.csv:98`, `README.md:80-81`) if the tidy dataset is meant to mirror the newer hard-coded report. Do not fold that data update into this rename audit.
3. **Future migration only:** if the benchmark must run against a fresh `pip install jamica`, update imports, runtime keys, result-schema/provenance policy, vendored code, and reproduction assets as one explicit repo-wide migration. The present patch correctly does none of that.
