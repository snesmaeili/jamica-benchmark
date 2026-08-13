# Rename audit: amica-python → jamica (display-only, chunk-size study)

**Reviewer:** Grok 4.6 (xAI)
**Tree audited:** workspace snapshot of `6fb4b63` (blobs match `HEAD` of the source repo). Net effect taken as `git diff 9dbf9b5..HEAD`.
**Verdict:** **Keep.** The scoped rename is correct. The revert of the repo-wide attempt left no residue. I found no breakage, no look-alike corruption, and no data change beyond the intended `amica`→`jamica` impl label (plus CRLF→LF on the CSV).

This is a correctness audit of the *net* saga (`0cd9e05` bad rename → `0d7a23e` revert → `6fb4b63` scoped display rename), not a style review.

---

## 1. Scope — pass

`git diff --name-status 9dbf9b5..HEAD` is exactly these 7 files, all under `benchmark/comparator/results/xperf_chunksize/`:

| File | Role |
|---|---|
| `README.md` | +10-line naming note |
| `chunk_sweep_data.csv` | 208-line CRLF+label rewrite |
| `gen_report.py` | display-key rename |
| `gen_showcase.py` | display-key rename |
| `xperf_chunk_report.html` | regenerated |
| `xperf_chunk_report_standalone.html` | regenerated |
| `realchunk_gpu_showcase.html` | regenerated |

Nothing outside that directory differs from `9dbf9b5`. In particular:

- `benchmark/comparator/implementation_perf.py` is **byte-identical** to pre-rename. It still registers `amica_python_jax` / `amica_python_jax_chunked` / `amica_python_numpy` against `runners/run_amica_python.py` (lines 447–449). **Zero** `jamica` hits in the orchestrator.
- `benchmark/comparator/runners/run_amica_python.py` still exists and still stamps `amica_python_jax*`. `run_jamica.py` is absent. That filename swap was unique to the reverted `0cd9e05` (`R080 run_amica_python.py → run_jamica.py`); HEAD restored the original path.
- Repo-wide `jamica` at HEAD lives only in those 7 study files. `git grep jamica HEAD -- ':!reviews'` outside `xperf_chunksize/` is empty. Sweeps, `paper_reproduction/`, vendored `amica_python/`, and committed `results/**/*.json` are clean.

The bad commit (`0cd9e05`) *did* leak into 29 paths (orchestrator, `xperf_ds004505/xperf_measures.csv`, `aggregate_pilot.py`, `cc_benchmark/submit_*.sh`, `zenodo_figures/`, etc.). That is gone.

## 2. No look-alike corruption — pass

Targeted greps for `pyjamica`, `jamica17`, `jamica-benchmark`, `scott_jamica`, `jamica_huberty` are empty at HEAD.

Intact on purpose:

- **scott:** still `scott-huberty` / `scott_huberty_torch` / `scott-huberty/amica-python` (imports as `amica`). Runner `run_scott_huberty.py` untouched.
- **pyamica / pamica / pAMICA:** CSV rows identical to `9dbf9b5` after line-ending normalize (25 / 30 rows). Generators still key `"pyamica"` / `"pamica"`.
- **Fortran:** still `amica17` / `fortran_amica17`. Report legend still renders `LABEL.get(..., "Fortran amica17")`. `xperf_measures.csv` still stamps `fortran_amica17`.
- **Vendored package:** `amica_python/` directory and `amica_python/__init__.py` unchanged (`from .solver import Amica, ...`).
- **Repo / cluster paths:** `amica-benchmark`, `/scratch/yorguin/amica-benchmark-repro/`, `/scratch/yorguin/amica_main_src` still spelled that way in the study README. Correct — those are real paths, not display labels.

A naive `amica`→`jamica` replace would have produced at least one of the corruption tokens above. It did not, because the scoped commit only rewrote explicit dict keys and prose in the study generators (and the impl column of the tidy CSV).

## 3. Revert was clean — pass

`git diff --exit-code 9dbf9b5..0d7a23e` is empty. The revert fully undid `0cd9e05`. `6fb4b63` then re-applied *only* the 7-file display rename. So:

```
9dbf9b5  ==  0d7a23e  (pre-rename / post-revert)
0d7a23e..HEAD         ==  the intended scoped rename
9dbf9b5..HEAD         ==  the same 7 files
```

Harness keys, committed measures (`results/rt_*`, `results/comparison/`, `results/cross_recording/`, `xperf_ds004505/xperf_measures.csv`), and the vendored package match the pre-campaign tree.

Committed JSON `implementation` stamps at HEAD are still exactly:

`amica_python_jax`, `amica_python_jax_chunked`, `amica_python_numpy`, `fortran_amica17`, `neuromechanist_numpy`, `pamica_torch`, `pyamica_torch`, `scott_huberty_torch`.

No JSON contains `jamica`.

## 4. Internal consistency of the display rename — pass

In both `gen_report.py` and `gen_showcase.py` the short dict key `"amica"` is now `"jamica"` everywhere it is used as a key.

**`gen_report.py`** — I executed the data bindings and confirmed:

| Binding | Keys include `jamica` | Leftover `"amica"` |
|---|---|---|
| `IMPLS` | yes (first) | no |
| `LABEL` / `KNOB` / `COLOR` / `COMMIT` | yes | no |
| `GPU` / `GPU_BAND` | yes | no |
| `CPU_RSS` / `CPU_FIT` / `CPU_FIT_MED` | yes | no |
| `REAL_OPT` | yes (`("jamica","full-batch",5.2,8.1)`) | no |
| `CPU_IMPLS` | yes (first) | no |

Also rewritten, not left as a landmine:

- highlight test `im=="jamica"` (`optrows`, line 136)
- CSS `.stat.jamica` (line 198) matching HTML `class="stat jamica"` (line 225)
- every prose mention that used to say “amica-python” / “amica” as *this* impl

`LABEL` has no `fortran` entry; that is pre-existing (`LABEL.get(i, "Fortran amica17")`) and still works.

**`gen_showcase.py`:** `IMPLS` / `LABEL` / `KNOB` / `COMMIT` / `COLOR` / `T` / `V` all key on `jamica`. No leftover `"amica"`.

**Generators actually run.** Copied off-tree and executed:

```
gen_report.py   → rc 0, 31037 + 31272 bytes
gen_showcase.py → rc 0, 12521 bytes
```

Regenerated HTML is **byte-identical** to the committed files. HTML parse is well-formed; no leftover `.stat.amica` / `"amica"` key. A missing rename in any of `IMPLS`/`GPU`/`COLOR`/`optrows` would have raised `KeyError` on generate. It does not.

## 5. CSV — pass (label + line-endings only)

`chunk_sweep_data.csv` is 104 data+header lines either side.

| | `9dbf9b5` | `HEAD` |
|---|---|---|
| bytes | 5402 | 5321 |
| line endings | CRLF (104) | LF (104) |
| impl set | `amica`, `pamica`, `pyamica`, `scott` | `jamica`, `pamica`, `pyamica`, `scott` |
| `amica`/`jamica` rows | 23 | 23 |
| leftover `amica` | — | 0 |

After CRLF→LF **and** remapping the impl field `amica`→`jamica`, the files are **byte-identical**. `pyamica` (25 rows), `pamica` (30), `scott` (25) are unchanged. No `pyjamica` / `pjamica` mangling.

The 208-line `git diff --stat` is exactly 104 lines deleted + 104 inserted (every line differs because of CRLF, and 23 of those also change the impl token). Confirmed with `git diff -w --ignore-cr-at-eol`: the only remaining hunks are those 23 impl-column substitutions.

**Stale-vs-report (pre-existing, not this rename):** the tidy CSV is still the *synthetic / early-real* table. `real_each_optimum,jamica,...,5.5,s` (auto chunk) and `pyamica 12.8s @16384` / `scott 45.7s @16384` do not match the hardcoded real-data report (`REAL_OPT`: jamica 5.2 s full-batch, pyamica 9.7 s, scott 10.1 s @65536). Same mismatch exists in the README “Real-workload” paragraph (still “amica 5.5s”). The rename did not invent this; it only relabeled the old `amica` rows.

Separately, and also pre-existing: `gen_showcase.py` `V["jamica"][65536] = 2.19` while `gen_report.py` `GPU["jamica"][65536] = (5.6, 3.68)`. The showcase VRAM series never steps up at 65536. Not introduced by the key rename; flagging only so it is not blamed on this commit.

## 6. Naming claims — mostly accurate; README is only half-updated

The new note in `xperf_chunksize/README.md` (lines 87–95) claims:

> amica-python → amica → jamica (`snesmaeili/jamica`, PyPI `jamica`, import `jamica`). Reports + `chunk_sweep_data.csv` display `jamica`. Harness keys `amica_python_jax*` stay. Vendored `amica_python/` and paper-repro are a repo-wide migration.

Checked against the live package, not just the brief:

- GitHub `snesmaeili/jamica` exists; `snesmaeili/amica` redirects to it. Package dir is `jamica/`.
- PyPI `jamica` 0.2.0 (2026-08-13) installs `from jamica import Amica, AmicaConfig`. Its own README: *“Renamed from `amica`. Releases up to 0.1.0 were `amica`. That name collided with scott-huberty’s `amica-python`.”*
- So the three-step history and the scott caveat are **right**, and they are the actual reason for the `j`.

Caveats in the note are right:

| Caveat | Evidence at HEAD |
|---|---|
| scott is a *different* `amica-python` | README table line 22; `setup_competitors.sh` still `git+https://github.com/scott-huberty/amica-python.git`; runner still `run_scott_huberty.py` |
| Fortran `amica17` | untouched; report still says “Fortran amica17” |
| cluster paths | `/scratch/yorguin/amica-benchmark-repro/`, `amica_main_src` not rewritten |
| ephemeral / committed raw JSONs keep machine keys | see §3 |

**Doc inconsistency (low, not a revert-reason):** the same README still presents the JAX impl as `amica-python` / `snesmaeili/amica` in the versions table (line 21), “amica-python’s *apparent* flat GPU curve” (line 12), and “amica 5.5s” (line 81). Runner script names (`gpu_main_amica.sbatch`) are fine — those are filenames. The versions-table package URL is the one that now disagrees with the note two screens later. Harmless (the old URL redirects) but the study README does not yet *display* jamica end-to-end.

**Out of scope but real future debt:** `run_amica_python.py` lines 48–57 still `from amica import …` with fallback `from amica_python import …`, and the comment still says `pip install amica`. A fresh `pip install jamica` will not satisfy that runner. The scoped rename correctly refused to touch this; a later harness migration needs a third import arm. Do not “fix” it in this commit.

## 7. Bottom line (prioritized)

1. **Keep `6fb4b63`.** Net `9dbf9b5..HEAD` is the intended 7-file display rename. Revert of `0cd9e05` was complete. Harness, vendored `amica_python/`, committed measures/JSONs, sweeps, and look-alike names (`pyamica`, `pamica`, `scott-huberty`, `amica17`, `amica-benchmark`) are untouched.
2. **No correctness bug in the generators.** Every keyed dict/list (`IMPLS`, `LABEL`, `KNOB`, `COLOR`, `COMMIT`, `GPU`, `GPU_BAND`, `CPU_RSS`, `CPU_FIT`, `CPU_FIT_MED`, `REAL_OPT`, `CPU_IMPLS`, plus showcase `T`/`V`) uses `jamica` with no leftover `"amica"`. Both scripts run and reproduce the committed HTML byte-for-byte.
3. **CSV is not a data rewrite.** 23 impl-column substitutions + CRLF→LF. `pyamica`/`pamica`/`scott` rows identical. Numeric payload of the old `amica` rows is intact under the new name.
4. **Non-blocking nits (do not revert over these):**
   - Study README still says `amica-python` / `snesmaeili/amica` in the lead sections; only the new footnote says jamica. Update the versions table if you want the display story to be complete.
   - CSV + README “real-opt” numbers remain stale vs the real-data report (5.5 s / 12.8 s / 45.7 s vs 5.2 / 9.7 / 10.1). Pre-existing.
   - Showcase VRAM at jamica@65536 (2.19 GB) disagrees with the report (3.68 GB). Pre-existing.
   - Runner import surface (`amica` / `amica_python`, not `jamica`) is the next migration, not this one.

**Safe to keep.** No defect in the rename itself.
