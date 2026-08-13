# Verify the amica-python → jamica rename: correct, scoped, and nothing broken

**Task (code-review / correctness audit, not a general review).** We renamed Sina's JAX AMICA impl
from *amica-python* to **jamica** in this benchmark repo. It must be **display-only and scoped to the
chunk-size study** — reports + that study's tidy data — and must **not** touch the shared harness, the
raw/committed result data, or accidentally rename look-alikes. Your job: confirm it is correct and
broke nothing, citing the actual files at HEAD. State your model + version first.

## Background (the saga you are auditing the NET result of)
1. `0cd9e05` — a mistaken **repo-wide** rename (harness keys, committed measures, everything). Reverted.
2. `0d7a23e` — `git revert` of `0cd9e05` (should have fully restored the pre-rename state).
3. `6fb4b63` (**HEAD**) — the intended **scoped, display-only** rename.

So the **net effect** of the whole saga is `git diff 9dbf9b5..HEAD` (9dbf9b5 = the commit *before* the
rename campaign). It should be **only** the 7 files under `benchmark/comparator/results/xperf_chunksize/`.

## What "correct" means here
- **Display = jamica**: the report/showcase *generators* (`gen_report.py`, `gen_showcase.py`), their
  rendered HTML, and the study's tidy `chunk_sweep_data.csv` show `jamica`.
- **Machinery + raw data = unchanged** (`amica_python_jax*`): the orchestrator
  `benchmark/comparator/implementation_perf.py`, the runner `benchmark/comparator/runners/run_amica_python.py`,
  all committed result JSONs / legacy measures, and the vendored `amica_python/` package — these keep the
  original run-time keys **on purpose**.

## Verification checklist (please actually run these against the HEAD tree)
1. **Scope.** Is the net change (vs `9dbf9b5`) confined to `results/xperf_chunksize/`? Grep the tree:
   the harness (`implementation_perf.py`, `run_amica_python.py`) must still contain `amica_python_jax`
   and must contain **no** `jamica`. Confirm `run_amica_python.py` still exists (was NOT renamed to
   `run_jamica.py` — that was part of the reverted commit).
2. **No look-alike corruption.** Grep the whole repo for accidental damage: `scott` is *also* an
   "amica-python" package — it must stay `scott-huberty` / `scott_huberty_torch`. Also confirm intact:
   `pyamica`, `pamica` / `pAMICA`, Fortran `amica17` / `fortran_amica17`, the vendored `amica_python/`
   dir, and paths like `amica-benchmark`. Look for corruption patterns (`pyjamica`, `jamica17`,
   `scott…jamica`, `jamica-benchmark`) — there should be none.
3. **Revert was clean.** Does HEAD match the pre-rename state for everything *except* the scoped study
   files? i.e., did `0d7a23e` fully undo `0cd9e05` with no residue in the harness / committed measures /
   vendored package?
4. **Internal consistency of the display rename.** In `gen_report.py` / `gen_showcase.py`, the JAX
   impl's short dict key was `"amica"` and is now `"jamica"` — verify it's consistent across **all**
   dicts/lists that key on it (`IMPLS`, `LABEL`, `KNOB`, `COLOR`, `COMMIT`, `GPU`, `GPU_BAND`, `CPU_RSS`,
   `CPU_FIT`, `CPU_FIT_MED`, `REAL_OPT`, `CPU_IMPLS`), with no leftover `"amica"` key that would raise a
   KeyError. The generators must still parse/run and emit valid HTML.
5. **CSV.** `chunk_sweep_data.csv` shows a 208-line diff; we believe that is line-ending normalization
   plus the `amica`→`jamica` impl-column rename, with the data otherwise identical. Please confirm it is
   **not** a data change/corruption, and that no `pyamica`/`pamica` row was mangled. (Separately, note if
   the CSV looks stale vs the real-data report — that's pre-existing, not caused by the rename.)
6. **Naming claims.** The README naming note states: amica-python → amica → **jamica**
   (`snesmaeili/jamica`, PyPI `jamica`, import `jamica`); harness keys deliberately unchanged. Is that
   accurate and are the caveats (scott's separate package, Fortran amica17, cluster paths, ephemeral raw
   JSONs) correct?
7. **Bottom line.** Is the rename correct and safe to keep, or is there a real defect? Give a prioritized
   verdict.

## Net stat (git diff 9dbf9b5..HEAD)
```
.../comparator/results/xperf_chunksize/README.md   |  10 +
 .../results/xperf_chunksize/chunk_sweep_data.csv   | 208 ++++++++++-----------
 .../results/xperf_chunksize/gen_report.py          |  64 +++----
 .../results/xperf_chunksize/gen_showcase.py        |  22 +--
 .../xperf_chunksize/realchunk_gpu_showcase.html    |  10 +-
 .../xperf_chunksize/xperf_chunk_report.html        |  40 ++--
 .../xperf_chunk_report_standalone.html             |  40 ++--
 7 files changed, 202 insertions(+), 192 deletions(-)
```

## Source diff (the generators + README — the meaningful source changes)
```diff
diff --git a/benchmark/comparator/results/xperf_chunksize/README.md b/benchmark/comparator/results/xperf_chunksize/README.md
index 3eda6fb..66e5b9a 100644
--- a/benchmark/comparator/results/xperf_chunksize/README.md
+++ b/benchmark/comparator/results/xperf_chunksize/README.md
@@ -83,3 +83,13 @@ Run through the orchestrator (`implementation_perf.py`) with each impl's best ch
 (`AMICA_CHUNK_SIZE`, `AMICA_PAMICA_BLOCK_SIZE`, `AMICA_PYAMICA_CHUNK`, `AMICA_SCOTT_BATCH`). The
 env-override for the three competitor runners is upstreamed in `../../runners/` (see git log);
 `run_amica_python.py` already honored `AMICA_CHUNK_SIZE`.
+
+## Naming: jamica (formerly amica-python)
+
+Sina's JAX AMICA was renamed **amica-python → amica → jamica** (`snesmaeili/jamica`, PyPI `jamica`,
+import `jamica`). This study's **reports and tidy data (`chunk_sweep_data.csv`) display the current
+name `jamica`**. The shared benchmark harness (runner impl keys `amica_python_jax*`, the orchestrator,
+and the committed legacy measures) is intentionally left on its **run-time keys** — renaming those
+would rewrite historical run-time provenance plus the vendored `amica_python/` package and the
+paper-reproduction bundle, which is a repo-wide migration best owned upstream. So: reports say
+`jamica`; the machine keys still reflect exactly what the runs stamped (`amica_python_jax*`).
diff --git a/benchmark/comparator/results/xperf_chunksize/gen_report.py b/benchmark/comparator/results/xperf_chunksize/gen_report.py
index ada77a1..37edf82 100644
--- a/benchmark/comparator/results/xperf_chunksize/gen_report.py
+++ b/benchmark/comparator/results/xperf_chunksize/gen_report.py
@@ -3,22 +3,22 @@
 
 Thesis: chunk size is the dominant performance variable; every implementation is strongly
 chunk-sensitive; defaults are footguns. On the real ds004505 workload (25-subject median, H100,
-latest main) amica-python is fastest at every chunk and leanest at full-batch. The synthetic
+latest main) jamica is fastest at every chunk and leanest at full-batch. The synthetic
 "flat curve" that once looked like robustness was a JIT-compile artifact (methodology section).
 CPU absolute timings carry a node-contention caveat (see NOTES_measurement.md); CPU *memory* is clean.
 """
 import math, os
 
 FULL = 262144
-IMPLS = ["amica", "pamica", "pyamica", "scott"]
-LABEL = {"amica":"amica-python (JAX)","pamica":"pAMICA (sccn)","pyamica":"pyamica","scott":"scott-huberty"}
-KNOB  = {"amica":"chunk_size","pamica":"block_size","pyamica":"chunk_t","scott":"batch_size","fortran":"block_size"}
-COMMIT= {"amica":"df18b5e","pamica":"0c4da39","pyamica":"a8a4d7e","scott":"e15e158","fortran":"665b577"}
-COLOR = {"amica":"#6366f1","pamica":"#d97706","pyamica":"#0d9488","scott":"#e11d48","fortran":"#7c3aed"}
+IMPLS = ["jamica", "pamica", "pyamica", "scott"]
+LABEL = {"jamica":"jamica","pamica":"pAMICA (sccn)","pyamica":"pyamica","scott":"scott-huberty"}
+KNOB  = {"jamica":"chunk_size","pamica":"block_size","pyamica":"chunk_t","scott":"batch_size","fortran":"block_size"}
+COMMIT= {"jamica":"df18b5e","pamica":"0c4da39","pyamica":"a8a4d7e","scott":"e15e158","fortran":"665b577"}
+COLOR = {"jamica":"#6366f1","pamica":"#d97706","pyamica":"#0d9488","scott":"#e11d48","fortran":"#7c3aed"}
 
 # ===== REAL ds004505, latest main, 25-subject median (GPU) : chunk -> (fit_s, vram_gb) =====
 GPU = {
- "amica":   {1024:(29.2,2.19),4096:(11.0,2.19),16384:(6.8,2.19),65536:(5.6,3.68),FULL:(5.2,8.14)},
+ "jamica":   {1024:(29.2,2.19),4096:(11.0,2.19),16384:(6.8,2.19),65536:(5.6,3.68),FULL:(5.2,8.14)},
  "pamica":  {1024:(139.6,0.58),4096:(36.9,0.64),16384:(14.3,0.86),65536:(11.6,1.75),FULL:(9.2,19.25)},
  "pyamica": {1024:(81.3,1.66),4096:(21.1,1.66),16384:(12.7,1.66),65536:(11.7,3.07),FULL:(9.7,28.30)},
  "scott":   {1024:(172.4,0.58),4096:(45.2,0.62),16384:(15.4,0.77),65536:(10.1,1.38)},  # full = OOM
@@ -26,14 +26,14 @@ GPU = {
 GPU_OOM = {"scott": "full"}
 # fit-time spread (min,max) across 25 subjects, for the band
 GPU_BAND = {
- "amica":   {1024:(20.8,34.7),4096:(9.1,29.9),16384:(6.0,43.0),65536:(5.0,10.5),FULL:(4.1,30.5)},
+ "jamica":   {1024:(20.8,34.7),4096:(9.1,29.9),16384:(6.0,43.0),65536:(5.0,10.5),FULL:(4.1,30.5)},
  "pamica":  {1024:(99.2,170.4),4096:(29.2,49.8),16384:(11.6,40.6),65536:(8.2,34.5),FULL:(7.0,33.5)},
  "pyamica": {1024:(58.5,98.8),4096:(15.0,26.2),16384:(9.4,15.8),65536:(8.4,14.3),FULL:(7.0,12.1)},
  "scott":   {1024:(123.3,213.5),4096:(36.8,72.4),16384:(13.3,85.7),65536:(7.4,16.8)},
 }
 # ===== REAL CPU memory (peak RSS, GB), 5-subject median : chunk -> rss (timing is contended, see note) =====
 CPU_RSS = {
- "amica":   {1024:2.23,4096:2.27,16384:2.23,65536:3.68,FULL:20.62},
+ "jamica":   {1024:2.23,4096:2.27,16384:2.23,65536:3.68,FULL:20.62},
  "pamica":  {1024:1.58,4096:1.74,16384:2.36,65536:2.63,FULL:19.57},
  "pyamica": {4096:1.94,16384:2.75,65536:3.81,FULL:27.21},
  "scott":   {1024:2.06,4096:2.06,16384:2.06,65536:2.31},
@@ -43,14 +43,14 @@ CPU_RSS = {
 # (best observed ≈ least-contended, the clean lower-bound curve); CPU_FIT_MED = median (still
 # carries residual node contention + per-subject data-size heterogeneity, shown for context).
 CPU_FIT = {  # min across subjects — the clean curve
- "amica":   {1024:155.0,4096:156.5,16384:162.3,65536:181.6,FULL:288.3},
+ "jamica":   {1024:155.0,4096:156.5,16384:162.3,65536:181.6,FULL:288.3},
  "scott":   {1024:195.0,4096:164.5,16384:172.3,65536:235.5},
  "pyamica": {4096:2008.4,16384:724.2,65536:776.9,FULL:903.2},
  "pamica":  {1024:241.3,4096:261.0,16384:341.7,65536:833.0,FULL:754.2},
  "fortran": {1024:639.8,4096:589.5,16384:730.3,65536:764.3,FULL:790.9},
 }
 CPU_FIT_MED = {  # median (contention-noisy, context only)
- "amica":{1024:276.4,4096:220.9,16384:186.6,65536:255.7,FULL:398.8},
+ "jamica":{1024:276.4,4096:220.9,16384:186.6,65536:255.7,FULL:398.8},
  "scott":{1024:257.9,4096:290.7,16384:268.9,65536:321.4},
  "pyamica":{4096:2345.0,16384:969.9,65536:980.8,FULL:980.7},
  "pamica":{1024:534.9,4096:431.3,16384:408.7,65536:911.7,FULL:815.8},
@@ -60,7 +60,7 @@ CPU_FIT_MED = {  # median (contention-noisy, context only)
 CPU_FIT_MISS = {("pyamica",1024):("&gt;1h","bad"),   # exceeded the 3600s runner timeout (pathological small-block)
                 ("scott",FULL):("OOM","bad")}         # full-batch out-of-memory (~30s failure)
 # Real each-at-own-optimum on GPU (25-subj median), fit + vram
-REAL_OPT = [("amica","full-batch",5.2,8.1),("pamica","full-batch",9.2,19.3),
+REAL_OPT = [("jamica","full-batch",5.2,8.1),("pamica","full-batch",9.2,19.3),
             ("pyamica","full-batch",9.7,28.3),("scott","65536",10.1,1.4)]
 # pAMICA default sensitivity (the 47x), GPU 25-subj median
 REAL_PAM = [("512 (default)",139.6,0.58,"artifact"),("16384 (tuned)",14.3,0.86,"tuned"),
@@ -122,7 +122,7 @@ gpu_t={im:{c:v[0] for c,v in GPU[im].items()} for im in IMPLS}
 gpu_v={im:{c:v[1] for c,v in GPU[im].items()} for im in IMPLS}
 c_gt=chart(gpu_t,GPU_BAND,"fit time (s, log)",True,"GPU · fit time vs chunk","real ds004505 · H100 · 100 iters · 25-subject median · band = min–max",IMPLS,oom={"scott":"full"})
 c_gv=chart(gpu_v,None,"peak VRAM (GB)",False,"GPU · memory vs chunk","real ds004505 · H100 · median peak VRAM · full-batch is the trap",IMPLS)
-CPU_IMPLS=["amica","pamica","pyamica","scott","fortran"]
+CPU_IMPLS=["jamica","pamica","pyamica","scott","fortran"]
 c_cr=chart(CPU_RSS,None,"peak RSS (GB)",False,"CPU · memory vs chunk","real ds004505 · 8 cores · 5-subject median · Fortran reference is leanest",CPU_IMPLS)
 c_ct=chart(CPU_FIT,None,"fit time (s, log)",True,"CPU · fit time vs chunk","real ds004505 · 8 cores · best-of-5 subjects (≈ least-contended) · optimum flips to small/mid chunks",CPU_IMPLS,oom={"scott":"full"})
 
@@ -133,7 +133,7 @@ def legend(impls):
 def optrows():
     base=REAL_OPT[0][2]; r=""
     for im,cfg,t,vram in REAL_OPT:
-        cls=' class="hi"' if im=="amica" else ""
+        cls=' class="hi"' if im=="jamica" else ""
         r+=f'<tr{cls}><td><span class="dot" style="background:{COLOR[im]}"></span>{LABEL[im]}</td><td><code>{cfg}</code></td><td class="num">{t:.1f}s</td><td class="num">{t/base:.1f}×</td><td class="num">{vram:.1f} GB</td></tr>'
     return r
 def pamrows():
@@ -195,7 +195,7 @@ tr.hi td{{background:color-mix(in srgb,var(--accent) 9%,transparent)}}tbody tr:l
 .stat{{background:var(--panel);border:1px solid var(--line);border-radius:14px;padding:18px;box-shadow:var(--shadow)}}
 .stat .big{{font-size:1.9rem;font-weight:800;letter-spacing:-.02em;font-variant-numeric:tabular-nums;line-height:1}}
 .stat .lab{{color:var(--mut);font-size:.86rem;margin-top:6px}}
-.stat.amica .big{{color:var(--accent)}}.stat.bad .big{{color:var(--bad)}}.stat.warn .big{{color:var(--warn)}}
+.stat.jamica .big{{color:var(--accent)}}.stat.bad .big{{color:var(--bad)}}.stat.warn .big{{color:var(--warn)}}
 p{{max-width:68ch}}.note{{font-size:.9rem;color:var(--mut)}}
 .warn-box{{background:color-mix(in srgb,var(--warn) 8%,var(--panel));border:1px solid var(--line);border-left:3px solid var(--warn);border-radius:10px;padding:14px 18px;margin:6px 0 14px;font-size:.92rem}}
 ul.tk{{max-width:68ch;padding-left:0;list-style:none}}ul.tk li{{padding:7px 0 7px 24px;position:relative;border-bottom:1px solid var(--line)}}
@@ -209,9 +209,9 @@ footer{{padding:34px 0 0;color:var(--mut);font-size:.86rem}}
   <div class="kick">Cross-implementation AMICA · ds004505 · real EEG · bleeding-edge</div>
   <h1>Chunk size is the <em>hidden variable</em></h1>
   <p class="lede">On real EEG, the batching knob moves fit time up to ~30× and peak memory up to ~30×.
-  Every implementation is strongly chunk-sensitive; the defaults are footguns. amica-python is fastest
+  Every implementation is strongly chunk-sensitive; the defaults are footguns. jamica is fastest
   at every chunk and leanest at full-batch.</p>
-  <div class="stamp"><span><b>Latest main:</b></span><span>amica <code>df18b5e</code></span><span>scott <code>e15e158</code></span><span>pyamica <code>a8a4d7e</code></span><span>pAMICA <code>0c4da39</code></span><span>Fortran ref <code>665b577</code></span><span>· 25 subj · 64 comp · 100 iters · H100 + 8-core Xeon</span></div>
+  <div class="stamp"><span><b>Latest main:</b></span><span>jamica <code>df18b5e</code></span><span>scott <code>e15e158</code></span><span>pyamica <code>a8a4d7e</code></span><span>pAMICA <code>0c4da39</code></span><span>Fortran ref <code>665b577</code></span><span>· 25 subj · 64 comp · 100 iters · H100 + 8-core Xeon</span></div>
 </header>
 
 <section>
@@ -222,7 +222,7 @@ footer{{padding:34px 0 0;color:var(--mut);font-size:.86rem}}
   <div class="callout">
     <div class="stat bad"><div class="big">140→9.2s</div><div class="lab">pAMICA on ds004505 (GPU median) from its 512 default to full-batch — a 15× self-speedup from one number.</div></div>
     <div class="stat warn"><div class="big">~30×</div><div class="lab">Range of every library across the chunk axis on real data. The default is never the optimum, and the optimum flips by device.</div></div>
-    <div class="stat amica"><div class="big">1.8×</div><div class="lab">amica's real lead over pAMICA when both run at their own optimum — not tens of ×.</div></div>
+    <div class="stat jamica"><div class="big">1.8×</div><div class="lab">jamica's real lead over pAMICA when both run at their own optimum — not tens of ×.</div></div>
   </div>
   <table><thead><tr><th><code>block_size</code></th><th>Fit time</th><th>Peak VRAM</th><th></th></tr></thead><tbody>{pamrows()}</tbody></table>
   <p class="note">pAMICA's torch backend does not auto-tune <code>block_size</code>; reporting its
@@ -237,12 +237,12 @@ footer{{padding:34px 0 0;color:var(--mut);font-size:.86rem}}
   <div class="grid2"><div class="card">{c_gt}</div><div class="card">{c_gv}</div></div>
   {legend(IMPLS)}
   <ul class="tk" style="margin-top:20px">
-    <li><b>amica-python is fastest at every chunk</b> — and its curve is a normal, steep U on real
+    <li><b>jamica is fastest at every chunk</b> — and its curve is a normal, steep U on real
     data (29→5 s), <em>not</em> flat (see methodology below).</li>
     <li><b>Defaults are footguns.</b> Small blocks are catastrophic (scott/pAMICA 140–170 s at 1024);
     the optimum is per-library and only found by sweeping.</li>
     <li><b>Full-batch is a memory trap:</b> pyamica <b>28 GB</b> / pAMICA <b>19 GB</b>, and
-    scott-huberty <b>OOMs</b> — while amica reaches its best time in ~8 GB (≈2 GB when chunked).</li>
+    scott-huberty <b>OOMs</b> — while jamica reaches its best time in ~8 GB (≈2 GB when chunked).</li>
   </ul>
 </section>
 
@@ -250,9 +250,9 @@ footer{{padding:34px 0 0;color:var(--mut);font-size:.86rem}}
   <h2>Fair comparison — each at its own optimum</h2>
   <p class="sub">The authoritative ranking: real ds004505, 25-subject median, every library at
   <em>its</em> best chunk on the H100.</p>
-  <table><thead><tr><th>Implementation</th><th>Batching</th><th>Fit time</th><th>vs amica</th><th>Peak VRAM</th></tr></thead><tbody>{optrows()}</tbody></table>
-  <p class="note">amica leads by <b>1.8×</b> over pAMICA at optima — not the tens-of-× a default
-  comparison implies. amica also pays the least memory for its speed.</p>
+  <table><thead><tr><th>Implementation</th><th>Batching</th><th>Fit time</th><th>vs jamica</th><th>Peak VRAM</th></tr></thead><tbody>{optrows()}</tbody></table>
+  <p class="note">jamica leads by <b>1.8×</b> over pAMICA at optima — not the tens-of-× a default
+  comparison implies. jamica also pays the least memory for its speed.</p>
 </section>
 
 <section>
@@ -263,11 +263,11 @@ footer{{padding:34px 0 0;color:var(--mut);font-size:.86rem}}
   <div class="grid2"><div class="card">{c_ct}</div><div class="card">{c_cr}</div></div>
   {legend(CPU_IMPLS)}
   <ul class="tk" style="margin-top:20px">
-    <li><b>The optimum flips by device.</b> CPU optima sit at <em>small/mid</em> chunks (amica 1024,
+    <li><b>The optimum flips by device.</b> CPU optima sit at <em>small/mid</em> chunks (jamica 1024,
     scott/Fortran ~4096, pyamica 16384) — the opposite of the GPU, where full-batch won. Small blocks
     fit CPU cache; on the H100 they starve the device. A single recommended chunk is wrong.</li>
-    <li><b>amica is fastest on CPU too</b> — best ~155 s vs scott 164, pAMICA 241, Fortran 589, pyamica 724.</li>
-    <li><b>Full-batch is a memory trap on CPU as well</b> (amica 21 GB, pyamica 27 GB); the Fortran
+    <li><b>jamica is fastest on CPU too</b> — best ~155 s vs scott 164, pAMICA 241, Fortran 589, pyamica 724.</li>
+    <li><b>Full-batch is a memory trap on CPU as well</b> (jamica 21 GB, pyamica 27 GB); the Fortran
     reference is the leanest everywhere (0.7 GB).</li>
     <li><b>Extremes fail:</b> pyamica@1024 exceeds the 1-hour runner timeout (~767 eager chunks/iter);
     scott full-batch OOMs — both absent from the curve.</li>
@@ -282,14 +282,14 @@ footer{{padding:34px 0 0;color:var(--mut);font-size:.86rem}}
 
 <section>
   <h2>Methodology — why the synthetic "flat curve" was a mirage</h2>
-  <p class="sub">An earlier synthetic microbenchmark (random data, 20 iters) showed amica with a
+  <p class="sub">An earlier synthetic microbenchmark (random data, 20 iters) showed jamica with a
   suspiciously flat GPU curve. An independent panel (GPT-5.6 + Grok-4.6) and a steady-state
   re-measurement traced it to a <b>JIT-compile artifact</b>, not robustness.</p>
-  <p>With <code>chunk_size</code> a JIT static argument, each of amica's fused-scan cells compiled a
+  <p>With <code>chunk_size</code> a JIT static argument, each of jamica's fused-scan cells compiled a
   different program; a fresh-process 20-iter wall clock folds a ~2.5 s compile into every point, which
   flattens a short run. Splitting compile from steady-state ms/iter (<code>(T₄₀−T₁₀)/30</code>) reveals
-  a normal <b>~19× steady-state spread</b> — amica is <em>not</em> chunk-robust. The real 100-iteration
-  workload above confirms it: a steep, ordinary U-curve. What amica genuinely has is a fused in-graph
+  a normal <b>~19× steady-state spread</b> — jamica is <em>not</em> chunk-robust. The real 100-iteration
+  workload above confirms it: a steep, ordinary U-curve. What jamica genuinely has is a fused in-graph
   blocked E-step (no Python-per-block penalty), memory-efficient full-batch, a correct <code>auto</code>
   pick per device, and the fastest real-workload time — not knob-insensitivity.</p>
   <div class="quote">"Flattest ⇒ most robust is a self-serving reading of a short, compile-contaminated
@@ -299,7 +299,7 @@ footer{{padding:34px 0 0;color:var(--mut);font-size:.86rem}}
 <section>
   <h2>Release vs main</h2>
   <p class="sub">All curves above are latest <code>main</code>. Release→main is performance-neutral on
-  GPU (measured identical); amica's one perf-relevant commit (<code>2cd81e4</code>, "make CPU fits
+  GPU (measured identical); jamica's one perf-relevant commit (<code>2cd81e4</code>, "make CPU fits
   faster and smaller") touches the CPU E-step only. Competitor <code>main</code> builds move only
   through the same batching knob; bumping to main rescues no one's default (pAMICA <code>main</code> is
   still 140 s at 512). The CPU curves above are the clean throttled <code>main</code> build; a
@@ -312,7 +312,7 @@ footer{{padding:34px 0 0;color:var(--mut);font-size:.86rem}}
     <dt>Dataset</dt><dd>ds004505 · 25 subjects (GPU) / 5 (CPU) · 64 PCA components</dd>
     <dt>GPU</dt><dd>NVIDIA H100 80GB · SciNet Trillium (def-kjerbi)</dd>
     <dt>CPU</dt><dd>8 cores · Alliance fir (rrg-kjerbi_cpu, bycore)</dd>
-    <dt>Commits</dt><dd>amica df18b5e · scott e15e158 · pyamica a8a4d7e · pAMICA 0c4da39 · Fortran 665b577</dd>
+    <dt>Commits</dt><dd>jamica df18b5e · scott e15e158 · pyamica a8a4d7e · pAMICA 0c4da39 · Fortran 665b577</dd>
     <dt>Workload</dt><dd>64 comp · 100 iters · per-subject median</dd>
     <dt>Runners</dt><dd>results/xperf_chunksize/sweeps/ + submit_cell_cpu.sh (atomic cells, cached input)</dd>
     <dt>Panel</dt><dd>panel/ (GPT-5.6 + Grok-4.6 + SYNTHESIS)</dd>
diff --git a/benchmark/comparator/results/xperf_chunksize/gen_showcase.py b/benchmark/comparator/results/xperf_chunksize/gen_showcase.py
index ecac4c1..6e996a6 100644
--- a/benchmark/comparator/results/xperf_chunksize/gen_showcase.py
+++ b/benchmark/comparator/results/xperf_chunksize/gen_showcase.py
@@ -3,21 +3,21 @@
 import math, os
 
 FULL = 262144
-IMPLS = ["amica", "pamica", "pyamica", "scott"]
-LABEL = {"amica":"amica-python (JAX)","pamica":"pAMICA (sccn)","pyamica":"pyamica","scott":"scott-huberty"}
-KNOB  = {"amica":"chunk_size","pamica":"block_size","pyamica":"chunk_t","scott":"batch_size"}
-COMMIT= {"amica":"df18b5e","pamica":"0c4da39","pyamica":"a8a4d7e","scott":"e15e158"}
-COLOR = {"amica":"#6366f1","pamica":"#d97706","pyamica":"#0d9488","scott":"#e11d48"}
+IMPLS = ["jamica", "pamica", "pyamica", "scott"]
+LABEL = {"jamica":"jamica","pamica":"pAMICA (sccn)","pyamica":"pyamica","scott":"scott-huberty"}
+KNOB  = {"jamica":"chunk_size","pamica":"block_size","pyamica":"chunk_t","scott":"batch_size"}
+COMMIT= {"jamica":"df18b5e","pamica":"0c4da39","pyamica":"a8a4d7e","scott":"e15e158"}
+COLOR = {"jamica":"#6366f1","pamica":"#d97706","pyamica":"#0d9488","scott":"#e11d48"}
 # GPU main, median of 25 subjects: chunk -> (median_s, min_s, max_s)
 T = {
- "amica":   {1024:(29.2,20.8,34.7),4096:(11.0,9.1,29.9),16384:(6.8,6.0,43.0),65536:(5.6,5.0,10.5),FULL:(5.2,4.1,30.5)},
+ "jamica":   {1024:(29.2,20.8,34.7),4096:(11.0,9.1,29.9),16384:(6.8,6.0,43.0),65536:(5.6,5.0,10.5),FULL:(5.2,4.1,30.5)},
  "pamica":  {1024:(139.6,99.2,170.4),4096:(36.9,29.2,49.8),16384:(14.3,11.6,40.6),65536:(11.6,8.2,34.5),FULL:(9.2,7.0,33.5)},
  "pyamica": {1024:(81.3,58.5,98.8),4096:(21.1,15.0,26.2),16384:(12.7,9.4,15.8),65536:(11.7,8.4,14.3),FULL:(9.7,7.0,12.1)},
  "scott":   {1024:(172.4,123.3,213.5),4096:(45.2,36.8,72.4),16384:(15.4,13.3,85.7),65536:(10.1,7.4,16.8)},  # full = OOM
 }
 # VRAM median (GB)
 V = {
- "amica":  
... (truncated; read the files at HEAD for the rest)
```

**Independence:** other panelists audit the same HEAD blind to your output. Your entire value is an
independent judgment from the actual files — do not read other files under `reviews/`.
