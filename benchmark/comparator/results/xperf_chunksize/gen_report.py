#!/usr/bin/env python3
"""Render the AMICA cross-implementation timing/memory report (self-contained HTML), real data.

Outsider-facing measurement report: fit time and peak memory for each Python AMICA implementation
on a real EEG dataset (ds004505), swept across each one's batch/chunk-size setting, on GPU and CPU.
Neutral and even-handed — it reports measurements, not a verdict. CPU absolute timings carry a
node-contention caveat (see NOTES_measurement.md); CPU *memory* and all GPU numbers are clean.
"""
import math, os, csv

FULL = 262144
IMPLS = ["jamica", "pamica", "pyamica", "scott"]
LABEL = {"jamica":"jamica","pamica":"pAMICA (sccn)","pyamica":"pyamica","scott":"scott-huberty"}
KNOB  = {"jamica":"chunk_size","pamica":"block_size","pyamica":"chunk_t","scott":"batch_size","fortran":"block_size"}
COMMIT= {"jamica":"df18b5e","pamica":"0c4da39","pyamica":"a8a4d7e","scott":"e15e158","fortran":"665b577"}
COLOR = {"jamica":"#6366f1","pamica":"#d97706","pyamica":"#0d9488","scott":"#e11d48","fortran":"#7c3aed"}

# ===== REAL ds004505, latest main, 25-subject median (GPU) : chunk -> (fit_s, vram_gb) =====
GPU = {
 "jamica":   {1024:(29.2,2.19),4096:(11.0,2.19),16384:(6.8,2.19),65536:(5.6,3.68),FULL:(5.2,8.14)},
 "pamica":  {1024:(139.6,0.58),4096:(36.9,0.64),16384:(14.3,0.86),65536:(11.6,1.75),FULL:(9.2,19.25)},
 "pyamica": {1024:(81.3,1.66),4096:(21.1,1.66),16384:(12.7,1.66),65536:(11.7,3.07),FULL:(9.7,28.30)},
 "scott":   {1024:(172.4,0.58),4096:(45.2,0.62),16384:(15.4,0.77),65536:(10.1,1.38)},  # full = OOM
}
GPU_OOM = {"scott": "full"}
# fit-time spread (min,max) across 25 subjects, for the band
GPU_BAND = {
 "jamica":   {1024:(20.8,34.7),4096:(9.1,29.9),16384:(6.0,43.0),65536:(5.0,10.5),FULL:(4.1,30.5)},
 "pamica":  {1024:(99.2,170.4),4096:(29.2,49.8),16384:(11.6,40.6),65536:(8.2,34.5),FULL:(7.0,33.5)},
 "pyamica": {1024:(58.5,98.8),4096:(15.0,26.2),16384:(9.4,15.8),65536:(8.4,14.3),FULL:(7.0,12.1)},
 "scott":   {1024:(123.3,213.5),4096:(36.8,72.4),16384:(13.3,85.7),65536:(7.4,16.8)},
}
# ===== REAL CPU memory (peak RSS, GB), 5-subject median : chunk -> rss (timing is contended, see note) =====
CPU_RSS = {
 "jamica":   {1024:2.23,4096:2.27,16384:2.23,65536:3.68,FULL:20.62},
 "pamica":  {1024:1.58,4096:1.74,16384:2.36,65536:2.63,FULL:19.57},
 "pyamica": {4096:1.94,16384:2.75,65536:3.81,FULL:27.21},
 "scott":   {1024:2.06,4096:2.06,16384:2.06,65536:2.31},
 "fortran": {1024:0.72,4096:0.75,16384:0.86,65536:1.32,FULL:9.97},
}
# REAL CPU fit-time (s), throttled %4 rerun, 5 subjects. CPU_FIT = min across subjects
# (best observed = fastest of 5; both contention and subject length vary); CPU_FIT_MED = median (still
# carries residual node contention + per-subject data-size heterogeneity, shown for context).
CPU_FIT = {  # min across subjects — the clean curve
 "jamica":   {1024:155.0,4096:156.5,16384:162.3,65536:181.6,FULL:288.3},
 "scott":   {1024:195.0,4096:164.5,16384:172.3,65536:235.5},
 "pyamica": {4096:2008.4,16384:724.2,65536:776.9,FULL:903.2},
 "pamica":  {1024:241.3,4096:261.0,16384:341.7,65536:833.0,FULL:754.2},
 "fortran": {1024:639.8,4096:589.5,16384:730.3,65536:764.3,FULL:790.9},
}
CPU_FIT_MED = {  # median (contention-noisy, context only)
 "jamica":{1024:276.4,4096:220.9,16384:186.6,65536:255.7,FULL:398.8},
 "scott":{1024:257.9,4096:290.7,16384:268.9,65536:321.4},
 "pyamica":{4096:2345.0,16384:969.9,65536:980.8,FULL:980.7},
 "pamica":{1024:534.9,4096:431.3,16384:408.7,65536:911.7,FULL:815.8},
 "fortran":{1024:759.7,4096:812.4,16384:845.0,65536:862.8,FULL:960.9},
}
# Why a CPU cell has no number (not just "missing"):
CPU_FIT_MISS = {("pyamica",1024):("&gt;1h","bad"),   # exceeded the 3600s runner timeout (pathological small-block)
                ("scott",FULL):("OOM","bad")}         # full-batch out-of-memory (~30s failure)
# Real each-at-own-optimum on GPU (25-subj median), fit + vram
REAL_OPT = [("jamica","full-batch",5.2,8.1),("pamica","full-batch",9.2,19.3),
            ("pyamica","full-batch",9.7,28.3),("scott","65536",10.1,1.4)]
# pAMICA default sensitivity (the 47x), GPU 25-subj median
REAL_PAM = [("1024 (near 512 default)",139.6,0.58,"artifact"),("16384 (tuned)",14.3,0.86,"tuned"),
            ("full-batch (best)",9.2,19.25,"best")]

def xlog(c): return math.log2(FULL*4 if c==FULL else c)
XT=[1024,4096,16384,65536,FULL]; XMIN,XMAX=math.log2(1024)-0.4,xlog(FULL)+0.4

def chart(series, band, ylab, ylog, title, sub, impls, oom=None):
    W,H=520,340; ml,mr,mt,mb=56,14,32,52; pw,ph=W-ml-mr,H-mt-mb
    def X(c): return ml+(xlog(c)-XMIN)/(XMAX-XMIN)*pw
    allv=[v for im in impls for v in series[im].values()]
    if band: allv+=[b for im in impls for c in series[im] if im in band and c in band[im] for b in band[im][c]]
    vmax=max(allv); vmin=min(v for v in allv if v>0)
    if ylog:
        lo,hi=math.log10(vmin*0.8),math.log10(vmax*1.25)
        def Y(v): return mt+ph-(math.log10(max(v,vmin*0.5))-lo)/(hi-lo)*ph
    else:
        hi=vmax*1.12; lo=0
        def Y(v): return mt+ph-(v-lo)/(hi-lo)*ph
    s=[f'<svg viewBox="0 0 {W} {H}" class="chart" role="img" aria-label="{title}">']
    s.append(f'<text x="{ml}" y="16" class="ct">{title}</text>')
    s.append(f'<text x="{ml}" y="{H-6}" class="cx">chunk / block size (samples) →</text>')
    s.append(f'<text transform="translate(14,{mt+ph/2}) rotate(-90)" class="cy">{ylab}</text>')
    if ylog:
        ticks=[]; d0=1
        while d0<=vmax*1.25:
            for m0 in (1,2,5):
                if vmin*0.7<=d0*m0<=vmax*1.25: ticks.append(d0*m0)
            d0*=10
    else:
        raw=hi/4; e=10**math.floor(math.log10(raw)); f=raw/e
        step=(1 if f<1.5 else 2 if f<3 else 5 if f<7 else 10)*e
        ticks=[]; t0=0.0
        while t0<=hi: ticks.append(t0); t0+=step
    for t in ticks:
        y=Y(t); s.append(f'<line x1="{ml}" y1="{y:.1f}" x2="{W-mr}" y2="{y:.1f}" class="grid"/>')
        lab=f'{t:.0f}' if t>=1 else f'{t:.1f}'
        s.append(f'<text x="{ml-6}" y="{y+3:.1f}" class="cyt">{lab}</text>')
    for c in XT:
        x=X(c); lab="full" if c==FULL else f'{c//1024}K'
        s.append(f'<line x1="{x:.1f}" y1="{mt}" x2="{x:.1f}" y2="{mt+ph}" class="grid vg"/>')
        s.append(f'<text x="{x:.1f}" y="{mt+ph+16}" class="cxt">{lab}</text>')
    for im in impls:
        cs=sorted(series[im])
        if not cs: continue
        if band and im in band:
            up=" ".join(f"{X(c):.1f},{Y(band[im][c][1]):.1f}" for c in cs if c in band[im])
            dn=" ".join(f"{X(c):.1f},{Y(band[im][c][0]):.1f}" for c in reversed(cs) if c in band[im])
            if up: s.append(f'<polygon points="{up} {dn}" fill="{COLOR[im]}" opacity="0.09"/>')
        path=" ".join((("M" if i==0 else "L")+f"{X(c):.1f},{Y(series[im][c]):.1f}") for i,c in enumerate(cs))
        s.append(f'<path d="{path}" fill="none" stroke="{COLOR[im]}" stroke-width="2.4"/>')
        for c in cs: s.append(f'<circle cx="{X(c):.1f}" cy="{Y(series[im][c]):.1f}" r="3" fill="{COLOR[im]}"/>')
        bc=min(cs,key=lambda k:series[im][k])
        s.append(f'<circle cx="{X(bc):.1f}" cy="{Y(series[im][bc]):.1f}" r="6" fill="none" stroke="{COLOR[im]}" stroke-width="2"/>')
    if oom:
        for im,cc in oom.items():
            s.append(f'<text x="{X(FULL):.1f}" y="{mt+13}" class="oom">{im} ✕ OOM</text>')
    s.append('</svg>')
    return f'<figure class="cf"><figcaption>{sub}</figcaption>{"".join(s)}</figure>'

gpu_t={im:{c:v[0] for c,v in GPU[im].items()} for im in IMPLS}
gpu_v={im:{c:v[1] for c,v in GPU[im].items()} for im in IMPLS}
c_gt=chart(gpu_t,GPU_BAND,"fit time (s, log)",True,"GPU · fit time vs chunk","real ds004505 · H100 · 100 iters · 25-subject median · band = min–max",IMPLS,oom={"scott":"full"})
c_gv=chart(gpu_v,None,"peak VRAM (GB)",False,"GPU · memory vs chunk","real ds004505 · H100 · median peak VRAM",IMPLS)
CPU_IMPLS=["jamica","pamica","pyamica","scott","fortran"]
c_cr=chart(CPU_RSS,None,"peak RSS (GB)",False,"CPU · memory vs chunk","real ds004505 · 8 cores · 5-subject median peak RSS",CPU_IMPLS)
c_ct=chart(CPU_FIT,None,"fit time (s, log)",True,"CPU · fit time vs chunk","real ds004505 · 8 cores · best-of-5 subjects (fastest observed)",CPU_IMPLS,oom={"scott":"full"})

def legend(impls):
    it="".join(f'<span class="lg"><i style="background:{COLOR[i]}"></i>{LABEL.get(i,"Fortran amica17")} <code>{KNOB[i]}</code> <span class="cm">@{COMMIT[i]}</span></span>' for i in impls)
    return f'<div class="legend">{it}<span class="lg">◯ each impl\'s optimum</span></div>'

def optrows():
    base=min(t for _,_,t,_ in REAL_OPT); r=""
    for im,cfg,t,vram in REAL_OPT:
        r+=f'<tr><td><span class="dot" style="background:{COLOR[im]}"></span>{LABEL[im]}</td><td><code>{cfg}</code></td><td class="num">{t:.1f}s</td><td class="num">{t/base:.1f}×</td><td class="num">{vram:.1f} GB</td></tr>'
    return r
def pamrows():
    bd={"artifact":'<span class="badge bad">near default</span>',"tuned":'<span class="badge ok">tuned</span>',"best":'<span class="badge best">best</span>'}
    return "".join(f'<tr><td><code>{cfg}</code></td><td class="num">{t:.1f}s</td><td class="num">{v:.2f} GB</td><td>{bd[tag]}</td></tr>' for cfg,t,v,tag in REAL_PAM)
def cpufitrows():
    r=""
    for im in CPU_IMPLS:
        cells=""
        for c in XT:
            if c in CPU_FIT[im]:
                cells+=f'<td class="num">{CPU_FIT[im][c]:.0f}s</td>'
            elif (im,c) in CPU_FIT_MISS:
                lab,cls=CPU_FIT_MISS[(im,c)]
                cells+=f'<td class="num"><span class="badge {cls}">{lab}</span></td>'
            else:
                cells+='<td class="num">—</td>'
        r+=f'<tr><td><span class="dot" style="background:{COLOR[im]}"></span>{LABEL.get(im,"Fortran amica17")}</td>{cells}</tr>'
    return r

HTML=f"""<title>AMICA implementations — fit time and peak memory (ds004505)</title>
<style>
:root{{--bg:#ffffff;--panel:#ffffff;--ink:#1a1d29;--mut:#5b6172;--line:#e5e7eb;--accent:#2563eb;--good:#0d9488;--bad:#e11d48;--warn:#d97706;--code:#f1f3f8;--shadow:0 1px 2px rgba(20,24,45,.05),0 8px 22px rgba(20,24,45,.06)}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font:16px/1.55 ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}}
.wrap{{max-width:1080px;margin:0 auto;padding:0 24px 96px}}
code{{font-family:ui-monospace,Menlo,Consolas,monospace;font-size:.85em;background:var(--code);padding:.06em .4em;border-radius:5px}}
.num{{font-variant-numeric:tabular-nums;text-align:right;white-space:nowrap}}
header.hero{{padding:60px 0 28px;border-bottom:1px solid var(--line)}}
.kick{{font-size:.72rem;letter-spacing:.15em;text-transform:uppercase;color:var(--accent);font-weight:700}}
h1{{font-size:clamp(2rem,4.4vw,3rem);line-height:1.05;letter-spacing:-.02em;margin:.28em 0 .3em;text-wrap:balance;font-weight:800}}
h1 em{{font-style:normal;color:var(--accent)}}
.lede{{font-size:1.14rem;color:var(--mut);max-width:66ch;margin:0}}
.stamp{{display:inline-flex;flex-wrap:wrap;gap:6px 14px;margin:16px 0 0;padding:10px 14px;background:var(--panel);border:1px solid var(--line);border-left:3px solid var(--accent);border-radius:10px;font-size:.83rem}}
.stamp b{{color:var(--accent)}}
section{{padding:44px 0 6px;border-bottom:1px solid var(--line)}}
h2{{font-size:1.5rem;letter-spacing:-.01em;margin:0 0 4px;font-weight:750}}
.sub{{color:var(--mut);margin:.1em 0 1.3em;max-width:68ch}}
.grid2{{display:grid;grid-template-columns:1fr 1fr;gap:20px}}@media(max-width:760px){{.grid2{{grid-template-columns:1fr}}}}
.card{{background:var(--panel);border:1px solid var(--line);border-radius:14px;box-shadow:var(--shadow);padding:12px 12px 4px}}
.cf{{margin:0}}.cf figcaption{{font-size:.82rem;color:var(--mut);padding:3px 5px 8px}}
svg.chart{{width:100%;height:auto;display:block;overflow:visible}}
.chart .grid{{stroke:var(--line)}}.chart .vg{{stroke-dasharray:2 3;opacity:.7}}
.chart .ct{{fill:var(--ink);font:700 13px ui-sans-serif}}.chart .cx,.chart .cy{{fill:var(--mut);font:11px ui-sans-serif}}
.chart .cyt,.chart .cxt{{fill:var(--mut);font:10px ui-monospace,monospace}}.chart .cyt{{text-anchor:end}}.chart .cxt{{text-anchor:middle}}
.chart .oom{{fill:#e11d48;font:600 10px ui-sans-serif;text-anchor:middle}}
.legend{{display:flex;flex-wrap:wrap;gap:8px 18px;margin:14px 2px 2px;font-size:.82rem;color:var(--mut);align-items:center}}
.lg{{display:inline-flex;align-items:center;gap:6px}}.lg i{{width:15px;height:3px;border-radius:2px}}.cm{{font-family:ui-monospace,monospace;font-size:.78em;opacity:.7}}
table{{width:100%;border-collapse:collapse;font-size:.95rem;margin:.2em 0 1em}}
th,td{{text-align:left;padding:9px 12px;border-bottom:1px solid var(--line)}}
th{{font-size:.74rem;letter-spacing:.05em;text-transform:uppercase;color:var(--mut);font-weight:650}}
tbody tr:last-child td{{border-bottom:none}}
.dot{{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:8px;vertical-align:middle}}
.badge{{font-size:.72rem;padding:2px 9px;border-radius:20px;font-weight:600;white-space:nowrap}}
.badge.bad{{background:color-mix(in srgb,var(--bad) 16%,transparent);color:var(--bad)}}.badge.ok{{background:color-mix(in srgb,var(--warn) 18%,transparent);color:var(--warn)}}.badge.best{{background:color-mix(in srgb,var(--good) 16%,transparent);color:var(--good)}}
.callout{{display:grid;grid-template-columns:repeat(3,1fr);gap:16px;margin:8px 0 20px}}@media(max-width:720px){{.callout{{grid-template-columns:1fr}}}}
.stat{{background:var(--panel);border:1px solid var(--line);border-radius:14px;padding:18px;box-shadow:var(--shadow)}}
.stat .big{{font-size:1.9rem;font-weight:800;letter-spacing:-.02em;font-variant-numeric:tabular-nums;line-height:1}}
.stat .lab{{color:var(--mut);font-size:.86rem;margin-top:6px}}
.stat.bad .big{{color:var(--bad)}}.stat.warn .big{{color:var(--warn)}}
p{{max-width:68ch}}.note{{font-size:.9rem;color:var(--mut)}}
.warn-box{{background:color-mix(in srgb,var(--warn) 8%,var(--panel));border:1px solid var(--line);border-left:3px solid var(--warn);border-radius:10px;padding:14px 18px;margin:6px 0 14px;font-size:.92rem}}
ul.tk{{max-width:68ch;padding-left:0;list-style:none}}ul.tk li{{padding:7px 0 7px 24px;position:relative;border-bottom:1px solid var(--line)}}
ul.tk li:before{{content:"";position:absolute;left:3px;top:14px;width:8px;height:8px;border-radius:50%;background:var(--accent)}}
footer{{padding:34px 0 0;color:var(--mut);font-size:.86rem}}
.prov{{display:grid;grid-template-columns:150px 1fr;gap:4px 16px;font-size:.9rem;margin-top:8px}}.prov dt{{color:var(--mut)}}.prov dd{{margin:0;font-family:ui-monospace,monospace;font-size:.86em}}
</style>
<div class="wrap">
<header class="hero">
  <div class="kick">Cross-implementation AMICA · ds004505 · real EEG</div>
  <h1>Fit time and peak memory across AMICA implementations</h1>
  <p class="lede">We measured how long each Python AMICA implementation takes to fit, and how much
  memory it needs, on a real EEG dataset — sweeping each one across its batch/chunk-size setting on
  both GPU and CPU. That single setting changes fit time by up to ~17× and peak memory by up to ~30×
  within one implementation, and the best setting differs by implementation and between GPU and CPU.</p>
  <div class="stamp"><span><b>Builds (main):</b></span><span>jamica <code>df18b5e</code></span><span>scott-huberty <code>e15e158</code></span><span>pyamica <code>a8a4d7e</code></span><span>pAMICA <code>0c4da39</code></span><span>Fortran ref <code>665b577</code></span><span>· 25 subj · 64 comp · 100 iters · H100 + 8-core Xeon</span></div>
</header>

<section>
  <h2>What we measured</h2>
  <p class="sub">Each implementation exposes a setting controlling how many samples are processed per
  pass — <code>chunk_size</code>, <code>block_size</code>, <code>chunk_t</code> or <code>batch_size</code>
  depending on the package. We swept it across five values (1024 → full-batch) and recorded fit time
  and peak memory, on the same subjects, for each device.</p>
  <div class="callout">
    <div class="stat warn"><div class="big">~17×</div><div class="lab">Widest fit-time range across the batch/chunk setting, within a single implementation. A setting that is fast on one device can be slow on the other.</div></div>
    <div class="stat warn"><div class="big">~30×</div><div class="lab">Widest peak-memory range across the setting. Memory grows sharply toward full-batch — enough to run some implementations out of memory on an 80&nbsp;GB card.</div></div>
    <div class="stat"><div class="big">GPU ⇄ CPU</div><div class="lab">The best setting flips by device: large / full-batch on the H100, small / mid on CPU. A single recommended value is wrong for both.</div></div>
  </div>
</section>

<section>
  <h2>GPU — fit time &amp; memory</h2>
  <p class="sub">Each implementation swept across its setting on real ds004505 (25-subject median,
  100 iterations, H100). ◯ marks each one's fastest measured setting; the shaded band on the timing
  plot is the min–max across subjects.</p>
  <div class="grid2"><div class="card">{c_gt}</div><div class="card">{c_gv}</div></div>
  {legend(IMPLS)}
  <ul class="tk" style="margin-top:20px">
    <li><b>Larger settings are faster on the GPU.</b> Fit time falls as the chunk grows; the fastest
    measured setting is at or near full-batch for jamica, pAMICA and pyamica, and 65K for scott-huberty.</li>
    <li><b>Small settings are slow.</b> At 1024, scott-huberty and pAMICA take 140–170&nbsp;s, versus
    ~9–10&nbsp;s at each one's own fastest setting. The fastest setting is per-implementation and only found by sweeping.</li>
    <li><b>Memory grows sharply toward full-batch.</b> At full-batch, pyamica peaks near <b>28&nbsp;GB</b>,
    pAMICA near <b>19&nbsp;GB</b> and jamica near <b>8&nbsp;GB</b>, and scott-huberty runs out of memory.
    Chunking trades a large amount of memory for a little time — all four drop to a few GB at smaller settings.</li>
  </ul>
</section>

<section>
  <h2>Each implementation at its own fastest setting</h2>
  <p class="sub">Fit time when every implementation runs at its own best measured setting on the H100
  (real ds004505, 25-subject median). These rows share the same dataset, subject set, iteration count
  (100), component count and fit-time metric — but equal iterations is not equal convergence, and the
  memory counters differ by framework (JAX <code>peak_bytes_in_use</code> vs Torch
  <code>max_memory_allocated</code>), so read the numbers as directional.</p>
  <table><thead><tr><th>Implementation</th><th>Setting</th><th>Fit time</th><th>Relative</th><th>Peak VRAM</th></tr></thead><tbody>{optrows()}</tbody></table>
  <p class="note">At their respective best settings the four fall within ~1.9× of each other. As one
  measured illustration of how far a single setting can sit from an implementation's optimum, pAMICA's
  <code>block_size</code> on the GPU:</p>
  <table><thead><tr><th><code>block_size</code></th><th>Fit time</th><th>Peak VRAM</th><th></th></tr></thead><tbody>{pamrows()}</tbody></table>
  <p class="note">pAMICA's shipped default is <code>block_size=512</code>, which we did not re-measure
  on this sweep; the nearest measured point, 1024, is already ~15× off its own fastest setting.</p>
</section>

<section>
  <h2>CPU — fit time &amp; memory</h2>
  <p class="sub">Real ds004505, 8 cores, 5 subjects. ◯ marks each implementation's fastest CPU setting.
  Includes the Fortran amica17 reference (CPU-only). Fit time is the best-observed (fastest) of the 5
  subjects per cell — see the measurement note below.</p>
  <div class="grid2"><div class="card">{c_ct}</div><div class="card">{c_cr}</div></div>
  {legend(CPU_IMPLS)}
  <ul class="tk" style="margin-top:20px">
    <li><b>The best setting is different on CPU.</b> CPU optima sit at <em>small/mid</em> chunks (jamica
    1024, scott-huberty/Fortran ~4096, pyamica 16384) — the opposite of the GPU, where large/full-batch
    won. Small blocks fit CPU cache; on the H100 they starve the device. The recommended setting depends
    on the device.</li>
    <li><b>Best-observed CPU times</b> (fastest of 5 subjects, approximate — see note): jamica and
    scott-huberty are closest at ~155–165&nbsp;s, then pAMICA ~240&nbsp;s, Fortran ~590&nbsp;s and pyamica ~720&nbsp;s.</li>
    <li><b>Full-batch is memory-heavy on CPU too</b> (jamica 21&nbsp;GB, pyamica 27&nbsp;GB); the Fortran
    reference uses the least memory at every measured setting (0.72&nbsp;GB at 1024, up to 9.97&nbsp;GB at full-batch).</li>
    <li><b>Two settings failed outright:</b> pyamica at 1024 exceeded the 1-hour timeout; scott-huberty
    at full-batch ran out of memory — both absent from the curve.</li>
  </ul>
  <div class="warn-box" style="margin-top:14px"><b>On the CPU timing numbers.</b> CPU fits were measured
  on shared cluster nodes, and AMICA fits are memory-bandwidth-bound, so jobs sharing a node interfere
  and per-cell absolute times are noisy. The 5 subjects also differ in recording length, so a per-cell
  median mixes both contention and data size and stays non-monotonic. We plot the best-observed (fastest)
  of the 5 per cell to recover a clean trend — but that minimum reflects both a quieter node and a shorter
  subject, so treat absolute CPU seconds as approximate, contention-inflated figures. Rely on the curve
  shapes and each implementation's fastest setting, and read the cross-implementation ordering as coarse.
  CPU <em>memory</em> is allocation-driven and clean regardless. Full detail: <code>NOTES_measurement.md</code>.</div>
</section>

<footer>
  <div class="kick" style="color:var(--mut)">Provenance &amp; reproduction</div>
  <dl class="prov">
    <dt>Dataset</dt><dd>ds004505 · 25 subjects (GPU) / 5 (CPU) · 64 PCA components</dd>
    <dt>GPU</dt><dd>NVIDIA H100 80GB · SciNet Trillium (def-kjerbi)</dd>
    <dt>CPU</dt><dd>8 cores · Alliance fir (rrg-kjerbi_cpu, bycore)</dd>
    <dt>Commits</dt><dd>jamica df18b5e · scott e15e158 · pyamica a8a4d7e · pAMICA 0c4da39 · Fortran 665b577</dd>
    <dt>Workload</dt><dd>64 comp · 100 iters · GPU 25-subj median · CPU best-of-5</dd>
    <dt>Runners</dt><dd>results/xperf_chunksize/sweeps/ + submit_cell_cpu.sh (atomic cells, cached input)</dd>
    <dt>Caveats</dt><dd>NOTES_measurement.md (CPU contention + noise estimate)</dd>
  </dl>
  <p class="note" style="margin-top:16px">All fit times are 100-iteration runs; for JIT-compiled
  implementations this amortizes the one-time compile so the numbers reflect steady-state throughput —
  on much shorter runs that compile cost dominates and the ranking can differ. GPU numbers are 25-subject
  medians and CPU <em>memory</em> is clean; CPU <em>fit-times</em> are the best-observed (fastest) of 5
  subjects, approximate — rely on the curve shapes and each implementation's fastest setting rather than
  the exact CPU seconds or ranking.</p>
</footer>
</div>"""
HERE=os.path.dirname(os.path.abspath(__file__))
# content version (no <!doctype>/<head>/<body> — the Artifact publish step adds those)
open(os.path.join(HERE,"xperf_chunk_report.html"),"w").write(HTML)
# standalone version — full document, opens directly in any browser
_title = HTML.split("<title>",1)[1].split("</title>",1)[0] if "<title>" in HTML else "AMICA chunk-size report"
STANDALONE = ('<!doctype html><html lang="en"><head><meta charset="utf-8">'
              '<meta name="viewport" content="width=device-width, initial-scale=1">'
              f'<title>{_title}</title></head><body>\n{HTML}\n</body></html>\n')
open(os.path.join(HERE,"xperf_chunk_report_standalone.html"),"w").write(STANDALONE)

# tidy dataset — regenerated from THIS report's real-data dicts (single source of truth),
# so chunk_sweep_data.csv never drifts from the rendered numbers.
def _cn(c): return "full" if c == FULL else str(c)
_rows = [("dataset", "impl", "knob", "chunk", "value", "unit", "note")]
for im in IMPLS:
    for c, (t, v) in sorted(GPU[im].items()):
        _rows.append(("gpu_fit_s_median", im, KNOB[im], _cn(c), t, "s", "real ds004505 25-subj median, H100"))
        _rows.append(("gpu_vram_gb_median", im, KNOB[im], _cn(c), v, "GB", ""))
    for c, (lo, hi) in sorted(GPU_BAND[im].items()):
        _rows.append(("gpu_fit_s_min", im, KNOB[im], _cn(c), lo, "s", "min across 25 subj"))
        _rows.append(("gpu_fit_s_max", im, KNOB[im], _cn(c), hi, "s", "max across 25 subj"))
# explicit GPU failure records (so OOM claims are checkable, not just an absent row)
for im, cc in GPU_OOM.items():
    _rows.append(("gpu_fit_s_median", im, KNOB[im], _cn(FULL if cc == "full" else cc), "OOM", "",
                  "full-batch out-of-memory on H100 80GB (~30s failure)"))
for im in CPU_IMPLS:
    for c, v in sorted(CPU_RSS[im].items()):
        _rows.append(("cpu_rss_gb_median", im, KNOB[im], _cn(c), v, "GB", "real ds004505 5-subj median, 8 cores"))
    for c, v in sorted(CPU_FIT[im].items()):
        _rows.append(("cpu_fit_s_min", im, KNOB[im], _cn(c), v, "s", "best-of-5 subj (fastest observed; contention + subject length both vary)"))
    for c, v in sorted(CPU_FIT_MED[im].items()):
        _rows.append(("cpu_fit_s_median", im, KNOB[im], _cn(c), v, "s", "median (contention-noisy)"))
for (im, c), (lab, _cls) in CPU_FIT_MISS.items():
    _rows.append(("cpu_fit_s_min", im, KNOB[im], _cn(c), lab.replace("&gt;", ">"), "", "no value: timeout / OOM"))
for im, cfg, t, vram in REAL_OPT:
    _rows.append(("real_each_optimum_gpu", im, KNOB[im], cfg, t, "s", "25-subj median, at own optimum"))
for cfg, t, v, tag in REAL_PAM:
    _rows.append(("real_pamica_sensitivity_gpu", "pamica", "block_size", cfg, t, "s", tag))
with open(os.path.join(HERE, "chunk_sweep_data.csv"), "w", newline="") as _f:
    csv.writer(_f).writerows(_rows)

print("wrote real-data report", len(HTML), "bytes (+ standalone", len(STANDALONE),
      "bytes) + chunk_sweep_data.csv", len(_rows) - 1, "rows")
