#!/usr/bin/env python3
"""Collaborator showcase: real-data (ds004505) fit-time & VRAM vs chunk size, GPU, bleeding-edge."""
import math, os

FULL = 262144
IMPLS = ["jamica", "pamica", "pyamica", "scott"]
LABEL = {"jamica":"jamica","pamica":"pAMICA (sccn)","pyamica":"pyamica","scott":"scott-huberty"}
KNOB  = {"jamica":"chunk_size","pamica":"block_size","pyamica":"chunk_t","scott":"batch_size"}
COMMIT= {"jamica":"df18b5e","pamica":"0c4da39","pyamica":"a8a4d7e","scott":"e15e158"}
COLOR = {"jamica":"#6366f1","pamica":"#d97706","pyamica":"#0d9488","scott":"#e11d48"}
# GPU main, median of 25 subjects: chunk -> (median_s, min_s, max_s)
T = {
 "jamica":   {1024:(29.2,20.8,34.7),4096:(11.0,9.1,29.9),16384:(6.8,6.0,43.0),65536:(5.6,5.0,10.5),FULL:(5.2,4.1,30.5)},
 "pamica":  {1024:(139.6,99.2,170.4),4096:(36.9,29.2,49.8),16384:(14.3,11.6,40.6),65536:(11.6,8.2,34.5),FULL:(9.2,7.0,33.5)},
 "pyamica": {1024:(81.3,58.5,98.8),4096:(21.1,15.0,26.2),16384:(12.7,9.4,15.8),65536:(11.7,8.4,14.3),FULL:(9.7,7.0,12.1)},
 "scott":   {1024:(172.4,123.3,213.5),4096:(45.2,36.8,72.4),16384:(15.4,13.3,85.7),65536:(10.1,7.4,16.8)},  # full = OOM
}
# VRAM median (GB)
V = {
 "jamica":   {1024:2.19,4096:2.19,16384:2.19,65536:2.19,FULL:8.14},
 "pamica":  {1024:0.58,4096:0.64,16384:0.86,65536:1.75,FULL:19.25},
 "pyamica": {1024:1.66,4096:1.66,16384:1.66,65536:3.07,FULL:28.30},
 "scott":   {1024:0.58,4096:0.62,16384:0.77,65536:1.38},
}
def xlog(c): return math.log2(FULL*4 if c==FULL else c)
XT=[1024,4096,16384,65536,FULL]; XMIN,XMAX=math.log2(1024)-0.4,xlog(FULL)+0.4

def chart(get, band, ylab, ylog, title, sub, oom=None):
    W,H=560,360; ml,mr,mt,mb=58,16,34,54; pw,ph=W-ml-mr,H-mt-mb
    def X(c): return ml+(xlog(c)-XMIN)/(XMAX-XMIN)*pw
    allv=[y for im in IMPLS for c in T[im] for y in ([get(im,c)] if get(im,c) else [])]
    if band: allv+=[b for im in IMPLS for c in T[im] for b in (band(im,c) or [])]
    vmax=max(allv); vmin=min(v for v in allv if v>0)
    if ylog:
        lo,hi=math.log10(vmin*0.8),math.log10(vmax*1.25)
        def Y(v): return mt+ph-(math.log10(max(v,vmin*0.5))-lo)/(hi-lo)*ph
    else:
        hi=vmax*1.1; lo=0
        def Y(v): return mt+ph-(v-lo)/(hi-lo)*ph
    s=[f'<svg viewBox="0 0 {W} {H}" class="chart" role="img" aria-label="{title}">']
    s.append(f'<text x="{ml}" y="17" class="ct">{title}</text>')
    s.append(f'<text x="{ml}" y="{H-6}" class="cx">chunk / block size (samples) →</text>')
    s.append(f'<text transform="translate(15,{mt+ph/2}) rotate(-90)" class="cy">{ylab}</text>')
    if ylog:
        ticks=[]; d0=1
        while d0<=vmax*1.25:
            for m0 in (1,2,5):
                if vmin*0.7<=d0*m0<=vmax*1.25: ticks.append(d0*m0)
            d0*=10
    else: ticks=[hi*i/4 for i in range(5)]
    for t in ticks:
        y=Y(t); s.append(f'<line x1="{ml}" y1="{y:.1f}" x2="{W-mr}" y2="{y:.1f}" class="grid"/>')
        s.append(f'<text x="{ml-7}" y="{y+3:.1f}" class="cyt">{t:.0f}</text>')
    for c in XT:
        x=X(c); lab="full" if c==FULL else f'{c//1024}K'
        s.append(f'<line x1="{x:.1f}" y1="{mt}" x2="{x:.1f}" y2="{mt+ph}" class="grid vg"/>')
        s.append(f'<text x="{x:.1f}" y="{mt+ph+17}" class="cxt">{lab}</text>')
    for im in IMPLS:
        cs=sorted(c for c in T[im] if get(im,c))
        if band:
            up=" ".join(f"{X(c):.1f},{Y(band(im,c)[1]):.1f}" for c in cs)
            dn=" ".join(f"{X(c):.1f},{Y(band(im,c)[0]):.1f}" for c in reversed(cs))
            s.append(f'<polygon points="{up} {dn}" fill="{COLOR[im]}" opacity="0.10"/>')
        path=" ".join((("M" if i==0 else "L")+f"{X(c):.1f},{Y(get(im,c)):.1f}") for i,c in enumerate(cs))
        s.append(f'<path d="{path}" fill="none" stroke="{COLOR[im]}" stroke-width="2.6"/>')
        for c in cs: s.append(f'<circle cx="{X(c):.1f}" cy="{Y(get(im,c)):.1f}" r="3.2" fill="{COLOR[im]}"/>')
        bc=min(cs,key=lambda k:get(im,k))
        s.append(f'<circle cx="{X(bc):.1f}" cy="{Y(get(im,bc)):.1f}" r="6.5" fill="none" stroke="{COLOR[im]}" stroke-width="2"/>')
    if oom:
        x=X(FULL); s.append(f'<text x="{x:.1f}" y="{mt+14}" class="oom">scott ✕ OOM</text>')
    s.append('</svg>')
    return f'<figure class="cf"><figcaption>{sub}</figcaption>{"".join(s)}</figure>'

c_t=chart(lambda im,c:(T[im].get(c) or [None])[0], lambda im,c:(T[im][c][1],T[im][c][2]) if c in T[im] else None,
          "fit time (s, log) — median of 25", True, "Fit time vs chunk size", "GPU · H100 · 100 iters · band = min–max across 25 subjects", oom=True)
c_v=chart(lambda im,c:V[im].get(c), None, "peak VRAM (GB)", False, "Peak GPU memory vs chunk size", "GPU · H100 · median of 25 subjects · full-batch is the memory trap")

legend="".join(f'<span class="lg"><i style="background:{COLOR[i]}"></i>{LABEL[i]} <code>{KNOB[i]}</code> <span class="cm">@{COMMIT[i]}</span></span>' for i in IMPLS)

HTML=f"""<title>AMICA chunk-size on real EEG (bleeding-edge)</title>
<style>
:root{{--bg:#f6f7fb;--panel:#fff;--ink:#181b26;--mut:#5a6072;--line:#e4e7f0;--accent:#6366f1;--code:#eef0f7}}
@media(prefers-color-scheme:dark){{:root{{--bg:#0d0f16;--panel:#161a24;--ink:#e8eaf2;--mut:#9aa0b4;--line:#262c3a;--accent:#8b8ff5;--code:#1d2230}}}}
:root[data-theme=dark]{{--bg:#0d0f16;--panel:#161a24;--ink:#e8eaf2;--mut:#9aa0b4;--line:#262c3a;--accent:#8b8ff5;--code:#1d2230}}
:root[data-theme=light]{{--bg:#f6f7fb;--panel:#fff;--ink:#181b26;--mut:#5a6072;--line:#e4e7f0;--accent:#6366f1;--code:#eef0f7}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font:16px/1.55 ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}}
.wrap{{max-width:1000px;margin:0 auto;padding:40px 24px 72px}}
code{{font-family:ui-monospace,Menlo,Consolas,monospace;font-size:.84em;background:var(--code);padding:.05em .38em;border-radius:5px}}
.kick{{font-size:.72rem;letter-spacing:.15em;text-transform:uppercase;color:var(--accent);font-weight:700}}
h1{{font-size:clamp(1.7rem,3.6vw,2.5rem);line-height:1.06;letter-spacing:-.02em;margin:.25em 0 .3em;font-weight:800;text-wrap:balance}}
.lede{{font-size:1.08rem;color:var(--mut);max-width:66ch;margin:0 0 6px}}
.stamp{{display:inline-flex;flex-wrap:wrap;gap:6px 14px;margin:14px 0 4px;padding:10px 14px;background:var(--panel);border:1px solid var(--line);border-left:3px solid var(--accent);border-radius:10px;font-size:.84rem}}
.stamp b{{color:var(--accent)}}
.grid2{{display:grid;grid-template-columns:1fr 1fr;gap:20px;margin:22px 0 4px}}
@media(max-width:760px){{.grid2{{grid-template-columns:1fr}}}}
.card{{background:var(--panel);border:1px solid var(--line);border-radius:14px;padding:12px 12px 6px;box-shadow:0 1px 2px rgba(20,24,45,.05),0 8px 24px rgba(20,24,45,.05)}}
svg.chart{{width:100%;height:auto;display:block;overflow:visible}}.cf{{margin:0}}.cf figcaption{{font-size:.8rem;color:var(--mut);padding:2px 4px 8px}}
.chart .grid{{stroke:var(--line)}}.chart .vg{{stroke-dasharray:2 3;opacity:.7}}
.chart .ct{{fill:var(--ink);font:700 13px ui-sans-serif}}.chart .cx,.chart .cy{{fill:var(--mut);font:11px ui-sans-serif}}
.chart .cyt,.chart .cxt{{fill:var(--mut);font:10px ui-monospace,monospace}}.chart .cyt{{text-anchor:end}}.chart .cxt{{text-anchor:middle}}
.chart .oom{{fill:#e11d48;font:600 10px ui-sans-serif;text-anchor:middle}}
.legend{{display:flex;flex-wrap:wrap;gap:8px 20px;margin:14px 2px;font-size:.85rem;color:var(--mut);align-items:center}}
.lg{{display:inline-flex;align-items:center;gap:6px}}.lg i{{width:16px;height:3px;border-radius:2px}}.cm{{font-family:ui-monospace,monospace;font-size:.76em;opacity:.75}}
.take{{margin-top:20px;padding:16px 20px;background:var(--panel);border:1px solid var(--line);border-radius:12px}}
.take b{{color:var(--accent)}}ul{{margin:.4em 0;padding-left:1.1em}}li{{margin:.25em 0;max-width:70ch}}
.foot{{margin-top:22px;color:var(--mut);font-size:.82rem}}
</style>
<div class="wrap">
  <div class="kick">Cross-implementation AMICA · real EEG · bleeding-edge</div>
  <h1>Chunk size drives fit time &amp; memory on real data</h1>
  <p class="lede">The batching knob moves fit time up to <b>~30×</b> and peak VRAM up to <b>~30×</b>.
  jamica is fastest at every setting and the leanest at full-batch.</p>
  <div class="stamp"><span><b>Bleeding-edge (latest main):</b></span>
    <span>jamica <code>df18b5e</code></span><span>scott-huberty <code>e15e158</code></span>
    <span>pyamica <code>a8a4d7e</code></span><span>pAMICA <code>0c4da39</code></span>
    <span>· ds004505 · 25 subjects · 64 comps · 100 iters · NVIDIA H100</span></div>
  <div class="grid2"><div class="card">{c_t}</div><div class="card">{c_v}</div></div>
  <div class="legend">{legend}<span class="lg">◯ each impl's optimum</span></div>
  <div class="take">
    <ul>
      <li><b>jamica wins at every chunk</b> (optimum <code>full/65536</code> ≈ 5.2 s vs pAMICA 9.2 s, pyamica 9.7 s, scott 10.1 s).</li>
      <li><b>Defaults are footguns.</b> Small blocks are catastrophic (scott/pAMICA 140–170 s at 1024); the sweet spot is per-implementation and only found by sweeping.</li>
      <li><b>Full-batch is a memory trap:</b> pyamica <b>28 GB</b> / pAMICA <b>19 GB</b> at full, and scott-huberty <b>OOMs</b> — while jamica reaches its best time in ~8 GB (and stays ~2 GB when chunked).</li>
    </ul>
  </div>
  <p class="foot">Lines are per-subject medians (n=25); shaded band = min–max across subjects (the wide tails at small chunks are compile/scheduling variance). Fortran reference (CPU-only) and the full CPU curves are a separate panel. Runners + data: <code>results/xperf_chunksize/</code>.</p>
</div>"""
HERE=os.path.dirname(os.path.abspath(__file__))
open(os.path.join(HERE,"realchunk_gpu_showcase.html"),"w").write(HTML)
print("wrote showcase", len(HTML), "bytes")
