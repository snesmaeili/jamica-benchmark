"""Is the local speed lead real, or inside the run-to-run spread?

A 15% gap on single runs is not a ranking when the competitor's own spread is of
similar size. This reads the per-seed results and reports median, spread, and
whether the fastest implementation's interval is actually separated from the
next one's.
"""
import glob
import json
import os
import sys
from collections import defaultdict

import numpy as np

root = sys.argv[1] if len(sys.argv) > 1 else "C:/amica-venvs/local_seeds/mne_sample"

times = defaultdict(list)
mems = defaultdict(list)
for f in sorted(glob.glob(os.path.join(root, "*_result.json"))):
    d = json.load(open(f))
    if "error" in d:
        print(f"  ! {os.path.basename(f)}: {str(d['error'])[:60]}")
        continue
    times[d["implementation"]].append(float(d["fit_time_s"]))
    mems[d["implementation"]].append(float(d["peak_rss_gb"]))

if not times:
    sys.exit(f"no results under {root}")

rows = []
for impl, ts in times.items():
    ts = np.array(ts)
    rows.append((float(np.median(ts)), impl, ts, np.array(mems[impl])))
rows.sort()

print(f"{'implementation':28} {'n':>2} {'median s':>9} {'min':>8} {'max':>8} "
      f"{'spread':>7} {'peak GiB':>9}")
print("-" * 78)
for med, impl, ts, ms in rows:
    spread = (ts.max() - ts.min()) / med * 100 if med else 0.0
    print(f"{impl:28} {len(ts):2d} {med:9.1f} {ts.min():8.1f} {ts.max():8.1f} "
          f"{spread:6.1f}% {np.median(ms):9.2f}")

print()
if len(rows) >= 2:
    (m1, i1, t1, _), (m2, i2, t2, _) = rows[0], rows[1]
    gap = (m2 - m1) / m1 * 100
    separated = t1.max() < t2.min()
    print(f"fastest      : {i1}  median {m1:.1f}s  range [{t1.min():.1f}, {t1.max():.1f}]")
    print(f"runner-up    : {i2}  median {m2:.1f}s  range [{t2.min():.1f}, {t2.max():.1f}]")
    print(f"median gap   : {gap:.1f}%")
    print()
    if separated:
        print("VERDICT: ranges do not overlap -- the lead survives run-to-run spread")
        print("         on this fixture and machine. Still one machine, one recording.")
    else:
        print("VERDICT: ranges OVERLAP -- the gap is within run-to-run variation.")
        print("         Report both as comparable rather than ranking them.")
