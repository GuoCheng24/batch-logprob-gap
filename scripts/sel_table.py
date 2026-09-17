"""Precision-budget table from selective_*.json: out-of-band % (Wilson 95% CI) vs fp32 parameter fraction."""
import os as _os
_RESULTS = _os.environ.get("BLG_DIR") or _os.path.join(
    _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "results")
import json, glob, math, sys, os
D = _RESULTS
suffix = sys.argv[1] if len(sys.argv) > 1 else ""
def wilson(p, n, z=1.96):
    p /= 100; c = p + z*z/(2*n); h = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n)); d = 1 + z*z/n
    return 100*(c-h)/d, 100*(c+h)/d
for f in sorted(glob.glob(f"{D}/selective_*{suffix}.json")):
    if suffix == "" and "_x" in os.path.basename(f): continue
    r = json.load(open(f)); n = r["n_tokens"]
    print(f"### {r['model']}  n_tokens={n}")
    print(f"  {'arm':<28} {'fp32 params':>11} {'out-of-band':>11}  95% CI")
    for a in r["arms"]:
        lo, hi = wilson(a["oob"], n)
        extra = f"  layers={a['layers']}" if a.get("delta_rank") else ""
        print(f"  {a['name']:<28} {100*a['fp32_frac']:>10.1f}% {a['oob']:>10.2f}%  [{lo:.2f}, {hi:.2f}]{extra}")
