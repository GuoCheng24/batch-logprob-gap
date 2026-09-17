"""One table per model merging the ladder (selective_<tag>_x4.json), the contiguous-window arms
(_x4_window.json), the fp16 arms (_x4_fp16.json) and, when present, the throughput benchmark
(selbench_<tag>.json). Out-of-band % with Wilson 95% CI; fp32 parameter fraction; relative time."""
import os as _os
_RESULTS = _os.environ.get("BLG_DIR") or _os.path.join(
    _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "results")
import json, glob, math, os, sys
D = _RESULTS
def wilson(p, n, z=1.96):
    p /= 100; c = p + z*z/(2*n); h = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n)); d = 1 + z*z/n
    return 100*(c-h)/d, 100*(c+h)/d
ORDER = ["baseline bf16", "head only (ScaleRL)", "fp16 all (no fp32)", "fp16 + head", "last 4 + head", "first 4 + head",
         "guided top4 + head", "guided window4 + head", "last 8 + head", "first 8 + head", "guided top8 + head",
         "guided window8 + head", "fp16 + guided top8 + head", "last 12 + head", "first 12 + head", "guided top12 + head",
         "guided window12 + head", "all layers + head (sanity)"]
tags = sorted({os.path.basename(f)[len("selective_"):].split("_x4")[0] for f in glob.glob(f"{D}/selective_*_x4*.json")})
for tag in (sys.argv[1:] or tags):
    arms, n, model = {}, None, None
    for f in sorted(glob.glob(f"{D}/selective_{tag}_x4*.json")):
        r = json.load(open(f)); n = n or r["n_tokens"]; model = r["model"]
        assert r["n_tokens"] == n, (f, r["n_tokens"], n)
        for a in r["arms"]:
            if a.get("skipped"): continue
            if a["name"] in arms and abs(arms[a["name"]]["oob"] - a["oob"]) > 1e-9:
                print(f"  ! {tag}: {a['name']} differs between files ({arms[a['name']]['oob']:.2f} vs {a['oob']:.2f}); keeping the later file")
            arms[a["name"]] = a
    bench = {}
    bf = f"{D}/selbench_{tag}.json"
    if os.path.exists(bf):
        for c in json.load(open(bf))["configs"]: bench[(tuple(c["layers"]), c["head"])] = c["rel_time"]
    print(f"### {model}  n_tokens={n}")
    print(f"  {'arm':<28} {'fp32 params':>11} {'out-of-band':>11}  {'95% CI':<16} {'time':>6}  layers")
    for name in ORDER:
        if name not in arms: continue
        a = arms[name]; lo, hi = wilson(a["oob"], n)
        t = bench.get((tuple(a["layers"]), a["norm_head"])); ts = f"x{t:.2f}" if t else "-"
        L = a["layers"]; ls = "none" if not L else (f"{L[0]}..{L[-1]}" if L == list(range(L[0], L[-1] + 1)) else str(L))
        dt = "" if a.get("base_dtype", "bfloat16") == "bfloat16" else f" [{a['base_dtype']}]"
        print(f"  {name:<28} {100*a['fp32_frac']:>10.1f}% {a['oob']:>10.2f}%  [{lo:.2f}, {hi:.2f}]{'':<{max(0,14-len(f'[{lo:.2f}, {hi:.2f}]'))}} {ts:>6}  {ls}{dt}")
    print()
