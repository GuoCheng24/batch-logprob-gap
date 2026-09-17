"""Summarise the four-arm GRPO grid: per arm, mean over seeds of late reward, reward AUC,
clip ratio and the trainer's own sampling-vs-old log-prob gap; paired differences vs arm A."""
import os as _os
_RESULTS = _os.environ.get("BLG_DIR") or _os.path.join(
    _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "results")
import json, glob, os, sys, statistics as st
D = sys.argv[1] if len(sys.argv) > 1 else _os.path.join(_RESULTS, "q1")
KEYS = ["reward", "clip_ratio/region_mean", "sampling/sampling_logp_difference/mean", "sampling/sampling_logp_difference/max", "kl", "entropy"]
runs = {}
for f in sorted(glob.glob(f"{D}/*_s*.json")):
    r = json.load(open(f)); h = [x for x in r["log_history"] if "reward" in x]
    if not h: continue
    n = len(h); k = max(1, n // 5)
    row = {"steps": n, "old_calls": r.get("n_old_calls"), "wall_min": r["wall_s"] / 60}
    for key in KEYS:
        v = [float(x[key]) for x in h if key in x]
        if v: row[key + "/late"] = sum(v[-k:]) / k; row[key + "/auc"] = sum(v) / len(v)
    runs[(r["arm"], r["seed"])] = row
if not runs: sys.exit("no runs")
arms = sorted({a for a, _ in runs}); seeds = sorted({s for _, s in runs})
print(f"runs: {len(runs)}  arms {arms}  seeds {seeds}")
def col(a, key): return [runs[(a, s)][key] for s in seeds if (a, s) in runs and key in runs[(a, s)]]
print(f"{'arm':<4}{'n':>3} {'reward late':>12} {'reward auc':>11} {'clip late':>10} {'|dlogp| mean':>13} {'|dlogp| max':>12} {'kl late':>9} {'min/run':>8}")
for a in arms:
    def m(key): v = col(a, key); return f"{st.mean(v):.4f}" + (f"±{st.stdev(v):.4f}" if len(v) > 1 else "") if v else "-"
    print(f"{a:<4}{len(col(a,'steps')):>3} {m('reward/late'):>12} {m('reward/auc'):>11} {m('clip_ratio/region_mean/late'):>10} "
          f"{m('sampling/sampling_logp_difference/mean/auc'):>13} {m('sampling/sampling_logp_difference/max/auc'):>12} {m('kl/late'):>9} {st.mean(col(a,'wall_min')):>8.1f}")
if "A" in arms:
    print("\npaired vs A (same seed):")
    for a in arms:
        if a == "A": continue
        d = [runs[(a, s)]["reward/late"] - runs[("A", s)]["reward/late"] for s in seeds if (a, s) in runs and ("A", s) in runs]
        da = [runs[(a, s)]["reward/auc"] - runs[("A", s)]["reward/auc"] for s in seeds if (a, s) in runs and ("A", s) in runs]
        if d: print(f"  {a}-A  late Δ {st.mean(d):+.4f} (n={len(d)}, per-seed {[f'{x:+.3f}' for x in d]})   auc Δ {st.mean(da):+.4f}")
