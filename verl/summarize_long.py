"""Summarise the 100-step arms: python summarize_long.py results/long_arms.json results/long_summary.json

The input is what collect.py writes from the long_* logs. Every number the README quotes for
these arms is one of the fields written here, so doubleblind can trace it.
"""
import json
import statistics
import sys

ARMS = ["long_nocorr_tp08", "long_geo_tp08", "long_geofix_tp08", "long_geo_tp10", "long_seqmis_tp08"]
# second runs of the arms whose outcome is random (vLLM sampling is unseeded; the prompt order is fixed)
ARMS += ["long_nocorr_tp08_r2", "long_geofix_tp08_r2", "long_geo_tp10_r2"]


def series(steps, key):
    return [steps[s][key] for s in sorted(steps, key=int) if steps[s].get(key) is not None]


def mean(xs):
    return statistics.fmean(xs) if xs else None


runs = json.load(open(sys.argv[1]))
out = {}
for arm in ARMS:
    if arm not in runs:
        continue
    steps = runs[arm]["steps"]
    order = sorted(steps, key=int)
    score = series(steps, "critic/score/mean")
    grad = series(steps, "actor/grad_norm")
    rejected = series(steps, "rollout_corr/rollout_rs_seq_masked_fraction")
    kl = series(steps, "rollout_corr/kl")
    kept = series(steps, "rollout_corr/log_kept_mass_mean")
    out[arm] = {
        "n_steps": len(order),
        "last_step": int(order[-1]) if order else None,
        "score_mean_steps_1_10": mean(score[:10]),
        "score_mean_last_20": mean(score[-20:]),
        "score_max": max(score) if score else None,
        "grad_norm_mean": mean(grad),
        "steps_with_zero_grad": sum(g == 0.0 for g in grad),
        "rejected_seq_fraction_mean": mean(rejected),
        "rejected_seq_fraction_min": min(rejected) if rejected else None,
        "steps_all_rejected": sum(r == 1.0 for r in rejected) if rejected else None,
        "kl_mean": mean(kl),
        "log_kept_mass_mean": mean(kept),
        "all_masked_warnings": runs[arm]["all_masked_warnings"],
    }
json.dump(out, open(sys.argv[2], "w"), indent=1, sort_keys=True)
for arm, s in out.items():
    print(arm, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in s.items()})
