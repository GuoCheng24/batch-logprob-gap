"""Plot the 100-step arms: python plot_long.py results/long_arms.json results/long_arms.png"""
import json
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ARMS = [  # (tag, label, colour)
    ("long_nocorr_tp08", "top_p=0.8, no correction", "#555555"),
    ("long_geo_tp08", "top_p=0.8, Geo-RS defaults", "#d62728"),
    ("long_geofix_tp08", "top_p=0.8, Geo-RS + support-size replay", "#1f77b4"),
    ("long_geo_tp10", "top_p=1.0, Geo-RS defaults", "#2ca02c"),
    ("long_seqmis_tp08", "top_p=0.8, Seq-MIS defaults", "#ff7f0e"),
]
PANELS = [
    ("critic/score/mean", "reward (mean score)", False),
    ("rollout_corr/rollout_rs_seq_masked_fraction", "sequences rejected", False),
    ("actor/grad_norm", "actor grad norm", False),
    ("rollout_corr/kl", "rollout_corr/kl", True),
]


def smooth(xs, w=5):
    return [sum(xs[max(0, i - w + 1) : i + 1]) / len(xs[max(0, i - w + 1) : i + 1]) for i in range(len(xs))]


runs = json.load(open(sys.argv[1]))
fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
for ax, (key, title, logy) in zip(axes.flat, PANELS):
    for tag, label, colour in ARMS:
        if tag not in runs:
            continue
        steps = runs[tag]["steps"]
        pts = [(int(s), steps[s][key]) for s in sorted(steps, key=int) if steps[s].get(key) is not None]
        if not pts:
            continue
        x, y = zip(*pts)
        if logy:
            y = [max(v, 1e-5) for v in y]
        ax.plot(x, y, color=colour, alpha=0.25, lw=0.8)
        ax.plot(x, smooth(list(y)), color=colour, lw=1.8, label=label)
        # a second run of the same arm, if there is one: dashed, 5-step mean only
        steps2 = runs.get(tag + "_r2", {}).get("steps", {})
        pts2 = [(int(s), steps2[s][key]) for s in sorted(steps2, key=int) if steps2[s].get(key) is not None]
        if pts2:
            x2, y2 = zip(*pts2)
            y2 = [max(v, 1e-5) for v in y2] if logy else list(y2)
            ax.plot(x2, smooth(list(y2)), color=colour, lw=1.2, ls="--")
    ax.set_title(title, fontsize=10)
    if logy:
        ax.set_yscale("log")
    ax.grid(alpha=0.3)
for ax in axes[1]:
    ax.set_xlabel("training step")
axes[0][0].legend(fontsize=8, loc="best")
fig.suptitle("Qwen2.5-1.5B-Instruct, GRPO on GSM8K, verl 6093e00 (replay arm: + prototype patch), "
             "2x RTX 4090 per arm; thin: per step, thick: 5-step mean, dashed: second run", fontsize=9)
fig.tight_layout()
fig.savefig(sys.argv[2], dpi=130)
print("wrote", sys.argv[2])
