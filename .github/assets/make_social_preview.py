"""Generate the GitHub social-preview card (1280x640). Reproducible: python3 make_social_preview.py

Designed for the size it is actually seen at. A social card is unfurled at roughly 360 px wide in
Slack and 500 px on X, so the first version of this file - a screenshot of terminal output with
five lines of 14 pt monospace - was unreadable noise in every place it would ever appear. What
survives that reduction is a large number and a shape, so that is what this draws.

Every figure is read from results/ at draw time rather than typed, so the card cannot drift from
the measurement.
"""
import json
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"

LABEL = {
    "pythia-160m": "pythia-160m", "pythia-410m": "pythia-410m",
    "Qwen2.5-0.5B-Instruct": "Qwen2.5-0.5B", "Qwen2.5-1.5B-Instruct": "Qwen2.5-1.5B",
    "Qwen2.5-3B-Instruct": "Qwen2.5-3B", "Qwen2.5-7B-Instruct": "Qwen2.5-7B",
    "granite-3.1-1b-a400m-instruct": "Granite-1B MoE", "Agents-A1-4B": "Agents-A1-4B",
}


def rate(model, key="bf16|b1_vs_b8"):
    r = json.loads((RESULTS / f"hfonly_{model}.json").read_text())["results"][key]
    return 100 * r["out"] / r["n"]


bars = sorted(((LABEL[m], rate(m)) for m in LABEL), key=lambda t: -t[1])
worst = bars[0][1]
fp32 = max(rate(m, "fp32|b1_vs_b8") for m in LABEL)

W, H = 12.8, 6.4
fig = plt.figure(figsize=(W, H), dpi=100)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W)
ax.set_ylim(0, H)
ax.axis("off")
ax.add_patch(plt.Rectangle((0, 0), W, H, color="#0d1117"))
SANS = "Liberation Sans"
INK, MUTE, RED, GREEN = "#e6edf3", "#8b949e", "#f85149", "#3fb950"

ax.text(0.7, 5.62, "batch-logprob-gap", fontsize=40, fontweight="bold", color=INK, family=SANS)
ax.text(0.7, 5.02, "Change only the batch shape. The log probabilities move.",
        fontsize=21, color=MUTE, family=SANS)

# The left column is budgeted to end before the bar labels begin. The first version of this
# file let a 44-character line run under "Qwen2.5-3B"; sciglyph's layout report now says so.
ax.text(0.7, 2.75, f"{worst:.0f}%", fontsize=118, fontweight="bold", color=RED, family=SANS)
ax.text(0.78, 2.18, "of importance ratios leave", fontsize=20, color=INK, family=SANS)
ax.text(0.78, 1.70, "the PPO clip band in bf16", fontsize=20, color=INK, family=SANS)
ax.text(0.78, 1.18, "same weights, same tokens", fontsize=16.5, color=MUTE, family=SANS)
ax.text(0.78, 0.60, f"{fp32:.2f}% in fp32, and on a rerun of the same batch",
        fontsize=19, fontweight="bold", color=GREEN, family=SANS)

# the shape: eight models, one bar each - legible as a silhouette even at thumbnail size
x0, w_max = 7.05, 4.6
y = 4.30
for name, v in bars:
    ax.barh(y, w_max * v / worst, height=0.30, left=x0, color=RED if v > 20 else "#f0883e",
            zorder=3)
    ax.text(x0 - 0.12, y, name, fontsize=13.5, color=MUTE, family=SANS, ha="right", va="center")
    ax.text(x0 + w_max * v / worst + 0.12, y, f"{v:.1f}", fontsize=13.5, color=INK,
            family=SANS, va="center")
    y -= 0.44
# "8 models" is not written: eight bars are countable, and the line it used to share with
# the subtitle read as cramped at the width a Slack unfurl actually gives this card.
ax.text(x0 - 0.12, 4.78, "bf16, batch 1 vs 8", fontsize=15.5, color=INK, family=SANS, ha="right")

try:                                   # the author's own layout checker, if it is on the path
    import sys
    sys.path.insert(0, str(pathlib.Path.home() / "sciglyph"))
    from sciglyph import report
    report(fig, ax)
except Exception:
    pass

out = pathlib.Path(__file__).parent / "social-preview.png"
fig.savefig(out)
print(f"written {out.name} 1280x640 (worst {worst:.2f}%, fp32 max {fp32:.2f}%)")
