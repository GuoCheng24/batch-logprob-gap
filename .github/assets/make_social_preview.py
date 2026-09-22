"""Generate the GitHub social-preview card (1280x640). Reproducible: python3 make_social_preview.py

Every number on the card is read from results/ at draw time rather than typed, so the card cannot
drift from the measurement the way the README's prose once did.
"""
import json
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"


def rate(model, key="bf16|b1_vs_b8"):
    r = json.loads((RESULTS / f"hfonly_{model}.json").read_text())["results"][key]
    return 100 * r["out"] / r["n"]


lo = rate("Agents-A1-4B")
hi = rate("pythia-160m")
fp16_qwen = rate("Qwen2.5-1.5B-Instruct", "fp16|b1_vs_b8")
fp32_qwen = rate("Qwen2.5-1.5B-Instruct", "fp32|b1_vs_b8")

W, H = 12.8, 6.4
fig = plt.figure(figsize=(W, H), dpi=100)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W)
ax.set_ylim(0, H)
ax.axis("off")
ax.add_patch(plt.Rectangle((0, 0), W, H, color="#0d1117"))
SANS, MONO = "Liberation Sans", "Liberation Mono"

ax.text(0.75, 5.55, "batch-logprob-gap", fontsize=34, fontweight="bold", color="#e6edf3", family=SANS)
ax.text(0.75, 4.92, "In bfloat16 the batch shape moves the importance ratio. The policy never changed.",
        fontsize=17, color="#8b949e", family=SANS)

ax.add_patch(FancyBboxPatch((0.72, 1.28), 11.36, 3.05, boxstyle="round,pad=0.12",
                            fc="#161b22", ec="#30363d", lw=1.5))
ax.text(0.95, 4.02, "$ python scripts/measure.py     # same checkpoint, same tokens, batch 1 vs 8",
        fontsize=13.5, color="#7d8590", family=MONO)
rows = [
    (f"bf16   {hi:5.2f}%  of importance ratios leave [0.9, 1.1]      pythia-160m", "#f85149"),
    (f"bf16   {lo:5.2f}%  ... and this is the quietest of eight models  Agents-A1-4B", "#f0883e"),
    (f"fp16   {fp16_qwen:5.2f}%  same 16 bits, three more of mantissa        Qwen2.5-1.5B", "#58a6ff"),
    (f"fp32   {fp32_qwen:5.2f}%  and a rerun of the same batch is bit-exact   Qwen2.5-1.5B", "#3fb950"),
]
y = 3.55
for txt, c in rows:
    ax.text(0.95, y, txt, fontsize=14, color=c, family=MONO)
    y -= 0.5
ax.text(0.95, y - 0.02,
        "8 models | 4 architectures | padding, batch size and membership separated | six-arm GRPO bound",
        fontsize=12, color="#7d8590", family=MONO)

ax.text(0.75, 0.62,
        "Answers the question open since May 2026 in verl#6280 - and says where it stops: "
        "at 1.5B it does not move the reward.",
        fontsize=12.5, color="#8b949e", family=SANS)

out = pathlib.Path(__file__).parent / "social-preview.png"
fig.savefig(out)
print(f"written {out.name} 1280x640 "
      f"(bf16 {lo:.2f}-{hi:.2f}%, fp16 {fp16_qwen:.2f}%, fp32 {fp32_qwen:.2f}%)")
