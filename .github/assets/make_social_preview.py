"""Generate the GitHub social-preview card (1200x630). Reproducible: python3 make_social_preview.py

Four models, one bar each, plus the fp32 line: the finding is that the rate is nonzero for every
architecture in bf16 and exactly zero in fp32, which is a shape rather than a sentence. Every
figure is read from results/ at draw time, so the card cannot drift from the measurement.

Eight bars were drawn first. At the 360 px a Slack unfurl gives a card the four smallest were
four indistinguishable stubs, so the four that span the range are here and the rest are in T1.
"""
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from cardkit import SANS, card  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
SHOWN = [("pythia-160m", "hfonly_pythia-160m"),
         ("pythia-410m", "hfonly_pythia-410m"),
         ("Granite MoE", "hfonly_granite-3.1-1b-a400m-instruct"),
         ("Qwen2.5-7B", "hfonly_Qwen2.5-7B-Instruct")]
ALL = ["hfonly_pythia-160m", "hfonly_pythia-410m", "hfonly_Qwen2.5-0.5B-Instruct",
       "hfonly_granite-3.1-1b-a400m-instruct", "hfonly_Agents-A1-4B",
       "hfonly_Qwen2.5-1.5B-Instruct", "hfonly_Qwen2.5-3B-Instruct",
       "hfonly_Qwen2.5-7B-Instruct"]


def rate(stem, key="bf16|b1_vs_b8"):
    r = json.loads((RESULTS / f"{stem}.json").read_text())["results"][key]
    return 100 * r["out"] / r["n"]


BARS = [(label, rate(stem)) for label, stem in SHOWN]
lo = min(rate(s) for s in ALL)
fp32 = max(rate(s, "fp32|b1_vs_b8") for s in ALL)


def chart(ax, accent):
    x0, span, top, step = 4.30, 4.15, 3.42, 0.60
    worst = max(v for _, v in BARS)
    for i, (name, value) in enumerate(BARS):
        y = top - i * step
        ax.barh(y, span * value / worst, height=0.40, left=x0,
                color="#cf222e" if value > 20 else "#d4801a", zorder=3)
        ax.text(x0 - 0.22, y, name, fontsize=34, color="#17181a", family=SANS,
                ha="right", va="center")
        ax.text(x0 + span * value / worst + 0.18, y, f"{value:.1f}%", fontsize=34,
                fontweight="bold", color="#17181a", family=SANS, va="center")
    # one line, left-aligned under the bars: two texts on the same baseline put the fp32
    # figure on top of a bar label and its own sentence on top of a bar value
    ax.text(0.78, top - 4.05 * step, f"{fp32:.2f}% in fp32, and on a rerun of the same batch",
            fontsize=34, fontweight="bold", color="#1a7f37", family=SANS, va="center")


out = card(
    out=str(pathlib.Path(__file__).parent / "social-preview.png"),
    accent="#cf222e", badge="G",
    kicker="RL POST-TRAINING  ·  8 models, 4 architectures",
    headline="Change the batch, move the ratio",
    evidence=f"ratios leaving [0.9, 1.1] in bf16, {lo:.1f}% to {max(v for _, v in BARS):.1f}%",
    chart=chart,
    footer="github.com/GuoCheng24/batch-logprob-gap",
    headline_size=44,
)
print(f"written {pathlib.Path(out).name}  range {lo:.2f}-{max(v for _, v in BARS):.2f}%, fp32 {fp32:.2f}%")
