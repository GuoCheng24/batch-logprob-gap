"""Rebuild every table in the report from the saved JSON, so no number is transcribed by hand."""
import os as _os
_RESULTS = _os.environ.get("BLG_DIR") or _os.path.join(
    _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "results")
_os.makedirs(_RESULTS, exist_ok=True)
import json, glob, os
D = _RESULTS
SHORT = {"pythia-160m": "pythia-160m (NeoX 0.16B)",
         "pythia-410m": "pythia-410m (NeoX 0.41B)",
         "Qwen2.5-0.5B-Instruct": "Qwen2.5-0.5B",
         "granite-3.1-1b-a400m-instruct": "Granite-3.1-1B-A400M (MoE)",
         "Agents-A1-4B": "Agents-A1-4B",
         "Qwen2.5-1.5B-Instruct": "Qwen2.5-1.5B",
         "Qwen2.5-3B-Instruct": "Qwen2.5-3B",
         "Qwen2.5-7B-Instruct": "Qwen2.5-7B"}
ORDER = ["pythia-160m", "pythia-410m", "Qwen2.5-0.5B-Instruct", "granite-3.1-1b-a400m-instruct", "Agents-A1-4B",
         "Qwen2.5-1.5B-Instruct", "Qwen2.5-3B-Instruct", "Qwen2.5-7B-Instruct"]


def pct(r):
    return f"{100*r['out']/r['n']:.2f}%"


print("## T1  dtype x batch, no inference engine involved\n")
print("| model | bf16 b1 vs b8 | fp16 b1 vs b8 | fp32 b1 vs b8 | bf16 b1 rerun | bf16 b1 vs b32 |")
print("|---|---|---|---|---|---|")
for k in ORDER:
    f = f"{D}/hfonly_{k}.json"
    if not os.path.exists(f):
        continue
    r = json.load(open(f))["results"]
    print(f"| {SHORT[k]} | {pct(r['bf16|b1_vs_b8'])} | {pct(r['fp16|b1_vs_b8'])} | "
          f"{pct(r['fp32|b1_vs_b8'])} | {pct(r['bf16|rerun'])} | {pct(r['bf16|b1_vs_b32'])} |")

print("\n## T2  is it padding, batch size, or who shares the batch\n")
print("| model | A padded ragged b1v8 | B equal-length no pad b1v8 | C same seq x8 vs alone | D two groupings of batch 8 |")
print("|---|---|---|---|---|")
for k in ORDER:
    f = f"{D}/pad_{k}.json"
    if not os.path.exists(f):
        continue
    r = json.load(open(f))["results"]
    print(f"| {SHORT[k]} | {pct(r['A pad ragged b1 vs b8'])} | {pct(r['B eq nopad b1 vs b8'])} | "
          f"{pct(r['C identical copies b8 vs b1'])} | {pct(r['D regrouped b8'])} |")

print("\n## T3  three ways to change the kernel path, in bf16 and fp32\n")
print("| model | change | bf16 | fp32 |")
print("|---|---|---|---|")
for k in ORDER:
    f = f"{D}/dtype_{k}.json"
    if not os.path.exists(f):
        continue
    r = json.load(open(f))["results"]
    for nm, key in (("batch shape [1,L] vs [8,L]", "batch"),
                    ("mask implicit vs explicit 4D", "mask"),
                    ("packing 1/row vs 8/row", "pack")):
        print(f"| {SHORT[k]} | {nm} | {pct(r['bf16|'+key])} | {pct(r['fp32|'+key])} |")

print("\n## T4  attention implementation and repeatability, bf16 batch 1 vs 8\n")
print("| model | implementation | out-of-band | 95% CI | per-seed | n |")
print("|---|---|---|---|---|---|")
for k in ORDER:
    f = f"{D}/robust_{k}.json"
    if not os.path.exists(f):
        continue
    r = json.load(open(f))["results"]
    for impl in ("eager", "sdpa", "flex_attention"):
        if impl not in r:
            continue
        v = r[impl]
        ps = ", ".join(f"{x:.2f}" for x in v["per_seed"])
        print(f"| {SHORT[k]} | {impl} | {v['rate']:.2f}% | [{v['ci'][0]:.2f}, {v['ci'][1]:.2f}] | {ps} | {v['n']} |")

print("\n## T5  does it reach the gradient: PPO clip-status flips\n")
print("| model | clipped at b1 | clipped at b8 | clip status flips | n |")
print("|---|---|---|---|---|")
for k in ORDER:
    f = f"{D}/grpo_{k}.json"
    if not os.path.exists(f):
        continue
    d = json.load(open(f))
    n = d["n"]
    print(f"| {SHORT[k]} | {100*d['clip_b1']/n:.2f}% | {100*d['clip_b8']/n:.2f}% | "
          f"**{100*d['flips']/n:.2f}%** | {n} |")

print("\n## T6  clip-flip rate depends on the advantage distribution\n")
print("| model | advantage | flips | 95% CI |")
print("|---|---|---|---|")
for k in ORDER:
    f = f"{D}/robust_{k}.json"
    if not os.path.exists(f):
        continue
    r = json.load(open(f))["results"]
    for a in [x for x in r if x.startswith("adv_")]:
        v = r[a]
        print(f"| {SHORT[k]} | {a[4:]} | {v['rate']:.2f}% | [{v['ci'][0]:.2f}, {v['ci'][1]:.2f}] |")


# ---- precision ladder (selective_*_x4*.json), per-layer attribution, fp16 overflow ----
import math as _math
def _wilson(p, n, z=1.96):
    p /= 100; c = p + z*z/(2*n); h = z*_math.sqrt(p*(1-p)/n + z*z/(4*n*n)); d = 1 + z*z/n
    return 100*(c-h)/d, 100*(c+h)/d
LADDER = ["baseline bf16", "head only (ScaleRL)", "last 8 + head", "first 8 + head", "guided top8 + head",
          "fp16 all (no fp32)", "fp16 + guided top8 + head", "all layers + head (sanity)"]
LSHORT = {"baseline bf16": "bf16", "head only (ScaleRL)": "fp32 head only", "last 8 + head": "last 8 + head",
          "first 8 + head": "first 8 + head", "guided top8 + head": "guided 8 + head", "fp16 all (no fp32)": "fp16, no fp32",
          "fp16 + guided top8 + head": "fp16 + guided 8", "all layers + head (sanity)": "all fp32 (sanity)"}
tags = sorted({os.path.basename(f)[len("selective_"):].split("_x4")[0] for f in glob.glob(f"{_RESULTS}/selective_*_x4*.json")})
print("\n## T7  precision ladder: where must fp32 go, bf16 batch 1 vs 8, 6,144 tokens\n")
print("| model | " + " | ".join(LSHORT[a] for a in LADDER) + " |")
print("|---|" + "---|" * len(LADDER))
for tag in tags:
    arms, n = {}, None
    for f in sorted(glob.glob(f"{_RESULTS}/selective_{tag}_x4*.json")):
        r = json.load(open(f)); n = r["n_tokens"]
        for a in r["arms"]:
            if not a.get("skipped"): arms[a["name"]] = a
    cells = []
    for a in LADDER:
        if a not in arms: cells.append("-"); continue
        lo, hi = _wilson(arms[a]["oob"], n)
        cells.append(f"{arms[a]['oob']:.2f}% ({100*arms[a]['fp32_frac']:.0f}%)")
    print(f"| {SHORT.get(tag, tag)} | " + " | ".join(cells) + " |")
print("\nOut-of-band rate, with the fraction of parameters held in fp32 in parentheses. 'guided 8' = the eight")
print("decoder layers with the largest per-layer divergence increment (T8), chosen on a separate 1,536-token calibration pass.")

bf = sorted(glob.glob(f"{_RESULTS}/selbench_*.json"))
if bf:
    print("\n## T7b  what each rung costs: teacher-forced scoring time relative to bf16, batch 8 x 512 tokens\n")
    names = ["base", "fp16", "head", "last8", "guided8", "full"]
    print("| model | " + " | ".join({"base": "bf16", "fp16": "fp16", "head": "fp32 head", "last8": "last 8 + head", "guided8": "guided 8 + head", "full": "all fp32"}[n] for n in names) + " | peak GiB bf16 -> all fp32 |")
    print("|---|" + "---|" * len(names) + "---|")
    for f in bf:
        r = json.load(open(f)); c = {x["name"]: x for x in r["configs"]}
        tag = r["model"].split("/")[-1]
        cells = [f"x{c[n]['rel_time']:.2f}" if n in c else "-" for n in names]
        print(f"| {SHORT.get(tag, tag)} | " + " | ".join(cells) + f" | {c['base']['peak_gib']:.1f} -> {c['full']['peak_gib']:.1f} |")
    print("\nWall time of the scoring forward including the fp32 log-softmax, 10 timed passes after 3 warm-up passes, on one RTX 4090")
    print("shared with another user's process at 30-40% utilisation; the ratios, not the absolute rates, are the measurement.")

print("\n## T8  where the divergence enters: b1-vs-b8 relative hidden-state divergence, by quarter of the stack\n")
print("| model | layers | Q1 | Q2 | Q3 | Q4 | after last layer | top-4 layers by increment |")
print("|---|---|---|---|---|---|---|---|")
for f in sorted(glob.glob(f"{_RESULTS}/attribution_*.json")):
    r = json.load(open(f)); v = [float(x) for x in r["per_layer_rel_div"]]; L = len(v) - 1
    d = [v[i + 1] - v[i] for i in range(L)]; q = max(1, L // 4)
    qs = [sum(d[k*q:(k+1)*q if k < 3 else L]) for k in range(4)]
    top = sorted(range(L), key=lambda i: -d[i])[:4]
    tag = os.path.basename(f)[len("attribution_"):-5]
    print(f"| {SHORT.get(tag, tag)} | {L} | " + " | ".join(f"{x:+.1e}" for x in qs) + f" | {v[-1]:.3f} | {top} |")
print("\nIncrement of the relative L2 divergence between the batch-1 and batch-8 hidden states, summed per quarter;")
print("a positive quarter is where the two batch shapes drift apart, a negative one is where later layers partly re-align them.")

fc = f"{_RESULTS}/fp16_check_3B_7B.json"
if os.path.exists(fc):
    print("\n## T9  does fp16 overflow on the larger models (8,192 tokens, 256-token continuations)\n")
    print("| model | fp16 non-finite log-probs | fp16 b1 vs b8 | fp16 vs bf16 at b1 | max abs hidden state (fp16 max 65,504) |")
    print("|---|---|---|---|---|")
    for m, r in json.load(open(fc)).items():
        tag = m.split("/")[-1]
        print(f"| {SHORT.get(tag, tag)} | {100*r['fp16_nonfinite']:.3f}% | {r['fp16_b1_vs_b8_oob']:.2f}% | {r['fp16_b1_vs_bf16_b1_oob']:.2f}% | {r['max_abs_hidden_fp16']:.0f} |")


# ---- downstream: the GRPO arms (results/q1/<arm>_s<seed>.json) ----
ARMN = {"A": "A default (bf16, trainer chunking)", "B": "B fp32 clone", "C": "C bf16, chunk = micro-batch",
        "D": "D bf16 + guided fp32 layers + head", "E": "E bf16 + fp32 head only", "F": "F fp16 clone"}
runs = {}
for f in sorted(glob.glob(f"{_RESULTS}/q1/*_s*.json")):
    r = json.load(open(f)); h = [x for x in r["log_history"] if "reward" in x]
    if len(h) < 30: continue
    rw = [float(x["reward"]) for x in h]; k = 30
    runs[(r["arm"], r["seed"])] = {"late": sum(rw[-k:]) / k, "auc": sum(rw) / len(rw),
        "clip": sum(float(x["clip_ratio/region_mean"]) for x in h) / len(h),
        "dl": sum(float(x["sampling/sampling_logp_difference/mean"]) for x in h) / len(h), "steps": len(h),
        "spm": r["wall_s"] / len(h)}
if runs:
    seeds = sorted({s for _, s in runs}); arms = sorted({a for a, _ in runs})
    print("\n## T10  does the old-log-prob precision reach the reward: GRPO on GSM8K, Qwen2.5-1.5B-Instruct, 150 steps\n")
    print("| arm | seeds | reward, last 30 steps | reward, mean over 150 | vs A, same seed | clip fraction | vLLM-vs-old abs dlogp | s/step |")
    print("|---|---|---|---|---|---|---|---|")
    for a in arms:
        rs = [runs[(a, s)] for s in seeds if (a, s) in runs]
        d = [runs[(a, s)]["late"] - runs[("A", s)]["late"] for s in seeds if (a, s) in runs and ("A", s) in runs]
        dv = ("-" if a == "A" or not d else " / ".join(f"{x:+.3f}" for x in d))
        m = lambda key: sum(x[key] for x in rs) / len(rs)
        print(f"| {ARMN.get(a, a)} | {len(rs)} | {m('late'):.3f} | {m('auc'):.3f} | {dv} | {m('clip'):.4f} | {m('dl'):.4f} | {m('spm'):.1f} |")
    print("\nSame seed means the same prompt order and the same vLLM sampling seed, so the arms start as near-replicas and")
    print("only the old-log-prob pass differs; the per-step reward noise is sd 0.14, so a 30-step mean carries an SE of about 0.026.")
