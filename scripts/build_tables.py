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
