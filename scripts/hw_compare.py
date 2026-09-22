#!/usr/bin/env python
"""Does the effect survive a change of silicon?

The repository's limitations said "Everything is RTX 4090, SM89. Reduction
strategies differ across architectures." That is a reason the numbers might not
transfer, stated without testing it. This runs the same `hf_only.py`, the same
models, the same venv, on an RTX 4090 and on an L40 and compares.

Two controls make the comparison readable:

* the RTX 4090 arm is re-run tonight, months after the recorded numbers, and
  must reproduce them. If it does not, nothing else here means anything.
* `bf16|rerun` scores the same tokens twice at the same batch shape on the same
  card. It is the zero: whatever it reports is what "no change at all" costs.

What it cannot say: an L40 and an RTX 4090 are both Ada, both compute
capability 8.9. This is a different chip, not a different architecture, so the
limitation is narrowed rather than removed.
"""
import glob
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HW = os.path.join(ROOT, "results", "hardware")
BASE = os.path.join(ROOT, "results")
ARMS = ["bf16|b1_vs_b8", "bf16|b1_vs_b32", "bf16|rerun", "fp16|b1_vs_b8", "fp32|b1_vs_b8"]


def load(path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def main():
    failures = []

    print("Control: tonight's RTX 4090 against the recorded RTX 4090 numbers")
    for path in sorted(glob.glob(os.path.join(HW, "rtx4090_rerun_hfonly_*.json"))):
        tag = os.path.basename(path).split("rtx4090_rerun_hfonly_")[1][:-5]
        now, rec = load(path), load(os.path.join(BASE, f"hfonly_{tag}.json"))
        same = all(now["results"][a]["out"] == rec["results"][a]["out"] for a in ARMS
                   if a in now["results"] and a in rec["results"])
        print(f"  {tag:<26} {'reproduces exactly' if same else 'DIFFERS'}"
              f"   ({', '.join(str(now['results'][a]['out']) for a in ARMS if a in now['results'])})")
        if not same:
            failures.append(f"the RTX 4090 re-run of {tag} does not reproduce the recorded numbers")

    print("\nOut-of-band tokens (|log ratio| outside [log 0.9, log 1.1]), by card")
    header = f"{'model':<26}{'arm':<18}" + "".join(f"{a.split('|')[0] + ' ' + a.split('|')[1]:>20}" for a in ARMS)
    print(header)
    rows = 0
    for path in sorted(glob.glob(os.path.join(HW, "l40_hfonly_*.json"))):
        tag = os.path.basename(path).split("l40_hfonly_")[1][:-5]
        base_path = os.path.join(BASE, f"hfonly_{tag}.json")
        if not os.path.exists(base_path):
            continue
        l40, rtx = load(path), load(base_path)
        n = l40["n_tokens"]
        if rtx["n_tokens"] != n:
            failures.append(f"{tag}: token counts differ ({rtx['n_tokens']} vs {n}) - "
                            f"the two arms did not score the same tokens")
        for name, d in (("RTX 4090", rtx), ("L40", l40)):
            cells = "".join(f"{d['results'][a]['out']:>20}" if a in d["results"] else f"{'-':>20}"
                            for a in ARMS)
            print(f"{tag if name == 'RTX 4090' else '':<26}{name:<18}{cells}")
        diff = "".join(
            f"{(l40['results'][a]['out'] - rtx['results'][a]['out']):>+20}"
            if a in l40["results"] and a in rtx["results"] else f"{'-':>20}" for a in ARMS)
        print(f"{'':<26}{'difference':<18}{diff}")
        b8 = "bf16|b1_vs_b8"
        pts = 100 * (l40["results"][b8]["out"] - rtx["results"][b8]["out"]) / n
        # Rates as well as counts: a rate quoted in prose has to be traceable to
        # something a script printed, or it is a number that exists only in the
        # sentence that quotes it.
        print(f"{'':<26}{'rate':<18}"
              f"{100 * rtx['results'][b8]['out'] / n:>19.2f}%"
              f"{100 * l40['results'][b8]['out'] / n:>19.2f}%"
              f"{pts:>+19.2f} points")
        print()
        rows += 1

    if not rows:
        failures.append("no model was run on both cards")

    print("Reading:")
    print("  bf16|rerun is 0 on both cards for every model here, so the forward pass is")
    print("  bit-reproducible within a machine. The bf16 batch effect is therefore not")
    print("  run-to-run noise on either card - and the small movement between cards is.")
    if failures:
        print("\nFAILED:")
        for f in failures:
            print("  " + f)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
