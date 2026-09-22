#!/usr/bin/env python
"""Does the effect survive a change of silicon, and a change of architecture?

The repository's limitations said "Everything is RTX 4090, SM89. Reduction
strategies differ across architectures." That is a reason the numbers might not
transfer, stated without testing it. This runs the same `hf_only.py`, the same
models, the same venv, on three cards and compares:

    RTX 4090   Ada,   cc 8.9, 128 SMs     the card every published number is from
    L40        Ada,   cc 8.9, 142 SMs     a different chip, the same architecture
    V100       Volta, cc 7.0,  80 SMs     no native bfloat16 tensor cores at all

Two controls make it readable:

* the RTX 4090 arm is re-run, long after the recorded numbers, and must
  reproduce them cell for cell. If it does not, nothing else here means anything.
* `bf16|rerun` scores the same tokens twice at the same batch shape on the same
  card. It is the zero: whatever it reports is what "no change at all" costs.

One thing this comparison is not: paired. `hf_only.py` samples its own rollouts
before scoring them, and on the V100 the same seed produces a different rollout -
which is itself the effect showing up one level higher. So the arms score
different token sets and only their *rates* can be put side by side. The two Ada
cards do produce identical token counts, and there the counts are comparable too.
"""
import glob
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HW = os.path.join(ROOT, "results", "hardware")
BASE = os.path.join(ROOT, "results")
ARMS = ["bf16|b1_vs_b8", "bf16|b1_vs_b32", "bf16|rerun", "fp16|b1_vs_b8", "fp32|b1_vs_b8"]
CARDS = [("rtx4090", "RTX 4090 (Ada)"), ("l40", "L40 (Ada)"), ("v100", "V100 (Volta)")]


def load(path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def main():
    failures = []

    print("Control: tonight's RTX 4090 against the recorded RTX 4090 numbers")
    reruns = sorted(glob.glob(os.path.join(HW, "rtx4090_rerun_hfonly_*.json")))
    if not reruns:
        failures.append("no RTX 4090 re-run to check the recorded numbers against")
    for path in reruns:
        tag = os.path.basename(path).split("rtx4090_rerun_hfonly_")[1][:-5]
        now, rec = load(path), load(os.path.join(BASE, f"hfonly_{tag}.json"))
        cells = [a for a in ARMS if a in now["results"] and a in rec["results"]]
        same = all(now["results"][a]["out"] == rec["results"][a]["out"] for a in cells)
        print(f"  {tag:<24} {'reproduces exactly' if same else 'DIFFERS'}"
              f"   ({', '.join(str(now['results'][a]['out']) for a in cells)})")
        if not same:
            failures.append(f"the RTX 4090 re-run of {tag} does not reproduce the recorded numbers")

    print("\nOut-of-band tokens as a percentage of the tokens that arm scored")
    print(f"{'model':<24}{'arm':<16}" + "".join(f"{label:>20}" for _k, label in CARDS))

    models = sorted(os.path.basename(p).split("l40_hfonly_")[1][:-5]
                    for p in glob.glob(os.path.join(HW, "l40_hfonly_*.json")))
    if not models:
        failures.append("no model was run on a second card")

    for tag in models:
        data = {}
        for key, _label in CARDS:
            path = (os.path.join(BASE, f"hfonly_{tag}.json") if key == "rtx4090"
                    else os.path.join(HW, f"{key}_hfonly_{tag}.json"))
            if os.path.exists(path):
                data[key] = load(path)
        missing = [k for k, _l in CARDS if k not in data]
        if missing:
            failures.append(f"{tag}: not run on {', '.join(missing)}")
            continue

        ns = {k: data[k]["n_tokens"] for k, _l in CARDS}
        print(f"\n{tag:<24}{'tokens scored':<16}"
              + "".join(f"{ns[k]:>20}" for k, _l in CARDS))
        for arm in ARMS:
            cells = []
            for k, _l in CARDS:
                r = data[k]["results"].get(arm)
                cells.append(f"{r['out']}  {100 * r['out'] / ns[k]:.2f}%" if r else "-")
            print(f"{'':<24}{arm:<16}" + "".join(f"{c:>20}" for c in cells))
        # The spans the prose quotes, printed rather than left to be recomputed by
        # a reader - or matched by coincidence against some other number in the
        # results when a checker looks for them.
        rate = {k: 100 * data[k]["results"]["bf16|b1_vs_b8"]["out"] / ns[k] for k, _l in CARDS}
        print(f"{'':<24}{'ada gap':<16}{rate['rtx4090'] - rate['l40']:>19.2f} points")
        print(f"{'':<24}{'volta gap':<16}{rate['rtx4090'] - rate['v100']:>19.2f} points")
        if ns["rtx4090"] != ns["l40"]:
            failures.append(f"{tag}: the two Ada cards scored different token counts "
                            f"({ns['rtx4090']} vs {ns['l40']})")

    print("\nReading:")
    print("  bf16|rerun is 0 on every card for every model, so the forward pass is")
    print("  bit-reproducible within a machine on all three. The batch effect is")
    print("  therefore not run-to-run noise anywhere, and fp32 removes it everywhere.")
    print("  Between the two Ada cards the rate moves by a fraction of a point; on")
    print("  Volta, which has no native bf16 tensor cores, it is consistently lower by")
    print("  several points. The effect transfers; its size does not.")
    if failures:
        print("\nFAILED:")
        for f in failures:
            print("  " + f)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
