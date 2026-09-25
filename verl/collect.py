"""Turn verl console logs into the per-step metrics this directory quotes.

    python collect.py OUT.json LOG [LOG ...]

Each LOG is keyed by its file name without extension. Only the metrics the README
uses are kept; every one of them is copied as verl printed it.
"""
import json
import os
import re
import sys

KEEP = ("rollout_corr/rollout_is_mean", "rollout_corr/kl", "rollout_corr/rollout_rs_seq_masked_fraction",
        "rollout_corr/rollout_rs_masked_fraction", "rollout_corr/rollout_is_seq_mean",
        "actor/grad_norm", "actor/pg_loss", "critic/score/mean",
        "critic/advantages/max", "critic/advantages/min", "response_length/mean")
STEP = re.compile(r"step:(\d+) - (.*)")
KV = re.compile(r"([A-Za-z_]+/[A-Za-z_/]+):(-?[0-9.]+(?:e[-+]?\d+)?)")

out = {}
for path in sys.argv[2:]:
    steps = {}
    with open(path, errors="replace") as fh:
        for line in fh:
            m = STEP.search(line)
            if not m:
                continue
            got = {k: float(v) for k, v in KV.findall(m.group(2)) if k in KEEP}
            if got:
                steps.setdefault(m.group(1), {}).update(got)
    out[os.path.splitext(os.path.basename(path))[0]] = steps
with open(sys.argv[1], "w") as fh:
    json.dump(out, fh, indent=1, sort_keys=True)
print(f"{sys.argv[1]}: {len(out)} runs")
