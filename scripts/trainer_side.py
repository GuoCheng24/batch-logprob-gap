"""Trainer-side logprobs for the exact token ids vLLM already sampled.

This is the second half of the gate. GRPO computes the importance ratio as
exp(logp_train - logp_rollout) on the same tokens; if the two kernels disagree the ratio
is not 1 even when the policy has not moved. Here the policy is identical by construction
(same checkpoint, no optimizer step), so any non-zero gap is implementation, not policy.
"""
import os as _os
_RESULTS = _os.environ.get("BLG_DIR") or _os.path.join(
    _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "results")
_os.makedirs(_RESULTS, exist_ok=True)
import os, json, math
os.environ.setdefault("HF_HUB_OFFLINE", "1"); os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

M = "Qwen/Qwen2.5-0.5B-Instruct"
D = _RESULTS
rows = json.load(open(f"{D}/rollout.json"))

tok = AutoTokenizer.from_pretrained(M)
model = AutoModelForCausalLM.from_pretrained(M, dtype=torch.bfloat16).to("cuda:0")
model.eval()

out = []
for i, r in enumerate(rows):
    ids = r["prompt_ids"] + r["gen_ids"]
    x = torch.tensor([ids], device="cuda:0")
    with torch.no_grad():
        logits = model(x).logits.float()          # [1, T, V]
    logp = torch.log_softmax(logits, dim=-1)
    np_ = len(r["prompt_ids"])
    # token at position np_+k is predicted by logits at np_+k-1
    tr = [logp[0, np_ + k - 1, t].item() for k, t in enumerate(r["gen_ids"])]
    gaps = [a - b for a, b in zip(tr, r["vllm_logprobs"])]
    ratios = [math.exp(g) for g in gaps]
    out.append({**r, "train_logprobs": tr, "gap": gaps, "ratio": ratios})
    am = max(abs(g) for g in gaps); mg = sum(gaps) / len(gaps)
    print(f"  [{i}] n={len(gaps)}  mean gap {mg:+.5f}  max|gap| {am:.5f}  "
          f"ratio range [{min(ratios):.4f}, {max(ratios):.4f}]")

json.dump(out, open(f"{D}/gate_pairs.json", "w"))
allg = [g for r in out for g in r["gap"]]
allr = [x for r in out for x in r["ratio"]]
allg.sort()
n = len(allg)
print(f"\n  all {n} tokens: mean {sum(allg)/n:+.6f}  median {allg[n//2]:+.6f}  "
      f"max|gap| {max(abs(g) for g in allg):.6f}")
print(f"  |gap| > 0.01: {sum(1 for g in allg if abs(g)>0.01)}/{n}   "
      f"> 0.1: {sum(1 for g in allg if abs(g)>0.1)}/{n}")
print(f"  ratio: min {min(allr):.4f}  max {max(allr):.4f}  "
      f"outside [0.9,1.1]: {sum(1 for x in allr if x<0.9 or x>1.1)}/{n}")
