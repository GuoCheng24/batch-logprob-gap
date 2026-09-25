"""How much is lost by replaying only the SIZE of vLLM's sampling support instead of the ids?

top-k, top-p and min-p all keep a prefix of the tokens sorted by probability, so the kept set
S_t is "the |S_t| most likely tokens" under the rollout engine. If the trainer ranks tokens the
same way, its own top-|S_t| logits give the same set, and the kept mass can be computed from one
integer per position instead of a list of ids. This measures, on the same sampled tokens:

  exact : trainer log-prob renormalised over vLLM's returned ids (return_sampling_mask)
  size  : trainer log-prob renormalised over the trainer's own top-|S_t| tokens

and reports how often the two sets are identical, how far the two kept masses differ, what the
importance ratio looks like under each, and how many integers each needs per generated token.

usage: python support_size_replay.py [MODEL] [OUT_JSON]
"""
import json
import os
import sys

import torch

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
from datasets import load_dataset  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402
from vllm import LLM, SamplingParams  # noqa: E402

MODEL = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen2.5-1.5B-Instruct"
OUT = sys.argv[2] if len(sys.argv) > 2 else "support_size_replay.json"
N_PROMPTS, MAX_TOKENS = 64, 256

tok = AutoTokenizer.from_pretrained(MODEL)
ds = load_dataset("openai/gsm8k", "main")["test"].select(range(N_PROMPTS))
instr = 'Let\'s think step by step and output the final answer after "####".'
prompts = [tok.apply_chat_template([{"role": "user", "content": q + " " + instr}], tokenize=False,
                                   add_generation_prompt=True) for q in ds["question"]]

llm = LLM(MODEL, dtype="bfloat16", gpu_memory_utilization=0.22, max_model_len=1024, seed=0,
          enforce_eager=True, logprobs_mode="processed_logprobs", return_sampling_mask=True)
hf = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to("cuda:0").eval()

# vLLM needs a finite top_k to bound the replayed mask; 1024 leaves top_p / min_p in charge.
ARMS = [
    ("top_p=0.8, T=1.0", dict(temperature=1.0, top_p=0.8, top_k=1024)),
    ("top_p=0.95, T=1.0", dict(temperature=1.0, top_p=0.95, top_k=1024)),
    ("top_p=0.9, T=0.7", dict(temperature=0.7, top_p=0.9, top_k=1024)),
    ("top_k=20, T=1.0", dict(temperature=1.0, top_k=20)),
    ("min_p=0.05, T=1.0", dict(temperature=1.0, min_p=0.05, top_k=1024)),
    ("top_p=1.0, T=1.0 (control)", dict(temperature=1.0, top_p=1.0, top_k=1024)),
]


def q(x, p):
    return torch.quantile(x, p).item() if x.numel() else float("nan")


res = {"model": MODEL, "n_prompts": N_PROMPTS, "max_tokens": MAX_TOKENS, "vllm": __import__("vllm").__version__,
       "note": "trainer = HF transformers bf16 at batch 1; ratios are exp(trainer - vLLM processed logprob)",
       "arms": {}}
for name, sp in ARMS:
    T = sp["temperature"]
    outs = llm.generate(prompts, SamplingParams(max_tokens=MAX_TOKENS, logprobs=0, seed=0, **sp))
    lk_exact, lk_size, lp_full, lp_vllm, sizes, same = [], [], [], [], [], []
    for p, o in zip(prompts, outs):
        c = o.outputs[0]
        gen = list(c.token_ids)
        if not gen:
            continue
        ids = tok(p, add_special_tokens=False).input_ids
        v = [next(x.logprob for t, x in lp.items() if t == g) for lp, g in zip(c.logprobs, gen)]
        masks = [list(m) for m in c.sampling_mask.token_ids]
        x = torch.tensor([ids + gen], device="cuda:0")
        with torch.no_grad():
            logits = hf(x).logits[0, len(ids) - 1:len(ids) - 1 + len(gen)].float() / T
        lse = torch.logsumexp(logits, -1)
        kmax = max(len(m) for m in masks)
        top_vals, top_idx = torch.topk(logits, kmax, dim=-1)
        prefix = torch.logcumsumexp(top_vals, dim=-1)          # logsumexp of the top-k, for every k
        for j, (g, m) in enumerate(zip(gen, masks)):
            k = len(m)
            keep = torch.tensor(m, device="cuda:0")
            lk_exact.append((torch.logsumexp(logits[j, keep], 0) - lse[j]).item())
            lk_size.append((prefix[j, k - 1] - lse[j]).item())
            same.append(set(top_idx[j, :k].tolist()) == set(m))
            lp_full.append((logits[j, g] - lse[j]).item())
            lp_vllm.append(v[j])
            sizes.append(k)
    le, ls, lf, lv = (torch.tensor(a) for a in (lk_exact, lk_size, lp_full, lp_vllm))
    sz = torch.tensor(sizes, dtype=torch.float)
    diff = (le - ls).abs()
    oob = lambda r: 100 * ((r < 0.9) | (r > 1.1)).float().mean().item()  # noqa: E731
    r_none = torch.exp(lf - lv)              # no correction: full-vocab trainer vs processed vLLM
    r_exact = torch.exp(lf - le - lv)        # trainer renormalised over the exact kept set
    r_size = torch.exp(lf - ls - lv)         # trainer renormalised over its own top-|S|
    a = {"n_tokens": len(sizes),
         "kept_size": {"mean": sz.mean().item(), "median": q(sz, .5), "p99": q(sz, .99), "max": sz.max().item()},
         "kept_mass_exact_mean": le.exp().mean().item(),
         "sets_identical_pct": 100 * sum(same) / len(same),
         "abs_diff_log_kept_mass": {"mean": diff.mean().item(), "p99": q(diff, .99), "max": diff.max().item()},
         "ratio_no_correction": {"mean": r_none.mean().item(), "oob_pct": oob(r_none)},
         "ratio_exact_mask": {"mean": r_exact.mean().item(), "oob_pct": oob(r_exact)},
         "ratio_support_size": {"mean": r_size.mean().item(), "oob_pct": oob(r_size)},
         "ints_per_token": {"exact_mask": sz.mean().item(), "support_size": 1.0}}
    res["arms"][name] = a
    print(f"### {name}: {a['n_tokens']} tokens, |S| mean {a['kept_size']['mean']:.1f} (p99 {a['kept_size']['p99']:.0f}), "
          f"kept mass {a['kept_mass_exact_mean']:.3f}\n"
          f"  sets identical {a['sets_identical_pct']:.2f}%  |dlog kept mass| mean {a['abs_diff_log_kept_mass']['mean']:.2e} "
          f"p99 {a['abs_diff_log_kept_mass']['p99']:.2e} max {a['abs_diff_log_kept_mass']['max']:.2e}\n"
          f"  ratio mean / out-of-band: none {a['ratio_no_correction']['mean']:.3f}/{a['ratio_no_correction']['oob_pct']:.1f}%  "
          f"exact {a['ratio_exact_mask']['mean']:.3f}/{a['ratio_exact_mask']['oob_pct']:.1f}%  "
          f"size {a['ratio_support_size']['mean']:.3f}/{a['ratio_support_size']['oob_pct']:.1f}%", flush=True)

json.dump(res, open(OUT, "w"), indent=1)
print("SUPPORT_SIZE_DONE")
