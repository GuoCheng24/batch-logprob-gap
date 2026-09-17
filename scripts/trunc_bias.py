"""Reproduce the truncated-support bias in the vLLM importance-sampling ratio and show the fix.

vLLM (processed_logprobs, top_p) returns log-probs normalised over the nucleus. The trainer recomputes
log-probs over the full vocabulary. The ratio exp(trainer - vllm) then carries -log(kept mass) per
token, which is not a policy change. With return_sampling_mask=True vLLM also returns the kept token
ids per position; renormalising the trainer log-prob over that set removes the term. Both are measured
here on the same sampled tokens, and a control with top_p=1.0 shows the gap collapse."""
import os as _os
_RESULTS = _os.environ.get("BLG_DIR") or _os.path.join(
    _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "results")
import os, sys, json, math, torch
os.environ.setdefault("HF_HUB_OFFLINE", "1"); os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
from vllm import LLM, SamplingParams
from transformers import AutoModelForCausalLM, AutoTokenizer
MODEL = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen2.5-1.5B-Instruct"
OUT = _RESULTS
PROMPTS = [f"{s} {t}." for s in ["Explain in two sentences why", "List three facts about", "Write one sentence about", "Give a short definition of"]
           for t in ["the ocean", "photosynthesis", "gravity", "the printing press", "antibiotics", "monsoons", "the transistor", "vaccination"]]
tok = AutoTokenizer.from_pretrained(MODEL)
llm = LLM(MODEL, dtype="bfloat16", gpu_memory_utilization=0.22, max_model_len=512, seed=0, enforce_eager=True,
          logprobs_mode="processed_logprobs", return_sampling_mask=True)
hf = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to("cuda:0").eval()
def mask_lists(sm):
    """Per generated position, the kept token ids, whatever layout vLLM's SamplingMask uses."""
    if hasattr(sm, "to_nested_list"): return sm.to_nested_list()
    if isinstance(getattr(sm, "token_ids", None), list): return [list(x) for x in sm.token_ids]   # vllm 0.28: dataclass SamplingMask(token_ids: list[list[int]])
    for attr in ("token_ids_per_step", "kept_token_ids", "masks", "lists", "data", "nested"):
        if hasattr(sm, attr): return [list(x) for x in getattr(sm, attr)]
    if hasattr(sm, "token_ids") and hasattr(sm, "offsets"):
        t, o = sm.token_ids, sm.offsets; return [list(t[int(o[i]):int(o[i + 1])]) for i in range(len(o) - 1)]
    if hasattr(sm, "__getitem__") and hasattr(sm, "__len__"): return [list(sm[i]) for i in range(len(sm))]
    raise TypeError(f"unknown SamplingMask layout: {[a for a in dir(sm) if not a.startswith('_')]}")
res = {"model": MODEL, "arms": {}}
# vLLM bounds the replayed mask with top_k (input_processor.py: "requires top_k > 0"); a large cap keeps top_p in charge
for name, sp in [("top_p=0.8, top_k=1024", dict(top_p=0.8, top_k=1024)), ("top_k=20", dict(top_k=20)), ("top_p=1.0, top_k=1024 (control)", dict(top_p=1.0, top_k=1024))]:
    T = 1.0
    outs = llm.generate(PROMPTS, SamplingParams(temperature=T, max_tokens=64, min_tokens=64, logprobs=0, seed=0, **sp))
    full, kept, vl, ntok, mass = [], [], [], 0, []
    for p, o in zip(PROMPTS, outs):
        c = o.outputs[0]; ids = tok(p).input_ids; gen = list(c.token_ids)
        v = [list(lp.values())[0].logprob if isinstance(lp, dict) else lp for lp in c.logprobs]  # sampled-token logprob per step
        v = [next(x.logprob for t, x in lp.items() if t == g) for lp, g in zip(c.logprobs, gen)]
        masks = mask_lists(c.sampling_mask)
        x = torch.tensor([ids + gen], device="cuda:0")
        with torch.no_grad(): logits = hf(x).logits[0, len(ids) - 1:len(ids) - 1 + len(gen)].float() / T
        lsm = torch.log_softmax(logits, -1)
        for j, (g, m) in enumerate(zip(gen, masks)):
            m = list(m) if not isinstance(m, list) else m
            keep = torch.tensor(m, device="cuda:0")
            full.append(lsm[j, g].item())
            kept.append((logits[j, g] - torch.logsumexp(logits[j, keep], 0)).item())
            vl.append(v[j]); mass.append(torch.logsumexp(lsm[j, keep], 0).exp().item()); ntok += 1
    f, k, vv = torch.tensor(full), torch.tensor(kept), torch.tensor(vl)
    r_full = torch.exp(f - vv); r_kept = torch.exp(k - vv)
    oob = lambda r: 100 * ((r < 0.9) | (r > 1.1)).float().mean().item()
    res["arms"][name] = {"n_tokens": ntok, "mean_kept_mass": sum(mass) / len(mass), "mean_kept_size": None,
        "ratio_full_vocab": {"mean": r_full.mean().item(), "median": r_full.median().item(), "oob": oob(r_full)},
        "ratio_kept_set": {"mean": r_kept.mean().item(), "median": r_kept.median().item(), "oob": oob(r_kept)},
        "abs_dlogp_full": (f - vv).abs().mean().item(), "abs_dlogp_kept": (k - vv).abs().mean().item()}
    a = res["arms"][name]
    print(f"### {name}: {ntok} tokens, mean kept mass {a['mean_kept_mass']:.3f}\n"
          f"  full-vocab trainer logp vs vLLM: ratio mean {a['ratio_full_vocab']['mean']:.3f} median {a['ratio_full_vocab']['median']:.3f} out-of-band {a['ratio_full_vocab']['oob']:.1f}%  |dlogp| {a['abs_dlogp_full']:.4f}\n"
          f"  kept-set renormalised       : ratio mean {a['ratio_kept_set']['mean']:.3f} median {a['ratio_kept_set']['median']:.3f} out-of-band {a['ratio_kept_set']['oob']:.1f}%  |dlogp| {a['abs_dlogp_kept']:.4f}", flush=True)
json.dump(res, open(f"{OUT}/trunc_bias_{MODEL.split('/')[-1]}.json", "w"), indent=1); print("TRUNC_DONE")
