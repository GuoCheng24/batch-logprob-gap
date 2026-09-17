"""Feasibility gate: can we align vLLM's rollout logprobs with a trainer-side forward pass?

RL frameworks that generate with vLLM and train with HF compute the same token's log
probability twice, in two different kernels. GRPO's importance ratio is exp of that
difference, so any systematic gap is an off-policy bias the objective does not know about.
This checks only that both numbers are obtainable for the same token ids, and prints the
raw gap for one small batch. No sweep, no claims.
"""
import os as _os
_RESULTS = _os.environ.get("BLG_DIR") or _os.path.join(
    _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "results")
_os.makedirs(_RESULTS, exist_ok=True)
import os, sys, json
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
import torch
from vllm import LLM, SamplingParams

M = "Qwen/Qwen2.5-0.5B-Instruct"
PROMPTS = [
    "Explain in two sentences why the sky is blue.",
    "List three prime numbers greater than 100.",
    "Write one sentence about the sea.",
    "What is the capital of France? Answer in one word.",
]

llm = LLM(model=M, gpu_memory_utilization=0.30, max_model_len=2048,
          enforce_eager=True, enable_prefix_caching=False, dtype="bfloat16")
sp = SamplingParams(temperature=1.0, top_p=1.0, max_tokens=48, seed=1234, logprobs=0)
outs = llm.generate(PROMPTS, sp)

rows = []
for i, o in enumerate(outs):
    c = o.outputs[0]
    lp = []
    for pos, tid in enumerate(c.token_ids):
        d = c.logprobs[pos] if c.logprobs else None
        lp.append(d[tid].logprob if d and tid in d else float("nan"))
    rows.append({"prompt_ids": list(o.prompt_token_ids), "gen_ids": list(c.token_ids),
                 "vllm_logprobs": lp})
    print(f"  [{i}] prompt {len(o.prompt_token_ids)} tok, gen {len(c.token_ids)} tok, "
          f"vllm logprobs {len(lp)}, first 3 = {[round(x,4) for x in lp[:3]]}")

json.dump(rows, open(_RESULTS + "/rollout.json", "w"))
print(f"\n  saved {len(rows)} sequences")
