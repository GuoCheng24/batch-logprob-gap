"""How much of GRPO's importance ratio is bf16 batch non-associativity?

For every sampled token the trainer recomputes a log probability the rollout engine already
produced. With an unchanged policy the ratio exp(logp_train - logp_rollout) should be 1.
This measures what it actually is, and splits the deviation into the factors that can
produce it without the policy moving: trainer precision, trainer batch composition, and the
rollout engine's own batch-variant kernels.
"""

import json
import math
import os
import os as _os
import statistics as st

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

M = "Qwen/Qwen2.5-0.5B-Instruct"
D = _os.path.join(
    _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "results"
)
tok = AutoTokenizer.from_pretrained(M)
PAD = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
rows = json.load(open(f"{D}/rollout_BI0.json"))
# teacher-forced scores of the SAME tokens under each kernel set
scored = {bi: json.load(open(f"{D}/scored_BI{bi}.json")) for bi in ("0", "1")}
for bi in ("0", "1"):
    assert [len(x) for x in scored[bi]] == [len(r["gen_ids"]) for r in rows], (
        f"BI={bi} length mismatch"
    )


def trainer_lp(model, batch):
    out = []
    for s in range(0, len(rows), batch):
        ch = rows[s : s + batch]
        seqs = [r["prompt_ids"] + r["gen_ids"] for r in ch]
        L = max(len(x) for x in seqs)
        ids = torch.tensor([[PAD] * (L - len(x)) + x for x in seqs], device="cuda:0")
        att = torch.tensor(
            [[0] * (L - len(x)) + [1] * len(x) for x in seqs], device="cuda:0"
        )
        with torch.no_grad():
            lg = model(ids, attention_mask=att).logits
        lp = torch.log_softmax(lg.float(), dim=-1)
        for j, r in enumerate(ch):
            off = L - len(seqs[j])
            np_ = off + len(r["prompt_ids"])
            out.append(
                [lp[j, np_ + k - 1, t].item() for k, t in enumerate(r["gen_ids"])]
            )
    return out


def rep(name, a, b):
    g = [x - y for ra, rb in zip(a, b) for x, y in zip(ra, rb)]
    n = len(g)
    rt = [math.exp(v) for v in g]
    ab = sorted(abs(v) for v in g)
    print(
        f"  {name:<46} mean {sum(g) / n:+.6f}  sd {st.pstdev(g):.6f}  "
        f"p95|gap| {ab[int(0.95 * n)]:.5f}  max {ab[-1]:.5f}  "
        f"out[0.9,1.1] {sum(1 for x in rt if x < 0.9 or x > 1.1):>4}/{n} "
        f"({100 * sum(1 for x in rt if x < 0.9 or x > 1.1) / n:.1f}%)"
    )
    return {
        "mean": sum(g) / n,
        "sd": st.pstdev(g),
        "p95": ab[int(0.95 * n)],
        "max": ab[-1],
        "out": sum(1 for x in rt if x < 0.9 or x > 1.1),
        "n": n,
    }


R = {}
print("loading bf16 ...", flush=True)
m = AutoModelForCausalLM.from_pretrained(M, dtype=torch.bfloat16).to("cuda:0").eval()
bf1, bf8, bf64 = trainer_lp(m, 1), trainer_lp(m, 8), trainer_lp(m, 64)
del m
torch.cuda.empty_cache()
print("loading fp32 ...", flush=True)
m = AutoModelForCausalLM.from_pretrained(M, dtype=torch.float32).to("cuda:0").eval()
fp1, fp8 = trainer_lp(m, 1), trainer_lp(m, 8)
del m
torch.cuda.empty_cache()
v_gen = [r["vllm_logprobs"] for r in rows]  # as produced during generation, BI=0
v0, v1 = scored["0"], scored["1"]  # teacher-forced re-score, BI off / on

print("\n== trainer against itself: no rollout engine involved ==")
R["fp32 b1 vs b8"] = rep("fp32 trainer, batch 1 vs 8", fp1, fp8)
R["bf16 b1 vs b8"] = rep("bf16 trainer, batch 1 vs 8", bf1, bf8)
R["bf16 b1 vs b64"] = rep("bf16 trainer, batch 1 vs 64", bf1, bf64)
R["bf16 vs fp32"] = rep("trainer batch 1, bf16 vs fp32", bf1, fp1)

print("\n== rollout engine against itself ==")
R["vllm BI0 vs BI1"] = rep("vLLM rescore, batch-invariant off vs on", v0, v1)
R["vllm gen vs rescore"] = rep("vLLM generation vs its own rescore (BI=0)", v_gen, v0)

print("\n== the ratio GRPO actually applies ==")
R["gen vs bf16 b8"] = rep("vLLM generation  vs  bf16 trainer b8", bf8, v_gen)
R["BI0 vs bf16 b8"] = rep("vLLM rescore BI=0  vs  bf16 trainer b8", bf8, v0)
R["BI1 vs bf16 b8"] = rep("vLLM BI=1  vs  bf16 trainer b8", bf8, v1)
R["BI0 vs fp32 b8"] = rep("vLLM BI=0  vs  fp32 trainer b8", fp8, v0)
R["BI1 vs fp32 b8"] = rep("vLLM BI=1  vs  fp32 trainer b8", fp8, v1)

json.dump(R, open(f"{D}/kernel_arms.json", "w"), indent=1)
print(f"\n  tokens per comparison: {R['bf16 b1 vs b8']['n']}")
