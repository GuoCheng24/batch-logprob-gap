"""Rule out the mundane explanations before attributing anything to a train/inference gap.

The gate showed importance ratios in [0.84, 1.25] on an unchanged policy. Four things could
produce that without any vLLM-vs-HF implementation difference:

  C1  trainer-side batch composition   HF forward, batch 1 vs padded batch of 4
  C2  trainer-side precision           HF bf16 vs HF fp32, identical inputs
  C3  rollout reproducibility          vLLM generated twice with the same seed
  C4  trainer-side determinism         HF forward run twice, identical inputs

Each control is a gap computed against a reference that shares everything except the one
factor. Whatever survives all four is the quantity GRPO's importance ratio actually carries.
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


def logprobs_for(model, rows, batch, dtype_cast_fp32):
    """Return per-row list of trainer logprobs for the already-sampled gen_ids."""
    out = []
    for s in range(0, len(rows), batch):
        chunk = rows[s:s + batch]
        seqs = [r["prompt_ids"] + r["gen_ids"] for r in chunk]
        L = max(len(x) for x in seqs)
        pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
        # left-pad so the last real token keeps its position relative to the end
        ids = torch.tensor([[pad] * (L - len(x)) + x for x in seqs], device="cuda:0")
        att = torch.tensor([[0] * (L - len(x)) + [1] * len(x) for x in seqs], device="cuda:0")
        with torch.no_grad():
            lg = model(ids, attention_mask=att).logits
        lg = lg.float() if dtype_cast_fp32 else lg
        lp = torch.log_softmax(lg.float(), dim=-1)
        for j, r in enumerate(chunk):
            off = L - len(seqs[j])
            np_ = off + len(r["prompt_ids"])
            out.append([lp[j, np_ + k - 1, t].item() for k, t in enumerate(r["gen_ids"])])
    return out


def stats(name, a, b):
    g = [x - y for ra, rb in zip(a, b) for x, y in zip(ra, rb)]
    n = len(g)
    ratios = [math.exp(v) for v in g]
    amax = max(abs(v) for v in g)
    print(f"  {name:<42} n={n}  mean {sum(g)/n:+.6f}  max|gap| {amax:.6f}  "
          f"|gap|>0.01 {sum(1 for v in g if abs(v)>0.01):>3}/{n}  "
          f"ratio outside[0.9,1.1] {sum(1 for x in ratios if x<0.9 or x>1.1):>3}/{n}")
    return g


print("loading bf16 ...", flush=True)
m_bf = AutoModelForCausalLM.from_pretrained(M, dtype=torch.bfloat16).to("cuda:0").eval()
bf_b1 = logprobs_for(m_bf, rows, 1, False)
bf_b1_again = logprobs_for(m_bf, rows, 1, False)
bf_b4 = logprobs_for(m_bf, rows, 4, False)
del m_bf; torch.cuda.empty_cache()

print("loading fp32 ...", flush=True)
m_fp = AutoModelForCausalLM.from_pretrained(M, dtype=torch.float32).to("cuda:0").eval()
fp_b1 = logprobs_for(m_fp, rows, 1, False)
fp_b4 = logprobs_for(m_fp, rows, 4, False)
del m_fp; torch.cuda.empty_cache()

vllm = [r["vllm_logprobs"] for r in rows]

print("\n== controls: trainer vs trainer (no vLLM involved) ==")
stats("C4 HF bf16 b=1 run1 vs run2", bf_b1, bf_b1_again)
stats("C1 HF bf16 b=1 vs b=4 (batch composition)", bf_b1, bf_b4)
stats("C2 HF b=1: bf16 vs fp32 (precision)", bf_b1, fp_b1)
stats("C1+C2 HF fp32 b=1 vs b=4", fp_b1, fp_b4)

print("\n== the quantity GRPO actually uses ==")
stats("vLLM bf16 rollout vs HF bf16 b=1", bf_b1, vllm)
stats("vLLM bf16 rollout vs HF bf16 b=4", bf_b4, vllm)
stats("vLLM bf16 rollout vs HF fp32 b=1", fp_b1, vllm)

json.dump({"bf_b1": bf_b1, "bf_b1_again": bf_b1_again, "bf_b4": bf_b4,
           "fp_b1": fp_b1, "fp_b4": fp_b4, "vllm": vllm},
          open(f"{D}/controls.json", "w"))
print(f"\n  saved {D}/controls.json")
