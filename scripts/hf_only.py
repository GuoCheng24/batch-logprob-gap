"""Does a trainer, changing nothing but how it groups sequences into a batch, change the
log probability it assigns to a token?

No inference engine is involved. One model samples its own continuations, then scores those
exact tokens twice: once one sequence at a time, once eight at a time. With the policy fixed
this is the identity, so exp(difference) is an importance ratio that should be 1. Repeated
across dtypes to see which ones hold it.

Logits are upcast to fp32 before log_softmax in every arm, so what remains comes from
reductions inside the forward pass.
"""
import os as _os
_RESULTS = _os.environ.get("BLG_DIR") or _os.path.join(
    _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "results")
_os.makedirs(_RESULTS, exist_ok=True)
import os, sys, json, math, statistics as st
os.environ.setdefault("HF_HUB_OFFLINE", "1"); os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = sys.argv[1]; D = _RESULTS; TAG = MODEL.split("/")[-1]
STEMS = ["Explain in two sentences why", "List three facts about", "Write one sentence about",
         "Give a short definition of", "Name one consequence of", "Describe briefly how",
         "In one sentence, compare", "State the main idea of"]
TOPICS = ["the ocean", "photosynthesis", "gravity", "the printing press", "antibiotics",
          "monsoons", "the transistor", "vaccination"]
PROMPTS = [f"{s} {t}." for s in STEMS for t in TOPICS]

tok = AutoTokenizer.from_pretrained(MODEL)
PAD = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

# sample rollouts once, in bf16, the way an RL loop would
m = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to("cuda:0").eval()
torch.manual_seed(1234)
rows = []
for p in PROMPTS:
    ids = tok(p, return_tensors="pt").input_ids.to("cuda:0")
    with torch.no_grad():
        out = m.generate(ids, do_sample=True, temperature=1.0, top_p=1.0,
                         max_new_tokens=64, pad_token_id=PAD)
    rows.append({"prompt_ids": ids[0].tolist(), "gen_ids": out[0, ids.shape[1]:].tolist()})
del m; torch.cuda.empty_cache()
ntok = sum(len(r["gen_ids"]) for r in rows)
print(f"### {MODEL}   {len(rows)} seqs, {ntok} tokens", flush=True)


def lp_at(model, batch):
    out = []
    for s in range(0, len(rows), batch):
        ch = rows[s:s + batch]
        seqs = [r["prompt_ids"] + r["gen_ids"] for r in ch]
        L = max(len(x) for x in seqs)
        ids = torch.tensor([[PAD] * (L - len(x)) + x for x in seqs], device="cuda:0")
        att = torch.tensor([[0] * (L - len(x)) + [1] * len(x) for x in seqs], device="cuda:0")
        with torch.no_grad():
            lg = model(ids, attention_mask=att).logits
        lp = torch.log_softmax(lg.float(), dim=-1)
        for j, r in enumerate(ch):
            off = L - len(seqs[j]); np_ = off + len(r["prompt_ids"])
            out.append([lp[j, np_ + k - 1, t].item() for k, t in enumerate(r["gen_ids"])])
    return out


def rep(name, a, b):
    g = [x - y for ra, rb in zip(a, b) for x, y in zip(ra, rb)]
    n = len(g); ab = sorted(abs(v) for v in g)
    o = sum(1 for v in g if math.exp(v) < .9 or math.exp(v) > 1.1)
    print(f"  {name:<36} sd {st.pstdev(g):.6f}  p95 {ab[int(.95*n)]:.5f}  max {ab[-1]:.5f}  "
          f"out {o:>4}/{n} ({100*o/n:5.2f}%)", flush=True)
    return {"sd": st.pstdev(g), "p95": ab[int(.95*n)], "max": ab[-1], "out": o, "n": n}


R = {}
for name, dt in (("bf16", torch.bfloat16), ("fp16", torch.float16), ("fp32", torch.float32)):
    mm = AutoModelForCausalLM.from_pretrained(MODEL, dtype=dt).to("cuda:0").eval()
    b1, b8 = lp_at(mm, 1), lp_at(mm, 8)
    R[f"{name}|b1_vs_b8"] = rep(f"{name}: batch 1 vs 8", b1, b8)
    if name == "bf16":
        R["bf16|rerun"] = rep("bf16: batch 1 run1 vs run2", b1, lp_at(mm, 1))
        R["bf16|b1_vs_b32"] = rep("bf16: batch 1 vs 32", b1, lp_at(mm, 32))
    del mm; torch.cuda.empty_cache()
json.dump({"model": MODEL, "n_tokens": ntok, "results": R},
          open(f"{D}/hfonly_{TAG}.json", "w"), indent=1)
