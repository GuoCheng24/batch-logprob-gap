"""Is every one of these path changes a bf16 effect, or do some survive in fp32?

Three ways to change the attention path without changing the mathematics:
  batch    [1,L] vs [8,L], implicit causal mask
  mask     implicit causal vs an explicit 4D mask holding the same causal pattern
  packing  1 sequence per row vs 8, both with block-diagonal 4D masks

If all three collapse to zero in fp32, they are one phenomenon: bf16 reductions that depend
on the kernel the shape selects. If any survives fp32 it is a real algorithmic difference
and must be described separately.
"""
import os as _os
_RESULTS = _os.environ.get("BLG_DIR") or _os.path.join(
    _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "results")
_os.makedirs(_RESULTS, exist_ok=True)
import os, sys, math, json, statistics as st
os.environ.setdefault("HF_HUB_OFFLINE", "1"); os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = sys.argv[1]; D = _RESULTS; TAG = MODEL.split("/")[-1]
tok = AutoTokenizer.from_pretrained(MODEL)
PAD = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
PROMPTS = [f"{s} {t}." for s in ["Explain in two sentences why", "List three facts about",
            "Write one sentence about", "Give a short definition of"]
           for t in ["the ocean", "photosynthesis", "gravity", "the printing press",
                     "antibiotics", "monsoons", "the transistor", "vaccination"]]

base = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to("cuda:0").eval()
torch.manual_seed(1234)
rows = []
for p in PROMPTS:
    ids = tok(p, return_tensors="pt").input_ids.to("cuda:0")
    with torch.no_grad():
        o = base.generate(ids, do_sample=True, temperature=1.0, top_p=1.0, max_new_tokens=48,
                          min_new_tokens=48, pad_token_id=PAD, eos_token_id=None)
    rows.append({"p": ids[0].tolist(), "g": o[0, ids.shape[1]:].tolist()})
del base; torch.cuda.empty_cache()
NP = min(len(r["p"]) for r in rows); rows = [{"p": r["p"][-NP:], "g": r["g"]} for r in rows]
NG = len(rows[0]["g"]); L = NP + NG
print(f"### {MODEL}  {len(rows)} seqs x {L} tok, {len(rows)*NG} scored", flush=True)


def run(m, batch, explicit_mask, pack):
    out = []
    step = batch
    for s in range(0, len(rows), step):
        ch = rows[s:s + step]
        if pack:
            flat = [t for r in ch for t in (r["p"] + r["g"])]
            pos = [i for r in ch for i in range(L)]
            ids = torch.tensor([flat], device="cuda:0")
            T = len(flat)
            seg = torch.tensor([j for j in range(len(ch)) for _ in range(L)], device="cuda:0")
            allow = torch.tril(torch.ones(T, T, dtype=torch.bool, device="cuda:0")) & (seg[:, None] == seg[None, :])
            am = torch.zeros(1, 1, T, T, dtype=m.dtype, device="cuda:0").masked_fill_(~allow, torch.finfo(m.dtype).min)
            with torch.no_grad():
                lg = m(ids, position_ids=torch.tensor([pos], device="cuda:0"), attention_mask=am).logits
            lp = torch.log_softmax(lg.float(), -1)
            for j, r in enumerate(ch):
                b = j * L
                out.append([lp[0, b + NP + k - 1, t].item() for k, t in enumerate(r["g"])])
        else:
            ids = torch.tensor([r["p"] + r["g"] for r in ch], device="cuda:0")
            if explicit_mask:
                T = ids.shape[1]
                allow = torch.tril(torch.ones(T, T, dtype=torch.bool, device="cuda:0"))
                am = torch.zeros(1, 1, T, T, dtype=m.dtype, device="cuda:0").masked_fill_(~allow, torch.finfo(m.dtype).min)
            else:
                am = torch.ones_like(ids)
            with torch.no_grad():
                lg = m(ids, attention_mask=am).logits
            lp = torch.log_softmax(lg.float(), -1)
            for j, r in enumerate(ch):
                out.append([lp[j, NP + k - 1, t].item() for k, t in enumerate(r["g"])])
    return out


def rep(name, a, b):
    g = [x - y for ra, rb in zip(a, b) for x, y in zip(ra, rb)]
    n = len(g); ab = sorted(abs(v) for v in g)
    o = sum(1 for v in g if math.exp(v) < .9 or math.exp(v) > 1.1)
    print(f"    {name:<34} sd {st.pstdev(g):.7f}  max {ab[-1]:.7f}  out {o:>4}/{n} ({100*o/n:5.2f}%)", flush=True)
    return {"sd": st.pstdev(g), "max": ab[-1], "out": o, "n": n}


R = {}
for dn, dt in (("bf16", torch.bfloat16), ("fp32", torch.float32)):
    m = AutoModelForCausalLM.from_pretrained(MODEL, dtype=dt).to("cuda:0").eval()
    print(f"  [{dn}]", flush=True)
    R[f"{dn}|batch"] = rep("batch  [1,L] vs [8,L]", run(m, 1, False, False), run(m, 8, False, False))
    R[f"{dn}|mask"] = rep("mask   implicit vs explicit 4D", run(m, 1, False, False), run(m, 1, True, False))
    R[f"{dn}|pack"] = rep("pack   1/row vs 8/row (both 4D)", run(m, 1, False, True), run(m, 8, False, True))
    del m; torch.cuda.empty_cache()
json.dump({"model": MODEL, "results": R}, open(f"{D}/dtype_{TAG}.json", "w"), indent=1)
