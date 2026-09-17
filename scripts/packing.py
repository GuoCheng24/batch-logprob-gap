"""Does sequence packing remove the effect? This is what verl does and TRL does not always.

verl runs its log-prob forward with use_remove_padding, which concatenates the batch into one
flat token sequence instead of a padded [B, L] rectangle. A contributor on verl#6280 measured
MBS=1 against MBS=2 in that configuration and got max_abs_diff exactly 0, which contradicts the
padded-batch measurements here. If the mechanism is shape-dependent kernel selection rather
than batch size as such, packing should reproduce their null while padding reproduces ours.

  P1  padded   [1, L] vs [8, L]     what a naive trainer does
  P2  packed   [1, 8L] vs [1, L]x8  what verl does with remove_padding
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
tok = AutoTokenizer.from_pretrained(MODEL)
PAD = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
PROMPTS = [f"{s} {t}." for s in
           ["Explain in two sentences why", "List three facts about", "Write one sentence about",
            "Give a short definition of", "Name one consequence of", "Describe briefly how",
            "In one sentence, compare", "State the main idea of"]
           for t in ["the ocean", "photosynthesis", "gravity", "the printing press",
                     "antibiotics", "monsoons", "the transistor", "vaccination"]]

m = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to("cuda:0").eval()
torch.manual_seed(1234)
rows = []
for p in PROMPTS:
    ids = tok(p, return_tensors="pt").input_ids.to("cuda:0")
    with torch.no_grad():
        o = m.generate(ids, do_sample=True, temperature=1.0, top_p=1.0, max_new_tokens=64,
                       min_new_tokens=64, pad_token_id=PAD, eos_token_id=None)
    rows.append({"p": ids[0].tolist(), "g": o[0, ids.shape[1]:].tolist()})
NP = min(len(r["p"]) for r in rows)
rows = [{"p": r["p"][-NP:], "g": r["g"]} for r in rows]     # equal length, no padding needed
NG = len(rows[0]["g"]); L = NP + NG
print(f"### {MODEL}   {len(rows)} seqs x (prompt {NP} + gen {NG}) = {len(rows)*NG} scored tokens",
      flush=True)


def padded(batch):
    out = []
    for s in range(0, len(rows), batch):
        ch = rows[s:s + batch]
        ids = torch.tensor([r["p"] + r["g"] for r in ch], device="cuda:0")
        with torch.no_grad():
            lg = m(ids, attention_mask=torch.ones_like(ids)).logits
        lp = torch.log_softmax(lg.float(), dim=-1)
        for j, r in enumerate(ch):
            out.append([lp[j, NP + k - 1, t].item() for k, t in enumerate(r["g"])])
    return out


def packed(nseq):
    """Concatenate nseq sequences into one flat row, the way remove_padding does.
    position_ids restart per sequence so each one is scored as if it were alone."""
    out = []
    for s in range(0, len(rows), nseq):
        ch = rows[s:s + nseq]
        flat, pos = [], []
        for r in ch:
            seq = r["p"] + r["g"]; flat += seq; pos += list(range(len(seq)))
        ids = torch.tensor([flat], device="cuda:0")
        pids = torch.tensor([pos], device="cuda:0")
        # block-diagonal causal mask: a packed sequence must not attend across its own
        # boundary, which is what flash-attn's cu_seqlens gives verl. Without it the
        # packed row is a different computation, not a re-batching of the same one.
        T = len(flat)
        seg = torch.tensor([i for i, r in enumerate(ch) for _ in range(L)], device="cuda:0")
        causal = torch.tril(torch.ones(T, T, dtype=torch.bool, device="cuda:0"))
        block = seg[:, None] == seg[None, :]
        allow = causal & block
        mask4d = torch.zeros(1, 1, T, T, dtype=m.dtype, device="cuda:0")
        mask4d.masked_fill_(~allow, torch.finfo(m.dtype).min)
        with torch.no_grad():
            lg = m(ids, position_ids=pids, attention_mask=mask4d).logits
        lp = torch.log_softmax(lg.float(), dim=-1)
        for j, r in enumerate(ch):
            base = j * L
            out.append([lp[0, base + NP + k - 1, t].item() for k, t in enumerate(r["g"])])
    return out


def rep(name, a, b):
    g = [x - y for ra, rb in zip(a, b) for x, y in zip(ra, rb)]
    n = len(g); ab = sorted(abs(v) for v in g)
    o = sum(1 for v in g if math.exp(v) < .9 or math.exp(v) > 1.1)
    print(f"  {name:<46} sd {st.pstdev(g):.6f}  max {ab[-1]:.6f}  "
          f"out {o:>4}/{n} ({100*o/n:5.2f}%)", flush=True)
    return {"sd": st.pstdev(g), "max": ab[-1], "out": o, "n": n}


R = {}
p1 = padded(1)
R["padded b1 vs b8"] = rep("P1 padded  [1,L] vs [8,L]", p1, padded(8))
k1 = packed(1)
R["packed x1 vs x8"] = rep("P2 packed  1 seq/row vs 8 seqs/row", k1, packed(8))
san = rep("   sanity: padded [1,L] vs packed 1/row", p1, k1)
R["padded1 vs packed1"] = san
assert san["max"] < 1e-6, f"packing path itself is wrong: max diff {san['max']}"
json.dump({"model": MODEL, "results": R}, open(f"{D}/pack_{TAG}.json", "w"), indent=1)
