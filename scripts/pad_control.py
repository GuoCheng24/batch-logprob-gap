"""Is the batch-1-vs-8 gap padding, or the batch itself?

The earlier runs left-padded short sequences to the longest in the batch, so batch 1 had no
padding and batch 8 did. That alone could explain everything. Here every sequence is
truncated to a common length, so no arm pads at all, and the batch is varied on its own.

  A  padded, ragged lengths, batch 1 vs 8        the original comparison, for reference
  B  equal length, no padding, batch 1 vs 8      padding removed, batch still varies
  C  equal length, 8 identical copies vs batch 1 batch size varies, content held fixed
  D  equal length, batch 8, two different groupings of the same sequences

If B is flat the effect was padding. If B and C survive, the batch itself changes the
reduction and the content it is grouped with is irrelevant. D separates "which sequences
share a batch" from "how many".
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
STEMS = ["Explain in two sentences why", "List three facts about", "Write one sentence about",
         "Give a short definition of", "Name one consequence of", "Describe briefly how",
         "In one sentence, compare", "State the main idea of"]
TOPICS = ["the ocean", "photosynthesis", "gravity", "the printing press", "antibiotics",
          "monsoons", "the transistor", "vaccination"]
PROMPTS = [f"{s} {t}." for s in STEMS for t in TOPICS]

m = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to("cuda:0").eval()
torch.manual_seed(1234)
raw = []
for p in PROMPTS:
    ids = tok(p, return_tensors="pt").input_ids.to("cuda:0")
    with torch.no_grad():
        o = m.generate(ids, do_sample=True, temperature=1.0, top_p=1.0,
                       max_new_tokens=64, min_new_tokens=64, pad_token_id=PAD,
                       eos_token_id=None)   # force a fixed length so the equal-length
                                            # arms keep all 64 positions, not just the
                                            # first token before some sequence hit EOS
    raw.append({"p": ids[0].tolist(), "g": o[0, ids.shape[1]:].tolist()})

# common length so no arm needs padding; keep the tail of the prompt + all generated tokens
NP = min(len(r["p"]) for r in raw)
NG = min(len(r["g"]) for r in raw)
eq = [{"p": r["p"][-NP:], "g": r["g"][:NG]} for r in raw]
assert NG >= 32, f"equal-length arm collapsed to {NG} generated tokens; comparison would be degenerate"
print(f"### {MODEL}  equal-length arms: prompt {NP} + gen {NG} = {NP+NG} tokens x {len(eq)} seqs "
      f"= {NG*len(eq)} scored", flush=True)


def score(rows, batch, pad, order=None):
    idx = list(range(len(rows))) if order is None else order
    out = {}
    for s in range(0, len(idx), batch):
        sel = [rows[i] for i in idx[s:s + batch]]
        seqs = [r["p"] + r["g"] for r in sel]
        L = max(len(x) for x in seqs)
        if pad:
            ids = torch.tensor([[PAD] * (L - len(x)) + x for x in seqs], device="cuda:0")
            att = torch.tensor([[0] * (L - len(x)) + [1] * len(x) for x in seqs], device="cuda:0")
        else:
            assert all(len(x) == L for x in seqs), "no-pad arm got ragged lengths"
            ids = torch.tensor(seqs, device="cuda:0")
            att = torch.ones_like(ids)
        with torch.no_grad():
            lg = m(ids, attention_mask=att).logits
        lp = torch.log_softmax(lg.float(), dim=-1)
        for j, i in enumerate(idx[s:s + batch]):
            off = L - len(seqs[j]); np_ = off + len(sel[j]["p"])
            out[i] = [lp[j, np_ + k - 1, t].item() for k, t in enumerate(sel[j]["g"])]
    return [out[i] for i in range(len(rows))]


def rep(name, a, b):
    g = [x - y for ra, rb in zip(a, b) for x, y in zip(ra, rb)]
    n = len(g); ab = sorted(abs(v) for v in g)
    o = sum(1 for v in g if math.exp(v) < .9 or math.exp(v) > 1.1)
    print(f"  {name:<52} sd {st.pstdev(g):.6f}  p95 {ab[int(.95*n)]:.5f}  max {ab[-1]:.5f}  "
          f"out {o:>4}/{n} ({100*o/n:5.2f}%)", flush=True)
    return {"sd": st.pstdev(g), "p95": ab[int(.95*n)], "max": ab[-1], "out": o, "n": n}


R = {}
R["A pad ragged b1 vs b8"] = rep("A  padded ragged:   batch 1 vs 8", score(raw, 1, True), score(raw, 8, True))
e1 = score(eq, 1, False)
R["B eq nopad b1 vs b8"] = rep("B  equal len, no pad: batch 1 vs 8", e1, score(eq, 8, False))
R["B32"] = rep("B' equal len, no pad: batch 1 vs 32", e1, score(eq, 32, False))
# C: eight identical copies of one sequence, scored as a batch of 8, vs that sequence alone
copies = [eq[0]] * 8
c8 = score(copies, 8, False)[0]
R["C identical copies b8 vs b1"] = rep("C  same sequence x8 in a batch vs alone", [e1[0]], [c8])
# D: same sequences, two different groupings into batches of 8
import random
rng = random.Random(7); order = list(range(len(eq))); rng.shuffle(order)
R["D regrouped b8"] = rep("D  batch 8, two different groupings",
                          score(eq, 8, False), score(eq, 8, False, order))
json.dump({"model": MODEL, "np": NP, "ng": NG, "results": R},
          open(f"{D}/pad_{TAG}.json", "w"), indent=1)
