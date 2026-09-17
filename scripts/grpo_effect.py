"""Does the batch-recomputation noise reach GRPO's gradient, or does clipping absorb it?

At the first inner epoch of GRPO the policy has not moved, so exp(logp_theta - logp_rollout)
is the identity and every token should sit at ratio 1, inside any clip range. The trainer
recomputes logp_theta in batches, which moves it. This asks the only question that matters
for training: how often does that movement change whether a token contributes gradient at
all, and by how much does it change the coefficient when it does.

Groups are real: G samples per prompt, advantages normalised within the group, exactly as
GRPO defines them. Rewards are synthetic - the advantage is a per-sequence scalar that does
not depend on how logp_theta was computed, so it cannot manufacture or hide the effect;
it only sets which tokens are eligible to be clipped on which side.
"""
import os as _os
_RESULTS = _os.environ.get("BLG_DIR") or _os.path.join(
    _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "results")
_os.makedirs(_RESULTS, exist_ok=True)
import os, sys, json, math, random, statistics as st
os.environ.setdefault("HF_HUB_OFFLINE", "1"); os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = sys.argv[1]; D = _RESULTS; TAG = MODEL.split("/")[-1]
G = 8                      # samples per prompt (GRPO group size)
NPROMPT = 8
EPS = 0.2                  # PPO clip epsilon, TRL's GRPOConfig default

tok = AutoTokenizer.from_pretrained(MODEL)
PAD = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
PROMPTS = ["Explain in two sentences why the ocean is salty.",
           "List three facts about photosynthesis.",
           "Write one sentence about gravity.",
           "Give a short definition of antibiotics.",
           "Name one consequence of deforestation.",
           "Describe briefly how a transistor works.",
           "In one sentence, compare rain and snow.",
           "State the main idea of vaccination."][:NPROMPT]

m = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to("cuda:0").eval()
torch.manual_seed(1234)
rows = []
for gi, p in enumerate(PROMPTS):
    ids = tok(p, return_tensors="pt").input_ids.to("cuda:0")
    for _ in range(G):
        with torch.no_grad():
            o = m.generate(ids, do_sample=True, temperature=1.0, top_p=1.0,
                           max_new_tokens=64, min_new_tokens=64, pad_token_id=PAD,
                           eos_token_id=None, return_dict_in_generate=True, output_scores=True)
        gen = o.sequences[0, ids.shape[1]:]
        # the sampler's own log prob for each emitted token: this is logp_rollout
        lp = [torch.log_softmax(s[0].float(), -1)[t].item() for s, t in zip(o.scores, gen)]
        rows.append({"group": gi, "p": ids[0].tolist(), "g": gen.tolist(), "rollout": lp})
NG = len(rows[0]["g"])
assert all(len(r["g"]) == NG for r in rows), "ragged generations"
print(f"### {MODEL}   {NPROMPT} prompts x {G} samples, {NG} tokens each = {len(rows)*NG} tokens",
      flush=True)


def score(batch):
    out = []
    for s in range(0, len(rows), batch):
        ch = rows[s:s + batch]
        seqs = [r["p"] + r["g"] for r in ch]
        L = max(len(x) for x in seqs)
        ids = torch.tensor([[PAD] * (L - len(x)) + x for x in seqs], device="cuda:0")
        att = torch.tensor([[0] * (L - len(x)) + [1] * len(x) for x in seqs], device="cuda:0")
        with torch.no_grad():
            lg = m(ids, attention_mask=att).logits
        lp = torch.log_softmax(lg.float(), dim=-1)
        for j, r in enumerate(ch):
            off = L - len(seqs[j]); np_ = off + len(r["p"])
            out.append([lp[j, np_ + k - 1, t].item() for k, t in enumerate(r["g"])])
    return out


theta_b1, theta_b8 = score(1), score(8)
rollout = [r["rollout"] for r in rows]

# group-normalised advantages from synthetic rewards, fixed across arms
rng = random.Random(0)
rew = [rng.gauss(0, 1) for _ in rows]
adv = []
for gi in range(NPROMPT):
    idx = [i for i, r in enumerate(rows) if r["group"] == gi]
    mu = sum(rew[i] for i in idx) / len(idx)
    sd = st.pstdev([rew[i] for i in idx]) or 1.0
    for i in idx:
        adv.append(None)
adv = [0.0] * len(rows)
for gi in range(NPROMPT):
    idx = [i for i, r in enumerate(rows) if r["group"] == gi]
    mu = sum(rew[i] for i in idx) / len(idx)
    sd = st.pstdev([rew[i] for i in idx]) or 1.0
    for i in idx:
        adv[i] = (rew[i] - mu) / sd


def coef(ratio, A):
    """d(loss)/d(logp) coefficient for GRPO's clipped surrogate, up to sign."""
    lo, hi = 1 - EPS, 1 + EPS
    if A >= 0:
        return 0.0 if ratio > hi else ratio * A      # clipped above when advantage positive
    return 0.0 if ratio < lo else ratio * A          # clipped below when advantage negative


flips = same = 0
rel = []
c1s, c8s = 0, 0
for i, r in enumerate(rows):
    A = adv[i]
    for k in range(NG):
        r1 = math.exp(theta_b1[i][k] - r["rollout"][k])
        r8 = math.exp(theta_b8[i][k] - r["rollout"][k])
        a, b = coef(r1, A), coef(r8, A)
        c1s += (a == 0.0); c8s += (b == 0.0)
        if (a == 0.0) != (b == 0.0):
            flips += 1
        else:
            same += 1
            if a != 0.0:
                rel.append(abs(b - a) / abs(a))
n = len(rows) * NG
rel.sort()
print(f"  ratio at first inner epoch should be 1.0 for every token")
print(f"  clipped (no gradient) : batch1 {c1s}/{n} ({100*c1s/n:.2f}%)   "
      f"batch8 {c8s}/{n} ({100*c8s/n:.2f}%)")
print(f"  clip status FLIPS between batch 1 and batch 8 : {flips}/{n} ({100*flips/n:.2f}%)")
if rel:
    print(f"  where both contribute, |dcoef|/coef : median {rel[len(rel)//2]:.5f}  "
          f"p95 {rel[int(.95*len(rel))]:.5f}  max {rel[-1]:.5f}")
json.dump({"model": MODEL, "n": n, "flips": flips, "clip_b1": c1s, "clip_b8": c8s,
           "rel_median": rel[len(rel)//2] if rel else None,
           "rel_p95": rel[int(.95*len(rel))] if rel else None},
          open(f"{D}/grpo_{TAG}.json", "w"), indent=1)
