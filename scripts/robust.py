"""Is the effect a property of one attention kernel, and how stable is the estimate?

Everything measured so far used whatever attention implementation transformers picks by
default. If the batch-1-vs-8 gap is an artefact of that one kernel family it should vanish
under another. And every number so far came from a single rollout, so it carries no interval.

  implementations   eager (explicit matmul+softmax), sdpa, flex_attention
  repeats           three independent rollout seeds, Wilson interval on the out-of-band rate
  advantage         the clip-flip metric is recomputed under three advantage distributions,
                    since a synthetic advantage decides which side a token can be clipped on
"""
import os as _os
_RESULTS = _os.environ.get("BLG_DIR") or _os.path.join(
    _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "results")
_os.makedirs(_RESULTS, exist_ok=True)
import os, sys, math, json, random, statistics as st
os.environ.setdefault("HF_HUB_OFFLINE", "1"); os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
import torch, warnings; warnings.filterwarnings("ignore")
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = sys.argv[1]; D = _RESULTS; TAG = MODEL.split("/")[-1]
tok = AutoTokenizer.from_pretrained(MODEL)
PAD = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
PROMPTS = [f"{s} {t}." for s in ["Explain in two sentences why", "List three facts about",
            "Write one sentence about", "Give a short definition of"]
           for t in ["the ocean", "photosynthesis", "gravity", "the printing press",
                     "antibiotics", "monsoons", "the transistor", "vaccination"]]


def wilson(k, n, z=1.96):
    if n == 0: return 0.0, 0.0
    p, d = k / n, 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** .5) / d
    return max(0., c - h), min(1., c + h)


def rollouts(seed):
    m = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to("cuda:0").eval()
    torch.manual_seed(seed)
    rs = []
    for p in PROMPTS:
        ids = tok(p, return_tensors="pt").input_ids.to("cuda:0")
        with torch.no_grad():
            o = m.generate(ids, do_sample=True, temperature=1.0, top_p=1.0, max_new_tokens=48,
                           min_new_tokens=48, pad_token_id=PAD, eos_token_id=None,
                           return_dict_in_generate=True, output_scores=True)
        g = o.sequences[0, ids.shape[1]:]
        lp = [torch.log_softmax(s[0].float(), -1)[t].item() for s, t in zip(o.scores, g)]
        rs.append({"p": ids[0].tolist(), "g": g.tolist(), "roll": lp})
    del m; torch.cuda.empty_cache()
    npad = min(len(r["p"]) for r in rs)
    return [{"p": r["p"][-npad:], "g": r["g"], "roll": r["roll"]} for r in rs], npad


def score(m, rows, npad, batch):
    out = []
    for s in range(0, len(rows), batch):
        ch = rows[s:s + batch]
        ids = torch.tensor([r["p"] + r["g"] for r in ch], device="cuda:0")
        with torch.no_grad():
            lg = m(ids, attention_mask=torch.ones_like(ids)).logits
        lp = torch.log_softmax(lg.float(), -1)
        for j, r in enumerate(ch):
            out.append([lp[j, npad + k - 1, t].item() for k, t in enumerate(r["g"])])
    return out


print(f"### {MODEL}", flush=True)
R = {}
SEEDS = [1234, 20260916, 7]
print("  [out-of-band rate for batch 1 vs 8, by attention implementation and seed]", flush=True)
for impl in ("eager", "sdpa", "flex_attention"):
    ks, ns = 0, 0
    per = []
    for sd in SEEDS:
        rows, npad = rollouts(sd)
        m = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16,
                                                 attn_implementation=impl).to("cuda:0").eval()
        a, b = score(m, rows, npad, 1), score(m, rows, npad, 8)
        del m; torch.cuda.empty_cache()
        g = [x - y for ra, rb in zip(a, b) for x, y in zip(ra, rb)]
        k = sum(1 for v in g if math.exp(v) < .9 or math.exp(v) > 1.1)
        ks += k; ns += len(g); per.append(100 * k / len(g))
    lo, hi = wilson(ks, ns)
    print(f"    {impl:<16} {100*ks/ns:6.2f}%  95%CI [{100*lo:5.2f}, {100*hi:5.2f}]  "
          f"per-seed {['%.2f' % x for x in per]}  n={ns}", flush=True)
    R[impl] = {"rate": 100 * ks / ns, "ci": [100 * lo, 100 * hi], "per_seed": per, "n": ns}

# advantage sensitivity for the clip-flip metric, on one seed, sdpa
rows, npad = rollouts(SEEDS[0])
m = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to("cuda:0").eval()
b1, b8 = score(m, rows, npad, 1), score(m, rows, npad, 8)
del m; torch.cuda.empty_cache()
print("  [clip-status flips, by advantage distribution]", flush=True)
rng = random.Random(0)
G = 8
for aname, draw in (("gaussian", lambda: rng.gauss(0, 1)),
                    ("binary +/-1", lambda: rng.choice([-1.0, 1.0])),
                    ("sparse (90% zero)", lambda: 0.0 if rng.random() < .9 else rng.gauss(0, 1))):
    rew = [draw() for _ in rows]
    adv = []
    for s in range(0, len(rows), G):
        blk = rew[s:s + G]; mu = sum(blk) / len(blk); sd_ = st.pstdev(blk) or 1.0
        adv += [(x - mu) / sd_ for x in blk]
    flips = n = 0
    for i, r in enumerate(rows):
        A = adv[i]
        for k in range(len(r["g"])):
            r1 = math.exp(b1[i][k] - r["roll"][k]); r8 = math.exp(b8[i][k] - r["roll"][k])
            c1 = 0. if ((A >= 0 and r1 > 1.2) or (A < 0 and r1 < .8)) else r1 * A
            c8 = 0. if ((A >= 0 and r8 > 1.2) or (A < 0 and r8 < .8)) else r8 * A
            flips += (c1 == 0.) != (c8 == 0.); n += 1
    lo, hi = wilson(flips, n)
    print(f"    {aname:<18} {100*flips/n:6.2f}%  95%CI [{100*lo:5.2f}, {100*hi:5.2f}]  n={n}", flush=True)
    R[f"adv_{aname}"] = {"rate": 100 * flips / n, "ci": [100 * lo, 100 * hi]}
json.dump({"model": MODEL, "results": R}, open(f"{D}/robust_{TAG}.json", "w"), indent=1)
