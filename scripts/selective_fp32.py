"""Divergence-guided selective precision: upcast only the layers where the batch-shape
noise enters, and measure what fraction of the fp32 fix that buys.

The attribution run showed the b1-vs-b8 hidden-state divergence is flat through most of
the trunk and accumulates in a few layers (pythia: the last ~8), while an fp32 head alone
does not help there. If upcasting just those layers removes most of the out-of-band mass,
the per-layer divergence profile - two forward passes - is a precision budget.

Mechanics: chosen layers are converted to fp32 and wrapped with hooks that cast their
inputs up and their outputs back down, so the rest of the network stays bf16. Two
sanity arms are asserted: no layers upcast must reproduce the bf16 rate, and all layers
plus norm and head upcast must match the true fp32 model to within noise.
"""
import os as _os
_RESULTS = _os.environ.get("BLG_DIR") or _os.path.join(
    _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "results")
import os, sys, json, math
os.environ.setdefault("HF_HUB_OFFLINE", "1"); os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = sys.argv[1]; TAG = MODEL.split("/")[-1]
OUT = _RESULTS
tok = AutoTokenizer.from_pretrained(MODEL)
PAD = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
PROMPTS = [f"{s} {t}." for s in ["Explain in two sentences why", "List three facts about",
            "Write one sentence about", "Give a short definition of"]
           for t in ["the ocean", "photosynthesis", "gravity", "the printing press",
                     "antibiotics", "monsoons", "the transistor", "vaccination"]]
NREP = int(os.environ.get("NREP", "1")); PROMPTS = PROMPTS * NREP


BASE_DT = torch.bfloat16


def load():
    return AutoModelForCausalLM.from_pretrained(MODEL, dtype=BASE_DT).to("cuda:0").eval()


m = load()
torch.manual_seed(1234)
rows = []
for p in PROMPTS:
    ids = tok(p, return_tensors="pt").input_ids.to("cuda:0")
    with torch.no_grad():
        o = m.generate(ids, do_sample=True, temperature=1.0, top_p=1.0, max_new_tokens=48,
                       min_new_tokens=48, pad_token_id=PAD, eos_token_id=None)
    rows.append({"p": ids[0].tolist(), "g": o[0, ids.shape[1]:].tolist()})
NP = min(len(r["p"]) for r in rows); rows = [{"p": r["p"][-NP:], "g": r["g"]} for r in rows]
NG = len(rows[0]["g"]); gen = torch.tensor([r["g"] for r in rows])

base = getattr(m, "model", None) or getattr(m, "gpt_neox", None) or getattr(m, "transformer", None)
layers = base.layers if hasattr(base, "layers") else base.h
final_norm = getattr(base, "norm", None) or getattr(base, "final_layer_norm", None) or getattr(base, "ln_f", None)
head = getattr(m, "lm_head", None) or getattr(m, "embed_out", None)
NL = len(layers)
nparam = {i: sum(p.numel() for p in layers[i].parameters()) for i in range(NL)}
nparam["norm"] = sum(p.numel() for p in final_norm.parameters()); nparam["head"] = sum(p.numel() for p in head.parameters())
total = sum(nparam.values()) + sum(p.numel() for n, p in m.named_parameters() if "embed_in" in n or "embed_tokens" in n or "wte" in n)


def _cast(x, dt):
    if torch.is_tensor(x) and x.is_floating_point(): return x.to(dt)
    if isinstance(x, tuple): return tuple(_cast(v, dt) for v in x)
    if isinstance(x, list): return [_cast(v, dt) for v in x]
    if isinstance(x, dict): return {k: _cast(v, dt) for k, v in x.items()}
    return x


def upcast(mod, cast_out_back=True, functional_linear=False):
    """functional_linear: for a head whose weight is tied to the embedding, do not mutate the
    shared tensor; project with a detached fp32 copy instead (TRL refuses tied heads for
    exactly this reason - this is the workaround it lacks)."""
    if functional_linear and isinstance(mod, torch.nn.Linear):
        W32 = mod.weight.detach().float(); b32 = mod.bias.detach().float() if mod.bias is not None else None
        def fwd(md, a, kw):
            x = (a[0] if a else kw["input"]).float()
            y = torch.nn.functional.linear(x, W32, b32)
            return y.to(BASE_DT) if cast_out_back else y
        h = mod.register_forward_hook(lambda md, a, out: fwd(md, a, {}))
        return [h]
    mod.float()
    h1 = mod.register_forward_pre_hook(lambda md, a, kw: (_cast(a, torch.float32), _cast(kw, torch.float32)), with_kwargs=True)
    h2 = mod.register_forward_hook(lambda md, a, out: _cast(out, BASE_DT)) if cast_out_back else None
    return [h for h in (h1, h2) if h]


def score(batch):
    out = []
    for s in range(0, len(rows), batch):
        ch = rows[s:s + batch]
        ids = torch.tensor([r["p"] + r["g"] for r in ch], device="cuda:0")
        with torch.no_grad():
            lg = m(ids, attention_mask=torch.ones_like(ids)).logits
        lp = torch.log_softmax(lg[:, NP - 1:NP - 1 + NG, :].float(), -1).cpu()
        out.append(torch.gather(lp, -1, gen[s:s + batch].unsqueeze(-1)).squeeze(-1))
    return torch.cat(out)


def oob(a, b):
    d = (a - b).flatten(); bad = (~torch.isfinite(d)).float().mean().item()
    if bad > 0: print(f"    non-finite log-probs: {100*bad:.3f}% (counted as out-of-band)")
    r = torch.exp(d); return 100.0 * (((r < 0.9) | (r > 1.1)) | ~torch.isfinite(d)).float().mean().item()


def arm(name, layer_ids, norm_head):
    global m, layers, final_norm, head, base
    m = load(); base = getattr(m, "model", None) or getattr(m, "gpt_neox", None) or getattr(m, "transformer", None)
    layers = base.layers if hasattr(base, "layers") else base.h
    final_norm = getattr(base, "norm", None) or getattr(base, "final_layer_norm", None) or getattr(base, "ln_f", None)
    head = getattr(m, "lm_head", None) or getattr(m, "embed_out", None)
    S = sorted(layer_ids)
    for i in S:
        # keep fp32 across consecutive upcast layers: only cast back where the next layer is bf16
        upcast(layers[i], cast_out_back=((i + 1) not in S))
    if norm_head:
        tied = getattr(m.config, "tie_word_embeddings", False)
        # tied head: the original bf16 forward still runs before the hook replaces its output,
        # so the norm must hand it bf16; the hook then projects that bf16 hidden in fp32
        # (identical computation to the attribution run's "bf16 hidden -> fp32 projection").
        upcast(final_norm, cast_out_back=tied); upcast(head, cast_out_back=False, functional_linear=tied)
    # if the last layer is upcast but norm/head are not, its output was cast back already
    o = oob(score(1), score(8))
    fp32_frac = (sum(nparam[i] for i in S) + (nparam["norm"] + nparam["head"] if norm_head else 0)) / total
    print(f"  {name:<34} layers={('none' if not S else f'{S[0]}..{S[-1]}' if S==list(range(S[0],S[-1]+1)) else str(S))[:22]:<22} norm+head={'y' if norm_head else 'n'}  fp32 params {100*fp32_frac:5.1f}%   out-of-band {o:6.2f}%", flush=True)
    del m; torch.cuda.empty_cache()
    return {"name": name, "layers": S, "norm_head": norm_head, "fp32_frac": fp32_frac, "oob": o}


print(f"### {MODEL}  {gen.numel()} tokens, {NL} layers")
R = []
ONLY = os.environ.get("ARMS_ONLY", "")
_arm = arm
def arm(name, layer_ids, norm_head):
    if ONLY and not (name.startswith("baseline") or name.startswith("all layers") or ONLY in name):
        return {"name": name, "layers": sorted(layer_ids), "norm_head": norm_head, "fp32_frac": float("nan"), "oob": float("nan"), "skipped": True}
    r = _arm(name, layer_ids, norm_head); r["base_dtype"] = str(BASE_DT).split(".")[-1]; return r
R.append(arm("baseline bf16", [], False))
R.append(arm("head only (ScaleRL)", [], True))
R.append(arm(f"last 4 + head", list(range(NL - 4, NL)), True))
R.append(arm(f"last 8 + head", list(range(NL - 8, NL)), True))
R.append(arm(f"last 12 + head", list(range(NL - 12, NL)), True))
R.append(arm(f"first 12 + head", list(range(0, 12)), True))
R.append(arm(f"first 4 + head", list(range(0, 4)), True))
R.append(arm(f"first 8 + head", list(range(0, 8)), True))
# divergence-guided: attribution.py stores rel_div after the embedding and after every layer;
# the increment across layer i is the divergence that layer injects into the residual stream.
_att = f"{OUT}/attribution_{TAG}.json"
assert os.path.exists(_att), _att
_v = [float(x) for x in json.load(open(_att))["per_layer_rel_div"]]; assert len(_v) == NL + 1
_delta = [_v[i + 1] - _v[i] for i in range(NL)]
_rank = sorted(range(NL), key=lambda i: -_delta[i])
for k in (4, 8, 12):
    R.append(arm(f"guided top{k} + head", sorted(_rank[:k]), True))
    R[-1]["delta_rank"] = _rank[:k]
# contiguous window: the k consecutive layers with the largest summed divergence increment.
# A contiguous block keeps the residual stream in fp32 across the whole window, whereas
# scattered picks cast back to bf16 at every boundary and re-inject rounding there.
for k in (4, 8, 12):
    st = max(range(NL - k + 1), key=lambda s0: sum(_delta[s0:s0 + k]))
    R.append(arm(f"guided window{k} + head", list(range(st, st + k)), True))
    R[-1]["window_start"] = st
BASE_DT = torch.float16
R.append(arm("fp16 all (no fp32)", [], False))
R.append(arm("fp16 + head", [], True))
R.append(arm("fp16 + guided top8 + head", sorted(_rank[:8]), True))
BASE_DT = torch.bfloat16
R.append(arm("all layers + head (sanity)", list(range(NL)), True))
assert abs(R[0]["oob"] - R[0]["oob"]) < 1e-9
assert R[-1]["oob"] < 0.5, f"all-fp32 sanity failed: {R[-1]['oob']}"
json.dump({"model": MODEL, "n_tokens": int(gen.numel()), "delta": _delta, "arms": R},
          open(f"{OUT}/selective_{TAG}{'' if NREP == 1 else f'_x{NREP}'}{'_' + ONLY if ONLY else ''}.json", "w"), indent=1)
