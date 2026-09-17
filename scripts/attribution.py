"""Where in the network does the batch-shape noise enter, and does an fp32 head remove it?

Same tokens scored at batch 1 and batch 8 in bf16. Two questions:
  (1) per layer, how much do the hidden states at the scored positions diverge between the
      two shapes - does the gap accumulate through the trunk or appear at the end;
  (2) if the final projection is computed in fp32 from the bf16 hidden states (what TRL's
      `cast_lm_head_to_fp32=True` does), does the out-of-band rate fall.
If (2) leaves the rate unchanged, the noise is upstream and an fp32 head cannot be the fix.
"""
import os as _os
_RESULTS = _os.environ.get("BLG_DIR") or _os.path.join(
    _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "results")
import os, sys, json, math, statistics as st
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
m = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to("cuda:0").eval()
torch.manual_seed(1234)
rows = []
for p in PROMPTS:
    ids = tok(p, return_tensors="pt").input_ids.to("cuda:0")
    with torch.no_grad():
        o = m.generate(ids, do_sample=True, temperature=1.0, top_p=1.0, max_new_tokens=48,
                       min_new_tokens=48, pad_token_id=PAD, eos_token_id=None)
    rows.append({"p": ids[0].tolist(), "g": o[0, ids.shape[1]:].tolist()})
NP = min(len(r["p"]) for r in rows); rows = [{"p": r["p"][-NP:], "g": r["g"]} for r in rows]
NG = len(rows[0]["g"])

# locate trunk / final norm / head generically
base = getattr(m, "model", None) or getattr(m, "gpt_neox", None) or getattr(m, "transformer", None)
final_norm = getattr(base, "norm", None) or getattr(base, "final_layer_norm", None) or getattr(base, "ln_f", None)
head = getattr(m, "lm_head", None) or getattr(m, "embed_out", None)
assert final_norm is not None and head is not None, "could not locate final norm / head"

cap = {}
def hook(name):
    def f(mod, inp, out):
        cap[name] = (inp[0] if isinstance(inp, tuple) else inp).detach(), (out[0] if isinstance(out, tuple) else out).detach()
    return f
h_norm = final_norm.register_forward_hook(hook("final_norm"))


def run(batch):
    """returns: per-layer hidden at scored positions [L+1][N, NG, H], pre-norm, post-norm, bf16 logprobs"""
    per_layer, pre, post, lp_bf = None, [], [], []
    for s in range(0, len(rows), batch):
        ch = rows[s:s + batch]
        ids = torch.tensor([r["p"] + r["g"] for r in ch], device="cuda:0")
        with torch.no_grad():
            out = m(ids, attention_mask=torch.ones_like(ids), output_hidden_states=True)
        hs = out.hidden_states                     # tuple len L+1, each [B, T, H]
        sl = slice(NP - 1, NP - 1 + NG)            # logits at these positions predict the generated tokens
        if per_layer is None: per_layer = [[] for _ in hs]
        for li, h in enumerate(hs): per_layer[li].append(h[:, sl, :].float().cpu())
        pre.append(cap["final_norm"][0][:, sl, :].cpu()); post.append(cap["final_norm"][1][:, sl, :].cpu())
        lp = torch.log_softmax(out.logits[:, sl, :].float(), -1)
        g = torch.tensor([r["g"] for r in ch])
        lp_bf.append(torch.gather(lp.cpu(), -1, g.unsqueeze(-1)).squeeze(-1))
    return [torch.cat(x) for x in per_layer], torch.cat(pre), torch.cat(post), torch.cat(lp_bf)


L1, pre1, post1, lp1 = run(1)
L8, pre8, post8, lp8 = run(8)
h_norm.remove()
gen = torch.tensor([r["g"] for r in rows])


def oob(a, b):
    d = (a - b).flatten()
    r = torch.exp(d)
    return 100.0 * ((r < 0.9) | (r > 1.1)).float().mean().item(), d.std().item()


res = {"model": MODEL, "n_tokens": int(gen.numel())}
# (1) per-layer relative divergence
div = [((a - b).norm(dim=-1) / (a.norm(dim=-1) + 1e-6)).mean().item() for a, b in zip(L1, L8)]
res["per_layer_rel_div"] = div
print(f"### {MODEL}  {gen.numel()} tokens, {len(div)-1} layers")
print("  per-layer relative divergence of hidden states (b1 vs b8), embeddings -> last:")
print("   " + " ".join(f"{x:.4f}" for x in div))
print(f"  ratio last/first-nonzero: {div[-1]/max(1e-9, next((x for x in div if x>0), 1e-9)):.1f}x")

# (2) head ablation: logits from bf16 post-norm hidden, projection in bf16 vs fp32
W = head.weight.detach()
def lp_from(hidden, dtype):
    with torch.no_grad():
        logits = torch.nn.functional.linear(hidden.to("cuda:0").to(dtype), W.to(dtype)).float()
    lp = torch.log_softmax(logits, -1).cpu()
    return torch.gather(lp, -1, gen.unsqueeze(-1)).squeeze(-1)
o_model, s_model = oob(lp1, lp8)
o_bf, s_bf = oob(lp_from(post1, torch.bfloat16), lp_from(post8, torch.bfloat16))
o_fp, s_fp = oob(lp_from(post1, torch.float32), lp_from(post8, torch.float32))
# also: fp32 norm + fp32 head from the pre-norm hidden
def norm_fp32(x):
    with torch.no_grad():
        mod = final_norm
        w = mod.weight.detach().float().cpu(); b = getattr(mod, "bias", None)
        eps = getattr(mod, "eps", getattr(mod, "variance_epsilon", 1e-5))
        xf = x.float()
        if b is not None:   # LayerNorm
            mu = xf.mean(-1, keepdim=True); var = xf.var(-1, unbiased=False, keepdim=True)
            return (xf - mu) / torch.sqrt(var + eps) * w + b.detach().float().cpu()
        return xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps) * w   # RMSNorm
o_nf, s_nf = oob(lp_from(norm_fp32(pre1), torch.float32), lp_from(norm_fp32(pre8), torch.float32))
res.update({"oob_model_bf16": o_model, "oob_head_bf16_recomputed": o_bf, "oob_head_fp32": o_fp, "oob_norm_and_head_fp32": o_nf,
            "sd_model": s_model, "sd_head_fp32": s_fp, "sd_norm_head_fp32": s_nf})
print(f"  out-of-band [0.9,1.1], b1 vs b8:")
print(f"    model as-is (bf16 head)                 {o_model:6.2f}%  sd {s_model:.4f}")
print(f"    bf16 hidden -> bf16 projection (recomp) {o_bf:6.2f}%  sd {s_bf:.4f}   (should match as-is)")
print(f"    bf16 hidden -> fp32 projection          {o_fp:6.2f}%  sd {s_fp:.4f}   (= TRL cast_lm_head_to_fp32)")
print(f"    bf16 pre-norm -> fp32 norm+projection   {o_nf:6.2f}%  sd {s_nf:.4f}")
json.dump(res, open(f"{OUT}/attribution_{TAG}.json", "w"), indent=1)
