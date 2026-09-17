"""Does fp16 scoring overflow on the larger models? For each model: sample 32 continuations with the
bf16 model, then teacher-force the same tokens in fp16 at batch 1 and batch 8 and in bf16 at batch 1.
Report the non-finite log-prob fraction in fp16, the fp16 b1-vs-b8 out-of-band rate, and the
max |hidden| seen in fp16 (fp16 max is 65504)."""
import os as _os
_RESULTS = _os.environ.get("BLG_DIR") or _os.path.join(
    _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "results")
import os, sys, json, torch
os.environ.setdefault("HF_HUB_OFFLINE", "1")
from transformers import AutoModelForCausalLM, AutoTokenizer
OUT = _RESULTS
PROMPTS = [f"{s} {t}." for s in ["Explain in two sentences why", "List three facts about", "Write one sentence about", "Give a short definition of"]
           for t in ["the ocean", "photosynthesis", "gravity", "the printing press", "antibiotics", "monsoons", "the transistor", "vaccination"]]
GEN = int(os.environ.get("GEN", "256"))
res = {}
for MODEL in sys.argv[1:]:
    tok = AutoTokenizer.from_pretrained(MODEL); PAD = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    m = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to("cuda:0").eval()
    torch.manual_seed(1234); rows = []
    for p in PROMPTS:
        ids = tok(p, return_tensors="pt").input_ids.to("cuda:0")
        with torch.no_grad():
            o = m.generate(ids, do_sample=True, temperature=1.0, top_p=1.0, max_new_tokens=GEN, min_new_tokens=GEN, pad_token_id=PAD, eos_token_id=None)
        rows.append({"p": ids[0].tolist(), "g": o[0, ids.shape[1]:].tolist()})
    NP = min(len(r["p"]) for r in rows); rows = [{"p": r["p"][-NP:], "g": r["g"]} for r in rows]; NG = len(rows[0]["g"])
    gen = torch.tensor([r["g"] for r in rows])
    def score(model, batch, track=None):
        out = []
        for s in range(0, len(rows), batch):
            ids = torch.tensor([r["p"] + r["g"] for r in rows[s:s + batch]], device="cuda:0")
            with torch.no_grad():
                o = model(ids, attention_mask=torch.ones_like(ids), output_hidden_states=track is not None)
            if track is not None:
                for h in o.hidden_states: track.append(h.float().abs().max().item())
            lp = torch.log_softmax(o.logits[:, NP - 1:NP - 1 + NG, :].float(), -1).cpu()
            out.append(torch.gather(lp, -1, gen[s:s + batch].unsqueeze(-1)).squeeze(-1))
        return torch.cat(out)
    bf1 = score(m, 1); del m; torch.cuda.empty_cache()
    m16 = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16).to("cuda:0").eval()
    hmax = []; f1 = score(m16, 1, hmax); f8 = score(m16, 8)
    del m16; torch.cuda.empty_cache()
    nonfin = (~torch.isfinite(f1)).float().mean().item()
    d = (f1 - f8).flatten(); r = torch.exp(d); oob = (((r < 0.9) | (r > 1.1)) | ~torch.isfinite(d)).float().mean().item()
    d2 = (f1 - bf1).flatten(); r2 = torch.exp(d2); oob2 = (((r2 < 0.9) | (r2 > 1.1)) | ~torch.isfinite(d2)).float().mean().item()
    res[MODEL] = {"n_tokens": int(gen.numel()), "fp16_nonfinite": nonfin, "fp16_b1_vs_b8_oob": 100 * oob, "fp16_b1_vs_bf16_b1_oob": 100 * oob2,
                  "max_abs_hidden_fp16": max(hmax), "n_layers_hidden_over_60000": sum(h > 60000 for h in hmax)}
    print(f"### {MODEL}: tokens={gen.numel()} fp16 non-finite {100*nonfin:.3f}%  fp16 b1-vs-b8 OOB {100*oob:.2f}%  fp16-vs-bf16(b1) OOB {100*oob2:.2f}%  max|hidden| {max(hmax):.0f} ({sum(h > 60000 for h in hmax)} states >60000)", flush=True)
json.dump(res, open(f"{OUT}/fp16_check.json", "w"), indent=1); print("FP16CHECK_DONE")
