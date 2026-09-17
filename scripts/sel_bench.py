"""Cost axis for the precision budget: teacher-forced scoring throughput (tokens/s) and peak
memory of each precision configuration, same hooks as selective_fp32.py. argv: model, then
configs as name=layerspec[,head] e.g. base= head=,h last8=16-23,h guided=14,17,18,19,20,21,22,23,h"""
import os as _os
_RESULTS = _os.environ.get("BLG_DIR") or _os.path.join(
    _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "results")
import os, sys, time, json, torch
os.environ.setdefault("HF_HUB_OFFLINE", "1")
sys.argv, cfgs = sys.argv[:2], sys.argv[2:]
os.environ["BLG_NOEXEC"] = "1"
from transformers import AutoModelForCausalLM, AutoTokenizer
MODEL = sys.argv[1]; TAG = MODEL.split("/")[-1]; OUT = _RESULTS
B, T = int(os.environ.get("BB", "8")), int(os.environ.get("TT", "512"))

def _cast(x, dt):
    if torch.is_tensor(x) and x.is_floating_point(): return x.to(dt)
    if isinstance(x, tuple): return tuple(_cast(v, dt) for v in x)
    if isinstance(x, dict): return {k: _cast(v, dt) for k, v in x.items()}
    return x
def upcast(mod, cast_out_back=True, functional_linear=False):
    if functional_linear:
        W32 = mod.weight.detach().float()
        mod.register_forward_hook(lambda md, a, out: torch.nn.functional.linear(a[0].float(), W32)); return
    mod.float()
    mod.register_forward_pre_hook(lambda md, a, kw: (_cast(a, torch.float32), _cast(kw, torch.float32)), with_kwargs=True)
    if cast_out_back: mod.register_forward_hook(lambda md, a, out: _cast(out, torch.bfloat16))

def build(spec):
    # a spec token "f16" scores the whole model in float16 (the Precision-RL baseline); everything else is bf16 + fp32 picks
    parts0 = [p for p in spec.split(",") if p]; f16 = "f16" in parts0; spec = ",".join(p for p in parts0 if p != "f16")
    m = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16 if f16 else torch.bfloat16).to("cuda:0").eval()
    base = getattr(m, "model", None) or getattr(m, "gpt_neox", None); layers = base.layers
    norm = getattr(base, "norm", None) or getattr(base, "final_layer_norm"); head = getattr(m, "lm_head", None) or getattr(m, "embed_out")
    parts = [p for p in spec.split(",") if p]; head_on = "h" in parts; S = []
    for p in parts:
        if p == "h": continue
        if "-" in p: a, b = map(int, p.split("-")); S += list(range(a, b + 1))
        else: S.append(int(p))
    S = sorted(set(S))
    for i in S: upcast(layers[i], cast_out_back=((i + 1) not in S))
    if head_on:
        tied = getattr(m.config, "tie_word_embeddings", False)
        upcast(norm, cast_out_back=tied); upcast(head, cast_out_back=False, functional_linear=tied)
    return m, S, head_on

def bench(m):
    ids = torch.randint(100, 20000, (B, T), device="cuda:0")
    with torch.no_grad():
        for _ in range(3): m(ids).logits.float().log_softmax(-1)
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); t0 = time.time(); n = 10
        for _ in range(n): lp = m(ids).logits.float().log_softmax(-1)
        torch.cuda.synchronize(); dt = time.time() - t0
    return B * T * n / dt, torch.cuda.max_memory_allocated() / 2**30

res = []
for c in cfgs:
    name, spec = c.split("=", 1)
    m, S, h = build(spec); tps, mem = bench(m)
    res.append({"name": name, "layers": S, "head": h, "f16": "f16" in spec.split(","), "tok_per_s": tps, "peak_gib": mem})
    print(f"  {name:<12} layers={str(S)[:28]:<28} head={'y' if h else 'n'}  {tps:>9.0f} tok/s  peak {mem:.2f} GiB", flush=True)
    del m; torch.cuda.empty_cache()
base = res[0]["tok_per_s"]
for r in res: r["rel_time"] = base / r["tok_per_s"]; print(f"  {r['name']:<12} time x{r['rel_time']:.2f}")
json.dump({"model": MODEL, "B": B, "T": T, "configs": res}, open(f"{OUT}/selbench_{TAG}.json", "w"), indent=1)
print("BENCH_DONE")
