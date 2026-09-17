"""Four-arm GRPO on GSM8K: does the trainer's own batch-shape log-prob noise change learning?

The arms differ only in how `old_per_token_logps` - the denominator of the importance
ratio TRL applies when generating with vLLM - is computed:

  A default   TRL as shipped: bf16, chunked by its own batch_size
  B fp32      the same tokens scored by an fp32 clone of the policy (oracle: ~0% out-of-band)
  C matched   bf16, chunk size forced to per_device_train_batch_size (the verl workaround)
  D selective bf16 clone with fp32 only in the divergence-guided layers + an fp32 head
              (functional, so it also works on tied embeddings where TRL's flag refuses)
  E head      bf16 clone with only the fp32 norm+head (the ScaleRL / cast_lm_head_to_fp32 recipe)
  F fp16      fp16 clone, no fp32 at all (the Precision-RL recipe applied to the scoring pass only)

Everything else - data order, seed, sampling, lr, steps - is identical. The clone is
synchronised from the live policy before every generation step and used only for that
one no-grad pass, so the training model and optimizer are untouched.
"""
import os as _os
_RESULTS = _os.environ.get("BLG_DIR") or _os.path.join(
    _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "results")
import os, sys, re, json, time, argparse
os.environ.setdefault("HF_HUB_OFFLINE", "1"); os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

ap = argparse.ArgumentParser()
ap.add_argument("--arm", choices=["A", "B", "C", "D", "E", "F"], required=True)
ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
ap.add_argument("--clone_device", default="cuda:1", help="arms B/D: where the scoring clone lives")
ap.add_argument("--steps", type=int, default=150)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--tail", type=int, default=8, help="arm D: number of last layers in fp32 (ignored if --layers given)")
ap.add_argument("--layers", default="", help="arm D: explicit comma-separated fp32 layer ids (the divergence-guided pick)")
ap.add_argument("--out", default=_os.path.join(_RESULTS, "q1"))
ap.add_argument("--nprompts", type=int, default=2048)
ap.add_argument("--parser", choices=["strict", "lenient"], default="lenient")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
RUN = f"{a.arm}_s{a.seed}"

SYSTEM = ("Solve the problem. Reason briefly, then give the final integer answer on its own line "
          "in the form '#### <number>'.")
ds = load_dataset("openai/gsm8k", "main")["train"].shuffle(seed=0).select(range(a.nprompts))
ds = ds.map(lambda x: {"prompt": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": x["question"]}],
                       "gold": x["answer"].split("####")[-1].strip().replace(",", "")},
            remove_columns=ds.column_names)


def accuracy_reward(completions, gold, **kw):
    out = []
    for c, g in zip(completions, gold):
        txt = c[0]["content"] if isinstance(c, list) else c
        if a.parser == "strict":
            m = re.findall(r"####\s*(-?\d[\d,]*)", txt)
            out.append(1.0 if m and m[-1].replace(",", "") == g else 0.0)
        else:
            m = re.findall(r"-?\d[\d,]*\.?\d*", txt)
            out.append(1.0 if m and m[-1].replace(",", "").rstrip(".").split(".")[0] == g else 0.0)
    return out


# ---------------- arm-specific old-logprob scoring ----------------
def _cast(x, dt):
    if torch.is_tensor(x) and x.is_floating_point(): return x.to(dt)
    if isinstance(x, tuple): return tuple(_cast(v, dt) for v in x)
    if isinstance(x, list): return [_cast(v, dt) for v in x]
    if isinstance(x, dict): return {k: _cast(v, dt) for k, v in x.items()}
    return x


class SelectiveClone:
    """bf16 copy of the policy with the last `tail` decoder layers, the final norm and the
    head evaluated in fp32. Hooks cast activations up on entry and back down where the next
    module is bf16. A tied head is projected with a detached fp32 weight copy, refreshed on sync."""
    def __init__(self, name, tail, device, dtype=torch.bfloat16, layers=None):
        self.dev = device
        self.m = AutoModelForCausalLM.from_pretrained(name, dtype=dtype).to(device).eval()
        base = getattr(self.m, "model", None) or getattr(self.m, "gpt_neox", None)
        self.layers = base.layers; self.norm = getattr(base, "norm", None) or getattr(base, "final_layer_norm")
        self.head = getattr(self.m, "lm_head", None) or getattr(self.m, "embed_out")
        self.tied = getattr(self.m.config, "tie_word_embeddings", False)
        NL = len(self.layers); S = sorted(layers) if layers is not None else (list(range(NL - tail, NL)) if tail > 0 else [])
        self.S = S
        for i in S:
            self.layers[i].float()
            self.layers[i].register_forward_pre_hook(lambda md, ar, kw: (_cast(ar, torch.float32), _cast(kw, torch.float32)), with_kwargs=True)
            if (i + 1) not in S:
                self.layers[i].register_forward_hook(lambda md, ar, out: _cast(out, torch.bfloat16))
        self.norm.float()
        self.norm.register_forward_pre_hook(lambda md, ar, kw: (_cast(ar, torch.float32), _cast(kw, torch.float32)), with_kwargs=True)
        if self.tied:
            self.norm.register_forward_hook(lambda md, ar, out: _cast(out, torch.bfloat16))
            self.W32 = self.head.weight.detach().float()
            self.head.register_forward_hook(lambda md, ar, out: torch.nn.functional.linear(ar[0].float(), self.W32))
        else:
            self.head.float()
            self.head.register_forward_pre_hook(lambda md, ar, kw: (_cast(ar, torch.float32), _cast(kw, torch.float32)), with_kwargs=True)
        self.dtypes = {k: v.dtype for k, v in self.m.state_dict().items()}

    def sync(self, src_state):
        with torch.no_grad():
            self.m.load_state_dict(src_state, strict=True)   # copy_ casts to each param's dtype
            if self.tied: self.W32 = self.head.weight.detach().float()


class ArmTrainer(GRPOTrainer):
    def __init__(self, *args, arm="A", clone=None, **kw):
        super().__init__(*args, **kw)
        self.arm = arm; self.clone = clone; self.n_old_calls = 0

    def _get_per_token_logps_and_entropies(self, model, input_ids, attention_mask, logits_to_keep,
                                           batch_size=None, *args, **kw):
        # the old-logprob pass is the no-grad call; the training forward has grad enabled.
        if torch.is_grad_enabled():
            return super()._get_per_token_logps_and_entropies(model, input_ids, attention_mask, logits_to_keep, batch_size, *args, **kw)
        self.n_old_calls += 1
        if self.arm == "A":
            return super()._get_per_token_logps_and_entropies(model, input_ids, attention_mask, logits_to_keep, batch_size, *args, **kw)
        if self.arm == "C":  # B/D/E fall through to the clone
            return super()._get_per_token_logps_and_entropies(model, input_ids, attention_mask, logits_to_keep,
                                                               self.args.per_device_train_batch_size, *args, **kw)
        # B / D: score with the synchronised clone
        src = self.accelerator.unwrap_model(model).state_dict()
        self.clone.sync(src)
        dev = input_ids.device
        r = super()._get_per_token_logps_and_entropies(self.clone.m, input_ids.to(self.clone.dev), attention_mask.to(self.clone.dev),
                                                        logits_to_keep, batch_size, *args, **kw)
        return tuple(x.to(dev) if torch.is_tensor(x) else x for x in r)


clone = None
if a.arm == "B":
    class _FP32: pass
    c = _FP32(); c.dev = a.clone_device
    c.m = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float32).to(a.clone_device).eval()
    c.sync = lambda st: c.m.load_state_dict(st, strict=True); clone = c
elif a.arm == "D":
    clone = SelectiveClone(a.model, a.tail, a.clone_device, layers=[int(x) for x in a.layers.split(",")] if a.layers else None)
    print("arm D fp32 layers:", clone.S, "tied:", clone.tied)
elif a.arm == "E":
    clone = SelectiveClone(a.model, 0, a.clone_device, layers=[])
elif a.arm == "F":
    class _FP16: pass
    c = _FP16(); c.dev = a.clone_device; c.S = []
    c.m = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float16).to(a.clone_device).eval()
    c.sync = lambda st: c.m.load_state_dict(st, strict=True); clone = c

cfg = GRPOConfig(
    output_dir=f"{a.out}/{RUN}", seed=a.seed, learning_rate=2e-6, lr_scheduler_type="constant",
    per_device_train_batch_size=4, gradient_accumulation_steps=8, num_generations=8,
    max_completion_length=384, max_steps=a.steps, gradient_checkpointing=True, logging_steps=1,
    save_strategy="no", report_to=[], bf16=True, temperature=1.0, beta=0.0, epsilon=0.2,
    use_vllm=True, vllm_mode="colocate", vllm_gpu_memory_utilization=0.18, vllm_max_model_length=1024,
    vllm_importance_sampling_correction=True, log_completions=False, disable_tqdm=True,
)
model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16)
tok = AutoTokenizer.from_pretrained(a.model)
tr = ArmTrainer(model=model, processing_class=tok, reward_funcs=accuracy_reward, args=cfg,
                train_dataset=ds, arm=a.arm, clone=clone)
t0 = time.time(); tr.train(); dt = time.time() - t0
hist = tr.state.log_history
json.dump({"arm": a.arm, "seed": a.seed, "model": a.model, "steps": a.steps, "tail": a.tail, "layers": (clone.S if a.arm in ("D", "E", "F") else None), "parser": a.parser,
           "wall_s": dt, "n_old_calls": tr.n_old_calls, "log_history": hist},
          open(f"{a.out}/{RUN}.json", "w"), indent=1)
rw = [h["reward"] for h in hist if "reward" in h]
print(f"### {RUN}  steps={len(rw)}  wall {dt/60:.1f} min  {dt/max(1,len(rw)):.1f}s/step  old_calls={tr.n_old_calls}")
if rw:
    k = max(1, len(rw)//5)
    print(f"  reward first{k} {sum(rw[:k])/k:.3f}  last{k} {sum(rw[-k:])/k:.3f}")
    for key in ["clip_ratio/region_mean", "sampling/sampling_logp_difference/mean", "sampling/sampling_logp_difference/max", "kl", "entropy"]:
        v = [h[key] for h in hist if key in h]
        if v: print(f"  {key:<44} first{k} {sum(v[:k])/k:.4f}  last{k} {sum(v[-k:])/k:.4f}")
