# Truncated sampling in verl: what it does to rollout correction, and a cheap fix

verl asks vLLM for `processed_logprobs`. With `top_p`, `top_k` or `min_p` truncating the
distribution, those are normalised over the tokens that survived, while the actor's
`old_log_probs` are normalised over the whole vocabulary. Every token's ratio then carries
the kept probability mass, with no policy change at all. This directory measures what that
does inside verl, and how little data a fix needs.

Everything here ran on verl `main` at 6093e00, vLLM 0.28.0, torch 2.13.0+cu126,
transformers 5.12.1, one NVIDIA L40, GRPO on GSM8K, 32 prompts x 4 samples per step,
temperature 1.0.

## 1. The term is in verl's own metrics

Qwen2.5-0.5B-Instruct, token-level rollout IS, 3 steps (`0a_*` in [results/gates.json](results/gates.json)):

| | `rollout_corr/rollout_is_mean`, steps 1-3 | `rollout_corr/kl` |
|---|---|---|
| `top_p=1.0` | 0.99994, 0.99984, 0.99995 | 0.00075, 0.00081, 0.00067 |
| `top_p=0.8` | 0.96371, 0.96385, 0.96802 | 0.03921, 0.03900, 0.03449 |

## 2. Two presets then reject everything, and training stops behind a generic warning

Qwen2.5-1.5B-Instruct, which earns reward on this task; FSDP2 with CPU offload; 2 steps
(`0c_*`). verl's sampler shuffles with an unseeded `torch.Generator`, whose seed is a fixed
default, so every arm sees the same batches; on them the uncorrected arm's advantages run
from -1.5 to 1.5.

| arm | rejected sequences | `actor/grad_norm` | `critic/score/mean` |
|---|---|---|---|
| `top_p=0.8`, no correction | - | 0.34501, 0.45507 | 0.13281, 0.18750 |
| Geo-RS defaults, `top_p=1.0` | 0.55469, 0.57031 | 0.15140, 0.26527 | 0.05469, 0.19531 |
| Geo-RS defaults, `top_p=0.8` | **1.0, 1.0** | **0.0, 0.0** | 0.08594, 0.16406 |
| Seq-MIS defaults, `top_p=0.8` | **1.0, 1.0** | **0.0, 0.0** | 0.07812, 0.21875 |

"Defaults" are `decoupled_geo_rs()` (`seq_mean_k1`, `0.999_1.001`) and `decoupled_seq_is_rs()`
(sequence IS at 2.0, `seq_sum_k1` at `0.5_2.0`). When every token is rejected, the only sign
is a generic warning at each step, `Response mask is all False, returning default advantage
metrics`, with `critic/advantages/*` logged as NaN. Nothing in it names the rejection, the
preset or the truncation behind it; the number that gives it away is `actor/grad_norm` at
exactly 0. The
rejection at `top_p=1.0` comes from the engine/trainer mismatch that remains without
truncation in this setup (sdpa, `use_remove_padding=False`); it was not checked with
flash-attn.

The same pattern on Qwen2.5-0.5B-Instruct (`0b_*`) adds a third preset, K3-RS
(`seq_mean_k3`, 0.01), which rejects nothing at either `top_p`: for a near-uniform shift K3
is second order in the log-ratio, so it does not see the term that is still in the weights.

## 3. The fix needs one integer per token

top-k, top-p and min-p all keep a prefix of the tokens sorted by probability, so the kept set
is "the |S| most likely tokens". vLLM 0.28 returns the kept ids (`return_sampling_mask=True`);
replaying only their count lets the trainer take its own top-|S| and renormalise. On the same
sampled tokens ([support_size_replay.py](support_size_replay.py), Qwen2.5-1.5B-Instruct,
64 GSM8K prompts, trainer = HF bf16 at batch 1, [results](results/support_size_replay.json)):

| sampling | mean \|S\| | trainer top-\|S\| equals vLLM's set | ratio mean, no correction | exact ids | size only | out of [0.9, 1.1]: none / ids / size |
|---|---|---|---|---|---|---|
| `top_p=0.8` | 1.31 | 99.52% | 0.9640 | 0.9999 | 0.9999 | 16.37% / 0.90% / 0.91% |
| `top_p=0.95` | 2.35 | 98.37% | 0.9869 | 0.9999 | 0.9999 | 2.63% / 2.04% / 2.03% |
| `top_p=0.9`, T=0.7 | 1.27 | 99.65% | 0.9873 | 0.9994 | 0.9994 | 3.82% / 2.34% / 2.35% |
| `top_k=20` | 20.00 | 72.09% | 0.9978 | 1.0002 | 1.0002 | 2.69% / 2.50% / 2.50% |
| `min_p=0.05` | 1.65 | 99.87% | 0.9822 | 0.9999 | 0.9999 | 5.16% / 1.58% / 1.58% |

Where the two sets differ, they differ at the boundary, on tokens with almost no mass, so
the two corrections agree to the fourth decimal. The count is 1 integer per token; the ids
are |S| per token, and 1024 under a `top_k=1024` cap with `top_p=1.0`.

## Reproducing

```bash
python verl/examples/data_preprocess/gsm8k.py --local_save_dir ~/data/gsm8k   # from a verl checkout
bash run_gates.sh 0a     # then 0b, 0c; each run writes logs/<tag>.log
python collect.py results/gates.json logs/*.log
python support_size_replay.py Qwen/Qwen2.5-1.5B-Instruct results/support_size_replay.json
```

`gate.sh` refuses to start on a GPU with less than 30 GB free. The raw logs are not committed:
they are mostly Ray's and vLLM's own output, full of machine-specific paths; `collect.py`
turns any set of them into the JSON above, and it reproduces the committed file from the
original logs value for value.

## What this does not show

Two steps per arm say that the gradient is zero, not what a long run does with it. Everything
is one GPU and at most 1.5B. The trainer in section 3 is HF transformers, not verl's actor, so
it bounds the replay error rather than measuring it in verl.
