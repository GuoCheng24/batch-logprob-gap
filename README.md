# batch-logprob-gap

**In bfloat16, a language model gives the same token a different log probability depending on
how many sequences share its batch.** The policy has not moved and the tokens are identical;
only the shape of the tensor around them changed. Across 4 architectures and 8 models the resulting
importance ratio `exp(logp_new - logp_old)` leaves `[0.9, 1.1]` for **1.8% to 60.0%** of tokens,
and in fp32 it leaves it for **0.00%**.

This matters to GRPO and every other clipped-surrogate RL objective, because at the first
inner epoch that ratio is the identity by construction. Whatever moves it there is an
off-policy correction the algorithm applies to a policy that never changed.

This repository is the measurement, the controls that survived, and the ones that killed my
own first three explanations. It also says where the effect stops: at 1.5B on GSM8K, six
different ways of computing that log probability — including removing the noise outright —
finish within 0.013 reward of each other after 150 steps on two seeds. The noise is real, it
is in the ratio, and at this scale it does not move what the model learns. Both halves of that
are measured here.

**Written up as a six-page report: [paper/report.pdf](paper/report.pdf)** — the same measurement
with every control in one place, assembled by [`paper/build.py`](paper/build.py), which splices the
tables in from [TABLES.md](TABLES.md) verbatim and refuses to build if a number in its prose does not
occur in a committed result file.

## What it is not

The phenomenon is already reported. [verl#6280](https://github.com/verl-project/verl/issues/6280)
has been open since 2026-05-08 with three people describing it in production, and it ends with
a question nobody has answered:

> Is this caused by kernels that are not batch-invariant?

These numbers are an attempt to answer that question, from a single-GPU harness small enough
that anyone can re-run it.

## The answer, as far as this harness can establish it

Yes, and specifically:

- it is **bfloat16**, not low precision generally: fp16 has the same 16 bits, three more of them
  mantissa, and on most of these models that is enough — 0.07% and below across the Qwen family,
  0.97% on the MoE — but on GPT-NeoX it only buys one to two orders of magnitude (60.0% to 22.0%
  at 160M, 36.7% to 2.5% at 410M). fp32 removes it everywhere;
- it is the **batch**, not the padding: equal-length sequences with no padding at all behave the
  same as ragged padded ones;
- it is **batch size**, not batch membership: eight identical copies of one sequence, scored
  together, differ from that sequence scored alone;
- **except under MoE routing**, where who else is in the batch matters too -- the only model here
  whose per-token log probabilities change when the same batch is merely regrouped;
- it survives **eager**, **sdpa** and **flex_attention**, so it is not one kernel's artefact;
- and it reaches the objective: the fraction of tokens that PPO clipping *excludes* barely
  moves, but **which** tokens are excluded moves by 1.7% to 10.1%;
- it enters **where the architecture puts it**: in the last third of the stack for GPT-NeoX and in
  the first five layers for Qwen2 and Qwen3, so an fp32 `lm_head` removes most of it on Qwen and
  almost none of it on pythia;
- **scoring in fp16 costs nothing and removes it** where fp16 is enough (0.93x to 0.99x the bf16
  time), and fp16 does not overflow on Qwen2.5 up to 7B;
- and at 1.5B on GSM8K it does **not** reach the reward: six ways of computing the old
  log-probabilities, same seeds, finish within -0.013 to +0.008 of each other after 150 steps on
  both seeds, with the clip fraction at zero throughout.

## Tables

All eleven tables are regenerated from `results/*.json` by `scripts/build_tables.py`; nothing in
[TABLES.md](TABLES.md) is typed by hand:

| | |
|---|---|
| [T1](TABLES.md#t1--dtype-x-batch-no-inference-engine-involved) dtype × batch, no inference engine | [T7](TABLES.md#t7--precision-ladder-where-must-fp32-go-bf16-batch-1-vs-8-6144-tokens) precision ladder: where must fp32 go |
| [T2](TABLES.md#t2--is-it-padding-batch-size-or-who-shares-the-batch) padding, batch size, or batch membership | [T7b](TABLES.md#t7b--what-each-rung-costs-teacher-forced-scoring-time-relative-to-bf16-batch-8-x-512-tokens) what each rung of that ladder costs |
| [T3](TABLES.md#t3--three-ways-to-change-the-kernel-path-in-bf16-and-fp32) three ways to change the kernel path | [T8](TABLES.md#t8--where-the-divergence-enters-b1-vs-b8-relative-hidden-state-divergence-by-quarter-of-the-stack) where the divergence enters, layer by layer |
| [T4](TABLES.md#t4--attention-implementation-and-repeatability-bf16-batch-1-vs-8) attention implementation and repeatability | [T9](TABLES.md#t9--does-fp16-overflow-on-the-larger-models-8192-tokens-256-token-continuations) does fp16 overflow at 3B and 7B |
| [T5](TABLES.md#t5--does-it-reach-the-gradient-ppo-clip-status-flips) does it reach the gradient: clip-status flips | [T10](TABLES.md#t10--does-the-old-log-prob-precision-reach-the-reward-grpo-on-gsm8k-qwen25-15b-instruct-150-steps) does it reach the reward: six GRPO arms |
| [T6](TABLES.md#t6--clip-flip-rate-depends-on-the-advantage-distribution) clip flips vs the advantage distribution | [T11](TABLES.md#t11--vllm-importance-ratio-under-truncated-sampling-full-vocabulary-trainer-log-probs-vs-the-same-log-probs-renormalised-over-vllms-replayed-support-qwen25-15b) the truncated-sampling term in the ratio |

Four of them are reproduced below; the report has all eleven with the prose that connects them.

### The effect is bf16, and it is not randomness

<!-- T1 -->
| model | bf16 b1 vs b8 | fp16 b1 vs b8 | fp32 b1 vs b8 | bf16 b1 rerun | bf16 b1 vs b32 |
|---|---|---|---|---|---|
| pythia-160m (NeoX 0.16B) | 60.03% | 22.02% | 0.00% | 0.00% | 59.70% |
| pythia-410m (NeoX 0.41B) | 36.70% | 2.49% | 0.00% | 0.00% | 36.65% |
| Qwen2.5-0.5B | 9.87% | 0.07% | 0.00% | 0.00% | 10.75% |
| Granite-3.1-1B-A400M (MoE) | 13.69% | 0.97% | 0.00% | 0.00% | 13.85% |
| Agents-A1-4B | 1.78% | 0.00% | 0.00% | 0.00% | 1.64% |
| Qwen2.5-1.5B | 9.37% | 0.00% | 0.00% | 0.00% | 8.87% |
| Qwen2.5-3B | 6.87% | 0.00% | 0.00% | 0.00% | 7.17% |
| Qwen2.5-7B | 5.60% | 0.00% | 0.00% | 0.00% | 5.71% |
<!-- /T1 -->

Percentage of scored tokens whose importance ratio falls outside `[0.9, 1.1]`, 3,518 to 4,096 tokens per
cell. The rerun column is the control that matters: repeating the *same* batch is bit-exact, so
this is not nondeterminism, it is a deterministic function of the batch shape.

### It is not one card either

<!-- T1b -->
| model | tokens | RTX 4090 | L40 | difference |
|---|---|---|---|---|
| pythia-410m | 3,970 | 36.70% | 36.47% | -0.23 |
| Qwen2.5-0.5B | 4,085 | 9.87% | 9.77% | -0.10 |
| Qwen2.5-1.5B | 3,959 | 9.37% | 8.66% | -0.71 |
<!-- /T1b -->

The same script, the same venv (torch 2.10.0+cu128, transformers 5.16.1), the same models, on an
RTX 4090 and on an L40. The 4090 arm was re-run alongside the L40 one, long after the numbers in T1
were recorded, and reproduces them **exactly** - 1457, 1455, 0, 99, 0 out-of-band tokens for
pythia-410m, every cell. So what follows is a comparison between two cards and not between two
environments.

The rate moves by 0.10 to 0.71 points. `bf16 b1 rerun` is 0 on both cards for all three models, so
the forward pass is bit-reproducible within a machine on either one, and fp32 is 0 on both. What
changes between the cards is the effect's exact size, not whether it is there and not whether fp32
removes it.

This narrows the limitation below rather than removing it: an L40 and an RTX 4090 are both Ada and
both compute capability 8.9. It is a different chip - 142 streaming multiprocessors against 128, a
different memory system - not a different architecture. `scripts/hw_compare.py` prints the table and
checks the control.

### Padding, batch size, and batch membership, separated

<!-- T2 -->
| model | A padded ragged b1v8 | B equal-length no pad b1v8 | C same seq x8 vs alone | D two groupings of batch 8 |
|---|---|---|---|---|
| pythia-410m (NeoX 0.41B) | 36.87% | 36.23% | 18.75% | 0.00% |
| Qwen2.5-0.5B | 9.74% | 10.06% | 6.25% | 0.00% |
| Granite-3.1-1B-A400M (MoE) | 14.09% | 14.21% | 20.31% | 12.60% |
| Agents-A1-4B | 1.78% | 3.25% | 6.25% | 0.00% |
<!-- /T2 -->

B kills padding as the explanation. C shows batch size alone is sufficient. D is zero for every
dense model and **12.60% for the MoE**, which matches the report in verl#6280 that the mismatch
appeared on Qwen3.5-35B-A3B and not on dense models: expert routing depends on batch
composition, so MoE has a second channel the dense models do not.

### Where it enters, and what precision buys

<!-- T7 -->
| model | bf16 | fp32 head only | last 8 + head | first 8 + head | guided 8 + head | fp16, no fp32 | fp16 + guided 8 | all fp32 (sanity) |
|---|---|---|---|---|---|---|---|---|
| DeepSeek-R1-Distill-Qwen-1.5B | 6.38% (0%) | 1.94% (13%) | 1.33% (34%) | 0.60% (34%) | 0.68% (34%) | 0.00% (0%) | 0.00% (34%) | 0.00% (87%) |
| Qwen2.5-0.5B | 8.92% (0%) | 2.72% (22%) | 1.30% (41%) | 0.81% (41%) | 0.86% (41%) | 0.02% (0%) | 0.00% (41%) | 0.00% (78%) |
| Qwen2.5-1.5B | 8.04% (0%) | 1.51% (13%) | 0.94% (34%) | 1.07% (34%) | 0.93% (34%) | 0.00% (0%) | 0.00% (34%) | 0.00% (87%) |
| Qwen3-1.7B | 5.58% (0%) | 1.99% (15%) | 1.42% (35%) | 1.04% (35%) | 1.06% (35%) | 0.00% (0%) | 0.00% (35%) | 0.00% (85%) |
| pythia-410m (NeoX 0.41B) | 34.26% (0%) | 32.24% (13%) | 1.12% (38%) | 32.31% (38%) | 0.41% (38%) | 0.47% (0%) | 0.00% (38%) | 0.00% (87%) |
<!-- /T7 -->

Out-of-band rate at batch 1 vs 8 on 6,144 tokens, with the share of parameters held in fp32 in
parentheses; `guided 8` is the eight decoder layers with the largest batch-1-vs-batch-8 divergence
increment, chosen on a separate 1,536-token pass (T8 in [TABLES.md](TABLES.md)).

The divergence between the two batch shapes can be followed layer by layer, and it is not the head.
On pythia-410m an fp32 `lm_head` leaves the rate at 32.24% (from 34.26%) because the residual stream
has already drifted apart in the last third of the stack: the last quarter of the layers contributes
+9.0e-2 of the 0.115 total. The Qwen2 and Qwen3 models are the mirror image. Their first five layers
inject almost all of it (+1.0e-2 to +1.1e-2 in the first quarter), the middle of the stack carries it
unchanged, and an fp32 head removes 64% to 81% of the out-of-band tokens there.

That profile says which layers are worth upcasting. Holding the eight layers with the largest
increment in fp32, plus the head, costs 34% to 41% of the parameters and reaches 0.41% on pythia and
0.68% to 1.06% on the Qwen models; the same budget spent on the *last* eight layers gives 1.12% and
0.94% to 1.42%, spent on the *first* eight gives 32.31% and 0.60% to 1.07%. The profile picks the
right end of the network without being told which family it is looking at.

It is still the wrong fix. Scoring the same tokens in fp16, with no fp32 anywhere and at 0.93x to
0.99x the bf16 time (T7b; the guided layers cost 1.7x to 2.0x, all-fp32 2.8x to 3.4x), gives 0.00% on
three of the four Qwen-family models, 0.02% on Qwen2.5-0.5B and 0.47% on pythia-410m; on Qwen2.5-3B
and 7B fp16 produces no non-finite log probability, and the largest hidden-state magnitude it meets
is 12,712 against a ceiling of 65,504 (T9). Under fp16 the guided layers bring pythia to 0.00%, which
is the only place they still add anything. The ladder stays here as the measurement of *where* bf16
loses batch invariance; the prescription it supports is the one sail-sg published for the whole
training loop ([arXiv 2510.26788](https://arxiv.org/abs/2510.26788)), applied to the scoring pass.

### The other term in the ratio: truncated sampling

The batch-shape noise is not the only thing that moves the importance ratio without a policy change. When
vLLM samples with `top_p`, `top_k` or `min_p`, the log probabilities it returns (`processed_logprobs`, which
is what TRL asks for) are normalised over the tokens that survived the truncation, while the trainer
normalises over the full vocabulary. The ratio then carries a factor equal to the kept probability mass:
at `top_p=0.8` on Qwen2.5-1.5B-Instruct it averages 0.896 for an unchanged policy and leaves `[0.9, 1.1]`
for 51.8% of tokens (T11 in [TABLES.md](TABLES.md)). vLLM 0.28 can return the kept token ids of every
generated token; renormalising the trainer's log probability over that set brings the ratio to 1.001 and
the out-of-band rate to 4.3%, the same floor as the untruncated control. `scripts/trunc_bias.py` is the
measurement; the trainer-side fix is a TRL change, not a change to this harness.

### Does it reach the reward

Six ways of computing the old log-probabilities in TRL's GRPO, run on the same seeds so that the
arms share prompt order and vLLM sampling and differ only in that one pass: the trainer's own bf16
pass (A), an fp32 copy of the policy (B), bf16 with the chunk size forced to the training
micro-batch (C), bf16 with the divergence-guided layers and the head in fp32 (D), bf16 with only
the fp32 head (E), and an fp16 copy of the policy (F). Qwen2.5-1.5B-Instruct on GSM8K, 4 prompts x
8 completions per step, lr 2e-6, one optimisation step per generation, 150 steps, two seeds.

<!-- T10 -->
| arm | seeds | reward, last 30 steps | reward, mean over 150 | vs A, same seed | clip fraction | vLLM-vs-old abs dlogp | s/step |
|---|---|---|---|---|---|---|---|
| A default (bf16, trainer chunking) | 2 | 0.657 | 0.638 | - | 0.0000 | 0.0113 | 9.3 |
| B fp32 clone | 2 | 0.658 | 0.641 | +0.004 / -0.003 | 0.0005 | 0.0092 | 11.8 |
| C bf16, chunk = micro-batch | 2 | 0.655 | 0.639 | -0.004 / +0.000 | 0.0000 | 0.0113 | 8.9 |
| D bf16 + guided fp32 layers + head | 2 | 0.651 | 0.647 | +0.001 / -0.013 | 0.0007 | 0.0104 | 10.9 |
| E bf16 + fp32 head only | 2 | 0.651 | 0.639 | -0.011 / -0.001 | 0.0000 | 0.0104 | 8.8 |
| F fp16 clone | 2 | 0.661 | 0.650 | +0.008 / +0.001 | 0.0006 | 0.0094 | 8.5 |
<!-- /T10 -->

None of them separates from the default. Per seed, the last-30-step reward sits within -0.013 to
+0.008 of arm A and the 150-step mean within +0.001 to +0.011, against a per-step noise of sd 0.14
(SE 0.026 on a 30-step mean); across the two seeds the arm means differ by less than the seed-to-seed
spread of any single arm (0.03 to 0.05). The clip fraction is 0.0000 to 0.0005 in every arm, because
at one optimisation step per generation the ratio is 1 up to exactly this noise, and the noise never
reaches the 0.2 clip band. The batch-shape noise is real and it is in the ratio; at this scale and
these defaults it does not move the reward, and neither does removing it (B, F) or removing only part
of it (D, E). That is the bound this harness can put on it: about a point, with no consistent sign,
on two seeds.

## What this harness cannot tell you

- **It is not verl's production path.** verl runs FSDP2 with static batching and
  `use_remove_padding`, i.e. flash-attn varlen kernels. `flash_attn` will not build here (no
  `nvcc`), so the one configuration that reported `max_abs_diff = 0` is exactly the one I cannot
  test. That null and these numbers are not in contradiction until someone runs both paths on
  the same hardware.
- **One architecture.** T1b adds an L40 and the rate moves by 0.10 to 0.71 points, with the
  bit-exact rerun control at 0 and fp32 at 0 on both cards. But both are Ada, compute capability
  8.9. A genuinely different architecture - Hopper, or anything pre-Ampere where bf16 is not
  native - is still untested, and reduction strategies are chosen per architecture.
- **Small models.** The largest here is Qwen2.5-7B; the production report was a 30B MoE. Within the Qwen2.5 family the rate drifts from 9.9%, 9.4%, 6.9%, 5.6% (0.5B, 1.5B, 3B, 7B), so size attenuates it slowly within a family, while the spread across families is far larger than the spread across sizes. The lowest rate in the set is Agents-A1-4B. What a 30B MoE does is not something this harness can say.
- **Synthetic advantages.** The clip-flip metric needs an advantage per sequence. It is drawn,
  not earned, and the rate depends on the distribution: 8.59% under Gaussian advantages, 2.47%
  when 90% of them are zero. Reported as a range for that reason.

## Three explanations this repository killed, mine

1. *"Enabling vLLM's batch-invariant kernels should shrink it."* It does not: 8.0% to 7.1%,
   while changing the trainer's precision goes to 3.7% in one step. Predicted in public, wrong.
   ([results/kernel_arms.json](results/kernel_arms.json), [logs/measure.log](logs/measure.log);
   3,916 tokens rescored under each kernel set by `scripts/measure.py`.)
2. *"Generate under both kernel settings and compare token by token."* Not a valid comparison --
   45 of 64 sequences diverge, median at generated token 21. The paired design had to be
   replaced by teacher-forced rescoring.
3. *"Sequence packing removes it, which is why verl sees zero."* Packing with an explicit
   block-diagonal mask gives 32.23% where padding gives 31.84%. Packing is a different path, not
   a safe one. (And the first version of that experiment had no block-diagonal mask at all,
   which an assertion caught: `sd 2.80`, obviously wrong rather than subtly wrong.)

## Reproducing

```bash
python scripts/hf_only.py   EleutherAI/pythia-410m    # T1: dtype x batch
python scripts/pad_control.py EleutherAI/pythia-410m  # T2: padding / size / membership
python scripts/dtype_paths.py EleutherAI/pythia-410m  # T3: three path changes, bf16 vs fp32
python scripts/robust.py    EleutherAI/pythia-410m    # T4, T6: implementations, seeds, advantages
python scripts/grpo_effect.py EleutherAI/pythia-410m  # T5: clip-status flips
python scripts/attribution.py EleutherAI/pythia-410m  # T8: per-layer divergence, fp32-head ablation
NREP=4 python scripts/selective_fp32.py EleutherAI/pythia-410m  # T7: precision ladder (needs T8 first)
python scripts/fp16_check.py Qwen/Qwen2.5-3B-Instruct  # T9: fp16 overflow check
python scripts/q1_grpo.py --arm A --seed 0             # T10: one GRPO arm (TRL + vLLM; arms B-F use a second GPU)
python scripts/build_tables.py                        # rebuild TABLES.md from results/
```

Single GPU, no distributed setup, models under 5B. `pythia-410m` shows the largest effect and
is the fastest to run.

## Citing it

[CITATION.cff](CITATION.cff) is the machine-readable form; GitHub's "Cite this repository" reads it.
In text:

> Guo Cheng. *In bfloat16 the batch shape moves the importance ratio: measurement, controls, and what
> removes it.* Technical report, September 2026. https://github.com/GuoCheng24/batch-logprob-gap

## License

MIT.
