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
own first three explanations. It is not a fix and not a paper.

## What it is not

The phenomenon is already reported. [verl#6280](https://github.com/verl-project/verl/issues/6280)
has been open since 2026-05-08 with three people describing it in production, and it ends with
a question nobody has answered:

> Is this caused by kernels that are not batch-invariant?

These numbers are an attempt to answer that question, from a single-GPU harness small enough
that anyone can re-run it.

## The answer, as far as this harness can establish it

Yes, and specifically:

- it is **bfloat16**, not low precision generally: fp16 has the same 16 bits and cuts the rate by
  an order of magnitude, fp32 removes it entirely;
- it is the **batch**, not the padding: equal-length sequences with no padding at all behave the
  same as ragged padded ones;
- it is **batch size**, not batch membership: eight identical copies of one sequence, scored
  together, differ from that sequence scored alone;
- **except under MoE routing**, where who else is in the batch matters too -- the only model here
  whose per-token log probabilities change when the same batch is merely regrouped;
- it survives **eager**, **sdpa** and **flex_attention**, so it is not one kernel's artefact;
- and it reaches the objective: the fraction of tokens that PPO clipping *excludes* barely
  moves, but **which** tokens are excluded moves by 1.7% to 10.1%.
- it enters **where the architecture puts it**: in the last third of the stack for GPT-NeoX and in
  the first five layers for Qwen2 and Qwen3, so an fp32 `lm_head` removes most of it on Qwen and
  almost none of it on pythia;
- **scoring in fp16 removes it** on every Qwen-family model here at bf16 cost, and fp16 does not
  overflow on Qwen2.5 up to 7B.
- and at 1.5B on GSM8K it does **not** reach the reward: five ways of computing the old
  log-probabilities, same seed, finish within -0.011 to +0.004 of each other after 150 steps, with
  the clip fraction at zero throughout.

## Tables

All six tables are regenerated from `results/*.json` by `scripts/build_tables.py`; nothing in
[TABLES.md](TABLES.md) is typed by hand. Two of them:

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
unchanged, and an fp32 head removes two thirds to four fifths of the out-of-band tokens there.

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

### Does it reach the reward

Five ways of computing the old log-probabilities in TRL's GRPO, run on the same seed so that the
arms share prompt order and vLLM sampling and differ only in that one pass: the trainer's own bf16
pass (A), an fp32 copy of the policy (B), bf16 with the chunk size forced to the training
micro-batch (C), bf16 with the divergence-guided layers and the head in fp32 (D), and bf16 with
only the fp32 head (E). Qwen2.5-1.5B-Instruct on GSM8K, 4 prompts x 8 completions per step,
lr 2e-6, one optimisation step per generation, 150 steps.

<!-- T10 -->
| arm | seeds | reward, last 30 steps | reward, mean over 150 | vs A, same seed | clip fraction | vLLM-vs-old abs dlogp | s/step |
|---|---|---|---|---|---|---|---|
| A default (bf16, trainer chunking) | 1 | 0.682 | 0.637 | - | 0.0000 | 0.0110 | 10.4 |
| B fp32 clone | 1 | 0.686 | 0.637 | +0.004 | 0.0005 | 0.0089 | 13.2 |
| C bf16, chunk = micro-batch | 1 | 0.678 | 0.639 | -0.004 | 0.0000 | 0.0110 | 9.7 |
| D bf16 + guided fp32 layers + head | 1 | 0.683 | 0.651 | +0.001 | 0.0007 | 0.0102 | 12.5 |
| E bf16 + fp32 head only | 1 | 0.671 | 0.640 | -0.011 | 0.0000 | 0.0102 | 9.0 |
<!-- /T10 -->

None of them separates from the default. The last-30-step reward sits within -0.011 to +0.004 of
arm A and the 150-step mean within +0.001 to +0.015, against a per-step noise of sd 0.14 (SE 0.026
on a 30-step mean); the clip fraction is 0.0000 to 0.0005 in every arm, because at one optimisation
step per generation the ratio is 1 up to exactly this noise, and the noise never reaches the 0.2
clip band. The batch-shape noise is real and it is in the ratio; at this scale and these defaults
it does not move the reward. That is the bound this harness can put on it: a few points at most,
with no consistent sign, on one seed. A second seed and an fp16 arm are running and enter T10 as
they finish.

## What this harness cannot tell you

- **It is not verl's production path.** verl runs FSDP2 with static batching and
  `use_remove_padding`, i.e. flash-attn varlen kernels. `flash_attn` will not build here (no
  `nvcc`), so the one configuration that reported `max_abs_diff = 0` is exactly the one I cannot
  test. That null and these numbers are not in contradiction until someone runs both paths on
  the same hardware.
- **One GPU generation.** Everything is RTX 4090, SM89. Reduction strategies differ across
  architectures.
- **Small models.** The largest here is Qwen2.5-7B; the production report was a 30B MoE. Within the Qwen2.5 family the rate drifts from 9.9%, 9.4%, 6.9%, 5.6% (0.5B, 1.5B, 3B, 7B), so size attenuates it slowly within a family, while the spread across families is far larger than the spread across sizes. The lowest rate in the set is Agents-A1-4B. What a 30B MoE does is not something this harness can say.
- **Synthetic advantages.** The clip-flip metric needs an advantage per sequence. It is drawn,
  not earned, and the rate depends on the distribution: 8.59% under Gaussian advantages, 2.47%
  when 90% of them are zero. Reported as a range for that reason.

## Three explanations this repository killed, mine

1. *"Enabling vLLM's batch-invariant kernels should shrink it."* It does not: 8.0% to 7.1%,
   while changing the trainer's precision goes to 3.7% in one step. Predicted in public, wrong.
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

## License

MIT.
