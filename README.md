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
python scripts/build_tables.py                        # rebuild TABLES.md from results/
```

Single GPU, no distributed setup, models under 5B. `pythia-410m` shows the largest effect and
is the fastest to run.

## License

MIT.
