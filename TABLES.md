## T1  dtype x batch, no inference engine involved

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

## T2  is it padding, batch size, or who shares the batch

| model | A padded ragged b1v8 | B equal-length no pad b1v8 | C same seq x8 vs alone | D two groupings of batch 8 |
|---|---|---|---|---|
| pythia-410m (NeoX 0.41B) | 36.87% | 36.23% | 18.75% | 0.00% |
| Qwen2.5-0.5B | 9.74% | 10.06% | 6.25% | 0.00% |
| Granite-3.1-1B-A400M (MoE) | 14.09% | 14.21% | 20.31% | 12.60% |
| Agents-A1-4B | 1.78% | 3.25% | 6.25% | 0.00% |

## T3  three ways to change the kernel path, in bf16 and fp32

| model | change | bf16 | fp32 |
|---|---|---|---|
| pythia-410m (NeoX 0.41B) | batch shape [1,L] vs [8,L] | 31.84% | 0.00% |
| pythia-410m (NeoX 0.41B) | mask implicit vs explicit 4D | 21.35% | 0.00% |
| pythia-410m (NeoX 0.41B) | packing 1/row vs 8/row | 32.23% | 0.00% |
| Qwen2.5-0.5B | batch shape [1,L] vs [8,L] | 9.18% | 0.00% |
| Qwen2.5-0.5B | mask implicit vs explicit 4D | 4.82% | 0.00% |
| Qwen2.5-0.5B | packing 1/row vs 8/row | 9.44% | 0.00% |

## T4  attention implementation and repeatability, bf16 batch 1 vs 8

| model | implementation | out-of-band | 95% CI | per-seed | n |
|---|---|---|---|---|---|
| pythia-410m (NeoX 0.41B) | eager | 53.91% | [52.46, 55.34] | 53.71, 54.30, 53.71 | 4608 |
| pythia-410m (NeoX 0.41B) | sdpa | 32.92% | [31.58, 34.29] | 31.84, 32.62, 34.31 | 4608 |
| pythia-410m (NeoX 0.41B) | flex_attention | 33.29% | [31.94, 34.66] | 33.40, 31.58, 34.90 | 4608 |
| Qwen2.5-0.5B | eager | 15.13% | [14.12, 16.19] | 14.13, 15.23, 16.02 | 4608 |
| Qwen2.5-0.5B | sdpa | 9.29% | [8.48, 10.16] | 9.18, 9.96, 8.72 | 4608 |
| Qwen2.5-0.5B | flex_attention | 9.18% | [8.38, 10.05] | 10.03, 9.31, 8.20 | 4608 |

## T5  does it reach the gradient: PPO clip-status flips

| model | clipped at b1 | clipped at b8 | clip status flips | n |
|---|---|---|---|---|
| pythia-410m (NeoX 0.41B) | 19.97% | 20.07% | **10.11%** | 4096 |
| Qwen2.5-0.5B | 21.41% | 21.48% | **1.68%** | 4096 |
| Granite-3.1-1B-A400M (MoE) | 3.15% | 2.86% | **2.78%** | 4096 |

## T6  clip-flip rate depends on the advantage distribution

| model | advantage | flips | 95% CI |
|---|---|---|---|
| pythia-410m (NeoX 0.41B) | gaussian | 8.59% | [7.29, 10.10] |
| pythia-410m (NeoX 0.41B) | binary +/-1 | 7.03% | [5.86, 8.42] |
| pythia-410m (NeoX 0.41B) | sparse (90% zero) | 2.47% | [1.81, 3.38] |
| Qwen2.5-0.5B | gaussian | 2.02% | [1.43, 2.85] |
| Qwen2.5-0.5B | binary +/-1 | 2.02% | [1.43, 2.85] |
| Qwen2.5-0.5B | sparse (90% zero) | 0.52% | [0.26, 1.02] |

## T7  precision ladder: where must fp32 go, bf16 batch 1 vs 8, 6,144 tokens

| model | bf16 | fp32 head only | last 8 + head | first 8 + head | guided 8 + head | fp16, no fp32 | fp16 + guided 8 | all fp32 (sanity) |
|---|---|---|---|---|---|---|---|---|
| DeepSeek-R1-Distill-Qwen-1.5B | 6.38% (0%) | 1.94% (13%) | 1.33% (34%) | 0.60% (34%) | 0.68% (34%) | 0.00% (0%) | 0.00% (34%) | 0.00% (87%) |
| Qwen2.5-0.5B | 8.92% (0%) | 2.72% (22%) | 1.30% (41%) | 0.81% (41%) | 0.86% (41%) | 0.02% (0%) | 0.00% (41%) | 0.00% (78%) |
| Qwen2.5-1.5B | 8.04% (0%) | 1.51% (13%) | 0.94% (34%) | 1.07% (34%) | 0.93% (34%) | 0.00% (0%) | 0.00% (34%) | 0.00% (87%) |
| Qwen3-1.7B | 5.58% (0%) | 1.99% (15%) | 1.42% (35%) | 1.04% (35%) | 1.06% (35%) | 0.00% (0%) | 0.00% (35%) | 0.00% (85%) |
| pythia-410m (NeoX 0.41B) | 34.26% (0%) | 32.24% (13%) | 1.12% (38%) | 32.31% (38%) | 0.41% (38%) | 0.47% (0%) | 0.00% (38%) | 0.00% (87%) |

Out-of-band rate, with the fraction of parameters held in fp32 in parentheses. 'guided 8' = the eight
decoder layers with the largest per-layer divergence increment (T8), chosen on a separate 1,536-token calibration pass.

## T8  where the divergence enters: b1-vs-b8 relative hidden-state divergence, by quarter of the stack

| model | layers | Q1 | Q2 | Q3 | Q4 | after last layer | top-4 layers by increment |
|---|---|---|---|---|---|---|---|
| DeepSeek-R1-Distill-Qwen-1.5B | 28 | +1.1e-02 | +3.7e-03 | +8.6e-04 | -1.1e-03 | 0.014 | [0, 2, 3, 1] |
| Qwen2.5-0.5B | 24 | +1.1e-02 | +2.1e-03 | -1.3e-03 | +5.0e-03 | 0.016 | [0, 23, 22, 2] |
| Qwen2.5-1.5B | 28 | +1.0e-02 | +1.4e-03 | +3.9e-04 | +8.3e-04 | 0.013 | [0, 27, 1, 2] |
| Qwen3-1.7B | 28 | +1.1e-02 | +4.7e-03 | -1.8e-03 | +1.5e-03 | 0.015 | [0, 27, 1, 2] |
| pythia-410m (NeoX 0.41B) | 24 | +9.6e-03 | +1.4e-03 | +1.4e-02 | +9.0e-02 | 0.115 | [23, 19, 21, 22] |

Increment of the relative L2 divergence between the batch-1 and batch-8 hidden states, summed per quarter;
a positive quarter is where the two batch shapes drift apart, a negative one is where later layers partly re-align them.

## T9  does fp16 overflow on the larger models (8,192 tokens, 256-token continuations)

| model | fp16 non-finite log-probs | fp16 b1 vs b8 | fp16 vs bf16 at b1 | max abs hidden state (fp16 max 65,504) |
|---|---|---|---|---|
| Qwen2.5-3B | 0.000% | 0.05% | 3.39% | 3356 |
| Qwen2.5-7B | 0.000% | 0.00% | 3.64% | 12712 |

## T10  does the old-log-prob precision reach the reward: GRPO on GSM8K, Qwen2.5-1.5B-Instruct, 150 steps

| arm | seeds | reward, last 30 steps | reward, mean over 150 | vs A, same seed | clip fraction | vLLM-vs-old abs dlogp | s/step |
|---|---|---|---|---|---|---|---|
| A default (bf16, trainer chunking) | 1 | 0.682 | 0.637 | - | 0.0000 | 0.0110 | 10.4 |
| B fp32 clone | 1 | 0.686 | 0.637 | +0.004 | 0.0005 | 0.0089 | 13.2 |
| C bf16, chunk = micro-batch | 1 | 0.678 | 0.639 | -0.004 | 0.0000 | 0.0110 | 9.7 |

Same seed means the same prompt order and the same vLLM sampling seed, so the arms start as near-replicas and
only the old-log-prob pass differs; the per-step reward noise is sd 0.14, so a 30-step mean carries an SE of about 0.026.
