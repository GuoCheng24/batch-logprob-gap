## T1  dtype x batch, no inference engine involved

| model | bf16 b1 vs b8 | fp16 b1 vs b8 | fp32 b1 vs b8 | bf16 b1 rerun | bf16 b1 vs b32 |
|---|---|---|---|---|---|
| pythia-160m (NeoX 0.16B) | 60.03% | 22.02% | 0.00% | 0.00% | 59.70% |
| pythia-410m (NeoX 0.41B) | 36.70% | 2.49% | 0.00% | 0.00% | 36.65% |
| Qwen2.5-0.5B | 9.87% | 0.07% | 0.00% | 0.00% | 10.75% |
| Granite-3.1-1B-A400M (MoE) | 13.69% | 0.97% | 0.00% | 0.00% | 13.85% |
| Agents-A1-4B | 1.78% | 0.00% | 0.00% | 0.00% | 1.64% |

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
