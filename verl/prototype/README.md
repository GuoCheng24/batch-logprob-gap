# Support-size replay, prototype patch for verl

`support_size_replay.patch` applies to verl `main` at 6093e00:

```bash
git -C verl checkout 6093e00 && git -C verl apply /path/to/support_size_replay.patch
```

vLLM (`return_sampling_mask=True`) reports the ids the truncated sampler kept at every
generated token; the server forwards only their count. In the old-log-prob forward the actor
computes `logsumexp(top-|S| logits / T) - logsumexp(logits / T)`, and the trainer adds it to
`rollout_log_probs` before any rollout correction, so both log-probs are on the
full-vocabulary support.

Scope of the prototype: vLLM rollout, single-turn agent loop, the v1 (TransferQueue) trainer,
the FSDP engine without fused kernels, decoupled mode. It is switched on by
`+actor_rollout_ref.rollout.engine_kwargs.vllm.return_sampling_mask=True`, which also makes
vLLM require `top_k > 0`. A proper config switch, the legacy trainer, multi-turn loops and
tests are for the PR.

Run end to end with Qwen2.5-1.5B-Instruct for 100 steps with `use_remove_padding=False`
([../README.md](../README.md), section 4), and with Qwen2.5-0.5B-Instruct for 2 steps on both
padding paths ([../results/replay_check_0p5b.json](../results/replay_check_0p5b.json)).
Ulysses sequence parallelism is handled like the per-token temperature but has not been run.

The patch modifies verl, which is licensed under the Apache License 2.0
([LICENSE-Apache-2.0](LICENSE-Apache-2.0)); the patch is offered under the same license.
The rest of this repository is MIT.
