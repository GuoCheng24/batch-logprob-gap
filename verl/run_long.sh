#!/usr/bin/env bash
# The 100-step arms behind results/long_*.json, each on two GPUs (FSDP shards the optimizer;
# Qwen2.5-1.5B-Instruct does not fit one 24 GB card).
#   bash run_long.sh GPUS ARM      e.g.  bash run_long.sh 0,1 geo_tp08
# ARM: nocorr_tp08 | geo_tp08 | geofix_tp08 | geo_tp10 | seqmis_tp08
# RUN=r2 tags a repeat run (long_<arm>_r2); the settings are identical.
# geofix_tp08 needs a verl checkout with prototype/support_size_replay.patch applied, put first
# on PYTHONPATH (VERL_PATCHED=/path/to/verl).
set -euo pipefail
cd "$(dirname "$0")"
GPUS=${1:?GPU list, e.g. 0,1}; ARM=${2:?arm}; S=${RUN:+_$RUN}
GEO="algorithm.rollout_correction.rollout_rs=seq_mean_k1 algorithm.rollout_correction.rollout_rs_threshold=0.999_1.001"
SEQMIS="algorithm.rollout_correction.rollout_is=sequence algorithm.rollout_correction.rollout_is_threshold=2.0 algorithm.rollout_correction.rollout_rs=seq_sum_k1 algorithm.rollout_correction.rollout_rs_threshold=0.5_2.0"
# vLLM's replay needs top_k > 0; at top_p=0.8 a cap of 1024 does not bind in practice
FIX="actor_rollout_ref.rollout.top_k=1024 +actor_rollout_ref.rollout.engine_kwargs.vllm.return_sampling_mask=True"
export MODEL=${MODEL:-Qwen/Qwen2.5-1.5B-Instruct} STEPS=100 OFFLOAD=0 MICRO=2
export GPU=$GPUS NGPU=$(echo "$GPUS" | tr ',' '\n' | wc -l)
# RTX 4090s have no peer access; without this NCCL 2.29 fails in FSDP's first broadcast
export NCCL_CUMEM_HOST_ENABLE=0
# fewer agent-loop workers and a smaller object store: host RAM was shared with other jobs
COMMON="actor_rollout_ref.rollout.agent.num_workers=2 +ray_kwargs.ray_init.object_store_memory=4000000000"
case "$ARM" in
  nocorr_tp08) bash gate.sh 0.8 long_nocorr_tp08$S $COMMON ;;
  geo_tp08)    bash gate.sh 0.8 long_geo_tp08$S $GEO $COMMON ;;
  geofix_tp08) PYTHONPATH=${VERL_PATCHED:?path to patched verl} bash gate.sh 0.8 long_geofix_tp08$S $GEO $FIX $COMMON ;;
  geo_tp10)    bash gate.sh 1.0 long_geo_tp10$S $GEO $COMMON ;;
  seqmis_tp08) bash gate.sh 0.8 long_seqmis_tp08$S $SEQMIS $COMMON ;;
  *) echo "unknown arm $ARM"; exit 2 ;;
esac
