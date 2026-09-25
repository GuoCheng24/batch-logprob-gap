#!/usr/bin/env bash
# The three gates behind results/gate0*.json, run sequentially on one GPU.
#   bash run_gates.sh 0a|0b|0c
set -euo pipefail
cd "$(dirname "$0")"
GEO="algorithm.rollout_correction.rollout_rs=seq_mean_k1 algorithm.rollout_correction.rollout_rs_threshold=0.999_1.001"   # decoupled_geo_rs() defaults
K3="algorithm.rollout_correction.rollout_rs=seq_mean_k3 algorithm.rollout_correction.rollout_rs_threshold=0.01"          # decoupled_k3_rs() defaults
SEQMIS="algorithm.rollout_correction.rollout_is=sequence algorithm.rollout_correction.rollout_is_threshold=2.0 algorithm.rollout_correction.rollout_rs=seq_sum_k1 algorithm.rollout_correction.rollout_rs_threshold=0.5_2.0"  # decoupled_seq_is_rs() defaults
TOKIS="algorithm.rollout_correction.rollout_is=token algorithm.rollout_correction.rollout_is_threshold=2.0"
case "${1:?0a|0b|0c}" in
  0a)  # is the kept-mass term in verl's own metrics?  Qwen2.5-0.5B-Instruct, 3 steps
       for tp in 1.0 0.8; do STEPS=3 GPU_UTIL=0.35 bash gate.sh $tp 0a_tp${tp/./} $TOKIS; done ;;
  0b)  # what do the sequence-level presets do with it?  Qwen2.5-0.5B-Instruct, 2 steps
       for p in GEO K3 SEQMIS; do for tp in 1.0 0.8; do
         GPU_UTIL=0.35 bash gate.sh $tp 0b_${p,,}_tp${tp/./} ${!p}; done; done ;;
  0c)  # does training stop?  Qwen2.5-1.5B-Instruct (earns reward, so advantages are non-zero)
       export MODEL=Qwen/Qwen2.5-1.5B-Instruct OFFLOAD=1
       bash gate.sh 0.8 0c_nocorr_tp08
       bash gate.sh 1.0 0c_geo_tp10 $GEO
       bash gate.sh 0.8 0c_geo_tp08 $GEO
       bash gate.sh 0.8 0c_seqmis_tp08 $SEQMIS ;;
esac
