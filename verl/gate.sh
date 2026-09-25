#!/usr/bin/env bash
# One short verl GRPO run with a given sampling truncation and rollout-correction preset.
#
#   bash gate.sh TOP_P TAG [hydra overrides for algorithm.rollout_correction ...]
#
# Environment (all optional):
#   MODEL        HF id or local path          default Qwen/Qwen2.5-0.5B-Instruct
#   DATA         dir with train/test.parquet  default $HOME/data/gsm8k (verl's examples/data_preprocess/gsm8k.py)
#   STEPS        training steps               default 2
#   OFFLOAD      1 = FSDP param+optimizer CPU offload (needed for 1.5B on one 46 GB card)
#   MIN_FREE_MB  only start on a GPU with at least this much free memory   default 30000
#   GPU_UTIL     vLLM gpu_memory_utilization  default 0.30 (0a and 0b used 0.35)
# Measured with verl main 6093e00, vLLM 0.28.0, torch 2.13.0+cu126, transformers 5.12.1.
set -euo pipefail
TOP_P=${1:?top_p}; TAG=${2:?tag}; shift 2; RC=("$@")
MODEL=${MODEL:-Qwen/Qwen2.5-0.5B-Instruct}
DATA=${DATA:-$HOME/data/gsm8k}
STEPS=${STEPS:-2}
mkdir -p logs

# never start on a card someone else is filling
GPU=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
      | awk -F', ' -v m="${MIN_FREE_MB:-30000}" '$2>=m{print $1; exit}')
[ -z "$GPU" ] && { echo "no GPU with ${MIN_FREE_MB:-30000} MB free, not starting"; exit 3; }
export CUDA_VISIBLE_DEVICES=$GPU OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 RAY_DEDUP_LOGS=0

OFFLOAD_ARGS=()
if [ "${OFFLOAD:-0}" = 1 ]; then
  OFFLOAD_ARGS=(actor_rollout_ref.actor.fsdp_config.param_offload=True
                actor_rollout_ref.actor.fsdp_config.optimizer_offload=True)
fi

python -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  algorithm.use_kl_in_reward=False \
  "${RC[@]}" \
  data.train_files="$DATA/train.parquet" \
  data.val_files="$DATA/test.parquet" \
  data.train_batch_size=32 \
  data.max_prompt_length=512 \
  data.max_response_length=512 \
  data.filter_overlong_prompts=True \
  data.truncation=error \
  actor_rollout_ref.model.path="$MODEL" \
  actor_rollout_ref.model.use_remove_padding=False \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.model.use_fused_kernels=False \
  +actor_rollout_ref.model.override_config.attn_implementation=sdpa \
  actor_rollout_ref.actor.strategy=fsdp2 \
  "${OFFLOAD_ARGS[@]}" \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.actor.ppo_mini_batch_size=32 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
  actor_rollout_ref.actor.use_kl_loss=False \
  actor_rollout_ref.actor.entropy_coeff=0 \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.gpu_memory_utilization="${GPU_UTIL:-0.30}" \
  actor_rollout_ref.rollout.n=4 \
  actor_rollout_ref.rollout.temperature=1.0 \
  actor_rollout_ref.rollout.top_p="$TOP_P" \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
  actor_rollout_ref.rollout.enforce_eager=True \
  actor_rollout_ref.rollout.free_cache_engine=True \
  trainer.logger='["console"]' \
  trainer.project_name=verl-truncated-sampling \
  trainer.experiment_name="$TAG" \
  trainer.n_gpus_per_node=1 \
  trainer.nnodes=1 \
  trainer.val_before_train=False \
  trainer.test_freq=-1 \
  trainer.save_freq=-1 \
  trainer.total_training_steps="$STEPS" \
  trainer.total_epochs=1 \
  +ray_kwargs.ray_init.num_cpus=32 \
  2>&1 | tee "logs/$TAG.log"
