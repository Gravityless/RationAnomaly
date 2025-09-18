#!/bin/bash

# Powered by XuSong
# 
# - 首先激活 Miniconda 虚拟环境
#   $ conda activate ~/miniconda3/envs/rationanomaly/
# - 选择合适显存空余容量的显卡
#   $ nvidia-smi --query-gpu=index,gpu_name,memory.used,memory.total --format=csv,noheader,nounits | awk -F ',' '{printf "显卡%s #%s 显存利用率: %.1f%%\n", $2, $1, ($3 / $4) * 100}'
#

set -x

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

export WANDB_MODE=offline
export HYDRA_FULL_ERROR=1

# 【临时调试】用于解决：Watchdog caught collective operation timeout: WorkNCCL(SeqNum=99123, OpType=ALLREDUCE, ...) ran for 1800076 milliseconds before timing out.
export NCCL_TIMEOUT=7200
# 下列内容不激活
# export NCCL_DEBUG=INFO
# export NCCL_DEBUG_SUBSYS=ALL
export TORCH_NCCL_TRACE_BUFFER_SIZE=16777216
# export TORCH_DISTRIBUTED_DEBUG=DETAIL


# 【临时调试】强制 CUDA 操作以同步方式执行，即 CPU 会等待每个 CUDA 操作完成后再继续执行后续代码。
# export CUDA_LAUNCH_BLOCKING=1

# 禁止在 ProcessPoolExecutor 内部对 reward_tensor 做并发操作，否则会引起:
#   RuntimeError: CUDA error: an illegal memory access was encountered
#   CUDA kernel errors might be asynchronously reported at some other API call, so the stacktrace below might be incorrect.

MODEL_PATH=./models/llama-2-7b-chat-hf

CURRENT_DATE_TIME=$(date  +"%Y%m%d_%H%M%S").log
CURRENT_LOGGING_FILE_NAME="verl_grpo_rationanomaly_$CURRENT_DATE_TIME"

python -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=datasets/bgl_train.parquet \
    data.val_files=datasets/bgl_validation.parquet \
    data.train_batch_size=32 \
    data.val_batch_size=32 \
    data.max_prompt_length=2048 \
    data.max_response_length=1024 \
    data.truncation=left \
    actor_rollout_ref.model.path=$MODEL_PATH\
    actor_rollout_ref.actor.optim.lr=4e-7 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=8 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.rollout.temperature=0.9 \
    actor_rollout_ref.rollout.n=16 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.critic_warmup=0 \
    trainer.logger=['wandb','console'] \
    trainer.project_name='RationAnomaly' \
    trainer.experiment_name='beginner_guide' \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.default_local_dir=targets/RationAnomaly \
    trainer.default_hdfs_dir=null \
    trainer.save_freq=10 \
    trainer.test_freq=-1 \
    trainer.val_before_train=false \
    trainer.total_epochs=2 \
    "$@" 2>&1 | tee -a $CURRENT_LOGGING_FILE_NAME

# 1. 快速后台启动
#    pyclean . && nohup bash model_train.sh > /dev/null 2>&1 & 
# 2. 快速查看当前进度
#    echo "$(tac $(ls verl_*.log | sort | tail -n 1) | grep -m 1 "Training Progress")"
