#!/bin/bash

export VLLM_ATTENTION_BACKEND=XFORMERS

VERL_CHECKPOINT_DIR=./targets/RationAnomaly/global_step_150/actor
RL_MODEL_OUTPUT_DIR=./models/RationAnomaly

# Checkpoints
# If you want to convert the model checkpoint into huggingface safetensor format, please refer to scripts/model_merger.py.
#  https://verl.readthedocs.io/en/latest/faq/faq.html#checkpoints
python -m verl.model_merger merge \
    --backend fsdp \
    --local_dir  $VERL_CHECKPOINT_DIR \
    --target_dir $RL_MODEL_OUTPUT_DIR
