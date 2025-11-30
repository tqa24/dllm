#!/usr/bin/env bash
# ===== Mandatory for proper import and evaluation =====
export PYTHONPATH=.:$PYTHONPATH             
export HF_ALLOW_CODE_EVAL=1                 # Allow code evaluation
export HF_DATASETS_TRUST_REMOTE_CODE=True

# ===== Optional but recommended for stability and debugging =====
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1    # Enable async error handling for multi-GPU communication to avoid deadlocks
export NCCL_DEBUG=warn                      # Show NCCL warnings for better diagnosis without flooding logs
export TORCH_DISTRIBUTED_DEBUG=DETAIL       # Provide detailed logging for PyTorch distributed debugging

# ===== Basic Settings =====
model_name_or_path="Tiny-A2D/Qwen3-0.6B-right-shift-opc-sft-stage1&2-epochs-10-bs-2048-len-1024"
num_gpu=1
while [[ $# -gt 0 ]]; do
  case "$1" in
    --model_name_or_path)
      model_name_or_path="$2"; shift 2 ;;
    --num_gpu)
      num_gpu="$2"; shift 2 ;;
    *) 
      echo "Error: Unknown argument: $1"; exit 1 ;;
  esac
done

# ===== Common arguments =====
common_args="--model llada --apply_chat_template"  # Tiny-A2D model uses chat template by default

# ===== Determine right_shift_logits from model path =====
if [[ "$model_name_or_path" == *"right-shift"* ]]; then
    right_shift_logits=True
elif [[ "$model_name_or_path" == *"non-shift"* ]]; then
    right_shift_logits=False
else
    echo "Warning: Could not determine right_shift_logits from model_name_or_path. Defaulting to False."
    right_shift_logits=False
fi

echo ">>> Using right_shift_logits=${right_shift_logits} for model: ${model_name_or_path}"

# =======================
# Tiny-A2D Tasks
# =======================

accelerate launch --num_processes "${num_gpu}" dllm/pipelines/llada/eval.py \
    --tasks mmlu_generative --num_fewshot 0 ${common_args} \
    --model_args "pretrained=${model_name_or_path},max_new_tokens=3,steps=3,block_size=3,cfg=0.0,right_shift_logits=${right_shift_logits}"

accelerate launch --num_processes "${num_gpu}" dllm/pipelines/llada/eval.py \
    --tasks mmlu_pro --num_fewshot 0 ${common_args} \
    --model_args "pretrained=${model_name_or_path},max_new_tokens=256,steps=256,block_size=256,cfg=0.0,right_shift_logits=${right_shift_logits}"

accelerate launch --num_processes "${num_gpu}" dllm/pipelines/llada/eval.py \
    --tasks hellaswag_gen --num_fewshot 0 ${common_args} \
    --model_args "pretrained=${model_name_or_path},max_new_tokens=3,steps=3,block_size=3,cfg=0.0,right_shift_logits=${right_shift_logits}"

accelerate launch --num_processes "${num_gpu}" dllm/pipelines/llada/eval.py \
    --tasks arc_challenge_chat --num_fewshot 0 ${common_args} \
    --model_args "pretrained=${model_name_or_path},max_new_tokens=512,steps=512,block_size=512,cfg=0.0,right_shift_logits=${right_shift_logits}"

accelerate launch --num_processes "${num_gpu}" dllm/pipelines/llada/eval.py \
    --tasks gpqa_diamond_generative_n_shot --num_fewshot 5 ${common_args} \
    --model_args "pretrained=${model_name_or_path},max_new_tokens=64,steps=64,block_size=64,cfg=0.0,right_shift_logits=${right_shift_logits}"

accelerate launch --num_processes "${num_gpu}" dllm/pipelines/llada/eval.py \
    --tasks gsm8k_cot --num_fewshot 5 ${common_args} \
    --model_args "pretrained=${model_name_or_path},max_new_tokens=512,steps=512,block_size=512,cfg=0.0,right_shift_logits=${right_shift_logits}"

accelerate launch --num_processes "${num_gpu}" dllm/pipelines/llada/eval.py \
    --tasks bbh --num_fewshot 3 ${common_args} \
    --model_args "pretrained=${model_name_or_path},max_new_tokens=1024,steps=1024,block_size=1024,cfg=0.0,right_shift_logits=${right_shift_logits}"

accelerate launch --num_processes "${num_gpu}" dllm/pipelines/llada/eval.py \
    --tasks minerva_math --num_fewshot 4 ${common_args} \
    --model_args "pretrained=${model_name_or_path},max_new_tokens=512,steps=512,block_size=512,cfg=0.0,right_shift_logits=${right_shift_logits}"

accelerate launch --num_processes "${num_gpu}" dllm/pipelines/llada/eval.py \
    --tasks humaneval_instruct_llada --num_fewshot 0 ${common_args} \
    --model_args "pretrained=${model_name_or_path},max_new_tokens=512,steps=512,block_size=512,cfg=0.0,right_shift_logits=${right_shift_logits}" \
    --confirm_run_unsafe_code

accelerate launch --num_processes "${num_gpu}" dllm/pipelines/llada/eval.py \
    --tasks mbpp_instruct_llada --num_fewshot 3 ${common_args} \
    --model_args "pretrained=${model_name_or_path},max_new_tokens=256,steps=256,block_size=256,cfg=0.0,right_shift_logits=${right_shift_logits}" \
    --confirm_run_unsafe_code
