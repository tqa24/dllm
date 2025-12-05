# A2D (AR-to-Diffusion)

[![Hugging Face Checkpoints](https://img.shields.io/badge/Hugging%20Face-Checkpoints-yellow)](https://huggingface.co/collections/dllm-collection/tiny-a2d)
[![W&B Report](https://img.shields.io/badge/W&B-Report-white?logo=weightsandbiases)](https://wandb.ai/asap-zzhou/dllm/reports/dLLM-Tiny-A2D--VmlldzoxNTI2NTEzOA)


This directory provides two key sets of resources:

- **Warmup ([MDLM](#warmup-mdlm) and [BD3LM](#warmup-bd3lm))**: Tutorials for continual pretraining and SFTing any autoregressive model on small datasets to generate text with MDLM (masked diffusion) or BD3LM (block diffusion).
- **[`Tiny-A2D`](#tiny-a2d)**: The exact training, inference, and evaluation scripts used to develop the 🤗checkpoints: [`Qwen3-0.6B-diffusion-bd3lm-v0.1`](https://huggingface.co/dllm-collection/Qwen3-0.6B-diffusion-bd3lm-v0.1) (MDLM, global masked diffusion) and [`Qwen3-0.6B-diffusion-mdlm-v0.1`](https://huggingface.co/dllm-collection/Qwen3-0.6B-diffusion-mdlm-v0.1) (BD3LM, blockwise masked diffusion with KV cache).
For a deep dive into experimental results, lessons learned, and more reproduction details, please see our full [![blog](https://img.shields.io/badge/W&B-white?logo=weightsandbiases) Tiny-A2D Report](https://wandb.ai/asap-zzhou/dllm/reports/dLLM-Tiny-A2D--VmlldzoxNTI2NTEzOA).

## Files overview
```
# example entry points for training / inference / evaluation
examples/a2d
├── bd3lm               # Block Discrete Denoising Diffusion Language Modeling (https://arxiv.org/abs/2503.09573)
│   ├── chat.py
│   ├── eval.sh
│   ├── pt.py
│   ├── sample.py
│   └── sft.py
├── mdlm                # Masked Diffusion Language Modeling (https://arxiv.org/abs/2406.07524)
│   ├── chat.py
│   ├── eval.sh
│   ├── pt.py
│   ├── sample.py
│   └── sft.py
└── README.md
```

## Setup 

1. **Customize modeling files**: You must first modify the original autoregressive modeling file to support non-causal attention. See [`modeling_qwen3.py`](/dllm/pipelines/a2d/models/qwen3/modeling_qwen3.py#L77-L108) for an example, and update [`__init__.py`](/dllm/pipelines/a2d/__init__.py) accordingly to register the new model config and architecture.

2. **Run unit tests**: Before proceeding with your customized models, ensure they pass:
    ```shell
    pytest scripts/tests/test_attention.py::test_a2d_attention_mask_invariance
    pytest scripts/tests/test_attention.py::test_a2d_fullmask_future_affects_past
    # Optional: only needed for BD3LM
    pytest scripts/tests/test_attention.py::test_a2d_staircase_attention_kvcache_equivalence
    ```

3. **Convert an AR model with customized attention**: For example, to convert `Qwen/Qwen3-0.6B` using its original weights but with the customized attention defined in [`modeling_qwen3.py`](/dllm/pipelines/a2d/models/qwen3/modeling_qwen3.py):
    ```shell
    python dllm/pipelines/a2d/convert.py --model_name_or_path "Qwen/Qwen3-0.6B" --output_dir "models/a2d/Qwen3-0.6B"
    ```

## Warmup: [MDLM](https://arxiv.org/abs/2406.07524)

In this section, we show toy examples of continual pretraining and SFTing [`Qwen/Qwen3-0.6B`](https://huggingface.co/Qwen/Qwen3-0.6B) on small datasets to generate text with [MDLM](https://arxiv.org/abs/2406.07524).

### Continual Pretraining

To adapat [`Qwen/Qwen3-0.6B`](https://huggingface.co/Qwen/Qwen3-0.6B) on the [`tiny-shakespeare`](https://huggingface.co/datasets/Trelis/tiny-shakespeare) dataset with [MDLM](https://arxiv.org/abs/2406.07524), run:

```shell
accelerate launch --config_file scripts/accelerate_configs/ddp.yaml --num_processes 1 \
    examples/a2d/mdlm/pt.py \
    --model_name_or_path "models/a2d/Qwen3-0.6B" \
    --dataset_args "Trelis/tiny-shakespeare" \
    --text_field "Text" \
    --insert_eos False \
    --max_length 128 \
    --learning_rate 1e-4 \
    --num_train_epochs 20 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 16 \
    --eval_steps 0.1 \
    --save_steps 0.1 \
    --output_dir "models/a2d/Qwen3-0.6B/mdlm/tiny-shakespeare"
```

To sample from the model interactively:
```shell
# Enter a prompt (e.g., "First citizen: Before we proceed any further, hear me speak."),
# or press Enter to let the model generate text from scratch.
python -u examples/a2d/mdlm/chat.py \
    --model_name_or_path "models/a2d/Qwen3-0.6B/mdlm/tiny-shakespeare/checkpoint-final" \
    --chat_template False --remasking "random" --steps 128 --max_new_tokens 128
```

### SFT

To adapat [`Qwen/Qwen3-0.6B`](https://huggingface.co/Qwen/Qwen3-0.6B) on the [`alpaca`](https://huggingface.co/datasets/tatsu-lab/alpaca) dataset with [MDLM](https://arxiv.org/abs/2406.07524), run:

```shell
accelerate launch --config_file scripts/accelerate_configs/zero2.yaml --num_processes 8 \
    examples/a2d/mdlm/sft.py \
    --model_name_or_path "models/a2d/Qwen3-0.6B" \
    --dataset_args "tatsu-lab/alpaca" \
    --max_length 512 \
    --learning_rate 1e-4 \
    --num_train_epochs 20 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 16 \
    --eval_steps 0.1 \
    --save_steps 0.1 \
    --output_dir "models/a2d/Qwen3-0.6B/mdlm/alpaca"
```

To chat with the model:
```shell
python -u examples/a2d/mdlm/chat.py \
    --model_name_or_path "models/a2d/Qwen3-0.6B/mdlm/alpaca/checkpoint-final" --block_size 32
```

## Warmup: [BD3LM](https://arxiv.org/abs/2503.09573)

In this section, we show toy examples of continual pretraining and SFTing [`Qwen/Qwen3-0.6B`](https://huggingface.co/Qwen/Qwen3-0.6B) on small datasets to generate text with [BD3LM](https://arxiv.org/abs/2503.09573).

### Continual Pretraining

To adapat [`Qwen/Qwen3-0.6B`](https://huggingface.co/Qwen/Qwen3-0.6B) on the [`tiny-shakespeare`](https://huggingface.co/datasets/Trelis/tiny-shakespeare) dataset with [BD3LM](https://arxiv.org/abs/2503.09573), run:

```shell
accelerate launch --config_file scripts/accelerate_configs/ddp.yaml --num_processes 1 \
    examples/a2d/bd3lm/pt.py \
    --model_name_or_path "models/a2d/Qwen3-0.6B" \
    --dataset_args "Trelis/tiny-shakespeare" \
    --text_field "Text" \
    --insert_eos False \
    --max_length 128 \
    --learning_rate 1e-4 \
    --num_train_epochs 20 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 16 \
    --eval_steps 0.1 \
    --save_steps 0.1 \
    --block_size 32 \
    --output_dir "models/a2d/Qwen3-0.6B/bd3lm/tiny-shakespeare"
```

To sample from the model interactively:
```shell
# Enter a prompt (e.g., "First citizen: Before we proceed any further, hear me speak."),
# or press Enter to let the model generate text from scratch.
python -u examples/a2d/bd3lm/chat.py \
    --model_name_or_path "models/a2d/Qwen3-0.6B/bd3lm/tiny-shakespeare/checkpoint-final" \
    --chat_template False --block_size 32 --remasking "random" --steps 128 --max_new_tokens 128
```

### SFT

To adapat [`Qwen/Qwen3-0.6B`](https://huggingface.co/Qwen/Qwen3-0.6B) on the [`alpaca`](https://huggingface.co/datasets/tatsu-lab/alpaca) dataset with [BD3LM](https://arxiv.org/abs/2503.09573), run:

```shell
accelerate launch --config_file scripts/accelerate_configs/zero2.yaml --num_processes 8 \
    examples/a2d/bd3lm/sft.py \
    --model_name_or_path "models/a2d/Qwen3-0.6B" \
    --dataset_args "tatsu-lab/alpaca" \
    --max_length 512 \
    --learning_rate 1e-4 \
    --num_train_epochs 20 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 16 \
    --eval_steps 0.1 \
    --save_steps 0.1 \
    --block_size 32 \
    --output_dir "models/a2d/Qwen3-0.6B/bd3lm/alpaca"
```

To chat with the model:
```shell
python -u examples/a2d/bd3lm/chat.py \
    --model_name_or_path "models/a2d/Qwen3-0.6B/bd3lm/alpaca/checkpoint-final" --block_size 32
```

## `Tiny-A2D`

Here we show the exact commands we use to train / interact with / evaluation the [`Tiny-A2D`](https://huggingface.co/collections/dllm-collection/tiny-a2d) models:
[`Qwen3-0.6B-diffusion-mdlm-v0.1`](https://huggingface.co/dllm-collection/Qwen3-0.6B-diffusion-mdlm-v0.1) and [`Qwen3-0.6B-diffusion-bd3lm-v0.1`](https://huggingface.co/dllm-collection/Qwen3-0.6B-diffusion-bd3lm-v0.1).
For training curves and other details, please see [![blog](https://img.shields.io/badge/W&B-white?logo=weightsandbiases) Tiny-A2D Report](https://wandb.ai/asap-zzhou/dllm/reports/dLLM-Tiny-A2D--VmlldzoxNTI2NTEzOA).

### Training
> Read [Useful tips for training](/README.md/#useful-tips-for-training) and [(optional) Slurm setup](/README.md/#optional-slurm-setup) before training.

The [`Tiny-A2D`](https://huggingface.co/collections/dllm-collection/tiny-a2d) models are trained purely with SFT.

To reproduce [`Qwen3-0.6B-diffusion-mdlm-v0.1`](https://huggingface.co/dllm-collection/Qwen3-0.6B-diffusion-mdlm-v0.1) (with MDLM & SFT), run the command below (about 10 hours on 64 GPUs):
```shell
WANDB_MODE=online sbatch --nodes=8 --gres=gpu:8 scripts/train.slurm.sh \
    --accelerate_config "zero2" \
    --script_path "examples/a2d/mdlm/sft.py" \
    --model_name_or_path "models/a2d/Qwen3-0.6B" \
    --dataset_args "allenai/tulu-3-sft-mixture+HuggingFaceTB/smoltalk+OpenCoder-LLM/opc-sft-stage1[lang:python]+OpenCoder-LLM/opc-sft-stage2[lang:python]" \
    --max_length 1024 \
    --num_train_epochs 10 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 16 \
    --gradient_accumulation_steps 2 \
    --save_steps 0.1 \
    --output_dir "models/a2d/Qwen3-0.6B/tulu-3-sft-mixture+smoltalk+opc-sft-stage1&2/epochs-10-bs-2048-len-1024"
```

To reproduce [`Qwen3-0.6B-diffusion-bd3lm-v0.1`](https://huggingface.co/dllm-collection/Qwen3-0.6B-diffusion-bd3lm-v0.1) (with BD3LM & SFT), run the command below (about 10 hours on 64 GPUs):
```shell
WANDB_MODE=online sbatch --nodes=8 --gres=gpu:8 scripts/train.slurm.sh \
    --accelerate_config "zero2" \
    --script_path "examples/a2d/bd3lm/sft.py" \
    --model_name_or_path "models/a2d/Qwen3-0.6B" \
    --dataset_args "allenai/tulu-3-sft-mixture+HuggingFaceTB/smoltalk+OpenCoder-LLM/opc-sft-stage1[lang:python]+OpenCoder-LLM/opc-sft-stage2[lang:python]" \
    --max_length 512 \
    --block_size 32 \
    --num_train_epochs 10 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 16 \
    --gradient_accumulation_steps 2 \
    --save_steps 0.1 \
    --output_dir "models/a2d/Qwen3-0.6B/tulu-3-sft-mixture+smoltalk+opc-sft-stage1&2/epochs-10-bs-2048-len-512-bls-32"
```

### Evaluation
