# A2D (Autoregressive-to-Diffusion)

[![Hugging Face Checkpoints](https://img.shields.io/badge/Hugging%20Face-Checkpoints-yellow)](https://huggingface.co/collections/dllm-collection/tiny-a2d)
[![W&B Report](https://img.shields.io/badge/W&B-Report-white?logo=weightsandbiases)]([TODO])


This directory provides two key sets of resources:

-  **[Warmup](#warmup)**: Tutorials for continual pretraining and SFTing any autoregressive model on small datasets to generate text.
<!-- -  **[BERT-Chat](#bert-chat)**: The exact training, inference, and evaluation scripts used to create the [`ModernBERT-base-chat-v0`](https://huggingface.co/dllm-collection/ModernBERT-base-chat-v0) and [`ModernBERT-large-chat-v0`](https://huggingface.co/dllm-collection/ModernBERT-large-chat-v0) checkpoints, two BERTs finetuned as Chatbots. For a deep dive into experimental results, lessons learned, and more reproduction details, please see our full [BERT-Chat W&B Report](https://api.wandb.ai/links/asap-zzhou/101h5xvg). -->

## Files overview
```
# example entry points for training / inference / evaluation
examples/a2d
├── bm3lm               # Block Discrete Denoising Diffusion Language Modeling (https://arxiv.org/abs/2503.09573)
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

## [TODO]

[TODO]: modeling files

```shell

python dllm/pipelines/a2d/convert.py --model_name_or_path "Qwen/Qwen3-0.6B" --output_dir "models/a2d/Qwen3-0.6B"
```

## Warmup: MDLM

### Continual Pretraining

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

### SFT

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

```shell
python -u examples/a2d/mdlm/sample.py --model_name_or_path "models/a2d/Qwen3-0.6B/mdlm/alpaca/checkpoint-final" --block_size 32
```

## Warmup: BM3LM

### Continual Pretraining

```shell
accelerate launch --config_file scripts/accelerate_configs/ddp.yaml --num_processes 1 \
    examples/a2d/bm3lm/pt.py \
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
    --output_dir "models/a2d/Qwen3-0.6B/bm3lm/tiny-shakespeare"
```

### SFT

```shell
accelerate launch --config_file scripts/accelerate_configs/zero2.yaml --num_processes 8 \
    examples/a2d/bm3lm/sft.py \
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
    --output_dir "models/a2d/Qwen3-0.6B/bm3lm/alpaca"
```

```shell
python -u examples/a2d/bm3lm/sample.py --model_name_or_path "models/a2d/Qwen3-0.6B/bm3lm/alpaca/checkpoint-final" --block_size 32 --chat True
```
