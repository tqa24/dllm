```shell
srun -p $PARTITION --quotatype=spot --gres=gpu:1 --time=03:00:00 --exclude=SH-IDCA1404-10-140-54-101 python dllm/pipelines/a2d/convert.py --model_name_or_path "Qwen/Qwen2.5-0.5B" --output_dir "models/a2d/Qwen2.5-0.5B"
```

```shell
srun -p $PARTITION --quotatype=spot --gres=gpu:8 --cpus-per-task=24 --time=03:00:00 --exclude=SH-IDCA1404-10-140-54-101 \
    accelerate launch --config_file scripts/accelerate_configs/zero2.yaml --num_processes 8 \
        examples/a2d/sft.py \
        --model_name_or_path "models/a2d/Qwen2.5-0.5B" \
        --dataset_args "tatsu-lab/alpaca" \
        --max_length 512 \
        --num_train_epochs 20 \
        --per_device_train_batch_size 32 \
        --per_device_eval_batch_size 32 \
        --save_steps 0.1 \
        --right_shift_logits True \
        --output_dir "models/a2d/Qwen2.5-0.5B/alpaca/right-shift"

srun -p $PARTITION --quotatype=spot --gres=gpu:8 --cpus-per-task=24 --time=03:00:00 --exclude=SH-IDCA1404-10-140-54-101 \
    accelerate launch --config_file scripts/accelerate_configs/zero2.yaml --num_processes 8 \
        examples/a2d/sft.py \
        --model_name_or_path "models/a2d/Qwen2.5-0.5B" \
        --dataset_args "tatsu-lab/alpaca" \
        --max_length 512 \
        --num_train_epochs 20 \
        --per_device_train_batch_size 32 \
        --per_device_eval_batch_size 32 \
        --save_steps 0.1 \
        --right_shift_logits False \
        --output_dir "models/a2d/Qwen2.5-0.5B/alpaca/non-right-shift"
```
