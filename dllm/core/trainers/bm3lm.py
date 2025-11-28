# If the mask is already 4D, simply return as-is (it was already prepared, or it is custom)
# if isinstance(attention_mask, (torch.Tensor, BlockMask)) and len(attention_mask.shape) == 4:
#     return True, attention_mask, None, None, None

# block-wise masked diffusion language modeling
from functools import partial
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from typing import Any

from .mdlm import MDLMTrainer
# from dllm.core.schedulers import BaseAlphaScheduler, LinearAlphaScheduler


@dataclass
class BM3LMCollator(transformers.DataCollatorForSeq2Seq):
    block_size: int = 32

    def __call__(self, features, return_tensors=None):
        # ---------- Step 1: list-of-dict 上按 block_size 各自补齐 ----------
        # 直接复用你上面定义的 pad_to_block_size
        # features = pad_to_block_size(
        #     features,
        #     eos_id=self.tokenizer.eos_token_id,
        #     block_size=self.block_size,
        # )
        # pad to the minimal multiple of block size
        for ex in features:
            ids = ex["input_ids"]
            labs = ex["labels"]

            assert isinstance(ids, list) and isinstance(labs, list)

            L = len(ids)
            target = (L + self.block_size - 1) // self.block_size * self.block_size
            pad_len = target - L
            if pad_len > 0:
                ex["input_ids"] = ids + [self.tokenizer.eos_token_id] * pad_len
                ex["labels"] = labs + [self.tokenizer.eos_token_id] * pad_len

        # ---------- Step 2: 用 Seq2Seq collator 做 batch padding ----------
        batch = super().__call__(features, return_tensors=return_tensors)
        return batch


def block_diff_mask(b, h, q_idx, kv_idx, block_size=None, n=None):
    """
    Constructs the specialized block diffusion attention mask for training
    composed of three masks:
    - **Block Diagonal Mask (M_BD)**: Self-attention within noised blocks
    - **Offset Block Causal Mask (M_OBC)**: Cross-attention for conditional context
    - **Block Causal Mask (M_BC)**: Attention to update x0

    Args:
        b, h: Batch and head indices (ignored for mask logic).
        q_idx, kv_idx: Query and Key indices.
        seq_len: Total sequence length.
        block_size: Defines the block structure.

    Returns:
        A boolean attention mask.
    """

    # Indicate whether token belongs to xt or x0
    x0_flag_q = (q_idx >= n)
    x0_flag_kv = (kv_idx >= n)

    # Compute block indices
    block_q = torch.where(x0_flag_q == 1,
                        (q_idx - n) // block_size,
                        q_idx // block_size)
    block_kv = torch.where(x0_flag_kv == 1,
                        (kv_idx - n) // block_size,
                        kv_idx // block_size)

    # **1. Block Diagonal Mask (M_BD) **
    block_diagonal = (block_q == block_kv) & (x0_flag_q == x0_flag_kv)

    # **2. Offset Block-Causal Mask (M_OBC) **
    offset_block_causal = (
    (block_q > block_kv)
    & (x0_flag_kv == 1)
    & (x0_flag_q == 0)
    )

    # **3. Block-Causal Mask (M_BC) **
    block_causal = (block_q >= block_kv) & (x0_flag_kv == 1) & (x0_flag_q == 1)

    # **4. Combine Masks **
    return block_diagonal | offset_block_causal | block_causal



class BM3LMTrainer(MDLMTrainer):

    def _preprocess_inputs(self, inputs):
        return inputs

    def _postprocess_outputs(self, outputs):
        return outputs

    def __init__(
        self,
        block_size: int = 32,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.block_size = block_size


    def compute_loss(
        self,
        model: transformers.PreTrainedModel | nn.Module,
        inputs: list[dict[str, Any]],   # 注意：这里现在是 list[dict]
        return_outputs: bool = False,
        **kwargs,
    ):
        assert self.processing_class.padding_side == "right"

        # # ---------- Step 1: list-of-dict 上按 block_size 各自补齐 ----------
        # # 每个样本 input_ids / labels 各自 pad 到 block_size 的倍数，仍然是 list[dict]
        # breakpoint()
        # inputs = pad_to_block_size(
        #     inputs,
        #     eos_id=self.processing_class.eos_token_id,
        #     block_size=self.block_size,
        # )

        # # ---------- Step 2: 用 Seq2Seq collator 做 batch padding ----------
        # # 假设 self.data_collator 是 DataCollatorForSeq2Seq 或其 wrapper，
        # # 它会把 labels 额外 pad 成 -100（label_pad_token_id）。
        # batch = self.data_collator(inputs)      # dict[str, torch.Tensor] on CPU
        # batch = self._prepare_inputs(batch)     # Trainer 自带：搬到 device / cast dtype

        assert self.processing_class.padding_side == "right"
        inputs = self._preprocess_inputs(inputs)
        input_ids, labels, attention_mask = (
            inputs["input_ids"],
            inputs["labels"],
            inputs.get("attention_mask", None),
        )
        b, l = input_ids.shape


        # === 1. Sample diffusion timesteps ===
        t = self.time_epsilon + (1 - self.time_epsilon) * torch.rand(
            b, device=input_ids.device
        )  # [b]
        alpha_t = self.scheduler(t)  # [b]
        p_mask = 1.0 - alpha_t.unsqueeze(1).expand(b, l)  # [b, l]

        # === 2. Apply stochastic masking ===
        masked_indices = (torch.rand((b, l), device=input_ids.device) < p_mask) & (
            labels != -100
        )
        noised_input_ids = torch.where(
            masked_indices, self.processing_class.mask_token_id, input_ids
        )

        # === 3. Forward pass through the model (block-diffusion) ===
        # We follow the paper and feed x_t ⊕ x_0 with a specialized block mask.

        # concat_input_ids: [b, 2l], first l are noisy (x_t), last l are clean (x_0)
        concat_input_ids = torch.cat([noised_input_ids, input_ids], dim=1)

        # TODO: others like flash attention 2
        if model.config._attn_implementation == "sdpa":
            attention_mask = block_diff_mask(
                b=None, h=None, q_idx=torch.arange(l*2)[:, None], 
                kv_idx=torch.arange(l*2)[None, :], block_size=self.block_size, n=l,
            )
            attention_mask = attention_mask.unsqueeze(0).unsqueeze(0).expand(1, 1, 2*l, 2*l)
            attention_mask = attention_mask.to(input_ids.device)
        elif model.config._attn_implementation == "flex_attention":
            from torch.nn.attention.flex_attention import create_block_mask
            attention_mask = create_block_mask(
                partial(block_diff_mask, block_size=self.block_size, n=l),
                B=None, H=None, Q_LEN=l*2, KV_LEN=l*2
            )
        else:
            raise NotImplementedError

        base_pos = torch.arange(l, device=input_ids.device).unsqueeze(0).expand(b, l)  # [B, L]
        concat_position_ids = torch.cat([base_pos, base_pos], dim=1)  # [B, 2L]

        # TODO: positional id, left half and right half should have the same positional id
        outputs = model(
            input_ids=concat_input_ids, 
            attention_mask=attention_mask,
            position_ids=concat_position_ids,
        )
        outputs = self._postprocess_outputs(outputs)
        logits = outputs.logits

        logits = logits[:, :l] # we only care about the first half for computing loss

        # === 4. Handle degenerate cases (no tokens masked) ===
        # If no positions were masked, return a zero loss to keep gradients valid.
        # This step is necessary for Deepspeed Zero-{2,3}
        if not masked_indices.any():
            return (
                (logits.sum() * 0.0, outputs) if return_outputs else logits.sum() * 0.0
            )

        # === 5. Compute per-token loss weights ===
        # Depending on the configuration, weights may depend on timestep t
        # (e.g., scheduler-based) or be uniform (ones).
        loss_weights = self._compute_loss_weights(
            t=t, inputs=inputs, masked_indices=masked_indices
        )

        # === 6. Compute weighted cross-entropy ===
        # Only masked tokens contribute to the loss.
        assert (input_ids[masked_indices] == labels[masked_indices]).all()
        token_loss = F.cross_entropy(
            logits[masked_indices], input_ids[masked_indices], reduction="none"
        )
        token_loss = token_loss * loss_weights[masked_indices]

        # === 7. Normalize loss per effective token length ===
        # Normalize each sequence’s contribution by its number of valid tokens,
        # then average over the batch for stability across variable-length inputs.
        effective_lengths = torch.sum(labels != -100, dim=1, keepdim=True).expand(b, l)
        loss = torch.sum(token_loss / effective_lengths[masked_indices]) / b

        # === 8. Return final loss (and optionally model outputs) ===
        return (loss, outputs) if return_outputs else loss
