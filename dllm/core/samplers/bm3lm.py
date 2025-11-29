"""
reference: https://github.com/ML-GSAI/LLaDA/blob/main/generate.py
"""

import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from dllm.core.samplers.base import (
    SamplerOutput,
    SamplerConfig,
    BaseSampler,
)
from dllm.core.samplers.utils import get_num_transfer_tokens


# def add_gumbel_noise(logits: torch.Tensor, temperature: float) -> torch.Tensor:
#     """
#     The Gumbel max is a method for sampling categorical distributions.
#     According to arXiv:2409.02908, for MDM, low-precision Gumbel Max improves perplexity score but reduces generation quality.
#     Thus, we use float64.
#     """
#     if temperature == 0:
#         return logits
#     logits = logits.to(torch.float64)
#     noise = torch.rand_like(logits, dtype=torch.float64)
#     gumbel_noise = (-torch.log(noise)) ** temperature
#     return logits.exp() / gumbel_noise


@dataclass
class BM3LMSamplerConfig(SamplerConfig):
    max_new_tokens: int = 128
    max_length: int = (
        None  # There's no explicit length_limit except for the tokenizer/model context
    )
    block_length: int = 128
    steps: int = 128
    temperature: float = 0.0
    remasking: str = "low_confidence"
    stochastic_transfer: bool = False
    cfg_scale: float = 0.0
    cfg_keep_tokens: list[int] | None = None
    right_shift_logits: bool = False


@dataclass
class BM3LMSampler(BaseSampler):

    @torch.no_grad()
    def sample(
        self,
        inputs: list[torch.Tensor | list],
        config: BM3LMSamplerConfig | None = None,
        **kwargs,
    ) -> SamplerOutput | torch.Tensor:

        if config is None:
            config = BM3LMSamplerConfig()

        # ---- config extraction ----
        steps = kwargs.get("steps", config.steps)
        max_new_tokens = kwargs.get("max_new_tokens", config.max_new_tokens)
        max_length = kwargs.get("max_length", config.max_length)
        block_length = kwargs.get("block_length", config.block_length)
        temperature = kwargs.get("temperature", config.temperature)
        cfg_scale = kwargs.get("cfg_scale", config.cfg_scale)
        cfg_keep_tokens = kwargs.get("cfg_keep_tokens", config.cfg_keep_tokens)
        remasking = kwargs.get("remasking", config.remasking)
        stochastic_transfer = kwargs.get("stochastic_transfer", config.stochastic_transfer)
        return_dict = kwargs.get(
            "return_dict", config.return_dict
        )
        right_shift_logits = kwargs.get("right_shift_logits", config.right_shift_logits)

        assert block_length >= 1
        assert steps >= 1

        mask_id = self.tokenizer.mask_token_id
        eos_id = self.tokenizer.eos_token_id

        # ---- prepare input tensors ----
        if isinstance(inputs[0], list):
            inputs = [
                torch.as_tensor(p, dtype=torch.long, device=self.model.device)
                for p in inputs
            ]

        prompt_lens = [p.shape[0] for p in inputs]

        # Determine how many new tokens will be generated
        if max_new_tokens:
            max_length = max_new_tokens + max(prompt_lens)
        else:
            max_new_tokens = max_length - max(prompt_lens)

        B = len(inputs)
        max_prompt_len = max(prompt_lens)

        # ==========================================================
        # NEW 1: Do NOT preallocate a full mask canvas.
        #        Only keep the prompt first; future blocks will be appended.
        # ==========================================================
        x = torch.full((B, max_prompt_len), eos_id, dtype=torch.long, device=self.model.device)
        for b, p in enumerate(inputs):
            x[b, : prompt_lens[b]] = p

        # ---- unconditional CFG preparation: prompt tokens considered "fixed" ----
        unmasked_index = (x != mask_id) & (x != eos_id)
        if cfg_keep_tokens:
            keep_mask = torch.isin(x, torch.as_tensor(cfg_keep_tokens, device=self.model.device))
            unmasked_index = unmasked_index & (~keep_mask)

        # ---- block scheduling ----
        num_blocks = math.ceil(max_new_tokens / block_length)
        steps = math.ceil(steps / num_blocks)
        histories = [x.clone()] if return_dict else None

        generated = 0  # number of appended tokens so far

        # ==========================================================
        # Block-wise generation loop
        # ==========================================================
        for b in range(num_blocks):
            cur_block_len = min(block_length, max_new_tokens - generated)
            if cur_block_len <= 0:
                break

            # ======================================================
            # NEW 2: append a fresh block of mask tokens
            # ======================================================
            new_block = torch.full(
                (B, cur_block_len), mask_id, dtype=torch.long, device=self.model.device
            )
            x = torch.cat([x, new_block], dim=1)

            # expand CFG mask status (new block tokens are NOT "given")
            unmasked_index = torch.cat(
                [
                    unmasked_index,
                    torch.zeros((B, cur_block_len), dtype=torch.bool, device=self.model.device),
                ],
                dim=1,
            )

            T = x.shape[1]

            # Build mask_index only for new block
            block_mask_index = torch.zeros((B, cur_block_len), dtype=torch.bool, device=x.device)
            for j in range(B):
                start = prompt_lens[j] + generated
                end = min(start + cur_block_len, T)
                if start < end:
                    block_mask_index[j, : (end - start)] = (x[j, start:end] == mask_id)

            # transfer schedule for diffusion steps
            num_transfer_tokens = get_num_transfer_tokens(
                mask_index=block_mask_index,
                steps=steps,
                scheduler=self.scheduler,
                stochastic=stochastic_transfer,
            )
            effective_steps = num_transfer_tokens.size(1)

            # ======================================================
            # NEW 3: build stair attention mask for current length T
            # ======================================================
            attention_mask = build_staircase_attention_mask(
                x=x,
                prompt_lens=prompt_lens,
                block_length=block_length,
            )  # [B, 1, T, T] boolean

            # ---- inner diffusion steps ----
            for i_step in range(effective_steps):
                mask_index = (x == mask_id)

                # ---- classifier-free guidance ----
                if cfg_scale > 0.0:
                    un_x = x.clone()
                    un_x[unmasked_index] = mask_id
                    x_ = torch.cat([x, un_x], dim=0)

                    # mask must also be duplicated along batch
                    attn = attention_mask.repeat(2, 1, 1, 1)

                    logits_all = self.model(x_, attention_mask=attn).logits
                    logits, un_logits = torch.chunk(logits_all, 2, dim=0)
                    logits = un_logits + (cfg_scale + 1) * (logits - un_logits)

                else:
                    logits = self.model(x, attention_mask=attention_mask).logits

                if right_shift_logits:
                    logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)

                # sample using Gumbel noise + argmax
                logits_noise = add_gumbel_noise(logits, temperature)
                x0 = torch.argmax(logits_noise, dim=-1)

                # confidence for masked positions
                if remasking == "low_confidence":
                    p = F.softmax(logits, dim=-1)
                    x0_p = torch.gather(p, dim=-1, index=x0.unsqueeze(-1)).squeeze(-1)
                elif remasking == "random":
                    x0_p = torch.rand((B, T), device=x.device)
                else:
                    raise NotImplementedError(remasking)

                # restrict selection to the current block region
                for j in range(B):
                    cutoff = prompt_lens[j] + (b + 1) * block_length
                    if cutoff < x0_p.size(1):
                        x0_p[j, cutoff:] = -np.inf

                # apply mask constraint (only update masked positions)
                x0 = torch.where(mask_index, x0, x)
                confidence = torch.where(mask_index, x0_p, -np.inf)

                # pick top-k masked positions per sample
                transfer_index = torch.zeros_like(x0, dtype=torch.bool)
                for j in range(B):
                    k = int(num_transfer_tokens[j, i_step].item())
                    k = min(k, (confidence[j] > -np.inf).sum().item())
                    if k > 0:
                        _, sel = torch.topk(confidence[j], k)
                        transfer_index[j, sel] = True

                # commit updates
                x[transfer_index] = x0[transfer_index]
                if histories is not None:
                    histories.append(x.clone())

            generated += cur_block_len

        # ---- output ----
        if not return_dict:
            return x
        return SamplerOutput(sequences=x, histories=histories)
