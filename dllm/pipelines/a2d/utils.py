from dataclasses import dataclass

import torch
import transformers


@dataclass
class A2DSFTCollator(transformers.DataCollatorForSeq2Seq):
    # right_shift_logits: bool = True

    def __call__(self, features, return_tensors=None):
        outputs = super().__call__(features, return_tensors=return_tensors)
        # fintune on padding <eos_token>; should not mask them out
        outputs.pop("attention_mask")

        input_ids, labels = (
            outputs["input_ids"],
            outputs["labels"],
        )
        bsz, seq_len = input_ids.shape

        # --- Add BOS token to the beginning of input_ids ---
        bos = torch.full(
            (bsz, 1),
            self.tokenizer.bos_token_id,
            dtype=input_ids.dtype,
            device=input_ids.device,
        )
        input_ids = torch.cat([bos, input_ids], dim=1)

        # --- Prepend zeros to labels instead of BOS ---
        ignore_labels = self.label_pad_token_id * torch.ones(
            (bsz, 1), dtype=labels.dtype, device=labels.device
        )
        labels = torch.cat([ignore_labels, labels], dim=1)

        # --- Update and return --`-
        outputs["input_ids"] = input_ids
        outputs["labels"] = labels

        return outputs
