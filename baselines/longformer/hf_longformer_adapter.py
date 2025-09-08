# hf_longformer_adapter.py
import torch
import torch.nn as nn
from transformers import LongformerConfig, LongformerForMaskedLM

class HFLongformerWrapper(nn.Module):
    """
    Thin wrapper so you can call: logits = model(input_ids, attn_mask=pad_mask)
    where attn_mask is your boolean (B,1,1,T) "True = mask out" mask.
    We convert it to HF's (B,T) attention_mask with 1 = keep/attend.
    We also build a global_attention_mask: first non-pad token is global.
    """

    def __init__(
        self,
        *,
        vocab_size: int,
        pad_token_id: int,
        hidden_size: int,
        num_heads: int,
        num_layers: int,
        max_position_embeddings: int,
        attention_window: int = 512,
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        self.pad_token_id = pad_token_id

        config = LongformerConfig(
            vocab_size=vocab_size,
            pad_token_id=pad_token_id,
            hidden_size=hidden_size,
            num_attention_heads=num_heads,
            num_hidden_layers=num_layers,
            max_position_embeddings=max_position_embeddings,
            attention_window=attention_window,   # can also be a per-layer list
            type_vocab_size=1,
        )

        self.longformer = LongformerForMaskedLM(config)
        self.longformer.gradient_checkpointing_enable()  # saves mem on long seqs
        # You can move to dtype/device outside to match your training code.

    @torch.no_grad()
    def _make_attention_masks(self, input_ids: torch.Tensor, attn_mask_4d=None):
        # HF wants (B,T) attention_mask with 1 where tokens are real.
        if attn_mask_4d is None:
            attention_mask = (input_ids != self.pad_token_id).long()
        else:
            # attn_mask_4d: (B,1,1,T) where True = mask OUT. Invert & squeeze.
            attention_mask = (~attn_mask_4d.squeeze(1).squeeze(1)).long()

        # Global attention: mark first non-pad token as global (1)
        # If everything is pad, argmax returns 0; attention_mask[*,0]==0 so no global bit gets set.
        first_nonpad = attention_mask.argmax(dim=1)  # (B,)
        global_attention_mask = torch.zeros_like(attention_mask)
        global_attention_mask.scatter_(1, first_nonpad.unsqueeze(1), 1)
        return attention_mask, global_attention_mask

    def forward(self, input_ids: torch.Tensor, attn_mask=None):
        attention_mask, global_attention_mask = self._make_attention_masks(input_ids, attn_mask)
        out = self.longformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            global_attention_mask=global_attention_mask,
            return_dict=True,
        )
        return out.logits  # (B, T, vocab)

    # for your FLOPs accounting
    def get_num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
