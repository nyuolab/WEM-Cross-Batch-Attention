"""
Heavily adapted from Karpathy's nanoGPT (https://github.com/karpathy/nanoGPT)
"""

from dataclasses import dataclass
from typing import Tuple, Optional
import numpy as np

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.attention.flex_attention import flex_attention, create_block_mask

from metrics import compute_tensor_stats
import torch.utils.checkpoint as checkpoint
from loader import PAD_TOKEN

class SwiGLU(nn.Module):
    """
    Used in LLaMA
    """
    def __init__(self):
        super().__init__()

    def forward(self, h):
        h, gate = h.chunk(2, dim=-1)
        return h * F.silu(gate)

# Modified from facebookresearch/llama/model.py
def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    #assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    assert freqs_cis.shape[-1] == x.shape[-1] # we allow variable sequence lengths
    freqs_cis = freqs_cis[:x.shape[1]] # truncate to sequence length
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


# Taken from facebookresearch/llama/model.py
def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)

    return xq_out.type_as(xq), xk_out.type_as(xk)

# Taken from facebookresearch/llama/model.py
def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (
        theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim)
    )
    t = torch.arange(end, device=freqs.device)  # type: ignore
    freqs = torch.outer(t, freqs).float()  # type: ignore
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64

    return freqs_cis

class LayerNorm(nn.Module):
    """ RMS Normalization """
    def __init__(self, n_embd):
        super().__init__()
        self.n_embd = n_embd
        self.alpha = nn.Parameter(torch.ones(n_embd))
        self.gamma = nn.Parameter(torch.zeros(n_embd))
        self.eps = 1e-5

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        x = self.alpha * (x - mean) / (std + self.eps) + self.gamma
        return x
    
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.attention.flex_attention import (
    flex_attention,
    create_block_mask,
)


import math
import torch
from torch import nn
from torch.nn.attention.flex_attention import (
    flex_attention,
    create_block_mask,
)

def _ceil_div(a, b):  # safe ceil division for ints
    return -(-a // b)

class BigBirdMaskBuilder:
    """
    Build a BlockMask that encodes BigBird sparsity:
      - window of 'window_blocks' around each query block
      - 'rand_blocks' random KV blocks per query block
      - 'global_tokens' (assumed at the beginning) act as global blocks
    """
    def __init__(
        self,
        block_size: int = 64,
        window_blocks: int = 3,
        rand_blocks: int = 3,
        global_tokens: int = 128,        # e.g., 2 * block_size as in BigBird base
        causal: bool = False,
        seed: int = 0,
        per_head_random: bool = True,
    ):
        self.block_size = block_size
        self.window_blocks = window_blocks
        self.rand_blocks = rand_blocks
        self.global_tokens = global_tokens
        self.causal = causal
        self.seed = seed
        self.per_head_random = per_head_random
        self._cache = {}  # (B,H,Lq,Lk,device) -> BlockMask

    def _make_rand_map(self, H, q_blocks, kv_blocks, device):
        """
        Returns int64 tensor of shape [H, q_blocks, rand_blocks] with the
        random KV block indices per query block and head.
        """
        g_blocks = _ceil_div(self.global_tokens, self.block_size)
        # Pre-allocate
        rand_map = torch.full((H, q_blocks, self.rand_blocks), -1,
                              device=device, dtype=torch.long)

        # We choose random blocks *outside* the local window and (optionally) excluding globals.
        # Deterministic per head for reproducibility.
        for h in range(H):
            gen = torch.Generator(device=device)
            # Use distinct seeds per head if requested
            gen.manual_seed(self.seed + (h if self.per_head_random else 0))

            for qb in range(q_blocks):
                # Window span (inclusive)
                left  = max(0, qb - self.window_blocks // 2)
                right = min(kv_blocks - 1, qb + self.window_blocks // 2)

                exclude = set(range(left, right + 1))
                # Exclude the same block and early/late blocks by causality when needed
                if self.causal:
                    # Only allow kv_blocks <= qb for causality
                    allow_pool = [kb for kb in range(kv_blocks) if kb <= qb]
                else:
                    allow_pool = list(range(kv_blocks))

                # (Optionally) exclude the global blocks from the random pool so they stay “special”.
                # If you prefer to allow them, just remove these two lines.
                exclude |= set(range(g_blocks))  # [0, g_blocks-1] are global

                candidate = [kb for kb in allow_pool if kb not in exclude]
                if len(candidate) == 0:
                    continue

                # Sample without replacement
                # Note: if fewer candidates than rand_blocks, we sample as many as available
                k = min(self.rand_blocks, len(candidate))
                idx = torch.randperm(len(candidate), generator=gen, device=device)[:k]
                chosen = torch.tensor([candidate[i.item()] for i in idx],
                                      device=device, dtype=torch.long)

                # If k < rand_blocks, pad (harmless duplicates)
                if k < self.rand_blocks:
                    pad = chosen[:1].repeat(self.rand_blocks - k)
                    chosen = torch.cat([chosen, pad], dim=0)

                rand_map[h, qb] = chosen
        return rand_map

    def _build_mask_mod(self, H, Q_LEN, KV_LEN, device):
        """
        Create a vectorized mask_mod closure that checks:
          - global tokens (as blocks at the beginning)
          - local window blocks
          - random blocks from rand_map
          - optional causality
        """
        BSZ = self.block_size
        q_blocks = _ceil_div(Q_LEN, BSZ)
        kv_blocks = _ceil_div(KV_LEN, BSZ)
        g_blocks = _ceil_div(self.global_tokens, BSZ)
        rand_map = self._make_rand_map(H, q_blocks, kv_blocks, device)  # [H, q_blocks, R]

        half_w = self.window_blocks // 2

        def mask_mod(b, h, q_idx, kv_idx):
            # token -> block ids
            qb = torch.div(q_idx, BSZ, rounding_mode='floor')
            kb = torch.div(kv_idx, BSZ, rounding_mode='floor')

            # Global: everyone may attend *to* global blocks; global queries attend to everyone.
            kv_is_global = kb < g_blocks
            q_is_global  = qb < g_blocks

            # Window: |kb - qb| <= half_w
            in_window = (kb >= (qb - half_w)) & (kb <= (qb + half_w))

            # Random: kb equals any of the preselected random blocks for (h, qb)
            # rand_map[h, qb] has shape [..., R]; compare against kb elementwise.
            # We need broadcasting to match rand_map indices per element in the tile.
            rsel = rand_map[h.long(), qb]                       # [..., R]
            in_random = (rsel == kb.unsqueeze(-1)).any(dim=-1)  # [...]

            allowed = kv_is_global | q_is_global | in_window | in_random

            if self.causal:
                allowed = allowed & (kv_idx <= q_idx)

            return allowed

        return mask_mod

    def get_block_mask(self, B, H, Q_LEN, KV_LEN, device):
        key = (B, H, Q_LEN, KV_LEN, device.index if isinstance(device, torch.device) else device)
        if key in self._cache:
            return self._cache[key]
        mask_mod = self._build_mask_mod(H, Q_LEN, KV_LEN, device)
        bm = create_block_mask(mask_mod, B, H, Q_LEN, KV_LEN, device=device, BLOCK_SIZE=self.block_size)
        self._cache[key] = bm
        return bm


class FlexBigBirdMHA(nn.Module):
    """
    Multi-head attention using BigBird sparsity + FlexAttention.
    - Self-attention; adapt easily to cross-attention by building mask with Q_LEN != KV_LEN.
    """
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        block_size: int = 64,
        window_blocks: int = 3,  # corresponds to w = 3*b tokens from BigBird base
        rand_blocks: int = 3,    # r = 3 random blocks, per paper
        global_tokens: int = 128,
        causal: bool = False,
        seed: int = 0,
        per_head_random: bool = True,
    ):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.h = num_heads
        self.d = embed_dim // num_heads
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.o_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.mask_builder = BigBirdMaskBuilder(
            block_size=block_size,
            window_blocks=window_blocks,
            rand_blocks=rand_blocks,
            global_tokens=global_tokens,
            causal=causal,
            seed=seed,
            per_head_random=per_head_random,
        )

    def forward(self, x):
        """
        x: [B, L, E]
        returns: [B, L, E]
        """
        B, L, E = x.shape
        H, D = self.h, self.d
        device = x.device

        q = self.q_proj(x).view(B, L, H, D).transpose(1, 2).contiguous()  # [B,H,L,D]
        k = self.k_proj(x).view(B, L, H, D).transpose(1, 2).contiguous()
        v = self.v_proj(x).view(B, L, H, D).transpose(1, 2).contiguous()

        bm = self.mask_builder.get_block_mask(B, H, L, L, device)
        y = flex_attention(q, k, v, block_mask=bm)                        # [B,H,L,D]
        y = y.transpose(1, 2).reshape(B, L, E)
        return self.o_proj(y)


class SparseSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.config = config
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.autoregressive = config.autoregressive
        self.register_buffer("freqs_cis", precompute_freqs_cis(self.n_embd // self.n_head, config.context_length))
        self.sparse_attn = FlexBigBirdMHA()

        ### Init weights
        self.c_attn.weight.data.normal_(std=self.config.param_std)
        self.c_proj.weight.data.normal_(std=self.config.param_std)

        if self.config.bias:
            self.c_attn.bias.data.zero_()
            self.c_proj.bias.data.zero_()

    def forward(self, x, attn_mask=None):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head) # (B, nh, hs, T)
        q = q.view(B, T, self.n_head, C // self.n_head) # (B, nh, hs, T)
        v = v.view(B, T, self.n_head, C // self.n_head) # (B, nh, hs, T)

        if getattr(self.config, "position_encoding", True) == "rope": # use gettr to maintain compatibility with older versions
            # apply RoPE
            q, k = apply_rotary_emb(q, k, self.freqs_cis)
        
        # transpose (B, nh, T, hs)
        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)

        y = torch.nn.functional.scaled_dot_product_attention(q, k, v, 
                                                            scale=8 / self.n_embd, # Changed for µP
                                                            attn_mask=attn_mask, # We don't attend to padding
                                                            dropout_p=self.dropout if self.training else 0, # training
                                                            is_causal=self.autoregressive)


        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))

        return y

class SelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.config = config
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.autoregressive = config.autoregressive
        self.register_buffer("freqs_cis", precompute_freqs_cis(self.n_embd // self.n_head, config.context_length))

        ### Init weights
        self.c_attn.weight.data.normal_(std=self.config.param_std)
        self.c_proj.weight.data.normal_(std=self.config.param_std)

        if self.config.bias:
            self.c_attn.bias.data.zero_()
            self.c_proj.bias.data.zero_()

    def forward(self, x, attn_mask=None):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head) # (B, nh, hs, T)
        q = q.view(B, T, self.n_head, C // self.n_head) # (B, nh, hs, T)
        v = v.view(B, T, self.n_head, C // self.n_head) # (B, nh, hs, T)

        if getattr(self.config, "position_encoding", True) == "rope": # use gettr to maintain compatibility with older versions
            # apply RoPE
            q, k = apply_rotary_emb(q, k, self.freqs_cis)
        
        # transpose (B, nh, T, hs)
        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)

        y = torch.nn.functional.scaled_dot_product_attention(q, k, v, 
                                                            scale=8 / self.n_embd, # Changed for µP
                                                            attn_mask=attn_mask, # We don't attend to padding
                                                            dropout_p=self.dropout if self.training else 0, # training
                                                            is_causal=self.autoregressive)


        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))

        return y

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.swiglu  = SwiGLU()
        self.c_proj  = nn.Linear(4 * config.n_embd // 2, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

        ### Init weights
        self.c_fc.weight.data.normal_(std=config.param_std)
        self.c_proj.weight.data.normal_(std=config.param_std / np.sqrt(2)) # sqrt(2) is because the input dim is double

        if config.bias:
            self.c_fc.bias.data.zero_()
            self.c_proj.bias.data.zero_()

    def forward(self, x):
        x = self.c_fc(x)
        x = self.swiglu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd)
        self.attn = SelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd)
        self.mlp = MLP(config)
        self.stats = []

    def forward(self, x, attn_mask, compute_stats=False):
        self.stats = []

        attn = self.attn(self.ln_1(x), attn_mask=attn_mask)
        x = x + attn

        mlp = self.mlp(self.ln_2(x))
        x = x + mlp

        if compute_stats:
            self.stats.append(compute_tensor_stats(attn, "attn.acts"))
            self.stats.append(compute_tensor_stats(mlp, "mlp.acts"))
            self.stats.append(compute_tensor_stats(x, "out"))

            def hook_fn(grad, name):
                self.stats.append(compute_tensor_stats(grad, name))

            if attn.requires_grad:
                attn.register_hook(lambda grad: hook_fn(grad, "attn.grads"))
            if mlp.requires_grad:
                mlp.register_hook(lambda grad: hook_fn(grad, "mlp.grads"))
            if x.requires_grad:
                x.register_hook(lambda grad: hook_fn(grad, "out.grads"))
        
        return x
    
class SparseAttentionBlock(nn.Module):
    """
    x -> LN -> SparseSelfAttention -> +res
       -> LN -> MLP -> +res
    Matches the API and stats behavior of `Block`.
    """
    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd)
        self.attn = SparseSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd)
        self.mlp = MLP(config)
        self.stats = []

    def forward(self, x, attn_mask, compute_stats=False):
        self.stats = []

        attn = self.attn(self.ln_1(x), attn_mask=attn_mask)
        x = x + attn

        mlp = self.mlp(self.ln_2(x))
        x = x + mlp

        if compute_stats:
            self.stats.append(compute_tensor_stats(attn, "attn.acts"))
            self.stats.append(compute_tensor_stats(mlp, "mlp.acts"))
            self.stats.append(compute_tensor_stats(x, "out"))

            def hook_fn(grad, name):
                self.stats.append(compute_tensor_stats(grad, name))

            if attn.requires_grad:
                attn.register_hook(lambda grad: hook_fn(grad, "attn.grads"))
            if mlp.requires_grad:
                mlp.register_hook(lambda grad: hook_fn(grad, "mlp.grads"))
            if x.requires_grad:
                x.register_hook(lambda grad: hook_fn(grad, "out.grads"))

        return x

@dataclass
class OmniBioTAConfig:
    context_length: int = 2048
    vocab_size: int = 2**16
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 1024
    dropout: float = 0.05
    bias: bool = False
    autoregressive: bool = False
    position_encoding: str = "rope"
    memory_efficient: bool = False
    padding_idx: int = PAD_TOKEN

class OmniBioTA(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.context_length is not None
        self.config = config
        self.config.lr_scale = 32 / config.n_embd # 32 is arbitrary but makes it easier if you set the overall LR to 0.01 (leads to an LR scale of 3.125e-4 at width 1024)
        self.config.param_std = 1 / np.sqrt(config.n_embd)

        assert config.position_encoding in ["rope", "learned"]

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd, padding_idx=config.padding_idx),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd),
        ))

        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.lm_head.weight.data.normal_(std=1e-5)

        if self.config.position_encoding == "learned":
            self.pos_emb = nn.Parameter(torch.zeros(1, config.context_length, config.n_embd))

        self.stats = []

        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wte.weight.numel()
            if self.config.position_encoding == "learned":
                n_params -= self.pos_emb.numel()
        return n_params

    def forward(self, idx, attn_mask=None, return_embeddings=False, compute_stats=False):
        '''
        Args:
            idx: a torch.LongTensor of shape (b, t) of token indices
            attn_mask: the attention mask to apply during flash attention
            return_embeddings: if True, return the token embeddings instead of the logits
        Returns:
        if return_embeddings=False:
            logits: a torch.FloatTensor of shape (b, t, vocab_size) of logits
        if return_embeddings=True:
            emb: a torch.FloatTensor of shape (b, t, n_embd) of token embeddings
        '''
        _, t = idx.size()
        assert t <= self.config.context_length, f"Cannot forward sequence of length {t}, block size is only {self.config.context_length}"

        self.stats = []

        def hook_fn(grad, name):
            self.stats.append(compute_tensor_stats(grad, name))

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)

        if compute_stats:
            self.stats.append(compute_tensor_stats(tok_emb, "tok_emb.acts"))

            if tok_emb.requires_grad:
                tok_emb.register_hook(lambda grad: hook_fn(grad, "tok_emb.grad"))

        if self.config.position_encoding == "learned":
            tok_emb = tok_emb + self.pos_emb[:, :t]

        x = self.transformer.drop(tok_emb)
        for block in self.transformer.h:
            if getattr(self.config, "memory_efficient", False):
                x = checkpoint.checkpoint(block, x, attn_mask, compute_stats, use_reentrant=False)
            else:
                x = block(x, attn_mask=attn_mask, compute_stats=compute_stats)

        emb = self.transformer.ln_f(x)

        if return_embeddings:
            return emb
        else:
            logits = self.lm_head(emb) * self.config.lr_scale

            if compute_stats:
                self.stats.append(compute_tensor_stats(logits, "logits"))
                if logits.requires_grad:
                    logits.register_hook(lambda grad: hook_fn(grad, "logits.grad"))

            return logits
    
    def get_stats(self):
        stats = []
        stats += self.stats
        for i, block in enumerate(self.transformer.h):
            for stat in block.stats:
                stat.name = f"block_{i}.{stat.name}"
                stats.append(stat)

        for name, param in self.named_parameters():
            if param.grad is not None:
                stats.append(compute_tensor_stats(param.grad, name + ".grad"))

        return stats
    
    @staticmethod
    def create_optimizer(model, lr=0.01, weight_decay=0.01, beta1=0.9, beta2=0.999, epsilon=1e-8):
        def fixed_lr(param, name):
            '''
            Determines how LR should scale with a param given muP scaling laws
            '''
            keys = ["lm_head"]
            if any([k in name for k in keys]):
                return True
            if len(param.shape) == 1:
                return True
            
            return False
        
        def no_wd(param, name):
            '''
            Determines which parameters should not have weight decay
            '''
            keys = ["ln", "wte", "pos_emb"]
            if any([k in name for k in keys]):
                return True
            
            return False

        fixed_lr_params = [p for n, p in model.named_parameters() if fixed_lr(p, n) and not no_wd(p, n)]
        inverse_dmodel_params = [p for n, p in model.named_parameters() if not fixed_lr(p, n) and not no_wd(p, n)]
        no_wd_params = [p for n, p in model.named_parameters() if no_wd(p, n)]

        fixed_lr = lr
        inverse_dmodel_lr = lr * 32 / model.config.n_embd
        
        param_groups = [
            {"params": inverse_dmodel_params, "lr": inverse_dmodel_lr,
             "weight_decay": weight_decay * model.config.n_embd / 32}, # scales weight decay by inverse LR to keep weight decay constant (since AdamW weight decay is wd * lr)
            {"params": fixed_lr_params, "lr": fixed_lr, "weight_decay": weight_decay},
            {"params": no_wd_params, "lr": fixed_lr, "weight_decay": 0.0}
        ]

        optimizer = torch.optim.AdamW(param_groups,
                                      betas=(beta1, beta2),
                                      eps=epsilon)
        
        return optimizer


@dataclass
class SparseAttentionConfig(OmniBioTAConfig):
    # BigBird/Longformer-style sparsity knobs
    block_size: int = 64
    local_span_blocks: int = 1
    num_random: int = 0
    random_per_head: bool = True
    global_blocks: Optional[Tuple[int, ...]] = None
    random_seed: int = 0

    # If you need *true* attention-dropout (dense SDPA fallback). Use only for short seqs.
    allow_dense_sdpa_dropout_fallback: bool = False

class SparseAttentionTransformer(nn.Module):
    """
    Same external interface as `OmniBioTA`, but with block-sparse FlexAttention.
    """
    def __init__(self, config: SparseAttentionConfig):
        super().__init__()
        assert config.vocab_size is not None
        assert config.context_length is not None
        self.config = config

        # Keep your scaling conventions
        self.config.lr_scale = 32 / config.n_embd
        self.config.param_std = 1 / np.sqrt(config.n_embd)

        assert config.position_encoding in ["rope", "learned"]

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd, padding_idx=config.padding_idx),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([SparseAttentionBlock(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd),
        ))

        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.lm_head.weight.data.normal_(std=1e-5)

        if self.config.position_encoding == "learned":
            self.pos_emb = nn.Parameter(torch.zeros(1, config.context_length, config.n_embd))

        self.stats = []

        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

    def get_num_params(self, non_embedding=True):
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wte.weight.numel()
            if self.config.position_encoding == "learned":
                n_params -= self.pos_emb.numel()
        return n_params

    def forward(self, idx, attn_mask=None, return_embeddings=False, compute_stats=False):
        """
        idx: [B, T] (LongTensor)
        attn_mask: optional [B, T] bool (True=real token). If None, we auto-build from padding_idx.
        """
        B, T = idx.size()
        assert T <= self.config.context_length, f"Cannot forward sequence of length {T}, block size is only {self.config.context_length}"

        self.stats = []

        # Auto-build padding mask if not provided
        if attn_mask is None and self.config.padding_idx is not None:
            attn_mask = (idx != self.config.padding_idx)

        tok_emb = self.transformer.wte(idx)  # [B, T, C]

        if compute_stats:
            self.stats.append(compute_tensor_stats(tok_emb, "tok_emb.acts"))
            if tok_emb.requires_grad:
                tok_emb.register_hook(lambda grad: self.stats.append(compute_tensor_stats(grad, "tok_emb.grad")))

        if self.config.position_encoding == "learned":
            tok_emb = tok_emb + self.pos_emb[:, :T]

        x = self.transformer.drop(tok_emb)

        for block in self.transformer.h:
            if getattr(self.config, "memory_efficient", False):
                x = checkpoint.checkpoint(block, x, attn_mask, compute_stats, use_reentrant=False)
            else:
                x = block(x, attn_mask=attn_mask, compute_stats=compute_stats)

        emb = self.transformer.ln_f(x)

        if return_embeddings:
            return emb

        logits = self.lm_head(emb) * self.config.lr_scale

        if compute_stats:
            self.stats.append(compute_tensor_stats(logits, "logits"))
            if logits.requires_grad:
                logits.register_hook(lambda grad: self.stats.append(compute_tensor_stats(grad, "logits.grad")))

        return logits

    def get_stats(self):
        stats = []
        stats += self.stats
        for i, block in enumerate(self.transformer.h):
            for stat in block.stats:
                stat.name = f"block_{i}.{stat.name}"
                stats.append(stat)
        for name, param in self.named_parameters():
            if param.grad is not None:
                stats.append(compute_tensor_stats(param.grad, name + ".grad"))
        return stats

    @staticmethod
    def create_optimizer(model, lr=0.01, weight_decay=0.01, beta1=0.9, beta2=0.999, epsilon=1e-8):
        """
        Same LR/wd partitioning you already use in OmniBioTA, for drop-in parity.
        """
        def fixed_lr(param, name):
            keys = ["lm_head"]
            if any([k in name for k in keys]):
                return True
            if len(param.shape) == 1:
                return True
            return False

        def no_wd(param, name):
            keys = ["ln", "wte", "pos_emb"]
            if any([k in name for k in keys]):
                return True
            return False

        fixed_lr_params = [p for n, p in model.named_parameters() if fixed_lr(p, n) and not no_wd(p, n)]
        inverse_dmodel_params = [p for n, p in model.named_parameters() if not fixed_lr(p, n) and not no_wd(p, n)]
        no_wd_params = [p for n, p in model.named_parameters() if no_wd(p, n)]

        fixed_lr = lr
        inverse_dmodel_lr = lr * 32 / model.config.n_embd

        param_groups = [
            {"params": inverse_dmodel_params, "lr": inverse_dmodel_lr,
             "weight_decay": weight_decay * model.config.n_embd / 32},
            {"params": fixed_lr_params, "lr": fixed_lr, "weight_decay": weight_decay},
            {"params": no_wd_params, "lr": fixed_lr, "weight_decay": 0.0}
        ]

        optimizer = torch.optim.AdamW(param_groups, betas=(beta1, beta2), eps=epsilon)
        return optimizer
