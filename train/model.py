from dataclasses import dataclass, field
from typing import Tuple
import numpy as np

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from flash_attn import flash_attn_qkvpacked_func
from memory import MemTracker
from metrics import compute_tensor_stats
from torch.distributed.nn.functional import all_to_all

class CrossBatchDistributed:

    def __init__(self, tp_rank, dp_rank, group, tp_size):
        self.tp_rank = tp_rank
        self.dp_rank = dp_rank
        self.group = group
        self.tp_size = tp_size

    def transpose_shard_2d_ddp(self, input_tensor, cut_dim=1, cat_dim=0):
        input_chunks = torch.chunk(input_tensor, self.tp_size, dim=cut_dim) 

        output_chunks = [torch.empty_like(input_chunks[0]) for _ in range(self.tp_size)]

        output_chunks = all_to_all(
            output_chunks,
            input_chunks,
            group=self.group,
        )

        output_tensor = torch.cat(output_chunks, dim=cat_dim)
        return output_tensor

    def scatter_tensor(self, x, dim):
        x_list = list(torch.chunk(x, self.tp_size, dim=dim))
        output = torch.empty_like(x_list[0])
        if self.tp_rank != 0:
            x_list = None
        dist.scatter(tensor=output, scatter_list=x_list, group_src=0, group=self.group)
        return output


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

# Taken from Llama 2 implementation
class LayerNorm(nn.Module):
    """ RMS Normalization """
    def __init__(self, n_embd):
        super().__init__()
        self.n_embd = n_embd
        self.weight = nn.Parameter(torch.ones(n_embd))
        self.eps = 1e-5

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

class SelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.config = config
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.autoregressive = config.autoregressive

        ### Init weights
        self.c_attn.weight.data.normal_(std=self.config.param_std)
        self.c_proj.weight.data.normal_(std=self.config.param_std)

        if self.config.bias:
            self.c_attn.bias.data.zero_()
            self.c_proj.bias.data.zero_()

    def forward(self, x, attn_mask=None, freqs_cis=None):
        # x can be any shape as long as the last dimension is n_embd
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        qkv = self.c_attn(x)
        y = flash_attn_qkvpacked_func(qkv.view(B, T, 3, self.n_embd).view(B, T, 3, self.n_head, C // self.n_head), softmax_scale=8 / self.n_embd, causal=self.autoregressive, dropout_p=self.dropout if self.training else 0)
        y = y.contiguous().view(B, T, C) # re-assemble all head outputs side by side

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

    def forward(self, x, attn_mask, compute_stats=False, freqs_cis=None):
        # print("Before Block", torch.cuda.memory_allocated() / 1024**2, "MB")
        
        self.stats = []
        attn = self.attn(self.ln_1(x), attn_mask=attn_mask, freqs_cis=freqs_cis)
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

    

class WidthBlock2D(nn.Module):
    """
    Processes attention along the WIDTH dimension.
    For each position in the length dimension, attends across all width positions.
    Shape: (B, L, W, C) → reshape → (B*L, W, C) → attention along W → reshape back.
    """
    def __init__(self, config):
        super().__init__()
        # Create a config for the inner block
        inner_config = Block1DConfig(
            n_head=config.n_head,
            n_embd=config.n_embd,
            dropout=config.dropout,
            bias=config.bias,
            autoregressive=config.autoregressive,
            position_encoding=config.position_encoding,
            context_length=config.width,
            param_std=config.param_std
        )
        self.inner = Block(inner_config)

        # Pre-compute rotary/pos-enc frequencies for width dimension
        self.register_buffer(
            "freqs_cis_width",
            precompute_freqs_cis(config.n_embd // config.n_head, config.width)
        )

    def forward(self, x, attn_mask=None, compute_stats: bool = False):
        B, L, W, C = x.shape
        x_reshaped = x.contiguous().view(B * L, W, C)

        # Expand the mask to match the (B*L, W) batch if provided
        if attn_mask is not None:
            attn_mask_expanded = attn_mask.repeat_interleave(L, dim=0)
        else:
            attn_mask_expanded = None

        out = self.inner(
            x_reshaped,
            attn_mask=attn_mask_expanded,
            compute_stats=compute_stats,
            freqs_cis=self.freqs_cis_width
        )

        return out.view(B, L, W, C).contiguous()
    

class LengthBlock2D(nn.Module):
    """
    Processes attention along the LENGTH dimension.
    For each position in the width dimension, attends across all length positions.
    Shape: (B, L, W, C) → transpose → (B*W, L, C) → attention along L → reshape back.
    """
    def __init__(self, config):
        super().__init__()
        # Create a config for the inner block
        inner_config = Block1DConfig(
            n_head=config.n_head,
            n_embd=config.n_embd,
            dropout=config.dropout,
            bias=config.bias,
            autoregressive=config.autoregressive,
            position_encoding=config.position_encoding,
            context_length=config.length,
            param_std=config.param_std
        )
        self.inner = Block(inner_config)

        # Pre-compute rotary/pos-enc frequencies for length dimension
        self.register_buffer(
            "freqs_cis_length",
            precompute_freqs_cis(config.n_embd // config.n_head, config.length)
        )

    def forward(self, x, attn_mask=None, compute_stats: bool = False):

        B, L, W, C = x.shape
        x_transposed = x.transpose(1, 2).contiguous().view(B * W, L, C)

        if attn_mask is not None:
            attn_mask_expanded = attn_mask.repeat_interleave(W, dim=0)
        else:
            attn_mask_expanded = None

        out = self.inner(
            x_transposed,
            attn_mask=attn_mask_expanded,
            compute_stats=compute_stats,
            freqs_cis=self.freqs_cis_length
        )

        out = out.view(B, W, L, C).transpose(1, 2).contiguous()
        return out

@dataclass
class Block1DConfig:          
    n_head:        int = 16
    n_embd:        int = 1024
    dropout:       float = 0.05
    bias:          bool = False
    autoregressive: bool = False
    position_encoding: str = "rope"
    context_length: int = 2048
    param_std: float = 0.02



@dataclass
class OLT2DConfig:
    length: int = 2048  
    width: int = 2048   
    vocab_size: int = 2**16
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 1024
    dropout: float = 0.05
    bias: bool = False
    autoregressive: bool = False
    position_encoding: str = "rope"
    type: str = "alternate"  
    freq: int = 1  
    last_n: int = 4  
    track_memory: bool = False       
    ckpt_embed:   bool = False

class OLT2D(nn.Module):
    def __init__(self, config, distributed_comms: CrossBatchDistributed):
        super().__init__()
        assert config.vocab_size is not None
        assert config.length is not None
        assert config.width is not None
        
        self.config = config
        self.config.lr_scale = 32 / config.n_embd
        self.config.param_std = 1 / np.sqrt(config.n_embd)
        self.layer_types = None

        assert config.position_encoding in ["rope", "learned"]

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList(),  # Will be populated by layer_composition_init
            ln_f = LayerNorm(config.n_embd),
        ))

        self.distributed_comms = distributed_comms

        # Initialize layer composition
        self.layer_composition_init(config)

        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.lm_head.weight.data.normal_(std=1e-5)

        if self.config.position_encoding == "learned":
            # For 2D, we need position embeddings for both dimensions
            self.pos_emb_length = nn.Parameter(torch.zeros(1, config.length, 1, config.n_embd))
            self.pos_emb_width = nn.Parameter(torch.zeros(1, 1, config.width, config.n_embd))

        self.stats = []

        # report number of parameters
        if config.track_memory:
            self._attach_memory_hooks()    
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

    def _attach_memory_hooks(self):
        """Register forward/backward hooks on every heavy sub-module."""
        tracker = MemTracker()
        # token embed & lm_head
        self.transformer.wte.register_forward_hook(tracker.fw("wte"))
        self.transformer.wte.register_full_backward_hook(tracker.bw("wte"))
        self.lm_head.register_forward_hook(tracker.fw("lm_head"))
        self.lm_head.register_full_backward_hook(tracker.bw("lm_head"))
        # each 2-D block
        for i, blk in enumerate(self.transformer.h):
            blk.register_forward_hook(tracker.fw(f"blk{i:03d}"))
            blk.register_full_backward_hook(tracker.bw(f"blk{i:03d}"))

    def layer_composition_init(self, config):
        """
        Initialize layers based on the composition configuration.
        
        Composition types:
        - "alternate": Alternates between length and width attention layers
          - freq=1: alternates every layer (length, width, length, width, ...)
          - freq=3: 3 length layers then 1 width layer (length, length, length, width, ...)
        - "later": Length layers for all except the last 'last_n' layers which are width layers
        """
        
        
        if config.type == "alternate":
            # Alternate between length and width attention layers
            for i in range(config.n_layer):
                # Determine if this should be a width or length layer
                cycle_length = config.freq + 1
                position_in_cycle = i % cycle_length
                
                if position_in_cycle == config.freq:  # Last position in cycle is width
                    layer = LengthBlock2D(config)
                else:
                    layer = WidthBlock2D(config)
                
                self.transformer.h.append(layer)
                
        elif config.type == "later":
            # All length layers except the last 'last_n' which are width layers
            for i in range(config.n_layer):
                if i >= config.n_layer - config.last_n:
                    layer = LengthBlock2D(config)
                else:
                    layer = WidthBlock2D(config)
                
                self.transformer.h.append(layer)
                
        else:
            raise ValueError(f"Unknown layer composition type: {config.type}")
        
        # Print layer composition for verification
        self.layer_types = []
        for layer in self.transformer.h:
            if isinstance(layer, WidthBlock2D):
                self.layer_types.append("W")
            else:
                self.layer_types.append("L")
        print(f"Layer composition ({config.type}): {' '.join(self.layer_types)}")

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wte.weight.numel()
            if self.config.position_encoding == "learned":
                n_params -= self.pos_emb_length.numel()
                n_params -= self.pos_emb_width.numel()
        return n_params
    
    def forward(self, idx, attn_mask=None, return_embeddings=False, compute_stats=False):
        b, l, w = idx.size()
        assert l <= self.config.length, f"Cannot forward sequence of length {l}, length is only {self.config.length}"
        assert w <= self.config.width, f"Cannot forward sequence of width {w}, width is only {self.config.width}"

        self.stats = []

        def hook_fn(grad, name):
            self.stats.append(compute_tensor_stats(grad, name))

        # Get token embeddings``
        with torch.cuda.amp.autocast():
            if self.config.ckpt_embed:
                # saves ~ (B·L·W·D) activations; recomputed during backward
                tok_emb = checkpoint(lambda i: self.transformer.wte(i), idx,
                                    use_reentrant=False)
            else:
                tok_emb = self.transformer.wte(idx)
        if compute_stats:
            self.stats.append(compute_tensor_stats(tok_emb, "tok_emb.acts"))
            tok_emb.register_hook(lambda grad: hook_fn(grad, "tok_emb.grad"))

        if self.config.position_encoding == "learned":
            # Add positional embeddings for both dimensions
            tok_emb = tok_emb + self.pos_emb_length[:, :l, :, :] + self.pos_emb_width[:, :, :w, :]

        x = self.transformer.drop(tok_emb)
        
        for idx, block in enumerate(self.transformer.h):
            if idx != 0 and self.layer_types[idx-1] != self.layer_types[idx]:
                if self.layer_types[idx] == 'W':
                    x = self.distributed_comms.transpose_shard_2d_ddp(x, 2, 1)
                else:
                    x = self.distributed_comms.transpose_shard_2d_ddp(x, 1, 2)

            x = checkpoint(
                lambda x_, m_=block, attn_mask_=attn_mask:            # m_ is frozen
                    m_(x_, attn_mask=attn_mask_, compute_stats=compute_stats),
                x,
                use_reentrant=False
            )
        if self.layer_types[-1] == 'L':
            x = self.distributed_comms.transpose_shard_2d_ddp(x, 2, 1)


        emb = self.transformer.ln_f(x)

        if return_embeddings:
            return emb
        else:
            # Reshape for linear layer (treating each width separately)
            b, l, w, c = emb.size()
            emb_reshaped = emb.reshape(b*l*w, c)
            logits_flat = self.lm_head(emb_reshaped) * self.config.lr_scale
            logits = logits_flat.reshape(b, l, w, -1)

            if compute_stats:
                self.stats.append(compute_tensor_stats(logits, "logits"))
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
    
    def __getstate__(self):
        state = self.__dict__.copy()
        state['distributed_comms'] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.distributed_comms = None