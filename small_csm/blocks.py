"""The transformer block, written once and used by both CSM transformers (backbone and depth decoder)."""

from typing import Callable

import torch
from transformers.models.csm.modeling_csm import apply_rotary_pos_emb

# write(k_cache, v_cache, k, v): store this step's keys/values in the caches
Write = Callable[[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], None]
# attend(k_cache, v_cache, q) -> attention output [B, heads, T, head_dim]
Attend = Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]


def decoder_block(layer, x, cos, sin, k_cache, v_cache, n_heads: int, n_kv_heads: int, head_dim: int,
                  write: Write, attend: Attend) -> torch.Tensor:
    """One Llama-style block (RMSNorm, GQA attention with RoPE, gated MLP), using a Hugging Face
    `CsmDecoderLayer` for the weights.

    The two transformers differ only in how they address their KV cache, so that part is injected:
    the backbone writes at dynamic positions into a 2048-slot cache, the depth decoder at static slices.
    x: [B, T, hidden]; cos/sin: rotary tables for these T positions.
    """
    h = layer.input_layernorm(x)
    a = layer.self_attn
    B, T, _ = h.shape
    q = a.q_proj(h).view(B, T, n_heads, head_dim).transpose(1, 2)
    k = a.k_proj(h).view(B, T, n_kv_heads, head_dim).transpose(1, 2)
    v = a.v_proj(h).view(B, T, n_kv_heads, head_dim).transpose(1, 2)
    q, k = apply_rotary_pos_emb(q, k, cos, sin)
    write(k_cache, v_cache, k, v)
    o = attend(k_cache, v_cache, q).transpose(1, 2).reshape(B, T, -1)
    x = x + a.o_proj(o)
    return x + layer.mlp(layer.post_attention_layernorm(x))
