"""RoPE (Rotary Position Embeddings) relative position encoding.

Implements the RoPE attention mechanism for relative position encoding
using rotational transformations.
"""

from __future__ import annotations

import math
from typing import Any

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ModuleNotFoundError:
    torch = None
    nn = None
    F = None


def _require_torch() -> None:
    """Raise error if PyTorch is not installed."""
    if torch is None or nn is None:
        raise ModuleNotFoundError(
            "PyTorch is required for positional encodings. Install torch before use."
        )


def _apply_rotary_embedding(x: Any, positions: Any) -> Any:
    """Apply rotary position embeddings to a projected attention tensor.
    
    Rotates the even-odd dimensions of the input based on absolute position,
    encoding relative position information implicitly.
    
    Args:
        x: Tensor of shape (..., d_model).
        positions: Position indices for each sequence element.
        
    Returns:
        Rotated tensor of same shape as x.
    """
    rotary_dim = x.shape[-1] - (x.shape[-1] % 2)
    if rotary_dim <= 0:
        return x

    x_rotary = x[..., :rotary_dim]
    x_pass = x[..., rotary_dim:]
    half_dim = rotary_dim // 2
    inv_freq = 1.0 / (
        10000 ** (torch.arange(0, half_dim, device=x.device, dtype=x.dtype) / max(half_dim, 1))
    )
    angles = positions.to(dtype=x.dtype).unsqueeze(-1) * inv_freq.unsqueeze(0)
    cos = angles.cos().unsqueeze(0).unsqueeze(0)
    sin = angles.sin().unsqueeze(0).unsqueeze(0)

    x_even = x_rotary[..., ::2]
    x_odd = x_rotary[..., 1::2]
    rotated = torch.stack(
        (x_even * cos - x_odd * sin, x_even * sin + x_odd * cos), dim=-1
    ).flatten(-2)
    if x_pass.shape[-1] == 0:
        return rotated
    return torch.cat([rotated, x_pass], dim=-1)


class RelativePositionSelfAttention(nn.Module if nn is not None else object):
    """Multi-head self-attention with RoPE (Rotary Position Embeddings).
    
    Implements relative position encoding using rotations applied to
    query and key projections.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.1,
    ):
        """Initialize RoPE self-attention.
        
        Args:
            d_model: Model dimension (must be divisible by num_heads).
            num_heads: Number of attention heads.
            dropout: Dropout rate.
        """
        _require_torch()
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.qkv_projection = nn.Linear(d_model, d_model * 3)
        self.output_projection = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Any, key_padding_mask: Any = None) -> Any:
        """Apply RoPE attention.
        
        Args:
            x: Input tensor of shape (batch, seq_len, d_model).
            key_padding_mask: Optional mask of shape (batch, seq_len) where True indicates padding.
            
        Returns:
            Output tensor of shape (batch, seq_len, d_model).
        """
        batch_size, seq_len, _ = x.shape
        qkv = self.qkv_projection(x)
        query, key, value = qkv.chunk(3, dim=-1)
        query = query.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        key = key.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE rotations to query and key
        positions = torch.arange(seq_len, device=x.device)
        query = _apply_rotary_embedding(query, positions)
        key = _apply_rotary_embedding(key, positions)

        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if key_padding_mask is not None:
            mask = key_padding_mask.view(batch_size, 1, 1, seq_len)
            scores = scores.masked_fill(mask, torch.finfo(scores.dtype).min)

        attention_weights = torch.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        attended = torch.matmul(attention_weights, value)
        attended = attended.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        return self.output_projection(attended)
