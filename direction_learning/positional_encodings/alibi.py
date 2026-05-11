"""ALiBi (Attention with Linear Biases) relative position encoding.

Implements the ALiBi attention mechanism for relative position encoding
without learnable embeddings.
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


def _alibi_slopes(num_heads: int) -> Any:
    """Build head-specific slopes for ALiBi-style relative bias.
    
    Each attention head gets a different slope for relative position bias,
    distributed geometrically following the ALiBi paper.
    
    Args:
        num_heads: Number of attention heads.
        
    Returns:
        torch.Tensor of shape (num_heads,) with slope values.
    """
    if num_heads < 1:
        raise ValueError("num_heads must be positive")

    def _slopes_power_of_two(power_of_two: int) -> list[float]:
        start = 2.0 ** (-2.0 ** -(math.log2(power_of_two) - 3.0))
        ratio = start
        return [start * (ratio ** index) for index in range(power_of_two)]

    if math.log2(num_heads).is_integer():
        slopes = _slopes_power_of_two(num_heads)
    else:
        closest_power_of_two = 2 ** math.floor(math.log2(num_heads))
        slopes = _slopes_power_of_two(closest_power_of_two)
        slopes.extend(
            _slopes_power_of_two(2 * closest_power_of_two)[0::2][: num_heads - closest_power_of_two]
        )

    return torch.tensor(slopes, dtype=torch.float32)


class RelativePositionSelfAttention(nn.Module if nn is not None else object):
    """Multi-head self-attention with ALiBi (Attention with Linear Biases).
    
    Implements relative position encoding using learned head-specific slopes
    applied to relative position distances.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.1,
    ):
        """Initialize ALiBi self-attention.
        
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
        self.register_buffer("alibi_slopes", _alibi_slopes(num_heads), persistent=False)

    def forward(self, x: Any, key_padding_mask: Any = None) -> Any:
        """Apply ALiBi attention.
        
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

        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # Apply ALiBi bias: slope * relative_distance
        positions = torch.arange(seq_len, device=x.device)
        relative_positions = positions.unsqueeze(0) - positions.unsqueeze(1)
        if seq_len > 1:
            relative_positions = relative_positions.to(dtype=x.dtype) / float(seq_len - 1)
        else:
            relative_positions = relative_positions.to(dtype=x.dtype)
        bias = (
            self.alibi_slopes.to(dtype=x.dtype).view(1, self.num_heads, 1, 1)
            * relative_positions.view(1, 1, seq_len, seq_len)
        )
        scores = scores + bias

        if key_padding_mask is not None:
            mask = key_padding_mask.view(batch_size, 1, 1, seq_len)
            scores = scores.masked_fill(mask, torch.finfo(scores.dtype).min)

        attention_weights = torch.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        attended = torch.matmul(attention_weights, value)
        attended = attended.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        return self.output_projection(attended)
