"""Positional encoding strategies for transformer models.

Provides multiple relative and absolute positional encoding implementations:
- absolute: Standard learnable positional embeddings (used in model.py)
- alibi: Attention with Linear Biases for relative position encoding
- rope: Rotary Position Embeddings for relative position encoding
"""

from .alibi import (
    RelativePositionSelfAttention as ALiBiRelativePositionSelfAttention,
    _alibi_slopes,
)
from .rope import (
    RelativePositionSelfAttention as RoPERelativePositionSelfAttention,
    _apply_rotary_embedding,
)

__all__ = [
    "_alibi_slopes",
    "ALiBiRelativePositionSelfAttention",
    "_apply_rotary_embedding",
    "RoPERelativePositionSelfAttention",
]
