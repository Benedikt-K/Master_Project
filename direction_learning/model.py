from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ModuleNotFoundError:
    torch = None
    nn = None
    F = None


@dataclass(frozen=True)
class DirectionTransformerConfig:
    """Immutable configuration for the SpacerDirectionTransformer model.
    
    Controls embedding dimensions, transformer architecture, dropout,
    maximum array size, and whether to include flanking sequences.
    """
    vocab_size: int
    token_dim: int = 64
    spacer_dim: int = 128
    transformer_dim: int = 128
    num_heads: int = 4
    num_layers: int = 4
    feedforward_dim: int | None = None
    dropout: float = 0.1
    activation: str = "gelu"
    max_spacers: int = 64
    include_flanks: bool = False
    positional_encoding: str = "absolute"
    pooling_strategy: str = "mean"


def _require_torch() -> None:
    """Raise error if PyTorch is not installed."""
    if torch is None or nn is None:
        raise ModuleNotFoundError(
            "PyTorch is required to instantiate direction_learning.model classes. "
            "Install torch before training the transformer."
        )


class SequenceEncoderBase(nn.Module if nn is not None else object):
    """Base class for variable-length sequence encoders with configurable pooling.
    
    Embeds tokens and applies a pooling strategy (mean, max, attention, or learnable)
    to produce fixed-size embeddings per sequence.
    """
    def __init__(self, vocab_size: int, token_dim: int, spacer_dim: int, dropout: float = 0.1, pooling_strategy: str = "mean"):
        """Initialize the sequence encoder.
        
        Args:
            vocab_size: Size of the token vocabulary.
            token_dim: Dimension of token embeddings.
            spacer_dim: Output dimension after projection.
            dropout: Dropout rate for regularization.
            pooling_strategy: One of 'mean', 'max', 'attention', 'learnable'.
        """
        _require_torch()
        super().__init__()
        if pooling_strategy not in {"mean", "max", "attention", "learnable"}:
            raise ValueError(f"Unknown pooling_strategy: {pooling_strategy}")
        
        self.pooling_strategy = pooling_strategy
        self.embedding = nn.Embedding(vocab_size, token_dim, padding_idx=0)
        self.projection = nn.Sequential(
            nn.Linear(token_dim, spacer_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        if pooling_strategy == "attention":
            self.attention_weights = nn.Linear(token_dim, 1)
        elif pooling_strategy == "learnable":
            self.pool_vector = nn.Parameter(torch.randn(token_dim))

    def forward(self, token_batch: Any, token_mask: Any = None) -> Any:
        """Encode token sequences to embeddings.
        
        Args:
            token_batch: Tensor of shape (batch_size, seq_length) with token IDs.
            token_mask: Optional binary mask (1=real, 0=padding).
            
        Returns:
            torch.Tensor: Shape (batch_size, spacer_dim) fixed-size embeddings.
        """
        if token_batch.shape[1] == 0:
            batch_size = token_batch.shape[0]
            output_dim = self.projection[0].out_features
            return torch.zeros((batch_size, output_dim), device=token_batch.device, dtype=self.embedding.weight.dtype)

        embedded = self.embedding(token_batch)  # (batch_size, seq_len, token_dim)
        
        if self.pooling_strategy == "mean":
            if token_mask is None:
                pooled = embedded.mean(dim=1)
            else:
                weights = token_mask.unsqueeze(-1).float()
                summed = (embedded * weights).sum(dim=1)
                denom = weights.sum(dim=1).clamp_min(1.0)
                pooled = summed / denom
        
        elif self.pooling_strategy == "max":
            if token_mask is not None:
                embedded = embedded.masked_fill(~token_mask.unsqueeze(-1).bool(), float('-inf'))
            pooled = embedded.max(dim=1)[0]
            pooled = torch.nan_to_num(
                pooled,
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )  # Handle all-padding case
        
        elif self.pooling_strategy == "attention":
            attn_scores = self.attention_weights(embedded).squeeze(-1)  # (batch_size, seq_len)
            if token_mask is not None:
                attn_scores = attn_scores.masked_fill(~token_mask.bool(), float('-inf'))
            attn_weights = F.softmax(attn_scores, dim=1)  # (batch_size, seq_len)
            attn_weights = torch.nan_to_num(attn_weights, nan=0.0)  # Handle all-padding case
            pooled = (embedded * attn_weights.unsqueeze(-1)).sum(dim=1)  # (batch_size, token_dim)
        
        elif self.pooling_strategy == "learnable":
            # Dot product with learnable vector, then weighted sum
            scores = torch.matmul(embedded, self.pool_vector)  # (batch_size, seq_len)
            if token_mask is not None:
                scores = scores.masked_fill(~token_mask.bool(), float('-inf'))
            weights = F.softmax(scores, dim=1)  # (batch_size, seq_len)
            weights = torch.nan_to_num(weights, nan=0.0)  # Handle all-padding case
            pooled = (embedded * weights.unsqueeze(-1)).sum(dim=1)  # (batch_size, token_dim)
        
        return self.projection(pooled)


# Backward-compatibility alias
class MeanPoolSequenceEncoder(SequenceEncoderBase):
    """Legacy alias for mean pooling. Use SequenceEncoderBase with pooling_strategy='mean' instead."""
    def __init__(self, vocab_size: int, token_dim: int, spacer_dim: int, dropout: float = 0.1):
        super().__init__(vocab_size, token_dim, spacer_dim, dropout, pooling_strategy="mean")


def _alibi_slopes(num_heads: int) -> Any:
    """Build head-specific slopes for ALiBi-style relative bias."""
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
        slopes.extend(_slopes_power_of_two(2 * closest_power_of_two)[0::2][: num_heads - closest_power_of_two])

    return torch.tensor(slopes, dtype=torch.float32)


def _apply_rotary_embedding(x: Any, positions: Any) -> Any:
    """Apply rotary position embeddings to a projected attention tensor."""
    rotary_dim = x.shape[-1] - (x.shape[-1] % 2)
    if rotary_dim <= 0:
        return x

    x_rotary = x[..., :rotary_dim]
    x_pass = x[..., rotary_dim:]
    half_dim = rotary_dim // 2
    inv_freq = 1.0 / (10000 ** (torch.arange(0, half_dim, device=x.device, dtype=x.dtype) / max(half_dim, 1)))
    angles = positions.to(dtype=x.dtype).unsqueeze(-1) * inv_freq.unsqueeze(0)
    cos = angles.cos().unsqueeze(0).unsqueeze(0)
    sin = angles.sin().unsqueeze(0).unsqueeze(0)

    x_even = x_rotary[..., ::2]
    x_odd = x_rotary[..., 1::2]
    rotated = torch.stack((x_even * cos - x_odd * sin, x_even * sin + x_odd * cos), dim=-1).flatten(-2)
    if x_pass.shape[-1] == 0:
        return rotated
    return torch.cat([rotated, x_pass], dim=-1)


class RelativePositionSelfAttention(nn.Module if nn is not None else object):
    """Multi-head self-attention with optional ALiBi or RoPE support."""

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1, positional_encoding: str = "alibi"):
        _require_torch()
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.positional_encoding = positional_encoding
        self.qkv_projection = nn.Linear(d_model, d_model * 3)
        self.output_projection = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        if positional_encoding == "alibi":
            self.register_buffer("alibi_slopes", _alibi_slopes(num_heads), persistent=False)
        else:
            self.register_buffer("alibi_slopes", None, persistent=False)

    def forward(self, x: Any, key_padding_mask: Any = None) -> Any:
        batch_size, seq_len, _ = x.shape
        qkv = self.qkv_projection(x)
        query, key, value = qkv.chunk(3, dim=-1)
        query = query.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        key = key.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        if self.positional_encoding == "rope":
            positions = torch.arange(seq_len, device=x.device)
            query = _apply_rotary_embedding(query, positions)
            key = _apply_rotary_embedding(key, positions)

        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if self.positional_encoding == "alibi":
            positions = torch.arange(seq_len, device=x.device)
            relative_positions = positions.unsqueeze(0) - positions.unsqueeze(1)
            if seq_len > 1:
                relative_positions = relative_positions.to(dtype=x.dtype) / float(seq_len - 1)
            else:
                relative_positions = relative_positions.to(dtype=x.dtype)
            bias = self.alibi_slopes.to(dtype=x.dtype).view(1, self.num_heads, 1, 1) * relative_positions.view(1, 1, seq_len, seq_len)
            scores = scores + bias

        if key_padding_mask is not None:
            mask = key_padding_mask.view(batch_size, 1, 1, seq_len)
            scores = scores.masked_fill(mask, torch.finfo(scores.dtype).min)

        attention_weights = torch.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        attended = torch.matmul(attention_weights, value)
        attended = attended.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        return self.output_projection(attended)


class RelativePositionTransformerEncoderLayer(nn.Module if nn is not None else object):
    """Transformer encoder layer that uses a relative-position attention stack."""

    def __init__(self, d_model: int, num_heads: int, dim_feedforward: int, dropout: float = 0.1, activation: str = "gelu", positional_encoding: str = "alibi"):
        _require_torch()
        super().__init__()
        self.self_attn = RelativePositionSelfAttention(d_model, num_heads, dropout=dropout, positional_encoding=positional_encoding)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = F.gelu if activation == "gelu" else F.relu

    def forward(self, x: Any, key_padding_mask: Any = None) -> Any:
        x = self.norm1(x + self.dropout1(self.self_attn(x, key_padding_mask=key_padding_mask)))
        x = self.norm2(x + self.dropout2(self.linear2(self.dropout(self.activation(self.linear1(x))))))
        if key_padding_mask is not None:
            x = x.masked_fill(key_padding_mask.unsqueeze(-1), 0.0)
        return x


class SpacerDirectionTransformer(nn.Module if nn is not None else object):
    """Hierarchical transformer model for CRISPR array direction prediction.
    
    Encodes individual spacers, applies the requested positional encoding to
    preserve array order, processes through a multi-head transformer encoder,
    and outputs a binary direction classification (logits) (Forward/Reverse)
    per array.
    """
    def __init__(self, config: DirectionTransformerConfig):
        """Initialize the transformer model.
        
        Args:
            config: DirectionTransformerConfig with architecture parameters.
        """
        _require_torch()
        super().__init__()
        self.config = config
        self.include_flanks = config.include_flanks
        self.positional_encoding = config.positional_encoding.lower()
        if self.positional_encoding not in {"absolute", "alibi", "rope"}:
            raise ValueError("positional_encoding must be one of: absolute, alibi, rope")
        self.sequence_encoder = SequenceEncoderBase(
            vocab_size=config.vocab_size,
            token_dim=config.token_dim,
            spacer_dim=config.spacer_dim,
            dropout=config.dropout,
            pooling_strategy=config.pooling_strategy,
        )
        self.flank_encoder = SequenceEncoderBase(
            vocab_size=config.vocab_size,
            token_dim=config.token_dim,
            spacer_dim=config.spacer_dim,
            dropout=config.dropout,
            pooling_strategy=config.pooling_strategy,
        )
        self.spacer_projection = nn.Linear(config.spacer_dim, config.transformer_dim)
        feedforward_dim = config.feedforward_dim if config.feedforward_dim is not None else config.transformer_dim * 4
        self.use_absolute_positional_encoding = self.positional_encoding == "absolute"
        self.spacer_position_embedding = nn.Embedding(config.max_spacers, config.transformer_dim) if self.use_absolute_positional_encoding else None
        if self.use_absolute_positional_encoding:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=config.transformer_dim,
                nhead=config.num_heads,
                dim_feedforward=feedforward_dim,
                dropout=config.dropout,
                batch_first=True,
                activation=config.activation,
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=config.num_layers)
            self.relative_transformer_layers = None
        else:
            self.transformer = None
            self.relative_transformer_layers = nn.ModuleList(
                [
                    RelativePositionTransformerEncoderLayer(
                        d_model=config.transformer_dim,
                        num_heads=config.num_heads,
                        dim_feedforward=feedforward_dim,
                        dropout=config.dropout,
                        activation=config.activation,
                        positional_encoding=self.positional_encoding,
                    )
                    for _ in range(config.num_layers)
                ]
            )
        self.flank_projection = nn.Linear(config.spacer_dim * 2, config.transformer_dim) if config.include_flanks else None
        self.layer_norm = nn.LayerNorm(config.transformer_dim)
        self.dropout = nn.Dropout(config.dropout)
        act_layer = nn.GELU() if config.activation == "gelu" else nn.ReLU()
        self.classifier = nn.Sequential(
            nn.Linear(config.transformer_dim, config.transformer_dim),
            act_layer,
            nn.Dropout(config.dropout),
            nn.Linear(config.transformer_dim, 1),
        )

    def forward(self, batch: dict[str, Any]) -> Any:
        """Predict direction logits for a batch of arrays.
        
        Args:
            batch: Dict with spacer_tokens (batch, max_spacers, max_spacer_len)
                and spacer_mask (batch, max_spacers, binary).
                
        Returns:
            torch.Tensor: Shape (batch_size,) with direction logits.
                Use BCEWithLogitsLoss for training with binary labels.
        """
        spacer_tokens = batch["spacer_tokens"]
        spacer_mask = batch["spacer_mask"]
        batch_size, max_spacers, _ = spacer_tokens.shape

        flat_tokens = spacer_tokens.view(batch_size * max_spacers, -1)
        flat_mask = flat_tokens.ne(0)
        spacer_embeddings = self.sequence_encoder(flat_tokens, flat_mask)
        spacer_embeddings = self.spacer_projection(spacer_embeddings)
        spacer_embeddings = spacer_embeddings.view(batch_size, max_spacers, -1)

        if self.include_flanks:
            left_flank_tokens = batch.get("left_flank_tokens")
            right_flank_tokens = batch.get("right_flank_tokens")
            if left_flank_tokens is None:
                left_flank_tokens = torch.zeros((batch_size, 0), dtype=spacer_tokens.dtype, device=spacer_tokens.device)
            if right_flank_tokens is None:
                right_flank_tokens = torch.zeros((batch_size, 0), dtype=spacer_tokens.dtype, device=spacer_tokens.device)

            left_mask = left_flank_tokens.ne(0)
            right_mask = right_flank_tokens.ne(0)
            left_embed = self.flank_encoder(left_flank_tokens, left_mask)
            right_embed = self.flank_encoder(right_flank_tokens, right_mask)
            flank_context = torch.cat([left_embed, right_embed], dim=-1)
            flank_context = self.flank_projection(flank_context).unsqueeze(1)
            spacer_embeddings = spacer_embeddings + flank_context

        if self.use_absolute_positional_encoding:
            positions = torch.arange(max_spacers, device=spacer_embeddings.device).unsqueeze(0).expand(batch_size, -1)
            spacer_embeddings = spacer_embeddings + self.spacer_position_embedding(positions)
        spacer_embeddings = self.layer_norm(spacer_embeddings)

        key_padding_mask = spacer_mask.eq(0)
        if self.use_absolute_positional_encoding:
            transformed = self.transformer(spacer_embeddings, src_key_padding_mask=key_padding_mask)
        else:
            transformed = spacer_embeddings
            for layer in self.relative_transformer_layers:
                transformed = layer(transformed, key_padding_mask=key_padding_mask)

        mask = spacer_mask.unsqueeze(-1).float()
        pooled = (transformed * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled).squeeze(-1)
        return logits


def build_model(
    vocab_size: int,
    include_flanks: bool = False,
    max_spacers: int = 64,
    dropout: float = 0.1,
    token_dim: int = 64,
    spacer_dim: int = 128,
    transformer_dim: int = 128,
    num_heads: int = 4,
    num_layers: int = 2,
    feedforward_dim: int | None = None,
    activation: str = "gelu",
    positional_encoding: str = "absolute",
    pooling_strategy: str = "mean",
) -> SpacerDirectionTransformer:
    """Instantiate a SpacerDirectionTransformer with default configuration.
    
    Creates a pre-configured transformer model suitable for binary direction
    classification on CRISPR arrays.
    
    Args:
        vocab_size: Token vocabulary size (should match dataset vocab).
        include_flanks: If True, configure model to accept flank sequences.
        max_spacers: Maximum number of spacers to support in embeddings.
        dropout: Dropout rate for regularization (default 0.1).
        token_dim: DNA token embedding dimension.
        spacer_dim: Projected spacer embedding dimension.
        transformer_dim: Transformer hidden dimension.
        num_heads: Number of attention heads.
        num_layers: Number of transformer encoder layers.
        feedforward_dim: Feedforward hidden dimension (default 4x transformer_dim).
        activation: Activation function for feedforward (default 'gelu').
        positional_encoding: Positional encoding strategy ('absolute', 'alibi', 'rope'; default 'absolute').
        pooling_strategy: Sequence pooling strategy ('mean', 'max', 'attention', 'learnable'; default 'mean').
        feedforward_dim: Transformer feedforward hidden dimension. If None,
            defaults to four times transformer_dim.
        activation: Transformer activation function (gelu or relu).
        positional_encoding: Positional encoding mode (absolute, alibi, or rope).
        
    Returns:
        SpacerDirectionTransformer: Initialized model ready for training.
        
    Raises:
        ModuleNotFoundError: If PyTorch is not installed.
    """
    _require_torch()
    config = DirectionTransformerConfig(
        vocab_size=vocab_size,
        token_dim=token_dim,
        spacer_dim=spacer_dim,
        transformer_dim=transformer_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        feedforward_dim=feedforward_dim,
        activation=activation,
        dropout=dropout,
        max_spacers=max_spacers,
        include_flanks=include_flanks,
        positional_encoding=positional_encoding,
        pooling_strategy=pooling_strategy,
    )
    return SpacerDirectionTransformer(config)
