from __future__ import annotations

from dataclasses import dataclass

try:
    import torch
    import torch.nn as nn
except ModuleNotFoundError:
    torch = None
    nn = None


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
    num_layers: int = 2
    dropout: float = 0.1
    max_spacers: int = 64
    include_flanks: bool = False


def _require_torch() -> None:
    """Raise error if PyTorch is not installed."""
    if torch is None or nn is None:
        raise ModuleNotFoundError(
            "PyTorch is required to instantiate direction_learning.model classes. "
            "Install torch before training the transformer."
        )


class MeanPoolSequenceEncoder(nn.Module if nn is not None else object):
    """Encodes variable-length DNA sequences to fixed-size embeddings via mean pooling.
    
    Embeds individual bases and applies mean pooling (optionally masked) to
    produce a single embedding per sequence. Used as the base encoder for
    spacers, repeats, and flanks.
    """
    def __init__(self, vocab_size: int, token_dim: int, spacer_dim: int, dropout: float = 0.1):
        """Initialize the sequence encoder.
        
        Args:
            vocab_size: Size of the token vocabulary.
            token_dim: Dimension of token embeddings.
            spacer_dim: Output dimension after projection.
            dropout: Dropout rate for regularization.
        """
        _require_torch()
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, token_dim, padding_idx=0)
        self.projection = nn.Sequential(
            nn.Linear(token_dim, spacer_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, token_batch: torch.Tensor, token_mask: torch.Tensor | None = None) -> torch.Tensor:
        """Encode token sequences to embeddings.
        
        Args:
            token_batch: Tensor of shape (batch_size, seq_length) with token IDs.
            token_mask: Optional binary mask (1=real, 0=padding).
            
        Returns:
            torch.Tensor: Shape (batch_size, spacer_dim) fixed-size embeddings.
        """
        embedded = self.embedding(token_batch)
        if token_mask is None:
            pooled = embedded.mean(dim=1)
        else:
            weights = token_mask.unsqueeze(-1).float()
            summed = (embedded * weights).sum(dim=1)
            denom = weights.sum(dim=1).clamp_min(1.0)
            pooled = summed / denom
        return self.projection(pooled)


class SpacerDirectionTransformer(nn.Module if nn is not None else object):
    """Hierarchical transformer model for CRISPR array direction prediction.
    
    Encodes individual spacers/repeats, applies positional embeddings to preserve
    array order, processes through a multi-head transformer encoder, and outputs
    a binary direction logit (Forward/Reverse) per array.
    """
    def __init__(self, config: DirectionTransformerConfig):
        """Initialize the transformer model.
        
        Args:
            config: DirectionTransformerConfig with architecture parameters.
        """
        _require_torch()
        super().__init__()
        self.config = config
        self.sequence_encoder = MeanPoolSequenceEncoder(
            vocab_size=config.vocab_size,
            token_dim=config.token_dim,
            spacer_dim=config.spacer_dim,
            dropout=config.dropout,
        )
        self.spacer_position_embedding = nn.Embedding(config.max_spacers, config.transformer_dim)
        self.spacer_projection = nn.Linear(config.spacer_dim, config.transformer_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.transformer_dim,
            nhead=config.num_heads,
            dim_feedforward=config.transformer_dim * 4,
            dropout=config.dropout,
            batch_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=config.num_layers)
        self.layer_norm = nn.LayerNorm(config.transformer_dim)
        self.dropout = nn.Dropout(config.dropout)
        self.classifier = nn.Sequential(
            nn.Linear(config.transformer_dim, config.transformer_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.transformer_dim, 1),
        )

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
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

        positions = torch.arange(max_spacers, device=spacer_embeddings.device).unsqueeze(0).expand(batch_size, -1)
        spacer_embeddings = spacer_embeddings + self.spacer_position_embedding(positions)
        spacer_embeddings = self.layer_norm(spacer_embeddings)

        key_padding_mask = spacer_mask.eq(0)
        transformed = self.transformer(spacer_embeddings, src_key_padding_mask=key_padding_mask)

        mask = spacer_mask.unsqueeze(-1).float()
        pooled = (transformed * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled).squeeze(-1)
        return logits


def build_model(vocab_size: int, include_flanks: bool = False, max_spacers: int = 64, dropout: float = 0.1) -> SpacerDirectionTransformer:
    """Instantiate a SpacerDirectionTransformer with default configuration.
    
    Creates a pre-configured transformer model suitable for binary direction
    classification on CRISPR arrays.
    
    Args:
        vocab_size: Token vocabulary size (should match dataset vocab).
        include_flanks: If True, configure model to accept flank sequences.
        max_spacers: Maximum number of spacers to support in embeddings.
        dropout: Dropout rate for regularization (default 0.1).
        
    Returns:
        SpacerDirectionTransformer: Initialized model ready for training.
        
    Raises:
        ModuleNotFoundError: If PyTorch is not installed.
    """
    _require_torch()
    config = DirectionTransformerConfig(vocab_size=vocab_size, include_flanks=include_flanks, max_spacers=max_spacers, dropout=dropout)
    return SpacerDirectionTransformer(config)
