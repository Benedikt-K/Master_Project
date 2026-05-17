from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ModuleNotFoundError:
    torch = None
    nn = None
    F = None

from ..dataset import encode_dna_sequence
from ..tokenization import DNA_VOCAB


@dataclass
class CNNTokConfig:
    output_dim: int = 256
    embed_dim: int = 8
    filters: int = 128
    kernels: list[int] = (3, 7)
    pooling: str = "max"
    activation: str = "gelu"


class CNNTokenizer(nn.Module if nn is not None else object):
    """CNN-based tokenizer that produces per-spacer embeddings.

    Usage:
        tokenizer = CNNTokenizer(config, vocab_size=len(DNA_VOCAB))
        embeddings = tokenizer.encode_sequences(list_of_spacer_strings, vocab)
    """

    def __init__(self, config: Optional[CNNTokConfig] = None, vocab_size: int | None = None):
        if torch is None or nn is None:
            raise ModuleNotFoundError("PyTorch is required for CNN tokenizer")
        super().__init__()
        self.config = CNNTokConfig() if config is None else config
        self.vocab_size = vocab_size or len(DNA_VOCAB)
        self.embedding = nn.Embedding(self.vocab_size, self.config.embed_dim, padding_idx=DNA_VOCAB["PAD"])
        # Create conv layers for each kernel size
        self.convs = nn.ModuleList(
            [nn.Conv1d(self.config.embed_dim, self.config.filters, kernel_size=k, padding=k // 2) for k in self.config.kernels]
        )
        self.fc = nn.Linear(self.config.filters * len(self.config.kernels), self.config.output_dim)
        self.act = nn.ReLU() if self.config.activation == "relu" else nn.GELU()

    def forward(self, padded_input: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """Forward compute on padded integer sequences.

        Args:
            padded_input: LongTensor shape (batch, seq_len)
            lengths: LongTensor shape (batch,) with true lengths
        Returns:
            Tensor shape (batch, output_dim)
        """
        x = self.embedding(padded_input)  # (batch, seq_len, embed_dim)
        x = x.transpose(1, 2)  # (batch, embed_dim, seq_len)
        conv_outs = []
        for conv in self.convs:
            c = conv(x)
            c = self.act(c)
            if self.config.pooling == "max":
                pooled = F.adaptive_max_pool1d(c, 1).squeeze(-1)
            else:
                pooled = F.adaptive_avg_pool1d(c, 1).squeeze(-1)
            conv_outs.append(pooled)
        cat = torch.cat(conv_outs, dim=-1)
        out = self.fc(cat)
        return out

    def encode_sequences(self, sequences: List[str], vocab: dict[str, int]) -> List[List[float]]:
        """Encode a list of DNA sequences to per-sequence embeddings (CPU).

        This pads sequences to the same length, runs them through the module,
        and returns a list of Python lists (floats).
        """
        # Convert sequences to integer lists
        int_seqs = [encode_dna_sequence(seq, vocab) for seq in sequences]
        if not int_seqs:
            return []
        max_len = max(len(s) for s in int_seqs)
        padded = [s + [vocab["PAD"]] * (max_len - len(s)) for s in int_seqs]
        device = next(self.parameters()).device if any(p is not None for p in self.parameters()) else torch.device("cpu")
        inp = torch.tensor(padded, dtype=torch.long, device=device)
        lengths = torch.tensor([len(s) for s in int_seqs], dtype=torch.long, device=device)
        with torch.no_grad():
            out = self.forward(inp, lengths)
        return out.cpu().tolist()
