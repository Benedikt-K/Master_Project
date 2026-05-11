"""Absolute positional encoding for transformer models.

Standard learnable absolute positional embeddings added to token embeddings.
"""

from __future__ import annotations

from typing import Any

try:
    import torch
    import torch.nn as nn
except ModuleNotFoundError:
    torch = None
    nn = None


def _require_torch() -> None:
    """Raise error if PyTorch is not installed."""
    if torch is None or nn is None:
        raise ModuleNotFoundError(
            "PyTorch is required for positional encodings. Install torch before use."
        )
