"""Training and evaluation module for CRISPR direction models.

Provides epoch training, model evaluation, and per-subtype performance analysis.
"""

from .loop import (
    evaluate,
    evaluate_per_subtype,
    train_one_epoch,
)

__all__ = [
    "train_one_epoch",
    "evaluate",
    "evaluate_per_subtype",
]
