"""PyTorch data loading utilities for CRISPR array examples.

Provides dataset wrappers, collation functions, and dataloader builders
for efficient training and evaluation loops.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable

try:
    import torch
    from torch.utils.data import DataLoader, Dataset
except ModuleNotFoundError:
    torch = None
    DataLoader = object
    Dataset = object

from ..dataset import DirectionJsonlDataset, encode_example, collate_encoded_examples
from ..utils import _require_torch

if TYPE_CHECKING:
    pass


class DirectionTorchDataset(Dataset if Dataset is not object else object):
    """PyTorch Dataset wrapper for indexed access to encoded CRISPR examples.
    
    Wraps a DirectionJsonlDataset and provides lazy encoding on-demand during
    iteration, allowing efficient memory usage with large datasets.
    """
    def __init__(
        self,
        base_dataset: DirectionJsonlDataset,
        indices: list[int],
        vocab: dict[str, int],
        augment_fn: Callable | None = None,
    ):
        """Initialize the PyTorch dataset.
        
        Args:
            base_dataset: Source DirectionJsonlDataset to wrap.
            indices: List of indices to use from base_dataset.
            vocab: Token vocabulary for encoding sequences.
            augment_fn: Optional function that takes DirectionExample and returns
                        augmented DirectionExample (for on-the-fly augmentation).
        """
        _require_torch()
        self.base_dataset = base_dataset
        self.indices = indices
        self.vocab = vocab
        self.augment_fn = augment_fn

    def __len__(self) -> int:
        """Return number of examples in this split."""
        return len(self.indices)

    def __getitem__(self, index: int) -> dict:
        """Get and encode a single example by index.

        Applies `augment_fn` (if present) before encoding so augmentation
        is applied only at data-loading time.
        """
        example = self.base_dataset[self.indices[index]]
        if self.augment_fn is not None:
            try:
                aug = self.augment_fn(example)
                # If augment_fn returns None or something falsy, fall back to original
                if aug:
                    example = aug
            except Exception:
                # Any augmentation error should not crash data loading; fall back.
                pass
        return encode_example(
            example,
            vocab=self.vocab,
            include_flanks=self.base_dataset.include_flanks,
        )


def batch_to_tensors(batch: dict[str, list]) -> dict[str, Any]:
    """Convert collated batch lists to PyTorch tensors.
    
    Takes output from collate_encoded_examples (lists of arrays) and
    converts to GPU-ready torch tensors with appropriate dtypes.
    
    Args:
        batch: Dict with spacer_tokens (3D list), spacer_mask (2D list),
            repeat_tokens (3D list), label (list).
            
    Returns:
        dict[str, torch.Tensor]: Same keys with tensor values.
    """
    _require_torch()
    return {
        "spacer_tokens": torch.tensor(batch["spacer_tokens"], dtype=torch.long),
        "spacer_mask": torch.tensor(batch["spacer_mask"], dtype=torch.bool),
        "repeat_tokens": torch.tensor(batch["repeat_tokens"], dtype=torch.long),
        "label": torch.tensor(batch["label"], dtype=torch.float32),
    }


def collate_for_training(batch: list[dict]) -> dict[str, Any]:
    """Collate and tensorize a batch for training.
    
    Composition of collate_encoded_examples and batch_to_tensors,
    ready to pass to the model forward() method.
    """
    return batch_to_tensors(collate_encoded_examples(batch))


def build_dataloader(
    dataset: DirectionTorchDataset,
    batch_size: int,
    shuffle: bool,
    weights: list[float] | None = None,
) -> Any:
    """Create a PyTorch DataLoader for a split of data.
    
    Args:
        dataset: DirectionTorchDataset for this split.
        batch_size: Number of examples per batch.
        shuffle: If True, shuffle examples during iteration.
        weights: Optional per-sample weights for WeightedRandomSampler.
        
    Returns:
        torch.utils.data.DataLoader: Ready for training/evaluation loops.
    """
    _require_torch()
    if weights is not None:
        # Use WeightedRandomSampler when per-sample weights are provided. When a sampler
        # is used, DataLoader must not be shuffled (sampler defines sampling order).
        try:
            from torch.utils.data import WeightedRandomSampler

            tensor_weights = torch.tensor(weights, dtype=torch.double)
            sampler = WeightedRandomSampler(
                tensor_weights, num_samples=len(tensor_weights), replacement=True
            )
            return DataLoader(
                dataset, batch_size=batch_size, sampler=sampler, collate_fn=collate_for_training
            )
        except Exception:
            # Fall back to standard loader if sampler unavailable
            return DataLoader(
                dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_for_training
            )

    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_for_training
    )
