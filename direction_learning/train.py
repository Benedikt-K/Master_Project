from __future__ import annotations

import argparse
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import torch
    from torch.utils.data import DataLoader, Dataset
except ModuleNotFoundError:
    torch = None
    DataLoader = object
    Dataset = object

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    plt = None

from .dataset import (
    DirectionExample,
    DirectionJsonlDataset,
    build_vocab_from_jsonl,
    collate_encoded_examples,
    encode_example,
)
from direction_learning.model import SpacerDirectionTransformer, build_model


def _require_torch() -> None:
    """Raise error if PyTorch is not installed."""
    if torch is None:
        raise ModuleNotFoundError(
            "PyTorch is required to run the direction training entrypoint. Install torch first."
        )


def split_groups(examples: list[DirectionExample], seed: int = 13, train_fraction: float = 0.7, val_fraction: float = 0.15) -> dict[str, list[int]]:
    """Split examples by group for train/val/test (preserves group cohesion).
    
    Groups examples by group_name (e.g., genome cluster), randomly shuffles
    groups, then assigns groups to splits to preserve biological coherence.
    
    Args:
        examples: List of DirectionExample objects to split.
        seed: Random seed for reproducibility.
        train_fraction: Fraction of groups to assign to training (default 0.7).
        val_fraction: Fraction of groups to assign to validation (default 0.15).
            Remaining fraction goes to test.
            
    Returns:
        dict[str, list[int]]: Mapping of split names ("train", "val", "test")
            to lists of indices into the examples list.
            
    Raises:
        ValueError: If train_fraction + val_fraction >= 1.0.
    """
    if train_fraction + val_fraction >= 1.0:
        raise ValueError("train_fraction + val_fraction must be smaller than 1.0")

    grouped_indices: dict[str, list[int]] = {}
    for index, example in enumerate(examples):
        grouped_indices.setdefault(example.group_name, []).append(index)

    groups = list(grouped_indices)
    rng = random.Random(seed)
    rng.shuffle(groups)

    n_groups = len(groups)
    n_train = max(1, round(n_groups * train_fraction))
    n_val = max(1, round(n_groups * val_fraction))
    if n_train + n_val >= n_groups:
        n_val = max(1, min(n_val, n_groups - n_train - 1))

    train_groups = groups[:n_train]
    val_groups = groups[n_train:n_train + n_val]
    test_groups = groups[n_train + n_val:]

    return {
        "train": [index for group in train_groups for index in grouped_indices[group]],
        "val": [index for group in val_groups for index in grouped_indices[group]],
        "test": [index for group in test_groups for index in grouped_indices[group]],
    }


def stratified_split_by_cas_subtype(
    examples: list[DirectionExample],
    seed: int = 13,
    train_fraction: float = 0.8,
    test_fraction: float = 0.1,
) -> dict[str, list[int]]:
    """Stratified 80/10/10 split by CRISPR cas_subtype for balanced cross-validation.
    
    Ensures each CRISPR subtype (e.g., I-F, I-E, I-C) is represented proportionally
    in train, validation, and test splits. Uses random stratified sampling to
    maintain label balance and CRISPR diversity across splits.
    
    Args:
        examples: List of DirectionExample objects with cas_subtype metadata.
        seed: Random seed for reproducibility (default 13).
        train_fraction: Fraction for training set (default 0.8, leaving 0.2 for val+test).
        test_fraction: Fraction for test set from remainder (default 0.1 of all).
            Validation gets 1 - train_fraction - test_fraction.
            
    Returns:
        dict[str, list[int]]: Split indices with keys "train", "val", "test".
        
    Example:
        >>> splits = stratified_split_by_cas_subtype(examples, seed=42)
        >>> train_indices = splits["train"]
        >>> val_indices = splits["val"]
        >>> test_indices = splits["test"]
    """
    if train_fraction + test_fraction > 1.0:
        raise ValueError("train_fraction + test_fraction must be <= 1.0")
    
    rng = random.Random(seed)
    
    # Group indices by cas_subtype
    subtype_indices: dict[str, list[int]] = {}
    for index, example in enumerate(examples):
        subtype = example.cas_subtype or "Unknown"
        subtype_indices.setdefault(subtype, []).append(index)
    
    # For each subtype, stratified split into train/val/test
    train_indices: list[int] = []
    val_indices: list[int] = []
    test_indices: list[int] = []
    
    for subtype, indices in subtype_indices.items():
        shuffled = list(indices)
        rng.shuffle(shuffled)
        
        n_total = len(shuffled)
        n_train = max(1, round(n_total * train_fraction))
        n_test = max(0, round(n_total * test_fraction))
        n_val = n_total - n_train - n_test
        
        train_indices.extend(shuffled[:n_train])
        test_indices.extend(shuffled[n_train:n_train + n_test])
        val_indices.extend(shuffled[n_train + n_test:])
    
    return {
        "train": train_indices,
        "val": val_indices,
        "test": test_indices,
    }


def stratified_train_test_by_cas_subtype(
    examples: list[DirectionExample],
    seed: int = 13,
    train_fraction: float = 0.8,
) -> dict[str, list[int]]:
    """Stratified train/test split by cas_subtype (default 80/20).

    Ensures each CRISPR subtype is split so that approximately `train_fraction`
    of examples from each subtype go to train and the remainder to test.

    Args:
        examples: List of DirectionExample objects with `cas_subtype` set.
        seed: Random seed for reproducibility.
        train_fraction: Fraction of examples per subtype to assign to train.

    Returns:
        dict[str, list[int]]: Mapping with keys "train" and "test".
    """
    if not (0.0 < train_fraction < 1.0):
        raise ValueError("train_fraction must be between 0 and 1")

    rng = random.Random(seed)
    subtype_indices: dict[str, list[int]] = {}
    for index, example in enumerate(examples):
        subtype = example.cas_subtype or "Unknown"
        subtype_indices.setdefault(subtype, []).append(index)

    train_indices: list[int] = []
    test_indices: list[int] = []
    for subtype, indices in subtype_indices.items():
        shuffled = list(indices)
        rng.shuffle(shuffled)
        n_total = len(shuffled)
        n_train = max(1, round(n_total * train_fraction)) if n_total > 1 else n_total
        train_indices.extend(shuffled[:n_train])
        test_indices.extend(shuffled[n_train:])

    return {"train": train_indices, "test": test_indices}


def stratified_train_test_and_val_by_cas_subtype(
    examples: list[DirectionExample], seed: int = 13, train_test_fraction: float = 0.8
) -> dict[str, list[int]]:
    """Split examples into a stratified (by cas_subtype) train+test group and validation.

    This creates a two-way split where approximately `train_test_fraction` of
    each subtype is assigned to the combined train+test set, and the remainder
    (1 - train_test_fraction) is used for validation.

    Args:
        examples: List of DirectionExample objects with `cas_subtype` set.
        seed: Random seed for reproducibility.
        train_test_fraction: Fraction of each subtype to keep for train+test.

    Returns:
        dict with keys `train_test` and `val` mapping to lists of indices.
    """
    if not (0.0 < train_test_fraction < 1.0):
        raise ValueError("train_test_fraction must be between 0 and 1")

    rng = random.Random(seed)

    # Build connected components so examples sharing a group_name OR identical
    # spacer/repeat signature stay in the same split, reducing leakage.
    n = len(examples)
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra = find(a)
        rb = find(b)
        if ra != rb:
            parent[rb] = ra

    first_by_group: dict[str, int] = {}
    first_by_signature: dict[tuple[tuple[str, ...], tuple[str, ...]], int] = {}
    for idx, example in enumerate(examples):
        group = example.group_name.strip()
        if group:
            if group in first_by_group:
                union(idx, first_by_group[group])
            else:
                first_by_group[group] = idx

        signature = (tuple(example.spacers), tuple(example.repeats))
        if signature in first_by_signature:
            union(idx, first_by_signature[signature])
        else:
            first_by_signature[signature] = idx

    components: dict[int, list[int]] = {}
    for idx in range(n):
        components.setdefault(find(idx), []).append(idx)

    # Stratify by subtype at component level (majority subtype in each component).
    subtype_components: dict[str, list[list[int]]] = {}
    for comp in components.values():
        subtype_counts = Counter((examples[i].cas_subtype or "Unknown") for i in comp)
        subtype = max(subtype_counts, key=subtype_counts.get)
        subtype_components.setdefault(subtype, []).append(comp)

    train_test_indices: list[int] = []
    val_indices: list[int] = []
    for subtype, comp_list in subtype_components.items():
        shuffled_components = list(comp_list)
        rng.shuffle(shuffled_components)
        n_components = len(shuffled_components)
        if n_components > 1:
            n_train_test_components = min(n_components - 1, max(1, round(n_components * train_test_fraction)))
        else:
            n_train_test_components = n_components

        train_components = shuffled_components[:n_train_test_components]
        val_components = shuffled_components[n_train_test_components:]
        for comp in train_components:
            train_test_indices.extend(comp)
        for comp in val_components:
            val_indices.extend(comp)

    return {"train_test": train_test_indices, "val": val_indices}


def stratified_train_test_and_val_by_label(
    examples: list[DirectionExample], seed: int = 13, train_test_fraction: float = 0.8
) -> dict[str, list[int]]:
    """Split examples stratified by label (Forward/Reverse) into train+test and validation.
    
    When cas_subtype is empty/unavailable, stratifies by binary label instead.
    Ensures both train and validation sets have balanced Forward/Reverse proportions.
    
    Args:
        examples: List of DirectionExample objects.
        seed: Random seed for reproducibility.
        train_test_fraction: Fraction of each label class to keep for train+test.
        
    Returns:
        dict with keys `train_test` and `val` mapping to lists of indices.
    """
    if not (0.0 < train_test_fraction < 1.0):
        raise ValueError("train_test_fraction must be between 0 and 1")
    
    rng = random.Random(seed)
    
    # Group indices by label
    label_indices: dict[int, list[int]] = {}
    for idx, example in enumerate(examples):
        label = example.label
        if label not in label_indices:
            label_indices[label] = []
        label_indices[label].append(idx)
    
    # Shuffle each label group independently
    for label in label_indices:
        rng.shuffle(label_indices[label])
    
    # Split each label group
    train_test_indices: list[int] = []
    val_indices: list[int] = []
    
    for label in sorted(label_indices.keys()):
        indices = label_indices[label]
        n = len(indices)
        n_train_test = max(1, round(n * train_test_fraction))
        
        train_test_indices.extend(indices[:n_train_test])
        val_indices.extend(indices[n_train_test:])
    
    return {"train_test": train_test_indices, "val": val_indices}


def stratified_train_test_and_val_by_cas_subtype_and_label(
    examples: list[DirectionExample], seed: int = 13, train_test_fraction: float = 0.8
) -> dict[str, list[int]]:
    """Split examples stratified by joint key (cas_subtype, label).

    This keeps both CRISPR subtype proportions and Forward/Reverse label
    proportions balanced across train+test vs validation.

    Args:
        examples: List of DirectionExample objects.
        seed: Random seed for reproducibility.
        train_test_fraction: Fraction of each stratum to keep for train+test.

    Returns:
        dict with keys `train_test` and `val` mapping to lists of indices.
    """
    if not (0.0 < train_test_fraction < 1.0):
        raise ValueError("train_test_fraction must be between 0 and 1")

    rng = random.Random(seed)

    strata_indices: dict[tuple[str, int], list[int]] = {}
    for idx, example in enumerate(examples):
        subtype = (example.cas_subtype or "Unknown").strip() or "Unknown"
        label = int(example.label)
        key = (subtype, label)
        strata_indices.setdefault(key, []).append(idx)

    train_test_indices: list[int] = []
    val_indices: list[int] = []
    for key in sorted(strata_indices.keys()):
        indices = list(strata_indices[key])
        rng.shuffle(indices)
        n = len(indices)
        if n == 1:
            n_train_test = 1
        else:
            n_train_test = min(n - 1, max(1, round(n * train_test_fraction)))
        train_test_indices.extend(indices[:n_train_test])
        val_indices.extend(indices[n_train_test:])

    return {"train_test": train_test_indices, "val": val_indices}


class DirectionTorchDataset(Dataset if Dataset is not object else object):
    """PyTorch Dataset wrapper for indexed access to encoded CRISPR examples.
    
    Wraps a DirectionJsonlDataset and provides lazy encoding on-demand during
    iteration, allowing efficient memory usage with large datasets.
    """
    def __init__(self, base_dataset: DirectionJsonlDataset, indices: list[int], vocab: dict[str, int]):
        """Initialize the PyTorch dataset.
        
        Args:
            base_dataset: Source DirectionJsonlDataset to wrap.
            indices: List of indices to use from base_dataset.
            vocab: Token vocabulary for encoding sequences.
        """
        _require_torch()
        self.base_dataset = base_dataset
        self.indices = indices
        self.vocab = vocab

    def __len__(self) -> int:
        """Return number of examples in this split."""
        return len(self.indices)

    def __getitem__(self, index: int) -> dict:
        """Get and encode a single example by index."""
        example = self.base_dataset[self.indices[index]]
        return encode_example(example, vocab=self.vocab, include_flanks=self.base_dataset.include_flanks)


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


def build_dataloader(dataset: DirectionTorchDataset, batch_size: int, shuffle: bool) -> Any:
    """Create a PyTorch DataLoader for a split of data.
    
    Args:
        dataset: DirectionTorchDataset for this split.
        batch_size: Number of examples per batch.
        shuffle: If True, shuffle examples during iteration.
        
    Returns:
        torch.utils.data.DataLoader: Ready for training/evaluation loops.
    """
    _require_torch()
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_for_training)


def train_one_epoch(model: SpacerDirectionTransformer, loader: Any, optimizer: Any, loss_fn: Any, device: Any) -> float:
    """Train model for one epoch on all batches from loader.
    
    Args:
        model: SpacerDirectionTransformer to train (moves to device internally).
        loader: DataLoader with training batches.
        optimizer: torch.optim optimizer (e.g., AdamW).
        loss_fn: Loss function (e.g., BCEWithLogitsLoss).
        device: torch.device to run on (CPU or CUDA).
        
    Returns:
        float: Average loss across all batches.
    """
    _require_torch()
    model.train()
    total_loss = 0.0
    total_items = 0
    for batch in loader:
        batch = {key: value.to(device) for key, value in batch.items()}
        optimizer.zero_grad(set_to_none=True)
        logits = model(batch)
        loss = loss_fn(logits, batch["label"])
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item()) * batch["label"].shape[0]
        total_items += int(batch["label"].shape[0])
    return total_loss / max(total_items, 1)


def evaluate(model: SpacerDirectionTransformer, loader: Any, loss_fn: Any, device: Any) -> dict[str, float]:
    """Evaluate model on all batches without gradient updates.
    
    Args:
        model: SpacerDirectionTransformer to evaluate.
        loader: DataLoader with validation/test batches.
        loss_fn: Loss function matching training.
        device: torch.device to run on.
        
    Returns:
        dict[str, float]: Metrics with keys "loss" and "accuracy".
            - loss: Average loss across batches.
            - accuracy: Fraction of predictions matching true labels
              (using 0.5 threshold on sigmoid).
    """
    _require_torch()
    model.eval()
    total_loss = 0.0
    total_items = 0
    correct = 0
    all_probs: list = []
    all_labels: list = []
    with torch.no_grad():
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            logits = model(batch)
            loss = loss_fn(logits, batch["label"])
            probs = torch.sigmoid(logits)
            predictions = (probs >= 0.5).long()
            correct += int((predictions == batch["label"].long()).sum().item())
            total_loss += float(loss.item()) * batch["label"].shape[0]
            total_items += int(batch["label"].shape[0])
            all_probs.append(probs.cpu().numpy())
            all_labels.append(batch["label"].cpu().numpy())

    # Concatenate numpy arrays
    import numpy as np

    if len(all_labels) == 0:
        return {
            "loss": float("nan"),
            "accuracy": float("nan"),
            "auc": float("nan"),
            "aupr": float("nan"),
            "precision": float("nan"),
            "recall": float("nan"),
            "f1": float("nan"),
        }

    y_prob = np.concatenate(all_probs)
    y_true = np.concatenate(all_labels)

    accuracy = correct / max(total_items, 1)
    loss_val = total_loss / max(total_items, 1)
    y_pred_bin = (y_prob >= 0.5).astype(int)
    positive_rate_true = float(y_true.mean()) if y_true.size > 0 else float("nan")
    positive_rate_pred = float(y_pred_bin.mean()) if y_pred_bin.size > 0 else float("nan")
    majority_baseline_accuracy = max(positive_rate_true, 1.0 - positive_rate_true)

    # Compute other metrics using sklearn if available, otherwise fallback to simple computations
    try:
        from sklearn.metrics import (
            roc_auc_score,
            average_precision_score,
            precision_score,
            recall_score,
            f1_score,
        )

        auc = float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else float("nan")
        aupr = float(average_precision_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else float("nan")
        precision = float(precision_score(y_true, y_pred_bin, zero_division=0))
        recall = float(recall_score(y_true, y_pred_bin, zero_division=0))
        f1 = float(f1_score(y_true, y_pred_bin, zero_division=0))
    except Exception:
        # Minimal safe fallbacks
        tp = int(((y_pred_bin == 1) & (y_true == 1)).sum())
        fp = int(((y_pred_bin == 1) & (y_true == 0)).sum())
        fn = int(((y_pred_bin == 0) & (y_true == 1)).sum())
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        auc = float("nan")
        aupr = float("nan")

    return {
        "loss": loss_val,
        "accuracy": accuracy,
        "auc": auc,
        "aupr": aupr,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "positive_rate_true": positive_rate_true,
        "positive_rate_pred": positive_rate_pred,
        "majority_baseline_accuracy": majority_baseline_accuracy,
    }


def evaluate_per_subtype(
    model: SpacerDirectionTransformer,
    base_dataset: DirectionJsonlDataset,
    indices: list[int],
    vocab: dict[str, int],
    loss_fn: Any,
    device: Any,
    batch_size: int,
) -> dict[str, dict[str, float]]:
    """Evaluate test performance separately for each cas_subtype.

    Args:
        model: Trained model to evaluate.
        base_dataset: Dataset containing original records and subtype metadata.
        indices: Dataset indices to evaluate (usually test split).
        vocab: Token vocabulary used during training.
        loss_fn: Loss function (unused for subtype metrics but kept for API parity).
        device: torch.device to run on.
        batch_size: Batch size for forward passes.

    Returns:
        Mapping cas_subtype -> metrics dict.
    """
    _require_torch()
    del loss_fn

    import numpy as np

    model.eval()
    probs_by_subtype: dict[str, list[float]] = {}
    labels_by_subtype: dict[str, list[float]] = {}

    with torch.no_grad():
        for start in range(0, len(indices), batch_size):
            batch_indices = indices[start:start + batch_size]
            encoded_batch = [
                encode_example(base_dataset[i], vocab=vocab, include_flanks=base_dataset.include_flanks)
                for i in batch_indices
            ]
            collated = collate_encoded_examples(encoded_batch)
            tensor_batch = batch_to_tensors(collated)
            tensor_batch = {key: value.to(device) for key, value in tensor_batch.items()}

            logits = model(tensor_batch)
            probs = torch.sigmoid(logits).cpu().numpy()
            labels = tensor_batch["label"].cpu().numpy()

            for dataset_idx, prob, label in zip(batch_indices, probs, labels):
                subtype = (base_dataset[dataset_idx].cas_subtype or "Unknown").strip() or "Unknown"
                probs_by_subtype.setdefault(subtype, []).append(float(prob))
                labels_by_subtype.setdefault(subtype, []).append(float(label))

    metrics_by_subtype: dict[str, dict[str, float]] = {}
    for subtype in sorted(probs_by_subtype.keys()):
        y_prob = np.array(probs_by_subtype[subtype], dtype=float)
        y_true = np.array(labels_by_subtype[subtype], dtype=float)
        y_pred_bin = (y_prob >= 0.5).astype(int)

        accuracy = float((y_pred_bin == y_true.astype(int)).mean()) if y_true.size > 0 else float("nan")
        positive_rate_true = float(y_true.mean()) if y_true.size > 0 else float("nan")
        majority_baseline_accuracy = max(positive_rate_true, 1.0 - positive_rate_true)

        try:
            from sklearn.metrics import (
                average_precision_score,
                f1_score,
                precision_score,
                recall_score,
                roc_auc_score,
            )

            auc = float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else float("nan")
            aupr = float(average_precision_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else float("nan")
            precision = float(precision_score(y_true, y_pred_bin, zero_division=0))
            recall = float(recall_score(y_true, y_pred_bin, zero_division=0))
            f1 = float(f1_score(y_true, y_pred_bin, zero_division=0))
        except Exception:
            tp = int(((y_pred_bin == 1) & (y_true == 1)).sum())
            fp = int(((y_pred_bin == 1) & (y_true == 0)).sum())
            fn = int(((y_pred_bin == 0) & (y_true == 1)).sum())
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            auc = float("nan")
            aupr = float("nan")

        metrics_by_subtype[subtype] = {
            "n": float(len(y_true)),
            "accuracy": accuracy,
            "auc": auc,
            "aupr": aupr,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "positive_rate_true": positive_rate_true,
            "majority_baseline_accuracy": majority_baseline_accuracy,
        }

    return metrics_by_subtype


def main() -> int:
    """Train the transformer
    
    Loads agreed-only JSONL dataset, performs stratified split by CRISPR
    subtype to balance train/val/test distributions, then trains for specified
    epochs with validation monitoring. Implements early stopping and checkpoints 
    the best model by validation loss.
    """
    parser = argparse.ArgumentParser(description="Train a CRISPR direction transformer on the agreed-only JSONL dataset.")
    parser.add_argument("--jsonl", default="output_dataset/direction_training_dataset.jsonl")
    parser.add_argument("--include_flanks", action="store_true")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training (default 16).")
    parser.add_argument("--epochs", type=int, default=5, help="Maximum number of training epochs (default 5).")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate for AdamW optimizer (default 3e-4).")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="L2 regularization strength (default 1e-5).")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate for regularization (default 0.1).")
    parser.add_argument("--early_stopping_patience", type=int, default=3, help="Stop if val_loss doesn't improve for N epochs (default 3).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility (default 42).")
    parser.add_argument(
        "--stratify_by",
        type=str,
        default="label",
        choices=["label", "cas_subtype"],
        help="Stratification method: 'label' (balanced classes) or 'cas_subtype' (CRISPR type). Default: label (recommended).",
    )
    parser.add_argument(
        "--stratify_by_cas_subtype_and_label",
        action="store_true",
        help=(
            "Use combined stratification on both cas_subtype and label. "
            "Overrides --stratify_by when provided."
        ),
    )
    parser.add_argument(
        "--test_within_train_fraction",
        type=float,
        default=0.0,
        help="Optional fraction of the combined train+test split to hold out as a test set (e.g. 0.1 for 10%%).",
    )
    args = parser.parse_args()

    _require_torch()

    base_dataset = DirectionJsonlDataset(args.jsonl, include_flanks=args.include_flanks)
    vocab = build_vocab_from_jsonl(args.jsonl)

    dataset_label_counts = Counter(example.label for example in base_dataset.records)
    if len(dataset_label_counts) < 2:
        raise ValueError(
            "Dataset contains only one class label. Training/validation accuracy is not informative. "
            f"Label counts: {dict(dataset_label_counts)}"
        )
    
    stratify_mode = (
        "cas_subtype_and_label" if args.stratify_by_cas_subtype_and_label else args.stratify_by
    )

    if stratify_mode == "label":
        splits = stratified_train_test_and_val_by_label(
            base_dataset.records, seed=args.seed, train_test_fraction=0.8
        )
        two_way_split = True
    elif stratify_mode == "cas_subtype":
        splits = stratified_split_by_cas_subtype(
            base_dataset.records, seed=args.seed, train_fraction=0.8, test_fraction=0.1
        )
        two_way_split = False
    else:
        splits = stratified_train_test_and_val_by_cas_subtype_and_label(
            base_dataset.records, seed=args.seed, train_test_fraction=0.8
        )
        two_way_split = True

    train_indices: list[int]
    test_indices: list[int]
    if not two_way_split:
        # splits contains keys 'train','val','test'
        train_indices = splits["train"]
        val_indices = splits["val"]
        test_indices = splits["test"]
    else:
        if args.test_within_train_fraction and 0.0 < args.test_within_train_fraction < 1.0:
            train_test_indices = splits["train_test"]
            train_test_examples = [base_dataset.records[i] for i in train_test_indices]
            inner_train_fraction = 1.0 - args.test_within_train_fraction

            if stratify_mode == "label":
                inner_splits = stratified_train_test_and_val_by_label(
                    train_test_examples, seed=args.seed, train_test_fraction=inner_train_fraction
                )
            else:
                inner_splits = stratified_train_test_and_val_by_cas_subtype_and_label(
                    train_test_examples, seed=args.seed, train_test_fraction=inner_train_fraction
                )
            train_indices = [train_test_indices[i] for i in inner_splits["train_test"]]
            test_indices = [train_test_indices[i] for i in inner_splits["val"]]
        else:
            train_indices = splits["train_test"]
            test_indices = []

        val_indices = splits["val"]

    train_dataset = DirectionTorchDataset(base_dataset, train_indices, vocab)
    val_dataset = DirectionTorchDataset(base_dataset, val_indices, vocab)
    test_dataset = DirectionTorchDataset(base_dataset, test_indices, vocab) if test_indices else None

    print(f"Split sizes: train={len(train_dataset)}, val={len(val_dataset)}, test={(len(test_dataset) if test_dataset else 0)}")

    train_label_counts = Counter(base_dataset.records[i].label for i in train_indices)
    val_label_counts = Counter(base_dataset.records[i].label for i in val_indices)
    print(f"Label distribution train={dict(train_label_counts)} val={dict(val_label_counts)}")
    if len(train_label_counts) < 2 or len(val_label_counts) < 2:
        raise ValueError(
            "Train/validation split contains only a single class."
            f"train={dict(train_label_counts)} val={dict(val_label_counts)}"
        )

    train_loader = build_dataloader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = build_dataloader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = build_dataloader(test_dataset, batch_size=args.batch_size, shuffle=False) if test_dataset else None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Determine maximum number of spacers in the dataset to size positional embeddings
    max_spacers_in_dataset = max((len(ex.spacers) for ex in base_dataset.records), default=64)
    model = build_model(vocab_size=len(vocab), include_flanks=args.include_flanks, max_spacers=max_spacers_in_dataset, dropout=args.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    # Training loop with early stopping.
    best_val_loss = float("inf")
    patience_counter = 0
    best_model_state = None
    train_losses = []
    val_losses = []
    
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        val_metrics = evaluate(model, val_loader, loss_fn, device)
        val_loss = val_metrics["loss"]
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        # if validation loss improves, checkpoint the model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = {k: v.cpu() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
        
        es_marker = " (BEST)" if patience_counter == 0 else (" (STOP)" if patience_counter >= args.early_stopping_patience else "")
        print(
            (
                "epoch={epoch} train_loss={train_loss:.4f} val_loss={val_loss:.4f}{es_marker} "
                "val_accuracy={val_accuracy:.4f} val_f1={val_f1:.4f} "
                "val_pos_rate={val_pos_rate:.4f} val_pred_pos_rate={val_pred_pos_rate:.4f} "
                "val_majority_baseline_acc={val_baseline:.4f}"
            ).format(
                epoch=epoch,
                train_loss=train_loss,
                val_loss=val_loss,
                es_marker=es_marker,
                val_accuracy=val_metrics["accuracy"],
                val_f1=val_metrics["f1"],
                val_pos_rate=val_metrics["positive_rate_true"],
                val_pred_pos_rate=val_metrics["positive_rate_pred"],
                val_baseline=val_metrics["majority_baseline_accuracy"],
            )
        )
        
        if patience_counter >= args.early_stopping_patience:
            print(f"Early stopping: validation loss did not improve for {args.early_stopping_patience} epochs.")
            break

    # Restore best model before final evaluation
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print("Restored best model (lowest validation loss).")
    
    # Plot training curves with parameters
    if plt is not None:
        fig, ax = plt.subplots(figsize=(12, 7))
        epochs_range = range(1, len(train_losses) + 1)
        ax.plot(epochs_range, train_losses, marker='o', label='Train Loss', linewidth=2)
        ax.plot(epochs_range, val_losses, marker='s', label='Val Loss', linewidth=2)
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Loss', fontsize=12)
        ax.set_title('Training vs Validation Loss', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        
        # param text box
        params_text = (
            f"batch_size={args.batch_size}\n"
            f"lr={args.lr}\n"
            f"weight_decay={args.weight_decay}\n"
            f"dropout={args.dropout}\n"
            f"early_stopping_patience={args.early_stopping_patience}\n"
            f"stratify_by={stratify_mode}\n"
            f"seed={args.seed}\n"
            f"train_size={len(train_dataset)}\n"
            f"val_size={len(val_dataset)}\n"
            f"epochs_completed={len(train_losses)}\n"
            f"best_val_loss={min(val_losses):.4f}"
        )
        ax.text(0.98, 0.97, params_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                family='monospace')
        
        output_path = Path("/tmp/training_curves.png")
        fig.savefig(str(output_path), dpi=100, bbox_inches='tight')
        print(f"Training curves saved to {output_path}")
        plt.close(fig)
    else:
        print("matplotlib not available; skipping training curves visualization.")
    
    if test_loader is not None:
        # print test metrics
        test_metrics = evaluate(model, test_loader, loss_fn, device)
        print(
            (
                "test_loss={loss:.4f} test_accuracy={accuracy:.4f} auc={auc} aupr={aupr} "
                "precision={precision:.4f} recall={recall:.4f} f1={f1:.4f}"
            ).format(
                loss=test_metrics["loss"],
                accuracy=test_metrics["accuracy"],
                auc=(f"{test_metrics['auc']:.4f}" if not (test_metrics['auc'] != test_metrics['auc']) else "nan"),
                aupr=(f"{test_metrics['aupr']:.4f}" if not (test_metrics['aupr'] != test_metrics['aupr']) else "nan"),
                precision=test_metrics["precision"],
                recall=test_metrics["recall"],
                f1=test_metrics["f1"],
            )
        )

        # print subtype test metrics
        per_subtype_metrics = evaluate_per_subtype(
            model=model,
            base_dataset=base_dataset,
            indices=test_indices,
            vocab=vocab,
            loss_fn=loss_fn,
            device=device,
            batch_size=args.batch_size,
        )
        print("Per-cas_subtype test metrics:")
        print(
            "{:<16} {:>7} {:>9} {:>9} {:>9} {:>9} {:>9} {:>9}".format(
                "cas_subtype", "n", "accuracy", "auc", "aupr", "precision", "recall", "f1"
            )
        )
        for subtype, metrics in per_subtype_metrics.items():
            auc_text = f"{metrics['auc']:.4f}" if not (metrics["auc"] != metrics["auc"]) else "nan"
            aupr_text = f"{metrics['aupr']:.4f}" if not (metrics["aupr"] != metrics["aupr"]) else "nan"
            print(
                "{:<16} {:>7} {:>9.4f} {:>9} {:>9} {:>9.4f} {:>9.4f} {:>9.4f}".format(
                    subtype,
                    int(metrics["n"]),
                    metrics["accuracy"],
                    auc_text,
                    aupr_text,
                    metrics["precision"],
                    metrics["recall"],
                    metrics["f1"],
                )
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
