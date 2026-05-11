"""Training and evaluation loops for CRISPR direction models.

Provides epoch training, validation/test evaluation, and per-subtype analysis.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ..dataset import DirectionJsonlDataset, collate_encoded_examples, encode_example
from ..data.loading import batch_to_tensors
from ..utils import _require_torch

try:
    import torch
except ModuleNotFoundError:
    torch = None

if TYPE_CHECKING:
    from ..model import SpacerDirectionTransformer


def train_one_epoch(
    model: SpacerDirectionTransformer,
    loader: Any,
    optimizer: Any,
    loss_fn: Any,
    device: Any,
) -> float:
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


def evaluate(
    model: SpacerDirectionTransformer,
    loader: Any,
    loss_fn: Any,
    device: Any,
) -> dict[str, float]:
    """Evaluate model on all batches without gradient updates.
    
    Args:
        model: SpacerDirectionTransformer to evaluate.
        loader: DataLoader with validation/test batches.
        loss_fn: Loss function matching training.
        device: torch.device to run on.
        
    Returns:
        dict[str, float]: Metrics with keys "loss", "accuracy", "auc", "aupr", "precision", "recall", "f1".
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
        Mapping cas_subtype -> metrics dict with keys:
        "n", "accuracy", "auc", "aupr", "precision", "recall", "f1",
        "positive_rate_true", "majority_baseline_accuracy".
    """
    _require_torch()
    del loss_fn

    import numpy as np

    model.eval()
    probs_by_subtype: dict[str, list[float]] = {}
    labels_by_subtype: dict[str, list[float]] = {}

    with torch.no_grad():
        for start in range(0, len(indices), batch_size):
            batch_indices = indices[start : start + batch_size]
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
