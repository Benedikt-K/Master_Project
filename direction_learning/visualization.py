"""Visualization helpers for direction_learning training and evaluation.

Centralizes plotting code (training curves, confusion matrix) so callers
can import and reuse the functions without embedding matplotlib code.
"""
from __future__ import annotations

from itertools import combinations
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Iterable

def _ensure_matplotlib():
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:  # pragma: no cover - optional dependency
        return None
    return plt


def _array_lengths(records: Sequence[object], indices: Sequence[int]) -> tuple[list[int], list[int]]:
    spacer_counts: list[int] = []
    bp_lengths: list[int] = []

    for index in indices:
        example = records[index]
        spacers = getattr(example, "spacers", []) or []
        repeats = getattr(example, "repeats", []) or []
        spacer_counts.append(len(spacers))
        bp_lengths.append(sum(len(sequence) for sequence in spacers) + sum(len(sequence) for sequence in repeats))

    return spacer_counts, bp_lengths


def _spacer_signature(example: object) -> set[str]:
    spacers = getattr(example, "spacers", []) or []
    return set(spacers)


def _jaccard_similarity(left: set[str], right: set[str]) -> float:
    union = left | right
    if not union:
        return 0.0
    return len(left & right) / len(union)


def _pairwise_spacer_similarities(records: Sequence[object], indices: Sequence[int]) -> list[float]:
    signatures = [(_spacer_signature(records[index])) for index in indices]
    similarities: list[float] = []

    for left, right in combinations(signatures, 2):
        similarities.append(_jaccard_similarity(left, right))

    return similarities


def _augmented_deletion_fractions(records: Sequence[object], indices: Sequence[int]) -> list[float]:
    fractions: list[float] = []
    for index in indices:
        example = records[index]
        source_count = int(getattr(example, "source_spacer_count", 0) or 0)
        deleted = int(getattr(example, "deleted_spacers", 0) or 0)
        fraction = getattr(example, "spacer_deletion_fraction", None)
        if fraction is not None and float(fraction) > 0:
            fractions.append(float(fraction))
        elif source_count > 0:
            fractions.append(deleted / source_count)
    return fractions


def _subtype_counts(records: Sequence[object], indices: Sequence[int]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for index in indices:
        example = records[index]
        subtype = (getattr(example, "cas_subtype", "") or "Unknown").strip() or "Unknown"
        counts[subtype] = counts.get(subtype, 0) + 1
    return counts


def _filter_subtypes_by_min_arrays(
    subtype_to_indices: dict[str, list[int]],
    min_arrays: int,
    reference_counts: dict[str, int] | None = None,
) -> tuple[dict[str, list[int]], list[str]]:
    if min_arrays <= 1:
        return subtype_to_indices, []

    if reference_counts is None:
        reference_counts = {subtype: len(indices) for subtype, indices in subtype_to_indices.items()}

    filtered = {
        subtype: indices
        for subtype, indices in subtype_to_indices.items()
        if reference_counts.get(subtype, 0) >= min_arrays
    }
    skipped = [subtype for subtype in subtype_to_indices.keys() if reference_counts.get(subtype, 0) < min_arrays]
    return filtered, skipped


def _format_stats(values: Sequence[int]) -> str:
    if not values:
        return "n=0\nmin=NA\nmax=NA\navg=NA"

    minimum = min(values)
    maximum = max(values)
    average = sum(values) / len(values)
    return f"n={len(values)}\nmin={minimum}\nmax={maximum}\navg={average:.2f}"


def _plot_length_histogram(ax, values: Sequence[int], title: str, xlabel: str, color: str) -> None:
    if not values:
        ax.text(0.5, 0.5, "No data available", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Count")
        ax.set_axis_off()
        return

    bins = min(30, max(5, len(set(values))))
    ax.hist(values, bins=bins, color=color, alpha=0.85, edgecolor="white")

    minimum = min(values)
    maximum = max(values)
    average = sum(values) / len(values)
    ax.axvline(minimum, color="#0f172a", linestyle=":", linewidth=2, label=f"min={minimum}")
    ax.axvline(average, color="#dc2626", linestyle="--", linewidth=2, label=f"avg={average:.2f}")
    ax.axvline(maximum, color="#0f172a", linestyle="-.", linewidth=2, label=f"max={maximum}")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")
    ax.legend(fontsize=9)
    ax.text(
        0.98,
        0.97,
        _format_stats(values),
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=10,
        family="monospace",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )


def plot_array_length_statistics(
    records: Sequence[object],
    indices: Sequence[int],
    title: str,
    output_path: Path | str,
) -> None:
    """Plot spacer-count and bp-length summaries for a sample subset.

    The bp statistic is the total length of all spacers and repeats in an array.
    """
    plt = _ensure_matplotlib()
    if plt is None:
        print(f"matplotlib not available; skipping length statistics visualization for {title}.")
        return

    spacer_counts, bp_lengths = _array_lengths(records, indices)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    _plot_length_histogram(axes[0], spacer_counts, "Spacer count per array", "Number of spacers", "#60a5fa")
    _plot_length_histogram(axes[1], bp_lengths, "Array length in bp", "Total bp (spacers + repeats)", "#34d399")
    fig.suptitle(title, fontsize=15, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])

    output_path = Path(output_path)
    fig.savefig(str(output_path), dpi=100, bbox_inches="tight")
    print(f"Length statistics saved to {output_path}")
    print(f"  spacers: {_format_stats(spacer_counts).replace(chr(10), ' | ')}")
    print(f"  bp: {_format_stats(bp_lengths).replace(chr(10), ' | ')}")
    plt.close(fig)


def plot_subtype_length_statistics(
    records: Sequence[object],
    indices: Sequence[int],
    title: str,
    output_path: Path | str,
    min_arrays: int = 0,
    reference_indices: Sequence[int] | None = None,
) -> None:
    """Plot per-subtype spacer-count and bp-length summaries for original samples.

    Each subtype is shown even if it only occurs once. The subtype count is
    included in the x-axis label so the plot makes class imbalance visible.
    """
    plt = _ensure_matplotlib()
    if plt is None:
        print(f"matplotlib not available; skipping subtype length statistics visualization for {title}.")
        return

    grouped_spacer_counts: dict[str, list[int]] = {}
    grouped_bp_lengths: dict[str, list[int]] = {}

    for index in indices:
        example = records[index]
        subtype = (getattr(example, "cas_subtype", "") or "Unknown").strip() or "Unknown"
        spacers = getattr(example, "spacers", []) or []
        repeats = getattr(example, "repeats", []) or []
        grouped_spacer_counts.setdefault(subtype, []).append(len(spacers))
        grouped_bp_lengths.setdefault(subtype, []).append(sum(len(sequence) for sequence in spacers) + sum(len(sequence) for sequence in repeats))

    reference_counts = _subtype_counts(records, reference_indices) if reference_indices is not None else None
    grouped_spacer_counts, skipped = _filter_subtypes_by_min_arrays(grouped_spacer_counts, min_arrays, reference_counts)
    grouped_bp_lengths = {subtype: grouped_bp_lengths[subtype] for subtype in grouped_spacer_counts.keys()}

    if skipped:
        print(f"Subtype length statistics: skipped subtypes below min_arrays={min_arrays}: {skipped}")

    subtype_order = sorted(grouped_spacer_counts.keys(), key=lambda key: (-len(grouped_spacer_counts[key]), key))
    labels = [f"{subtype}\nn={len(grouped_spacer_counts[subtype])}" for subtype in subtype_order]

    def _subplot(ax, grouped_values: dict[str, list[int]], metric_title: str, ylabel: str, color: str) -> None:
        values = [grouped_values[subtype] for subtype in subtype_order]
        box = ax.boxplot(values, labels=labels, showmeans=True, patch_artist=True)
        for patch in box["boxes"]:
            patch.set_facecolor(color)
            patch.set_alpha(0.75)
        for median in box["medians"]:
            median.set_color("#111827")
            median.set_linewidth(2)
        for mean in box["means"]:
            mean.set_marker("o")
            mean.set_markerfacecolor("white")
            mean.set_markeredgecolor("#111827")
            mean.set_markersize(5)

        ax.set_title(metric_title)
        ax.set_ylabel(ylabel)
        ax.grid(True, axis="y", alpha=0.3)
        ax.tick_params(axis="x", labelrotation=30)

        global_max = max(max(item) for item in values if item)
        y_offset = max(0.5, global_max * 0.06)
        for position, subtype in enumerate(subtype_order, start=1):
            subtype_values = grouped_values[subtype]
            minimum = min(subtype_values)
            maximum = max(subtype_values)
            average = sum(subtype_values) / len(subtype_values)
            ax.text(
                position,
                maximum + y_offset,
                f"min={minimum}\nmax={maximum}\navg={average:.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
                rotation=0,
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.75),
            )

        ax.set_ylim(top=global_max + y_offset * 5)

    fig, axes = plt.subplots(2, 1, figsize=(max(14, 1.2 * len(subtype_order) + 6), 12))
    _subplot(axes[0], grouped_spacer_counts, "Spacer count per subtype", "Number of spacers", "#60a5fa")
    _subplot(axes[1], grouped_bp_lengths, "Array length in bp per subtype", "Total bp (spacers + repeats)", "#34d399")
    fig.suptitle(title, fontsize=15, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    output_path = Path(output_path)
    fig.savefig(str(output_path), dpi=100, bbox_inches="tight")
    subtype_counts = Counter((getattr(records[index], "cas_subtype", "") or "Unknown").strip() or "Unknown" for index in indices)
    print(f"Subtype length statistics saved to {output_path}")
    print(f"  subtype_counts={dict(sorted(subtype_counts.items(), key=lambda kv: kv[0]))}")
    plt.close(fig)


def plot_spacer_similarity_statistics(
    records: Sequence[object],
    indices: Sequence[int],
    title: str,
    output_path: Path | str,
    min_arrays: int = 0,
    reference_indices: Sequence[int] | None = None,
) -> None:
    """Plot average spacer similarity overall and per subtype for original samples.

    Spacer similarity is computed as the Jaccard similarity between the sets of
    spacer sequences in each pair of arrays. The figure also includes an intra-
    subtype distribution panel.
    """
    plt = _ensure_matplotlib()
    if plt is None:
        print(f"matplotlib not available; skipping spacer similarity visualization for {title}.")
        return

    subtype_to_indices: dict[str, list[int]] = {}
    for index in indices:
        example = records[index]
        subtype = (getattr(example, "cas_subtype", "") or "Unknown").strip() or "Unknown"
        subtype_to_indices.setdefault(subtype, []).append(index)

    reference_counts = _subtype_counts(records, reference_indices) if reference_indices is not None else None
    subtype_to_indices, skipped = _filter_subtypes_by_min_arrays(subtype_to_indices, min_arrays, reference_counts)
    if skipped:
        print(f"Spacer similarity statistics: skipped subtypes below min_arrays={min_arrays}: {skipped}")

    subtype_order = sorted(subtype_to_indices.keys(), key=lambda key: (-len(subtype_to_indices[key]), key))
    overall_similarities = _pairwise_spacer_similarities(records, indices)
    overall_mean = sum(overall_similarities) / len(overall_similarities) if overall_similarities else 0.0

    subtype_means: list[float] = []
    subtype_pair_counts: list[int] = []
    subtype_similarity_values: list[list[float]] = []
    for subtype in subtype_order:
        subtype_sims = _pairwise_spacer_similarities(records, subtype_to_indices[subtype])
        subtype_similarity_values.append(subtype_sims)
        subtype_means.append(sum(subtype_sims) / len(subtype_sims) if subtype_sims else 0.0)
        subtype_pair_counts.append(len(subtype_sims))

    fig, axes = plt.subplots(1, 2, figsize=(max(16, 1.3 * len(subtype_order) + 8), 6))

    # Panel 1: average similarity by subtype with overall reference line.
    ax = axes[0]
    if subtype_order:
        x_positions = list(range(len(subtype_order)))
        bars = ax.bar(x_positions, subtype_means, color="#60a5fa", alpha=0.85, edgecolor="white")
        ax.axhline(overall_mean, color="#dc2626", linestyle="--", linewidth=2, label=f"overall mean={overall_mean:.3f}")
        ax.set_xticks(x_positions)
        ax.set_xticklabels(
            [f"{subtype}\nn={len(subtype_to_indices[subtype])}" for subtype in subtype_order],
            rotation=30,
            ha="right",
        )
        ax.set_ylim(0, max(0.05, max([overall_mean] + subtype_means) * 1.25))
        ax.set_ylabel("Mean pairwise Jaccard similarity")
        ax.set_title("Average spacer similarity by subtype")
        ax.grid(True, axis="y", alpha=0.25)
        ax.legend(fontsize=9)
        for bar, subtype, pair_count in zip(bars, subtype_order, subtype_pair_counts):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(0.005, overall_mean * 0.03),
                f"pairs={pair_count}\navg={bar.get_height():.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.75),
            )
    else:
        ax.text(0.5, 0.5, "No data available", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()

    # Panel 2: intra-subtype pairwise distribution.
    ax = axes[1]
    box_values = [values for values in subtype_similarity_values if values]
    box_labels = [f"{subtype}\npairs={len(values)}" for subtype, values in zip(subtype_order, subtype_similarity_values) if values]
    if box_values:
        box = ax.boxplot(box_values, labels=box_labels, showmeans=True, patch_artist=True)
        for patch in box["boxes"]:
            patch.set_facecolor("#34d399")
            patch.set_alpha(0.75)
        for median in box["medians"]:
            median.set_color("#111827")
            median.set_linewidth(2)
        for mean in box["means"]:
            mean.set_marker("o")
            mean.set_markerfacecolor("white")
            mean.set_markeredgecolor("#111827")
            mean.set_markersize(5)

        ax.set_title("Intra-subtype spacer similarity distribution")
        ax.set_ylabel("Pairwise Jaccard similarity")
        ax.grid(True, axis="y", alpha=0.25)
        ax.tick_params(axis="x", labelrotation=30)
        ax.text(
            0.98,
            0.97,
            f"overall_pairs={len(overall_similarities)}\noverall_mean={overall_mean:.3f}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=10,
            family="monospace",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )
    else:
        ax.text(0.5, 0.5, "Not enough samples per subtype for pairwise similarity", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()

    fig.suptitle(title, fontsize=15, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])

    output_path = Path(output_path)
    fig.savefig(str(output_path), dpi=100, bbox_inches="tight")
    print(f"Spacer similarity statistics saved to {output_path}")
    print(f"  overall_pairwise_mean={overall_mean:.4f} overall_pairs={len(overall_similarities)}")
    for subtype, values in zip(subtype_order, subtype_similarity_values):
        mean = sum(values) / len(values) if values else 0.0
        print(
            f"  {subtype}: n={len(subtype_to_indices[subtype])} pairwise_pairs={len(values)} pairwise_mean={mean:.4f}"
        )
    plt.close(fig)


def plot_augmented_spacer_deletion_statistics(
    records: Sequence[object],
    indices: Sequence[int],
    title: str,
    output_path: Path | str,
    min_arrays: int = 0,
    reference_indices: Sequence[int] | None = None,
) -> None:
    """Plot spacer deletion fractions for augmented samples overall and by subtype.

    Deletion fraction is defined as deleted_spacers / original_spacer_count for
    each augmented sample.
    """
    plt = _ensure_matplotlib()
    if plt is None:
        print(f"matplotlib not available; skipping deletion statistics visualization for {title}.")
        return

    subtype_to_indices: dict[str, list[int]] = {}
    for index in indices:
        example = records[index]
        subtype = (getattr(example, "cas_subtype", "") or "Unknown").strip() or "Unknown"
        subtype_to_indices.setdefault(subtype, []).append(index)

    reference_counts = _subtype_counts(records, reference_indices) if reference_indices is not None else None
    subtype_to_indices, skipped = _filter_subtypes_by_min_arrays(subtype_to_indices, min_arrays, reference_counts)
    if skipped:
        print(f"Deletion-fraction statistics: skipped subtypes below min_arrays={min_arrays}: {skipped}")

    subtype_order = sorted(subtype_to_indices.keys(), key=lambda key: (-len(subtype_to_indices[key]), key))
    overall_fractions = _augmented_deletion_fractions(records, indices)
    overall_mean = sum(overall_fractions) / len(overall_fractions) if overall_fractions else 0.0

    subtype_fraction_values: list[list[float]] = []
    subtype_means: list[float] = []
    subtype_counts: list[int] = []
    for subtype in subtype_order:
        values = _augmented_deletion_fractions(records, subtype_to_indices[subtype])
        subtype_fraction_values.append(values)
        subtype_means.append(sum(values) / len(values) if values else 0.0)
        subtype_counts.append(len(values))

    fig, axes = plt.subplots(1, 2, figsize=(max(16, 1.3 * len(subtype_order) + 8), 6))

    ax = axes[0]
    if subtype_order:
        x_positions = list(range(len(subtype_order)))
        bars = ax.bar(x_positions, subtype_means, color="#f59e0b", alpha=0.85, edgecolor="white")
        ax.axhline(overall_mean, color="#2563eb", linestyle="--", linewidth=2, label=f"overall mean={overall_mean:.3f}")
        ax.set_xticks(x_positions)
        ax.set_xticklabels(
            [f"{subtype}\nn={len(subtype_to_indices[subtype])}" for subtype in subtype_order],
            rotation=30,
            ha="right",
        )
        ax.set_ylim(0, max(0.05, max([overall_mean] + subtype_means) * 1.25))
        ax.set_ylabel("Mean deletion fraction")
        ax.set_title("Average spacer deletion fraction by subtype")
        ax.grid(True, axis="y", alpha=0.25)
        ax.legend(fontsize=9)
        for bar, pair_count in zip(bars, subtype_counts):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(0.005, overall_mean * 0.03),
                f"n={pair_count}\navg={bar.get_height():.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.75),
            )
    else:
        ax.text(0.5, 0.5, "No data available", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()

    ax = axes[1]
    box_values = [values for values in subtype_fraction_values if values]
    box_labels = [f"{subtype}\nn={len(values)}" for subtype, values in zip(subtype_order, subtype_fraction_values) if values]
    if box_values:
        box = ax.boxplot(box_values, labels=box_labels, showmeans=True, patch_artist=True)
        for patch in box["boxes"]:
            patch.set_facecolor("#fb7185")
            patch.set_alpha(0.75)
        for median in box["medians"]:
            median.set_color("#111827")
            median.set_linewidth(2)
        for mean in box["means"]:
            mean.set_marker("o")
            mean.set_markerfacecolor("white")
            mean.set_markeredgecolor("#111827")
            mean.set_markersize(5)

        ax.set_title("Intra-subtype spacer deletion fraction distribution")
        ax.set_ylabel("Deletion fraction")
        ax.grid(True, axis="y", alpha=0.25)
        ax.tick_params(axis="x", labelrotation=30)
        ax.text(
            0.98,
            0.97,
            f"overall_mean={overall_mean:.3f}\noverall_n={len(overall_fractions)}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=10,
            family="monospace",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )
    else:
        ax.text(0.5, 0.5, "No augmented samples available", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()

    fig.suptitle(title, fontsize=15, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])

    output_path = Path(output_path)
    fig.savefig(str(output_path), dpi=100, bbox_inches="tight")
    print(f"Deletion-fraction statistics saved to {output_path}")
    print(f"  overall_mean={overall_mean:.4f} overall_n={len(overall_fractions)}")
    for subtype, values in zip(subtype_order, subtype_fraction_values):
        mean = sum(values) / len(values) if values else 0.0
        print(f"  {subtype}: n={len(values)} mean={mean:.4f}")
    plt.close(fig)


def plot_training_curves(
    train_losses: Iterable[float],
    val_losses: Iterable[float],
    test_losses: Iterable[float] | None,
    args: object,
    train_dataset: object,
    val_dataset: object,
    stratify_mode: str,
    output_path: Path | str = "/tmp/training_curves.png",
) -> None:
    """Plot and save training/validation (and optional test) loss curves.

    Saves to `output_path`. This function is resilient when matplotlib is
    not available and will print a short message instead.
    """
    plt = _ensure_matplotlib()
    if plt is None:
        print("matplotlib not available; skipping training curves visualization.")
        return

    fig, ax = plt.subplots(figsize=(12, 7))
    epochs_range = range(1, len(train_losses) + 1)
    ax.plot(epochs_range, train_losses, marker='o', label='Train Loss', linewidth=2)
    ax.plot(epochs_range, val_losses, marker='s', label='Val Loss', linewidth=2)
    if test_losses is not None and len(test_losses) > 0:
        test_plot = [test_losses[i] if i < len(test_losses) else float('nan') for i in range(len(train_losses))]
        ax.plot(epochs_range, test_plot, marker='^', label='Test Loss', linewidth=2)

    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('Training vs Validation Loss', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    params_text = (
        f"batch_size={getattr(args, 'batch_size', 'NA')}\n"
        f"lr={getattr(args, 'lr', 'NA')}\n"
        f"weight_decay={getattr(args, 'weight_decay', 'NA')}\n"
        f"dropout={getattr(args, 'dropout', 'NA')}\n"
        f"early_stopping_patience={getattr(args, 'early_stopping_patience', 'NA')}\n"
        f"stratify_by={stratify_mode}\n"
        f"seed={getattr(args, 'seed', 'NA')}\n"
        f"train_size={len(train_dataset) if train_dataset is not None else 'NA'}\n"
        f"val_size={len(val_dataset) if val_dataset is not None else 'NA'}\n"
        f"epochs_completed={len(train_losses)}\n"
        f"best_val_loss={min(val_losses):.4f}"
    )
    ax.text(
        0.98, 0.97, params_text, transform=ax.transAxes, fontsize=10,
        verticalalignment='top', horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
        family='monospace'
    )

    output_path = Path(output_path)
    fig.savefig(str(output_path), dpi=100, bbox_inches='tight')
    print(f"Training curves saved to {output_path}")
    plt.close(fig)


def plot_confusion_matrix(
    model: object,
    test_loader: object,
    device: object,
    output_path: Path | str = "/tmp/confusion_matrix.png",
    threshold: float = 0.5,
):
    """Compute predictions on `test_loader` and save a confusion matrix plot.

    The model is expected to return logits when called with a batch dict.
    """
    plt = _ensure_matplotlib()
    if plt is None:
        print("matplotlib not available; skipping confusion matrix visualization.")
        return

    try:
        import numpy as np
        from sklearn.metrics import confusion_matrix
        from matplotlib.colors import LinearSegmentedColormap
    except Exception as e:  # pragma: no cover - optional deps
        print(f"Could not generate confusion matrix: {e}")
        return

    import torch

    all_probs = []
    all_labels = []
    model.eval()
    with torch.no_grad():
        for batch in test_loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            logits = model(batch)
            probs = torch.sigmoid(logits).cpu().numpy()
            labels = batch["label"].cpu().numpy()
            all_probs.extend(probs.flatten())
            all_labels.extend(labels.flatten())

    y_pred = (np.array(all_probs) >= threshold).astype(int)
    y_true = np.array(all_labels, dtype=int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    blue_purple = LinearSegmentedColormap.from_list(
        "blue_purple", ["#dbeafe", "#a78bfa", "#7c3aed"], N=256
    )
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation='nearest', cmap=blue_purple)

    total_examples = int(cm.sum()) if int(cm.sum()) > 0 else 1
    row_sums = cm.sum(axis=1)
    class_names = [
        f"Backward (n={int(row_sums[0])}, {row_sums[0] / total_examples:.1%})",
        f"Forward (n={int(row_sums[1])}, {row_sums[1] / total_examples:.1%})",
    ]
    tick_marks = np.arange(len(class_names))
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)

    thresh = cm.max() * 0.65
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            count = cm[i, j]
            pct_all = (count / total_examples) * 100.0
            pct_row = (count / row_sums[i]) * 100.0 if row_sums[i] > 0 else 0.0
            ax.text(
                j, i, f'{count}\n{pct_all:.1f}% total\n{pct_row:.1f}% row',
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=10, fontweight='bold'
            )

    ax.set_ylabel('True Label', fontsize=12)
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_title('Test Set Confusion Matrix', fontsize=14, fontweight='bold')
    plt.colorbar(im, ax=ax, label='Count')

    output_path = Path(output_path)
    fig.savefig(str(output_path), dpi=100, bbox_inches='tight')
    print(f"Confusion matrix saved to {output_path}")
    plt.close(fig)
