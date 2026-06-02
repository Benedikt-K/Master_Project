"""Visualization helpers for direction_learning training and evaluation.

Centralizes plotting code (training curves, confusion matrix) so callers
can import and reuse the functions without embedding matplotlib code.
"""
from __future__ import annotations

from itertools import combinations
import random
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Iterable
import re

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


def _sample_indices(indices: Sequence[int], max_samples: int, rng: random.Random) -> list[int]:
    if max_samples <= 0 or len(indices) <= max_samples:
        return list(indices)
    return rng.sample(list(indices), max_samples)


def _sampled_pairwise_spacer_similarities(
    records: Sequence[object],
    indices: Sequence[int],
    max_pair_samples: int,
    rng: random.Random,
) -> tuple[list[float], int]:
    if len(indices) < 2:
        return [], 0

    signatures = [_spacer_signature(records[index]) for index in indices]
    n = len(signatures)
    total_pairs = n * (n - 1) // 2

    if max_pair_samples <= 0 or total_pairs <= max_pair_samples:
        similarities: list[float] = []
        for left, right in combinations(signatures, 2):
            similarities.append(_jaccard_similarity(left, right))
        return similarities, total_pairs

    similarities = []
    seen_pairs: set[tuple[int, int]] = set()
    while len(similarities) < max_pair_samples:
        left = rng.randrange(n)
        right = rng.randrange(n - 1)
        if right >= left:
            right += 1
        if right < left:
            left, right = right, left
        key = (left, right)
        if key in seen_pairs:
            continue
        seen_pairs.add(key)
        similarities.append(_jaccard_similarity(signatures[left], signatures[right]))
    return similarities, total_pairs


def _sampled_cross_spacer_similarities(
    records: Sequence[object],
    left_indices: Sequence[int],
    right_indices: Sequence[int],
    max_pair_samples: int,
    rng: random.Random,
) -> tuple[list[float], int]:
    if not left_indices or not right_indices:
        return [], 0

    left_signatures = [_spacer_signature(records[index]) for index in left_indices]
    right_signatures = [_spacer_signature(records[index]) for index in right_indices]
    total_pairs = len(left_signatures) * len(right_signatures)

    if max_pair_samples <= 0 or total_pairs <= max_pair_samples:
        similarities = [
            _jaccard_similarity(left, right)
            for left in left_signatures
            for right in right_signatures
        ]
        return similarities, total_pairs

    similarities = []
    seen_pairs: set[tuple[int, int]] = set()
    while len(similarities) < max_pair_samples:
        left = rng.randrange(len(left_signatures))
        right = rng.randrange(len(right_signatures))
        key = (left, right)
        if key in seen_pairs:
            continue
        seen_pairs.add(key)
        similarities.append(_jaccard_similarity(left_signatures[left], right_signatures[right]))
    return similarities, total_pairs


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


def _safe_filename_component(text: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", text.strip())
    cleaned = cleaned.strip("._-")
    return cleaned or "unknown"


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


def _plot_discrete_histogram(ax, values: Sequence[int], title: str, xlabel: str, color: str) -> None:
    if not values:
        ax.text(0.5, 0.5, "No data available", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Count")
        ax.set_axis_off()
        return

    minimum = min(values)
    maximum = max(values)
    average = sum(values) / len(values)
    bins = list(range(minimum, maximum + 2))
    ax.hist(values, bins=bins, color=color, alpha=0.88, edgecolor="white", rwidth=0.9)
    try:
        from matplotlib.ticker import MaxNLocator

        ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=10))
    except Exception:
        ax.set_xticks(list(range(minimum, maximum + 1, max(1, (maximum - minimum) // 10 or 1))))
    ax.set_xlim(minimum - 0.5, maximum + 0.5)
    ax.axvline(minimum, color="#0f172a", linestyle=":", linewidth=2, label=f"min={minimum}")
    ax.axvline(average, color="#dc2626", linestyle="--", linewidth=2, label=f"avg={average:.2f}")
    ax.axvline(maximum, color="#0f172a", linestyle="-.", linewidth=2, label=f"max={maximum}")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")
    ax.tick_params(axis="x", labelrotation=0, labelsize=9)
    ax.grid(True, axis="y", alpha=0.25)
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
    output_dir: Path | str,
    min_arrays: int = 0,
    reference_indices: Sequence[int] | None = None,
) -> None:
    """Plot per-subtype spacer-count histograms.

    Only the number of spacers is considered here, not basepair length.
    A separate PNG is written for each subtype so the charts stay readable.
    """
    plt = _ensure_matplotlib()
    if plt is None:
        print(f"matplotlib not available; skipping subtype length statistics visualization for {title}.")
        return

    grouped_spacer_counts: dict[str, list[int]] = {}

    for index in indices:
        example = records[index]
        subtype = (getattr(example, "cas_subtype", "") or "Unknown").strip() or "Unknown"
        spacers = getattr(example, "spacers", []) or []
        grouped_spacer_counts.setdefault(subtype, []).append(len(spacers))

    reference_counts = _subtype_counts(records, reference_indices) if reference_indices is not None else None
    grouped_spacer_counts, skipped = _filter_subtypes_by_min_arrays(grouped_spacer_counts, min_arrays, reference_counts)

    if skipped:
        print(f"Subtype length statistics: skipped subtypes below min_arrays={min_arrays}: {skipped}")

    if not grouped_spacer_counts:
        print(f"Subtype length statistics: no subtypes met the min_arrays={min_arrays} threshold for {title}.")
        return

    subtype_order = sorted(grouped_spacer_counts.keys(), key=lambda key: (-len(grouped_spacer_counts[key]), key))
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    subtype_counts = Counter((getattr(records[index], "cas_subtype", "") or "Unknown").strip() or "Unknown" for index in indices)
    print(f"Subtype length statistics: saving per-subtype spacer-count histograms to {output_dir}")
    print(f"  subtype_counts={dict(sorted(subtype_counts.items(), key=lambda kv: kv[0]))}")

    for subtype in subtype_order:
        values = grouped_spacer_counts[subtype]
        subtype_file = output_dir / f"{_safe_filename_component(subtype)}.png"
        fig, ax = plt.subplots(figsize=(7.5, 5.5), constrained_layout=True)
        _plot_discrete_histogram(
            ax,
            values,
            f"{title}: {subtype}",
            "Number of spacers per array",
            "#60a5fa",
        )
        fig.savefig(str(subtype_file), dpi=180, bbox_inches="tight", pad_inches=0.12)
        print(f"  saved {subtype_file} (n={len(values)})")
        plt.close(fig)


def plot_spacer_similarity_statistics(
    records: Sequence[object],
    indices: Sequence[int],
    title: str,
    output_path: Path | str,
    min_arrays: int = 0,
    reference_indices: Sequence[int] | None = None,
    max_samples_per_subtype: int = 1000,
    max_pair_samples: int = 50000,
    random_seed: int = 42,
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

    rng = random.Random(random_seed)
    subtype_order = sorted(subtype_to_indices.keys(), key=lambda key: (-len(subtype_to_indices[key]), key))
    sampled_overall_indices = _sample_indices(indices, max_samples_per_subtype, rng)
    overall_similarities, overall_total_pairs = _sampled_pairwise_spacer_similarities(
        records,
        sampled_overall_indices,
        max_pair_samples=max_pair_samples,
        rng=rng,
    )
    overall_mean = sum(overall_similarities) / len(overall_similarities) if overall_similarities else 0.0

    subtype_means: list[float] = []
    subtype_pair_counts: list[int] = []
    subtype_total_pairs: list[int] = []
    subtype_sample_sizes: list[int] = []
    subtype_original_sizes: list[int] = []
    subtype_similarity_values: list[list[float]] = []
    for subtype in subtype_order:
        original_indices = subtype_to_indices[subtype]
        sampled_indices = _sample_indices(original_indices, max_samples_per_subtype, rng)
        subtype_sims, total_pairs = _sampled_pairwise_spacer_similarities(
            records,
            sampled_indices,
            max_pair_samples=max_pair_samples,
            rng=rng,
        )
        subtype_similarity_values.append(subtype_sims)
        subtype_means.append(sum(subtype_sims) / len(subtype_sims) if subtype_sims else 0.0)
        subtype_pair_counts.append(len(subtype_sims))
        subtype_total_pairs.append(total_pairs)
        subtype_sample_sizes.append(len(sampled_indices))
        subtype_original_sizes.append(len(original_indices))

    fig, axes = plt.subplots(1, 2, figsize=(max(16, 1.3 * len(subtype_order) + 8), 6))

    # Panel 1: average similarity by subtype with overall reference line.
    ax = axes[0]
    if subtype_order:
        x_positions = list(range(len(subtype_order)))
        bars = ax.bar(x_positions, subtype_means, color="#60a5fa", alpha=0.85, edgecolor="white")
        ax.axhline(overall_mean, color="#dc2626", linestyle="--", linewidth=2, label=f"overall mean={overall_mean:.3f}")
        ax.set_xticks(x_positions)
        ax.set_xticklabels(
            [
                f"{subtype}\nn={sampled_n}/{original_n}"
                for subtype, sampled_n, original_n in zip(subtype_order, subtype_sample_sizes, subtype_original_sizes)
            ],
            rotation=30,
            ha="right",
        )
        ax.set_ylim(0, max(0.05, max([overall_mean] + subtype_means) * 1.25))
        ax.set_ylabel("Mean pairwise Jaccard similarity")
        ax.set_title("Average spacer similarity by subtype")
        ax.grid(True, axis="y", alpha=0.25)
        ax.legend(fontsize=9)
        for bar, subtype, pair_count, total_pairs in zip(bars, subtype_order, subtype_pair_counts, subtype_total_pairs):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(0.005, overall_mean * 0.03),
                f"pairs={pair_count}/{total_pairs}\navg={bar.get_height():.3f}",
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
            f"overall_pairs={len(overall_similarities)}/{overall_total_pairs}\noverall_mean={overall_mean:.3f}",
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
    print(
        f"  overall_pairwise_mean={overall_mean:.4f} "
        f"overall_pairs={len(overall_similarities)}/{overall_total_pairs} "
        f"overall_n={len(sampled_overall_indices)}/{len(indices)}"
    )
    for subtype, values, sampled_n, original_n, pair_count, total_pairs in zip(
        subtype_order,
        subtype_similarity_values,
        subtype_sample_sizes,
        subtype_original_sizes,
        subtype_pair_counts,
        subtype_total_pairs,
    ):
        mean = sum(values) / len(values) if values else 0.0
        print(
            f"  {subtype}: n={sampled_n}/{original_n} pairwise_pairs={pair_count}/{total_pairs} pairwise_mean={mean:.4f}"
        )
    plt.close(fig)


def plot_split_spacer_similarity_statistics(
    records: Sequence[object],
    train_indices: Sequence[int],
    val_indices: Sequence[int],
    test_indices: Sequence[int],
    title: str,
    output_path: Path | str,
    max_samples_per_split: int = 1000,
    max_pair_samples: int = 50000,
    random_seed: int = 42,
) -> None:
    """Plot train/val/test spacer-similarity statistics after augmentation.

    Uses bounded sampling so the visualization remains tractable for large
    augmented datasets.
    """
    plt = _ensure_matplotlib()
    if plt is None:
        print(f"matplotlib not available; skipping split similarity visualization for {title}.")
        return

    rng = random.Random(random_seed)
    split_indices_map: dict[str, list[int]] = {
        "train": _sample_indices(train_indices, max_samples_per_split, rng),
        "val": _sample_indices(val_indices, max_samples_per_split, rng),
        "test": _sample_indices(test_indices, max_samples_per_split, rng),
    }
    split_order = ["train", "val", "test"]

    split_intra_means: dict[str, float] = {}
    split_sampled_pairs: dict[str, int] = {}
    split_total_pairs: dict[str, int] = {}
    for split_name in split_order:
        values, total_pairs = _sampled_pairwise_spacer_similarities(
            records,
            split_indices_map[split_name],
            max_pair_samples=max_pair_samples,
            rng=rng,
        )
        split_intra_means[split_name] = (sum(values) / len(values)) if values else 0.0
        split_sampled_pairs[split_name] = len(values)
        split_total_pairs[split_name] = total_pairs

    matrix_values: list[list[float]] = []
    matrix_labels: list[list[str]] = []
    for left_name in split_order:
        row_values: list[float] = []
        row_labels: list[str] = []
        for right_name in split_order:
            if left_name == right_name:
                row_values.append(split_intra_means[left_name])
                row_labels.append(
                    f"{split_sampled_pairs[left_name]}/{split_total_pairs[left_name]}"
                )
                continue

            values, total_pairs = _sampled_cross_spacer_similarities(
                records,
                split_indices_map[left_name],
                split_indices_map[right_name],
                max_pair_samples=max_pair_samples,
                rng=rng,
            )
            mean = (sum(values) / len(values)) if values else 0.0
            row_values.append(mean)
            row_labels.append(f"{len(values)}/{total_pairs}")
        matrix_values.append(row_values)
        matrix_labels.append(row_labels)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    ax = axes[0]
    bar_values = [split_intra_means[split_name] for split_name in split_order]
    bars = ax.bar(split_order, bar_values, color=["#3b82f6", "#10b981", "#f59e0b"], alpha=0.85, edgecolor="white")
    ax.set_ylim(0, max(0.05, max(bar_values) * 1.3))
    ax.set_ylabel("Mean pairwise Jaccard similarity")
    ax.set_title("Intra-split spacer similarity")
    ax.grid(True, axis="y", alpha=0.25)
    for bar, split_name in zip(bars, split_order):
        sampled_n = len(split_indices_map[split_name])
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(0.005, max(bar_values) * 0.04),
            (
                f"n={sampled_n}\n"
                f"pairs={split_sampled_pairs[split_name]}/{split_total_pairs[split_name]}\n"
                f"avg={bar.get_height():.3f}"
            ),
            ha="center",
            va="bottom",
            fontsize=8,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.75),
        )

    ax = axes[1]
    image = ax.imshow(matrix_values, vmin=0.0, vmax=1.0, cmap="YlGnBu")
    ax.set_xticks(list(range(len(split_order))))
    ax.set_yticks(list(range(len(split_order))))
    ax.set_xticklabels(split_order)
    ax.set_yticklabels(split_order)
    ax.set_title("Mean Jaccard similarity matrix")
    for i, left_name in enumerate(split_order):
        for j, right_name in enumerate(split_order):
            color = "white" if matrix_values[i][j] > 0.5 else "black"
            ax.text(
                j,
                i,
                f"{matrix_values[i][j]:.3f}\n({matrix_labels[i][j]})",
                ha="center",
                va="center",
                color=color,
                fontsize=8,
            )
    plt.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="Mean Jaccard")

    fig.suptitle(title, fontsize=15, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])

    output_path = Path(output_path)
    fig.savefig(str(output_path), dpi=100, bbox_inches="tight")
    print(f"Split spacer similarity statistics saved to {output_path}")
    for split_name in split_order:
        print(
            f"  {split_name}: n={len(split_indices_map[split_name])} "
            f"pair_samples={split_sampled_pairs[split_name]}/{split_total_pairs[split_name]} "
            f"intra_mean={split_intra_means[split_name]:.4f}"
        )
    plt.close(fig)


def plot_subtype_split_spacer_similarity_statistics(
    records: Sequence[object],
    train_indices: Sequence[int],
    val_indices: Sequence[int],
    test_indices: Sequence[int],
    title: str,
    output_dir: Path | str,
    min_arrays: int = 0,
    reference_indices: Sequence[int] | None = None,
    max_samples_per_split: int = 400,
    max_pair_samples: int = 20000,
    random_seed: int = 42,
    subtypes_per_figure: int = 4,
) -> None:
    """Plot per-subtype train/val/test similarity matrices in paginated figures.

    For each subtype, a 3x3 matrix is generated where diagonal cells are
    intra-split similarities and off-diagonal cells are cross-split
    similarities. Sampling caps keep runtime bounded on large datasets.
    """
    plt = _ensure_matplotlib()
    if plt is None:
        print(f"matplotlib not available; skipping per-subtype split similarity visualization for {title}.")
        return

    def _get_subtype(index: int) -> str:
        example = records[index]
        return (getattr(example, "cas_subtype", "") or "Unknown").strip() or "Unknown"

    train_by_subtype: dict[str, list[int]] = {}
    val_by_subtype: dict[str, list[int]] = {}
    test_by_subtype: dict[str, list[int]] = {}
    combined_by_subtype: dict[str, list[int]] = {}

    for index in train_indices:
        subtype = _get_subtype(index)
        train_by_subtype.setdefault(subtype, []).append(index)
        combined_by_subtype.setdefault(subtype, []).append(index)
    for index in val_indices:
        subtype = _get_subtype(index)
        val_by_subtype.setdefault(subtype, []).append(index)
        combined_by_subtype.setdefault(subtype, []).append(index)
    for index in test_indices:
        subtype = _get_subtype(index)
        test_by_subtype.setdefault(subtype, []).append(index)
        combined_by_subtype.setdefault(subtype, []).append(index)

    reference_counts = _subtype_counts(records, reference_indices) if reference_indices is not None else None
    filtered_subtypes, skipped = _filter_subtypes_by_min_arrays(combined_by_subtype, min_arrays, reference_counts)
    if skipped:
        print(f"Per-subtype split similarity: skipped subtypes below min_arrays={min_arrays}: {skipped}")

    subtype_order = sorted(filtered_subtypes.keys())
    if not subtype_order:
        print("Per-subtype split similarity: no subtypes to plot after filtering.")
        return

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    split_order = ["train", "val", "test"]
    rng = random.Random(random_seed)

    def _matrix_for_subtype(subtype: str) -> tuple[list[list[float]], list[list[str]], dict[str, tuple[int, int]]]:
        original_map = {
            "train": train_by_subtype.get(subtype, []),
            "val": val_by_subtype.get(subtype, []),
            "test": test_by_subtype.get(subtype, []),
        }
        sampled_map = {
            split_name: _sample_indices(original_map[split_name], max_samples_per_split, rng)
            for split_name in split_order
        }
        split_n = {
            split_name: (len(sampled_map[split_name]), len(original_map[split_name]))
            for split_name in split_order
        }

        matrix_values: list[list[float]] = []
        matrix_labels: list[list[str]] = []
        for left_name in split_order:
            row_values: list[float] = []
            row_labels: list[str] = []
            for right_name in split_order:
                if left_name == right_name:
                    values, total_pairs = _sampled_pairwise_spacer_similarities(
                        records,
                        sampled_map[left_name],
                        max_pair_samples=max_pair_samples,
                        rng=rng,
                    )
                else:
                    values, total_pairs = _sampled_cross_spacer_similarities(
                        records,
                        sampled_map[left_name],
                        sampled_map[right_name],
                        max_pair_samples=max_pair_samples,
                        rng=rng,
                    )
                mean = (sum(values) / len(values)) if values else 0.0
                row_values.append(mean)
                row_labels.append(f"{len(values)}/{total_pairs}")
            matrix_values.append(row_values)
            matrix_labels.append(row_labels)

        return matrix_values, matrix_labels, split_n

    total_subtypes = len(subtype_order)
    per_figure = max(1, subtypes_per_figure)
    total_pages = (total_subtypes + per_figure - 1) // per_figure

    for page in range(total_pages):
        start = page * per_figure
        page_subtypes = subtype_order[start:start + per_figure]

        cols = 2
        rows = (len(page_subtypes) + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(7.2 * cols, 5.2 * rows))
        axes_list = axes.flatten() if hasattr(axes, "flatten") else [axes]

        for axis, subtype in zip(axes_list, page_subtypes):
            matrix_values, matrix_labels, split_n = _matrix_for_subtype(subtype)
            image = axis.imshow(matrix_values, vmin=0.0, vmax=1.0, cmap="YlGnBu")
            axis.set_xticks(list(range(len(split_order))))
            axis.set_yticks(list(range(len(split_order))))
            axis.set_xticklabels(split_order)
            axis.set_yticklabels(split_order)
            axis.set_title(
                (
                    f"{subtype}\n"
                    f"train n={split_n['train'][0]}/{split_n['train'][1]} | "
                    f"val n={split_n['val'][0]}/{split_n['val'][1]} | "
                    f"test n={split_n['test'][0]}/{split_n['test'][1]}"
                ),
                fontsize=10,
            )

            for i in range(len(split_order)):
                for j in range(len(split_order)):
                    color = "white" if matrix_values[i][j] > 0.5 else "black"
                    axis.text(
                        j,
                        i,
                        f"{matrix_values[i][j]:.3f}\n({matrix_labels[i][j]})",
                        ha="center",
                        va="center",
                        color=color,
                        fontsize=8,
                    )

            fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)

        for axis in axes_list[len(page_subtypes):]:
            axis.set_axis_off()

        fig.suptitle(f"{title} (part {page + 1}/{total_pages})", fontsize=15, fontweight="bold")
        fig.tight_layout(rect=[0, 0, 1, 0.94])

        out_path = output_dir / f"subtype_split_spacer_similarity_after_augmentation_part{page + 1}.png"
        fig.savefig(str(out_path), dpi=100, bbox_inches="tight")
        print(f"Per-subtype split similarity statistics saved to {out_path}")
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
