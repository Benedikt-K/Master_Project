"""Reverse-complement augmentation for CRISPR array examples.

Provides functionality to generate reverse-complemented versions of CRISPR arrays,
with automatic label flipping and test set safeguards.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .similarity import _candidate_passes_similarity_filter, _example_signature

if TYPE_CHECKING:
    from ..dataset import DirectionExample, DirectionJsonlDataset

from ..tokenization import reverse_complement


def _reverse_complement_example(example: DirectionExample) -> DirectionExample:
    """Return the reverse-complemented counterpart of a CRISPR array example.
    
    Flips the direction label (Forward <-> Reverse), reverses spacer/repeat order,
    reverse-complements all sequences, and swaps flanks.
    
    Args:
        example: Original DirectionExample.
        
    Returns:
        New DirectionExample with RC sequences and flipped label.
    """
    from ..dataset import DirectionExample

    flipped_label = 1 - int(example.label)
    flipped_direction = "Forward" if flipped_label == 1 else "Reverse"
    return DirectionExample(
        array_name=example.array_name,
        group_name=example.group_name,
        agreement=example.agreement,
        evor_direction=flipped_direction,
        label=flipped_label,
        orientation_variant="reverse_complement",
        source_variant=example.orientation_variant,
        spacers=[reverse_complement(seq) for seq in reversed(example.spacers)],
        repeats=[reverse_complement(seq) for seq in reversed(example.repeats)],
        cas_subtype=example.cas_subtype,
        left_flank=reverse_complement(example.right_flank),
        right_flank=reverse_complement(example.left_flank),
        source_json=example.source_json,
    )


def _materialize_reverse_complement_augmentation(
    base_dataset: DirectionJsonlDataset,
    source_indices: list[int],
    test_signatures: set[tuple[tuple[str, ...], tuple[str, ...]]] | None = None,
    test_signatures_by_idx: dict[int, tuple[tuple[str, ...], tuple[str, ...]]] | None = None,
    test_token_sets: dict[int, set[str]] | None = None,
    inverted_index: dict[str, set[int]] | None = None,
    similarity_metric: str = "jaccard",
    min_distance: float = 0.0,
) -> tuple[list[int], dict[str, int]]:
    """Duplicate a split with reverse-complemented examples.
    
    Applies similarity filtering to prevent test set leakage.
    
    Args:
        base_dataset: Dataset to add RC examples to.
        source_indices: Indices to generate RC versions of.
        test_signatures: Optional set of exact test signatures to exclude.
        test_signatures_by_idx: Optional dict from test index to (spacers, repeats).
        test_token_sets: Optional dict from test index to token set.
        inverted_index: Optional inverted index from token to test indices.
        similarity_metric: "jaccard" or "overlap".
        min_distance: Minimum acceptable distance from test set.
        
    Returns:
        (new_indices, stats) where new_indices are RC example indices and stats contains
        "added" (examples added) and "blocked_similarity" (examples rejected).
    """
    new_indices: list[int] = []
    stats = {"added": 0, "blocked_similarity": 0}

    for idx in source_indices:
        candidate_example = _reverse_complement_example(base_dataset.records[idx])
        if not _candidate_passes_similarity_filter(
            candidate_example,
            test_token_sets=test_token_sets,
            inverted_index=inverted_index,
            metric=similarity_metric,
            min_distance=min_distance,
            test_signatures_by_idx=test_signatures_by_idx,
        ):
            stats["blocked_similarity"] += 1
            continue

        if test_signatures is not None:
            candidate_sig = _example_signature(candidate_example)
            if candidate_sig in test_signatures:
                stats["blocked_similarity"] += 1
                continue

        base_dataset.records.append(candidate_example)
        new_indices.append(len(base_dataset.records) - 1)
        stats["added"] += 1

    return new_indices, stats
