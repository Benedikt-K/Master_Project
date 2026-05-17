"""Subarray deletion augmentation for CRISPR array examples.

Provides spacer subset selection with optional diversity maximization
and test set safeguards.
"""

from __future__ import annotations

import math
import random
from typing import TYPE_CHECKING

from .similarity import _candidate_passes_similarity_filter, _example_signature

if TYPE_CHECKING:
    from ..dataset import DirectionExample, DirectionJsonlDataset

from ..dataset import DirectionExample
from ..utils import _print_ts


def _keep_overlap_ratio(a: tuple[int, ...], b: tuple[int, ...]) -> float:
    """Calculate Jaccard-like overlap ratio between two sets of indices."""
    set_a = set(a)
    set_b = set(b)
    union = set_a | set_b
    if not union:
        return 0.0
    return len(set_a & set_b) / len(union)


def _select_diverse_keep_sets(
    candidates: list[tuple[int, ...]],
    target: int,
    rng: random.Random,
) -> list[tuple[int, ...]]:
    """Choose a diverse subset of keep-indices by maximizing pairwise distance.
    
    Uses precomputed overlap matrix to select candidates that are maximally different
    from each other, improving augmentation diversity.
    
    Args:
        candidates: Pool of candidate keep-index tuples.
        target: Number of items to select.
        rng: Random number generator for tie-breaking.
        
    Returns:
        Diverse subset of keep-indices (up to target items).
    """
    unique_candidates = list(dict.fromkeys(candidates))
    if target <= 0 or len(unique_candidates) <= target:
        rng.shuffle(unique_candidates)
        return unique_candidates

    # Limit pool to avoid excessive computation
    if len(unique_candidates) > 500:
        rng.shuffle(unique_candidates)
        unique_candidates = unique_candidates[:500]

    n = len(unique_candidates)
    # Pre-compute all pairwise overlaps once (O(n²) one-time cost)
    overlaps: dict[tuple[int, int], float] = {}
    for i in range(n):
        for j in range(i + 1, n):
            overlap = _keep_overlap_ratio(unique_candidates[i], unique_candidates[j])
            overlaps[(i, j)] = overlap
            overlaps[(j, i)] = overlap

    selected_indices = [rng.randint(0, n - 1)]
    remaining = set(range(n)) - {selected_indices[0]}

    while remaining and len(selected_indices) < target:
        best_score = -1.0
        best_candidate = None

        for idx in remaining:
            # Fast O(selected) lookup: check precomputed overlaps
            min_distance = min(1.0 - overlaps.get((idx, selected), 0.0) for selected in selected_indices)
            if min_distance > best_score:
                best_score = min_distance
                best_candidate = idx

        if best_candidate is not None:
            selected_indices.append(best_candidate)
            remaining.remove(best_candidate)

    return [unique_candidates[i] for i in selected_indices]


def make_subarray_augment_fn(prob: float = 1.0, seed: int = 42):
    """Return an augment_fn that deletes a random subset of spacers (preserving order).

    The augment_fn returns either the original example or a new DirectionExample
    with a subset of spacers/repeats chosen uniformly by size and indices.
    
    Args:
        prob: Probability to apply augmentation per example.
        seed: Random seed for reproducibility.
        
    Returns:
        Function that takes DirectionExample and returns augmented or original.
    """
    rng = random.Random(seed)

    def augment_fn(example: DirectionExample) -> DirectionExample:
        try:
            if rng.random() >= float(prob):
                return example

            n = len(example.spacers)
            # Nothing to drop if <=1 spacer
            if n <= 1:
                return example

            # choose subset size between 1 and n-1 (so we produce a proper subarray)
            k = rng.randint(1, n - 1)
            keep_indices = sorted(rng.sample(range(n), k))
            new_spacers = [example.spacers[i] for i in keep_indices]
            new_repeats = [example.repeats[i] for i in keep_indices]

            return DirectionExample(
                array_name=example.array_name,
                group_name=example.group_name,
                agreement=example.agreement,
                evor_direction=example.evor_direction,
                label=example.label,
                orientation_variant=example.orientation_variant,
                source_variant=example.source_variant,
                spacers=new_spacers,
                repeats=new_repeats,
                cas_subtype=example.cas_subtype,
                left_flank=example.left_flank,
                right_flank=example.right_flank,
                source_json=example.source_json,
            )
        except Exception:
            return example

    return augment_fn


def make_subarray_augment_fn_with_similarity_filter(
    prob: float = 1.0,
    seed: int = 42,
    max_attempts: int = 5,
    test_signatures: set[tuple[tuple[str, ...], tuple[str, ...]]] | None = None,
    test_signatures_by_idx: dict[int, tuple[tuple[str, ...], tuple[str, ...]]] | None = None,
    test_token_sets: dict[int, set[str]] | None = None,
    inverted_index: dict[str, set[int]] | None = None,
    similarity_metric: str = "jaccard",
    min_distance: float = 0.0,
):
    """Return an augment_fn that also filters candidates against the test set.

    This is only used when the optional similarity safeguard is enabled.
    
    Args:
        prob: Probability to apply augmentation.
        seed: Random seed.
        max_attempts: Maximum retry attempts if candidate fails filter.
        test_signatures: Optional set of exact test signatures to avoid.
        test_signatures_by_idx: Optional dict from test index to (spacers, repeats).
        test_token_sets: Optional dict from test index to token set.
        inverted_index: Optional inverted index from token to test indices.
        similarity_metric: "jaccard" or "overlap".
        min_distance: Minimum acceptable distance from test set.
        
    Returns:
        Function that takes DirectionExample and returns augmented or original.
    """
    rng = random.Random(seed)

    stats = {
        "accepted": 0,
        "blocked_exact": 0,
        "blocked_distance": 0,
        "attempts": 0,
    }

    def augment_fn(example: DirectionExample) -> DirectionExample:
        try:
            if rng.random() >= float(prob):
                return example

            n = len(example.spacers)
            if n <= 1:
                return example

            for _ in range(max(1, max_attempts)):
                stats["attempts"] += 1
                k = rng.randint(1, n - 1)
                keep_indices = sorted(rng.sample(range(n), k))
                new_spacers = [example.spacers[i] for i in keep_indices]
                new_repeats = [example.repeats[i] for i in keep_indices]
                candidate = DirectionExample(
                    array_name=example.array_name,
                    group_name=example.group_name,
                    agreement=example.agreement,
                    evor_direction=example.evor_direction,
                    label=example.label,
                    orientation_variant=example.orientation_variant,
                    source_variant=example.source_variant,
                    spacers=new_spacers,
                    repeats=new_repeats,
                    cas_subtype=example.cas_subtype,
                    left_flank=example.left_flank,
                    right_flank=example.right_flank,
                    source_json=example.source_json,
                )

                candidate_sig = _example_signature(candidate)
                if test_signatures is not None and candidate_sig in test_signatures:
                    stats["blocked_exact"] += 1
                    continue
                if not _candidate_passes_similarity_filter(
                    candidate,
                    test_token_sets=test_token_sets,
                    inverted_index=inverted_index,
                    metric=similarity_metric,
                    min_distance=min_distance,
                    test_signatures_by_idx=test_signatures_by_idx,
                ):
                    stats["blocked_distance"] += 1
                    continue

                stats["accepted"] += 1
                return candidate

            return example
        except Exception:
            return example

    augment_fn.similarity_stats = stats  # type: ignore[attr-defined]
    return augment_fn


def _materialize_subarray_augmentations(
    base_dataset: DirectionJsonlDataset,
    source_indices: list[int],
    seen_signatures: set[tuple[tuple[str, ...], tuple[str, ...]]],
    test_signatures: set[tuple[tuple[str, ...], tuple[str, ...]]] | None,
    test_signatures_by_idx: dict[int, tuple[tuple[str, ...], tuple[str, ...]]] | None,
    test_token_sets: dict[int, set[str]] | None,
    inverted_index: dict[str, set[int]] | None,
    seed: int,
    mode: str,
    prob: float,
    min_spacers: int,
    max_per_array: int,
    split_name: str,
    use_diversity: bool = True,
    similarity_metric: str = "jaccard",
    min_distance: float = 0.0,
    target_additions: int = 0,
    balance_per_array: bool = False,
) -> tuple[list[int], dict[str, int]]:
    """Materialize subarray deletion augmentation for a split.

    Augmented signatures are globally deduplicated against `seen_signatures`
    so train/val/test do not gain overlapping spacer/repeat pairs.
    
    Args:
        base_dataset: Dataset to add augmented examples to.
        source_indices: Indices to augment.
        seen_signatures: Global set of signatures to avoid duplicates.
        test_signatures: Optional set of exact test signatures.
        test_signatures_by_idx: Optional dict from test index to (spacers, repeats).
        test_token_sets: Optional dict from test index to token set.
        inverted_index: Optional inverted index from token to test indices.
        seed: Random seed.
        mode: "random" or "enumerate".
        prob: Probability for random mode.
        min_spacers: Minimum spacers in augmented example.
        max_per_array: Maximum augmentations per source array (0 for unlimited).
        split_name: Name for logging.
        use_diversity: Whether to select diverse keep-sets.
        similarity_metric: "jaccard" or "overlap".
        min_distance: Minimum acceptable distance from test set.
        target_additions: If > 0, stop once this many examples added (for balanced augmentation).
        balance_per_array: If True and target_additions > 0, distribute evenly across sources.
        
    Returns:
        (new_indices, stats) where new_indices are indices of added examples.
    """
    base_rng = random.Random(seed)
    new_indices: list[int] = []
    stats = {
        "added": 0,
        "blocked_overlap": 0,
        "blocked_similarity": 0,
        "capped_arrays": 0,
        "skipped_short": 0,
        "source_examples": len(source_indices),
    }

    _print_ts(f"Augmentation: {split_name} starting materialization of {len(source_indices)} source examples...")
    source_list = list(source_indices)

    if max_per_array <= 0:
        _print_ts(f"Augmentation: {split_name} disabled because max_per_array={max_per_array}")
        return new_indices, stats
    
    # When balancing, distribute augmentations evenly across all source arrays
    effective_max_per_array = max_per_array
    if balance_per_array and target_additions > 0 and len(source_list) > 0:
        # Calculate fair per-array cap: distribute target evenly across all sources
        per_array_target = math.ceil(target_additions / len(source_list))
        effective_max_per_array = min(max_per_array, per_array_target) if max_per_array > 0 else per_array_target
        _print_ts(f"Augmentation: {split_name} balancing per-array distribution: need {target_additions} total across {len(source_list)} arrays, cap per-array to {effective_max_per_array}")
    
    for i_example, orig_idx in enumerate(source_list, 1):
        if i_example % 50 == 1 or i_example == 1:
            _print_ts(f"  Augmentation: {split_name} processing example {i_example}/{len(source_list)} (added so far: {stats['added']})")
        ex = base_dataset.records[orig_idx]
        n = len(ex.spacers)
        if n <= min_spacers:
            stats["skipped_short"] += 1
            continue

        local_rng = random.Random(base_rng.randint(0, 2**31 - 1))

        if mode == "random":
            if local_rng.random() >= float(prob):
                continue
            k = local_rng.randint(min_spacers, n - 1)
            keep_sets = [tuple(sorted(local_rng.sample(range(n), k)))]
        else:
            candidate_goal = effective_max_per_array if effective_max_per_array > 0 else 0
            if use_diversity:
                # Generate a larger candidate pool than the final cap so we can
                # choose a more diverse subset of subarrays.
                if candidate_goal > 0:
                    pool_goal = max(candidate_goal * 4, candidate_goal + 32, 64)
                else:
                    pool_goal = max(64, n * 16)
                max_attempts = max(100, pool_goal * 50)
            else:
                # Fast mode: just generate candidate_goal samples without diversity
                pool_goal = candidate_goal if candidate_goal > 0 else min(64, n * 4)
                max_attempts = max(50, pool_goal * 10)
            candidates: list[tuple[int, ...]] = []
            seen_local: set[tuple[int, ...]] = set()
            attempts = 0

            while len(candidates) < pool_goal and attempts < max_attempts:
                attempts += 1
                k = local_rng.randint(min_spacers, n - 1)
                keep = tuple(sorted(local_rng.sample(range(n), k)))
                if keep not in seen_local:
                    seen_local.add(keep)
                    candidates.append(keep)

            keep_sets = (
                _select_diverse_keep_sets(candidates, candidate_goal, local_rng)
                if use_diversity
                else candidates[:candidate_goal]
            )
            if candidate_goal > 0 and len(keep_sets) >= candidate_goal:
                stats["capped_arrays"] += 1

        for keep in keep_sets:
            new_spacers = [ex.spacers[i] for i in keep]
            new_repeats = [ex.repeats[i] for i in keep]
            aug_sig = (tuple(new_spacers), tuple(new_repeats))
            if aug_sig in seen_signatures:
                stats["blocked_overlap"] += 1
                continue

            if test_signatures is not None and aug_sig in test_signatures:
                stats["blocked_similarity"] += 1
                continue

            candidate_example = DirectionExample(
                array_name=ex.array_name,
                group_name=ex.group_name,
                agreement=ex.agreement,
                evor_direction=ex.evor_direction,
                label=ex.label,
                orientation_variant=ex.orientation_variant,
                source_variant=ex.source_variant,
                spacers=new_spacers,
                repeats=new_repeats,
                cas_subtype=ex.cas_subtype,
                left_flank=ex.left_flank,
                right_flank=ex.right_flank,
                source_json=ex.source_json,
            )
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

            base_dataset.records.append(candidate_example)
            new_indices.append(len(base_dataset.records) - 1)
            seen_signatures.add(aug_sig)
            stats["added"] += 1
            
            # Early exit if we've reached the target number of additions
            if target_additions > 0 and stats["added"] >= target_additions:
                break
        
        # Early exit outer loop if we've reached the target
        if target_additions > 0 and stats["added"] >= target_additions:
            break

    _print_ts(
        f"Augmentation: {split_name} added {stats['added']} examples "
        f"(blocked_overlap={stats['blocked_overlap']}, blocked_similarity={stats['blocked_similarity']}, "
        f"skipped_short={stats['skipped_short']})"
    )
    if mode == "enumerate" and max_per_array > 0:
        _print_ts(f"Augmentation: {split_name} cap_hit_on={stats['capped_arrays']} arrays")

    return new_indices, stats
