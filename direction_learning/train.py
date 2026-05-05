from __future__ import annotations

import argparse
import random
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

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


def make_subarray_augment_fn(prob: float = 1.0, seed: int = 42):
    """Return an augment_fn that deletes a random subset of spacers (preserving order).

    The augment_fn returns either the original example or a new DirectionExample
    with a subset of spacers/repeats chosen uniformly by size and indices.
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


def _example_signature(example: DirectionExample) -> tuple[tuple[str, ...], tuple[str, ...]]:
    return (tuple(example.spacers), tuple(example.repeats))


def _keep_overlap_ratio(a: tuple[int, ...], b: tuple[int, ...]) -> float:
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
    """Choose a diverse subset of keep-indices by maximizing pairwise distance (with precomputed overlaps)."""
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


def _materialize_subarray_augmentations(
    base_dataset: DirectionJsonlDataset,
    source_indices: list[int],
    seen_signatures: set[tuple[tuple[str, ...], tuple[str, ...]]],
    seed: int,
    mode: str,
    prob: float,
    min_spacers: int,
    max_per_array: int,
    split_name: str,
    use_diversity: bool = True,
) -> tuple[list[int], dict[str, int]]:
    """Materialize subarray deletion augmentation for a split.

    Augmented signatures are globally deduplicated against `seen_signatures`
    so train/val/test do not gain overlapping spacer/repeat pairs.
    """
    base_rng = random.Random(seed)
    new_indices: list[int] = []
    stats = {
        "added": 0,
        "blocked_overlap": 0,
        "capped_arrays": 0,
        "skipped_short": 0,
        "source_examples": len(source_indices),
    }

    for orig_idx in list(source_indices):
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
            candidate_goal = max_per_array if max_per_array > 0 else 0
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

            new_ex = DirectionExample(
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
            base_dataset.records.append(new_ex)
            new_indices.append(len(base_dataset.records) - 1)
            seen_signatures.add(aug_sig)
            stats["added"] += 1

    print(
        f"Augmentation: {split_name} added {stats['added']} examples "
        f"(blocked_overlap={stats['blocked_overlap']}, skipped_short={stats['skipped_short']})"
    )
    if mode == "enumerate" and max_per_array > 0:
        print(f"Augmentation: {split_name} cap_hit_on={stats['capped_arrays']} arrays")

    return new_indices, stats


def summarize_cas_subtypes(
    records: list[DirectionExample],
    indices: list[int],
) -> tuple[int, dict[str, int]]:
    """Return unique subtype count and per-subtype counts for a split."""
    counts = Counter((records[i].cas_subtype or "Unknown") for i in indices)
    return len(counts), dict(sorted(counts.items(), key=lambda kv: kv[0]))


def build_cv_folds_by_signature(
    examples: list[DirectionExample],
    pool_indices: list[int],
    n_folds: int,
    seed: int,
    stratify_mode: str,
) -> list[list[int]]:
    """Build CV folds from pool indices, keeping exact signatures together.

    Args:
        examples: Full list of examples.
        pool_indices: Indices allowed for CV (development pool).
        n_folds: Number of CV folds.
        seed: RNG seed.
        stratify_mode: One of "label", "cas_subtype", "cas_subtype_and_label".

    Returns:
        List of folds where each entry is a list of validation indices for that fold.
    """
    if n_folds < 2:
        raise ValueError("n_folds must be at least 2")

    rng = random.Random(seed)
    pool_set = set(pool_indices)

    signature_groups: dict[tuple[tuple[str, ...], tuple[str, ...]], list[int]] = {}
    for idx in pool_indices:
        ex = examples[idx]
        sig = (tuple(ex.spacers), tuple(ex.repeats))
        signature_groups.setdefault(sig, []).append(idx)

    strata_groups: dict[Any, list[list[int]]] = {}
    for group in signature_groups.values():
        rep = examples[group[0]]
        subtype = (rep.cas_subtype or "Unknown").strip() or "Unknown"
        label = int(rep.label)

        if stratify_mode == "label":
            key: Any = label
        elif stratify_mode == "cas_subtype":
            key = subtype
        else:
            key = (subtype, label)

        strata_groups.setdefault(key, []).append(group)

    folds: list[list[int]] = [[] for _ in range(n_folds)]
    for key in sorted(strata_groups.keys(), key=str):
        groups = list(strata_groups[key])
        rng.shuffle(groups)
        for j, group in enumerate(groups):
            fold_idx = j % n_folds
            folds[fold_idx].extend(group)

    # Keep only indices from the pool and stabilize ordering for reproducibility
    for i in range(n_folds):
        folds[i] = sorted(idx for idx in folds[i] if idx in pool_set)

    return folds


def stratified_holdout_by_mode(
    examples: list[DirectionExample],
    seed: int,
    holdout_fraction: float,
    stratify_mode: str,
) -> tuple[list[int], list[int]]:
    """Split indices into development/train and held-out test sets.

    Exact spacer/repeat signatures stay together, and the stratification key
    follows the selected mode.
    """
    if not (0.0 <= holdout_fraction < 1.0):
        raise ValueError("holdout_fraction must be in [0.0, 1.0)")
    if holdout_fraction == 0.0:
        return list(range(len(examples))), []

    rng = random.Random(seed)

    signature_groups: dict[tuple[tuple[str, ...], tuple[str, ...]], list[int]] = {}
    for idx, example in enumerate(examples):
        signature = (tuple(example.spacers), tuple(example.repeats))
        signature_groups.setdefault(signature, []).append(idx)

    strata_groups: dict[Any, list[list[int]]] = {}
    for group in signature_groups.values():
        rep = examples[group[0]]
        subtype = (rep.cas_subtype or "Unknown").strip() or "Unknown"
        label = int(rep.label)

        if stratify_mode == "label":
            key: Any = label
        elif stratify_mode == "cas_subtype":
            key = subtype
        else:
            key = (subtype, label)

        strata_groups.setdefault(key, []).append(group)

    dev_indices: list[int] = []
    test_indices: list[int] = []
    for key in sorted(strata_groups.keys(), key=str):
        groups = list(strata_groups[key])
        rng.shuffle(groups)
        n_groups = len(groups)
        n_test = 0 if n_groups == 1 else min(n_groups - 1, max(1, round(n_groups * holdout_fraction)))
        for group in groups[:n_test]:
            test_indices.extend(group)
        for group in groups[n_test:]:
            dev_indices.extend(group)

    return sorted(dev_indices), sorted(test_indices)


def split_dev_pool_by_mode(
    examples: list[DirectionExample],
    pool_indices: list[int],
    seed: int,
    stratify_mode: str,
) -> tuple[list[int], list[int]]:
    """Split a development pool into train and validation indices."""
    pool_examples = [examples[i] for i in pool_indices]
    if stratify_mode == "label":
        splits = stratified_train_test_and_val_by_label(pool_examples, seed=seed, train_test_fraction=0.8)
        train_indices = [pool_indices[i] for i in splits["train_test"]]
        val_indices = [pool_indices[i] for i in splits["val"]]
    elif stratify_mode == "cas_subtype":
        splits = stratified_split_by_cas_subtype(pool_examples, seed=seed, train_fraction=0.8, test_fraction=0.0)
        train_indices = [pool_indices[i] for i in splits["train"]]
        val_indices = [pool_indices[i] for i in splits["val"]]
    else:
        splits = stratified_train_test_and_val_by_cas_subtype_and_label(
            pool_examples, seed=seed, train_test_fraction=0.8
        )
        train_indices = [pool_indices[i] for i in splits["train_test"]]
        val_indices = [pool_indices[i] for i in splits["val"]]

    return sorted(train_indices), sorted(val_indices)


def _build_signature_components(examples: list[DirectionExample]) -> dict[int, list[int]]:
    """Group indices into connected components by exact spacer/repeat signature."""
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

    first_by_signature: dict[tuple[tuple[str, ...], tuple[str, ...]], int] = {}
    for idx, example in enumerate(examples):
        signature = (tuple(example.spacers), tuple(example.repeats))
        if signature in first_by_signature:
            union(idx, first_by_signature[signature])
        else:
            first_by_signature[signature] = idx

    components: dict[int, list[int]] = {}
    for idx in range(n):
        components.setdefault(find(idx), []).append(idx)
    return components


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
    components = _build_signature_components(examples)

    # Stratify by cas_subtype at component level so exact duplicate signatures stay together.
    subtype_components: dict[str, list[list[int]]] = {}
    for comp in components.values():
        subtype_counts = Counter((examples[i].cas_subtype or "Unknown") for i in comp)
        subtype = max(subtype_counts, key=subtype_counts.get)
        subtype_components.setdefault(subtype, []).append(comp)

    train_indices: list[int] = []
    val_indices: list[int] = []
    test_indices: list[int] = []

    for subtype, comp_list in subtype_components.items():
        shuffled = list(comp_list)
        rng.shuffle(shuffled)

        n_total = len(shuffled)
        n_train = max(1, round(n_total * train_fraction))
        n_test = max(0, round(n_total * test_fraction))
        n_val = n_total - n_train - n_test

        for comp in shuffled[:n_train]:
            train_indices.extend(comp)
        for comp in shuffled[n_train:n_train + n_test]:
            test_indices.extend(comp)
        for comp in shuffled[n_train + n_test:]:
            val_indices.extend(comp)
    
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
    components = _build_signature_components(examples)

    subtype_components: dict[str, list[list[int]]] = {}
    for comp in components.values():
        subtype_counts = Counter((examples[i].cas_subtype or "Unknown") for i in comp)
        subtype = max(subtype_counts, key=subtype_counts.get)
        subtype_components.setdefault(subtype, []).append(comp)

    train_indices: list[int] = []
    test_indices: list[int] = []
    for subtype, comp_list in subtype_components.items():
        shuffled = list(comp_list)
        rng.shuffle(shuffled)
        n_total = len(shuffled)
        n_train = max(1, round(n_total * train_fraction)) if n_total > 1 else n_total
        for comp in shuffled[:n_train]:
            train_indices.extend(comp)
        for comp in shuffled[n_train:]:
            test_indices.extend(comp)

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
    components = _build_signature_components(examples)

    label_components: dict[int, list[list[int]]] = {}
    for comp in components.values():
        label_counts = Counter(examples[i].label for i in comp)
        label = max(label_counts, key=label_counts.get)
        label_components.setdefault(label, []).append(comp)

    train_test_indices: list[int] = []
    val_indices: list[int] = []

    for label in sorted(label_components.keys()):
        comp_list = list(label_components[label])
        rng.shuffle(comp_list)
        n = len(comp_list)
        n_train_test = max(1, round(n * train_test_fraction))

        for comp in comp_list[:n_train_test]:
            train_test_indices.extend(comp)
        for comp in comp_list[n_train_test:]:
            val_indices.extend(comp)
    
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
    def __init__(self, base_dataset: DirectionJsonlDataset, indices: list[int], vocab: dict[str, int], augment_fn: Callable | None = None):
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
        # augment_fn: Callable that accepts a DirectionExample and returns
        # either the original or an augmented DirectionExample. Only used
        # for on-the-fly training augmentation (e.g., spacer-subset deletion).
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
        "--test_size",
        "--test_within_train_fraction",
        dest="test_size",
        type=float,
        default=0.0,
        help="Fraction of the full dataset to hold out as the final test set (e.g. 0.1 for 10%%).",
    )
    parser.add_argument(
        "--cv_folds",
        type=int,
        default=0,
        help="If >1, run a single CV fold split on the development pool (train+val), keeping test untouched.",
    )
    parser.add_argument(
        "--cv_fold_index",
        type=int,
        default=0,
        help="Validation fold index to use when --cv_folds > 1 (0-based).",
    )
    parser.add_argument(
        "--augment_subarrays",
        action="store_true",
        help="If set, perform spacer-subset augmentation for train and validation splits (test untouched).",
    )
    parser.add_argument(
        "--augment_subarrays_prob",
        type=float,
        default=1.0,
        help="Per-example probability to apply spacer-subset augmentation (0.0-1.0).",
    )
    parser.add_argument(
        "--augment_subarrays_mode",
        type=str,
        default="enumerate",
        choices=["random", "enumerate"],
        help="Augmentation mode: 'random' (per-sample random subset) or 'enumerate' (add all subarrays to train set).",
    )
    parser.add_argument(
        "--augment_subarrays_min_spacers",
        type=int,
        default=2,
        help="Minimum number of spacers to keep when enumerating subarrays (default 2).",
    )
    parser.add_argument(
        "--augment_subarrays_max_per_array",
        type=int,
        default=256,
        help="Maximum number of augmented subarrays to add per original array in enumerate mode (default 256; <=0 means unlimited).",
    )
    parser.add_argument(
        "--augment_subarrays_enumerate_fast",
        action="store_true",
        help="Skip diversity computation in enumerate mode; randomly sample max_per_array subarrays instead (faster, less diverse).",
    )
    parser.add_argument(
        "--plot_test_curve",
        action="store_true",
        help="If set and a test set is present, evaluate the model on the test set once per epoch and include test loss as a third line in the training curves plot.",
    )
    args = parser.parse_args()

    _require_torch()

    base_dataset = DirectionJsonlDataset(args.jsonl, include_flanks=args.include_flanks)
    base_len = len(base_dataset.records)
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

    use_explicit_test_holdout = args.test_size and 0.0 < args.test_size < 1.0
    if args.test_size and not (0.0 < args.test_size < 1.0):
        raise ValueError("test_size must be in (0.0, 1.0)")

    if use_explicit_test_holdout:
        dev_indices, test_indices = stratified_holdout_by_mode(
            base_dataset.records,
            seed=args.seed,
            holdout_fraction=args.test_size,
            stratify_mode=stratify_mode,
        )
    else:
        # Legacy behavior when no explicit final test holdout is requested.
        if stratify_mode == "label":
            splits = stratified_train_test_and_val_by_label(
                base_dataset.records, seed=args.seed, train_test_fraction=0.8
            )
            train_indices = splits["train_test"]
            val_indices = splits["val"]
            test_indices = []
            dev_indices = sorted(set(list(train_indices) + list(val_indices)))
        elif stratify_mode == "cas_subtype":
            splits = stratified_split_by_cas_subtype(
                base_dataset.records, seed=args.seed, train_fraction=0.8, test_fraction=0.1
            )
            train_indices = splits["train"]
            val_indices = splits["val"]
            test_indices = splits["test"]
            dev_indices = sorted(set(list(train_indices) + list(val_indices)))
        else:
            splits = stratified_train_test_and_val_by_cas_subtype_and_label(
                base_dataset.records, seed=args.seed, train_test_fraction=0.8
            )
            train_indices = splits["train_test"]
            val_indices = splits["val"]
            test_indices = []
            dev_indices = sorted(set(list(train_indices) + list(val_indices)))

    train_indices: list[int]
    test_indices = list(test_indices)
    if use_explicit_test_holdout:
        if args.cv_folds > 1:
            if not (0 <= args.cv_fold_index < args.cv_folds):
                raise ValueError(
                    f"cv_fold_index must be in [0, {args.cv_folds - 1}] when cv_folds={args.cv_folds}"
                )

            cv_folds = build_cv_folds_by_signature(
                examples=base_dataset.records,
                pool_indices=dev_indices,
                n_folds=args.cv_folds,
                seed=args.seed,
                stratify_mode=stratify_mode,
            )
            val_indices = cv_folds[args.cv_fold_index]
            train_indices = [idx for j, fold in enumerate(cv_folds) if j != args.cv_fold_index for idx in fold]

            print(
                f"CV mode: folds={args.cv_folds}, using fold {args.cv_fold_index} as validation "
                f"(dev pool={len(dev_indices)}; train={len(train_indices)}, val={len(val_indices)}, "
                f"test={len(test_indices)})"
            )
        else:
            train_indices, val_indices = split_dev_pool_by_mode(
                examples=base_dataset.records,
                pool_indices=dev_indices,
                seed=args.seed,
                stratify_mode=stratify_mode,
            )
    elif args.cv_folds > 1:
        if not (0 <= args.cv_fold_index < args.cv_folds):
            raise ValueError(
                f"cv_fold_index must be in [0, {args.cv_folds - 1}] when cv_folds={args.cv_folds}"
            )

        dev_indices = sorted(set(list(train_indices) + list(val_indices)))
        cv_folds = build_cv_folds_by_signature(
            examples=base_dataset.records,
            pool_indices=dev_indices,
            n_folds=args.cv_folds,
            seed=args.seed,
            stratify_mode=stratify_mode,
        )
        val_indices = cv_folds[args.cv_fold_index]
        train_indices = [idx for j, fold in enumerate(cv_folds) if j != args.cv_fold_index for idx in fold]

        print(
            f"CV mode: folds={args.cv_folds}, using fold {args.cv_fold_index} as validation "
            f"(dev pool={len(dev_indices)}; train={len(train_indices)}, val={len(val_indices)}, "
            f"test={(len(test_indices) if test_indices else 0)})"
        )

    augment_fn = None
    if getattr(args, "augment_subarrays", False):
        mode = args.augment_subarrays_mode
        min_spacers = max(1, args.augment_subarrays_min_spacers)
        max_per_array = args.augment_subarrays_max_per_array

        print(
            "Augmentation: spacer deletion enabled for train and validation "
            f"(test untouched; mode={mode}, min_spacers={min_spacers}, "
            f"max_per_array={max_per_array if max_per_array > 0 else 'unlimited'})"
        )

        if mode == "random":
            # Random mode: on-the-fly augmentation during training, not materialized
            augment_fn = make_subarray_augment_fn(prob=args.augment_subarrays_prob, seed=args.seed)
            print(f"Augmentation: random subarray deletion (on-the-fly) enabled for train and validation")
        else:
            # Enumerate mode: materialize diverse subarrays upfront for train and val
            seen_signatures = {
                _example_signature(example)
                for example in base_dataset.records
            }

            train_new_indices, _ = _materialize_subarray_augmentations(
                base_dataset=base_dataset,
                source_indices=list(train_indices),
                seen_signatures=seen_signatures,
                seed=args.seed,
                mode=mode,
                prob=args.augment_subarrays_prob,
                min_spacers=min_spacers,
                max_per_array=max_per_array,
                split_name="train",
                use_diversity=not args.augment_subarrays_enumerate_fast,
            )
            train_indices = list(train_indices) + train_new_indices

            val_new_indices, _ = _materialize_subarray_augmentations(
                base_dataset=base_dataset,
                source_indices=list(val_indices),
                seen_signatures=seen_signatures,
                seed=args.seed + 1,
                mode=mode,
                prob=args.augment_subarrays_prob,
                min_spacers=min_spacers,
                max_per_array=max_per_array,
                split_name="val",
                use_diversity=not args.augment_subarrays_enumerate_fast,
            )
            val_indices = list(val_indices) + val_new_indices

    train_dataset = DirectionTorchDataset(base_dataset, train_indices, vocab, augment_fn=augment_fn)
    val_dataset = DirectionTorchDataset(base_dataset, val_indices, vocab, augment_fn=augment_fn)
    test_dataset = DirectionTorchDataset(base_dataset, test_indices, vocab) if test_indices else None

    print(f"Split sizes: train={len(train_dataset)}, val={len(val_dataset)}, test={(len(test_dataset) if test_dataset else 0)}")

    train_label_counts = Counter(base_dataset.records[i].label for i in train_indices)
    val_label_counts = Counter(base_dataset.records[i].label for i in val_indices)
    print(f"Label distribution train={dict(train_label_counts)} val={dict(val_label_counts)}")
    if test_indices:
        test_label_counts = Counter(base_dataset.records[i].label for i in test_indices)
        print(f"Label distribution test={dict(test_label_counts)}")

    train_type_n, train_type_counts = summarize_cas_subtypes(base_dataset.records, train_indices)
    val_type_n, val_type_counts = summarize_cas_subtypes(base_dataset.records, val_indices)
    print(f"CRISPR types train (unique={train_type_n}): {train_type_counts}")
    print(f"CRISPR types val (unique={val_type_n}): {val_type_counts}")
    if test_indices:
        test_type_n, test_type_counts = summarize_cas_subtypes(base_dataset.records, test_indices)
        print(f"CRISPR types test (unique={test_type_n}): {test_type_counts}")

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
    test_losses = []
    
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        val_metrics = evaluate(model, val_loader, loss_fn, device)
        val_loss = val_metrics["loss"]
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        # Optionally evaluate on the held-out test set once per epoch for plotting
        test_loss = float("nan")
        if getattr(args, "plot_test_curve", False) and test_loader is not None:
            test_metrics_epoch = evaluate(model, test_loader, loss_fn, device)
            test_loss = test_metrics_epoch.get("loss", float("nan"))
            test_losses.append(test_loss)
        
        # if validation loss improves, checkpoint the model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = {k: v.cpu() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
        
        es_marker = " (BEST)" if patience_counter == 0 else (" (STOP)" if patience_counter >= args.early_stopping_patience else "")
        timestamp = datetime.now().strftime('%H:%M:%S')
        print(
            (
                "[{timestamp}] epoch={epoch} train_loss={train_loss:.4f} val_loss={val_loss:.4f} test_loss={test_loss:.4f}{es_marker} "
                "val_accuracy={val_accuracy:.4f} val_f1={val_f1:.4f} "
                "val_pos_rate={val_pos_rate:.4f} val_pred_pos_rate={val_pred_pos_rate:.4f} "
                "val_majority_baseline_acc={val_baseline:.4f}"
            ).format(
                timestamp=timestamp,
                epoch=epoch,
                train_loss=train_loss,
                val_loss=val_loss,
                test_loss=test_loss,
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
        if getattr(args, "plot_test_curve", False) and test_loader is not None and len(test_losses) > 0:
            # Align test losses to the same epoch x-axis. If for any reason test_losses
            # is shorter than train_losses (shouldn't be), pad with NaN so plotting
            # uses the same epoch indices for all curves.
            test_plot = [test_losses[i] if i < len(test_losses) else float('nan') for i in range(len(train_losses))]
            ax.plot(epochs_range, test_plot, marker='^', label='Test Loss', linewidth=2)
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
