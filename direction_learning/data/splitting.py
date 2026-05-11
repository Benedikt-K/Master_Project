"""Stratified data splitting for CRISPR array examples.

Provides flexible stratification strategies to ensure balanced train/val/test splits
while preserving CRISPR subtype diversity and binary label balance.
"""

from __future__ import annotations

import random
from collections import Counter
from typing import TYPE_CHECKING, Any

from ..utils import _build_signature_components

if TYPE_CHECKING:
    from ..dataset import DirectionExample


def split_groups(
    examples: list[DirectionExample],
    seed: int = 13,
    train_fraction: float = 0.7,
    val_fraction: float = 0.15,
) -> dict[str, list[int]]:
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


def stratified_holdout_by_mode(
    examples: list[DirectionExample],
    seed: int,
    holdout_fraction: float,
    stratify_mode: str,
) -> tuple[list[int], list[int]]:
    """Split indices into development/train and held-out test sets.

    Exact spacer/repeat signatures stay together, and the stratification key
    follows the selected mode (label, cas_subtype, or both).
    
    Args:
        examples: List of DirectionExample objects.
        seed: Random seed for reproducibility.
        holdout_fraction: Fraction of examples to hold out for test.
        stratify_mode: One of "label", "cas_subtype", "cas_subtype_and_label".
        
    Returns:
        (dev_indices, test_indices) where dev will be split further into train/val.
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
    """Split a development pool into train and validation indices.
    
    Args:
        examples: Full list of DirectionExample objects.
        pool_indices: Indices in the development pool to split.
        seed: Random seed.
        stratify_mode: Stratification strategy.
        
    Returns:
        (train_indices, val_indices)
    """
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
