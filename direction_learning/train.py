"""Main training CLI for CRISPR direction prediction transformer.

Loads JSONL dataset, performs stratified splits, applies augmentations,
trains transformer with validation monitoring, and reports metrics.
"""
from __future__ import annotations

import argparse
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

try:
    import torch
except ModuleNotFoundError:
    torch = None

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    plt = None

# Import from new modular structure
from .augmentation import (
    materialize_reverse_complement_augmentation,
    materialize_subarray_augmentations,
    example_signature,
    build_test_similarity_index,
    make_subarray_augment_fn,
    make_subarray_augment_fn_with_similarity_filter,
)
from .data import (
    split_dev_pool_by_mode,
    stratified_holdout_by_mode,
    stratified_train_test_and_val_by_label,
    stratified_split_by_cas_subtype,
    stratified_train_test_and_val_by_cas_subtype_and_label,
    build_cv_folds_by_signature,
    DirectionTorchDataset,
    build_dataloader,
)
from .training import (
    train_one_epoch,
    evaluate,
    evaluate_per_subtype,
)
from .utils import (
    require_torch,
    timestamp,
    print_ts,
    summarize_cas_subtypes,
)
from .dataset import (
    DirectionJsonlDataset,
    build_vocab_from_jsonl,
)
from .model import build_model


def main() -> int:
    """Train the transformer.
    
    Loads JSONL dataset, performs stratified split by CRISPR subtype to balance
    train/val/test distributions, then trains for specified epochs with validation
    monitoring. Implements early stopping and checkpoints the best model by validation loss.
    """
    parser = argparse.ArgumentParser(
        description="Train a CRISPR direction transformer on the agreed-only JSONL dataset."
    )
    parser.add_argument(
        "--jsonl",
        default="output_dataset/direction_training_dataset.jsonl"
    )
    parser.add_argument("--include_flanks", action="store_true")
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size for training (default 16)."
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Maximum number of training epochs (default 5)."
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=3e-4,
        help="Learning rate for AdamW optimizer (default 3e-4)."
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=1e-5,
        help="L2 regularization strength (default 1e-5)."
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.1,
        help="Dropout rate for regularization (default 0.1)."
    )
    parser.add_argument(
        "--positional_encoding",
        type=str,
        default="absolute",
        choices=["absolute", "alibi", "rope"],
        help="Positional encoding to use for spacer order (default: absolute).",
    )
    parser.add_argument(
        "--pooling_strategy",
        type=str,
        default="mean",
        choices=["mean", "max", "attention", "learnable"],
        help="Sequence pooling strategy to use in encoder (default: mean).",
    )
    parser.add_argument(
        "--activation",
        type=str,
        default="gelu",
        choices=["gelu", "relu"],
        help="Activation function to use in transformer feedforward (default: gelu).",
    )
    parser.add_argument(
        "--reverse_complement_mode",
        type=str,
        default="none",
        choices=["none", "before", "after", "initial_only"],
        help=(
            "When and how to apply reverse-complement augmentation to train/val (test always stays untouched):\n"
            "  none: Do not apply reverse-complement augmentation (default).\n"
            "  before: Add reverse complements before subarray augmentation; augment all including RC examples.\n"
            "  after: Apply subarray augmentation first, then add reverse complements of all resulting arrays.\n"
            "  initial_only: Apply subarray augmentation, then add reverse complements only of the initial (non-augmented) arrays."
        ),
    )
    parser.add_argument(
        "--early_stopping_patience",
        type=int,
        default=3,
        help="Stop if val_loss doesn't improve for N epochs (default 3)."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default 42)."
    )
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
        help="Use combined stratification on both cas_subtype and label. Overrides --stratify_by when provided.",
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
        "--weighted_sampling",
        action="store_true",
        help="Enable weighted sampling to upweight under-represented strata instead of augmenting as much.",
    )
    parser.add_argument(
        "--weighted_sampling_by",
        type=str,
        default="cas_subtype",
        choices=["cas_subtype", "label"],
        help="Which key to base sampling weights on (default: cas_subtype).",
    )
    parser.add_argument(
        "--weighted_sampling_alpha",
        type=float,
        default=1.0,
        help="Aggressiveness exponent for inverse-frequency weighting (default 1.0).",
    )
    parser.add_argument(
        "--weighted_sampling_max_weight",
        type=float,
        default=10.0,
        help="Maximum allowed sample weight to avoid extreme oversampling (default 10.0).",
    )
    parser.add_argument(
        "--augment_subtypes_balance",
        action="store_true",
        help=(
            "Materialize enumerate-mode augmentations to balance cas_subtype counts in train/val "
            "so each subtype has the same number of examples (uses existing augmentation flags; "
            "requires --augment_subarrays with enumerate mode)."
        ),
    )
    parser.add_argument(
        "--augment_subtypes_balance_target",
        type=int,
        default=0,
        help=(
            "Optional target count per subtype when --augment_subtypes_balance is set. "
            "Default 0 means use the current maximum subtype count per split."
        ),
    )
    parser.add_argument(
        "--aug_similarity",
        type=str,
        default="",
        choices=["", "jaccard", "overlap"],
        help="Optional test-set similarity safeguard for augmented subarrays. Choose 'jaccard' or 'overlap' to reject candidates that are too close to any held-out test example. Leave empty to keep current behavior.",
    )
    parser.add_argument(
        "--aug_similarity_min_distance",
        type=float,
        default=0.30,
        help="Minimum allowed distance to the nearest test example when --aug_similarity is set (higher is stricter).",
    )
    parser.add_argument(
        "--plot_test_curve",
        action="store_true",
        help="If set and a test set is present, evaluate the model on the test set once per epoch and include test loss as a third line in the training curves plot.",
    )
    args = parser.parse_args()

    require_torch()
    
    augmentation_elapsed = 0.0
    training_elapsed = 0.0

    base_dataset = DirectionJsonlDataset(args.jsonl, include_flanks=args.include_flanks)
    base_len = len(base_dataset.records)
    vocab = build_vocab_from_jsonl(args.jsonl)

    dataset_label_counts = Counter(example.label for example in base_dataset.records)
    if len(dataset_label_counts) < 2:
        raise ValueError(
            f"Dataset contains only one class label. Label counts: {dict(dataset_label_counts)}"
        )

    stratify_mode = (
        "cas_subtype_and_label" if args.stratify_by_cas_subtype_and_label else args.stratify_by
    )

    # Determine if explicit test holdout is being used
    use_explicit_test_holdout = args.test_size and 0.0 < args.test_size < 1.0
    if args.test_size and not (0.0 < args.test_size < 1.0):
        raise ValueError("test_size must be in (0.0, 1.0)")

    # Perform splitting
    if use_explicit_test_holdout:
        dev_indices, test_indices = stratified_holdout_by_mode(
            base_dataset.records,
            seed=args.seed,
            holdout_fraction=args.test_size,
            stratify_mode=stratify_mode,
        )
    else:
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

    # Setup similarity filtering if requested
    similarity_metric = (args.aug_similarity or "").strip().lower()
    use_similarity_filter = bool(similarity_metric)
    if use_similarity_filter and not test_indices:
        print("Augmentation similarity safeguard requested, but no test split is present; similarity filtering will be skipped.")

    test_signatures = None
    test_signatures_by_idx = None
    test_token_sets = None
    inverted_index = None
    if use_similarity_filter and test_indices:
        test_signatures = {
            example_signature(base_dataset.records[idx])
            for idx in test_indices
        }
        test_signatures_by_idx = {
            idx: example_signature(base_dataset.records[idx]) for idx in test_indices
        }
        test_token_sets, inverted_index = build_test_similarity_index(
            base_dataset.records, test_indices
        )

    # Handle CV and final train/val split
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
            train_indices = [
                idx for j, fold in enumerate(cv_folds) if j != args.cv_fold_index for idx in fold
            ]
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
        train_indices = [
            idx for j, fold in enumerate(cv_folds) if j != args.cv_fold_index for idx in fold
        ]
        print(
            f"CV mode: folds={args.cv_folds}, using fold {args.cv_fold_index} as validation "
            f"(dev pool={len(dev_indices)}; train={len(train_indices)}, val={len(val_indices)}, "
            f"test={(len(test_indices) if test_indices else 0)})"
        )

    # Handle reverse-complement in "before" mode
    rc_mode = getattr(args, "reverse_complement_mode", "none")
    initial_train_indices = None
    initial_val_indices = None

    if rc_mode == "before":
        print_ts("Augmentation: reverse-complement mode=before (add RC before subarray augmentation; augment all)")
        train_rc_indices, train_rc_stats = materialize_reverse_complement_augmentation(
            base_dataset=base_dataset,
            source_indices=list(train_indices),
            test_signatures=test_signatures,
            test_signatures_by_idx=test_signatures_by_idx,
            test_token_sets=test_token_sets,
            inverted_index=inverted_index,
            similarity_metric=similarity_metric or "jaccard",
            min_distance=args.aug_similarity_min_distance,
        )
        val_rc_indices, val_rc_stats = materialize_reverse_complement_augmentation(
            base_dataset=base_dataset,
            source_indices=list(val_indices),
            test_signatures=test_signatures,
            test_signatures_by_idx=test_signatures_by_idx,
            test_token_sets=test_token_sets,
            inverted_index=inverted_index,
            similarity_metric=similarity_metric or "jaccard",
            min_distance=args.aug_similarity_min_distance,
        )
        train_indices = list(train_indices) + train_rc_indices
        val_indices = list(val_indices) + val_rc_indices
        print_ts(
            "Augmentation: reverse-complement duplication summary "
            f"train_added={train_rc_stats['added']} train_blocked={train_rc_stats['blocked_similarity']} "
            f"val_added={val_rc_stats['added']} val_blocked={val_rc_stats['blocked_similarity']}"
        )
    elif rc_mode in ("after", "initial_only"):
        initial_train_indices = list(train_indices)
        initial_val_indices = list(val_indices)
        print_ts(f"Augmentation: reverse-complement mode={rc_mode} (will apply after subarray augmentation)")

    # Setup subarray augmentation
    augment_fn = None
    skip_standard_augment_for_balancing = (
        getattr(args, "augment_subarrays", False)
        and getattr(args, "augment_subtypes_balance", False)
        and args.augment_subarrays_mode == "enumerate"
    )

    augmentation_started_at = None

    if getattr(args, "augment_subarrays", False):
        augmentation_started_at = time.perf_counter()
        mode = args.augment_subarrays_mode
        min_spacers = max(1, args.augment_subarrays_min_spacers)
        max_per_array = args.augment_subarrays_max_per_array

        print_ts(
            "Augmentation: spacer deletion enabled for train and validation "
            f"(test untouched; mode={mode}, min_spacers={min_spacers}, "
            f"max_per_array={max_per_array if max_per_array > 0 else 'unlimited'})"
        )

        if mode == "random":
            if use_similarity_filter:
                augment_fn = make_subarray_augment_fn_with_similarity_filter(
                    prob=args.augment_subarrays_prob,
                    seed=args.seed,
                    max_attempts=5,
                    test_signatures=test_signatures,
                    test_signatures_by_idx=test_signatures_by_idx,
                    test_token_sets=test_token_sets,
                    inverted_index=inverted_index,
                    similarity_metric=similarity_metric,
                    min_distance=args.aug_similarity_min_distance,
                )
            else:
                augment_fn = make_subarray_augment_fn(
                    prob=args.augment_subarrays_prob, seed=args.seed
                )
            print_ts("Augmentation: random subarray deletion (on-the-fly) enabled for train and validation")
        elif not skip_standard_augment_for_balancing:
            seen_signatures = {
                example_signature(example) for example in base_dataset.records
            }

            train_new_indices, train_aug_stats = materialize_subarray_augmentations(
                base_dataset=base_dataset,
                source_indices=list(train_indices),
                seen_signatures=seen_signatures,
                test_signatures=test_signatures,
                test_signatures_by_idx=test_signatures_by_idx,
                test_token_sets=test_token_sets,
                inverted_index=inverted_index,
                seed=args.seed,
                mode=mode,
                prob=args.augment_subarrays_prob,
                min_spacers=min_spacers,
                max_per_array=max_per_array,
                split_name="train",
                use_diversity=not args.augment_subarrays_enumerate_fast,
                similarity_metric=similarity_metric or "jaccard",
                min_distance=args.aug_similarity_min_distance,
            )
            train_indices = list(train_indices) + train_new_indices

            val_new_indices, val_aug_stats = materialize_subarray_augmentations(
                base_dataset=base_dataset,
                source_indices=list(val_indices),
                seen_signatures=seen_signatures,
                test_signatures=test_signatures,
                test_signatures_by_idx=test_signatures_by_idx,
                test_token_sets=test_token_sets,
                inverted_index=inverted_index,
                seed=args.seed + 1,
                mode=mode,
                prob=args.augment_subarrays_prob,
                min_spacers=min_spacers,
                max_per_array=max_per_array,
                split_name="val",
                use_diversity=not args.augment_subarrays_enumerate_fast,
                similarity_metric=similarity_metric or "jaccard",
                min_distance=args.aug_similarity_min_distance,
            )
            val_indices = list(val_indices) + val_new_indices
        else:
            seen_signatures = {
                example_signature(example) for example in base_dataset.records
            }
            print_ts("Augmentation: skipping standard enumerate pass; will use subtype-aware balancing instead")

    # Subtype balancing augmentation
    if getattr(args, "augment_subtypes_balance", False):
        if not getattr(args, "augment_subarrays", False):
            print_ts("augment_subtypes_balance requested but --augment_subarrays not set; skipping balancing.")
        elif args.augment_subarrays_mode != "enumerate":
            print_ts("augment_subtypes_balance requires --augment_subarrays_mode enumerate; skipping balancing.")
        else:
            if 'seen_signatures' not in locals():
                seen_signatures = {example_signature(example) for example in base_dataset.records}

            print_ts("Augmentation: subtype-aware balancing—computing per-subtype targets and materializing only what's needed")

            min_spacers = max(1, args.augment_subarrays_min_spacers)
            max_per_array = args.augment_subarrays_max_per_array

            def _subtype_of(idx: int) -> str:
                return (base_dataset.records[idx].cas_subtype or "Unknown").strip() or "Unknown"

            train_size = len(train_indices)
            val_size = len(val_indices)
            val_to_train_ratio = val_size / train_size if train_size > 0 else 0.2
            print_ts(f"Augmentation: val/train ratio = {val_to_train_ratio:.3f}")

            for split_name in ["train", "val"]:
                split_indices = train_indices if split_name == "train" else val_indices
                counts = Counter(_subtype_of(i) for i in list(split_indices))
                if not counts:
                    continue

                base_target = args.augment_subtypes_balance_target or max(counts.values())
                if split_name == "val":
                    target = int(base_target * val_to_train_ratio)
                else:
                    target = base_target

                if target <= 0:
                    continue

                subtype_needs = {
                    subtype: target - cnt
                    for subtype, cnt in sorted(counts.items())
                    if cnt < target
                }

                if not subtype_needs:
                    print_ts(f"Balancing {split_name}: all subtypes already at or above target {target}")
                    continue

                print_ts(f"Balancing {split_name}: target={target}, subtype needs: {subtype_needs}")

                for subtype, needed in sorted(subtype_needs.items()):
                    print_ts(
                        f"Balancing {split_name}: augmenting subtype={subtype} "
                        f"(current={counts[subtype]}, need={needed} more)"
                    )

                    source_pool = [i for i in list(split_indices) if _subtype_of(i) == subtype]
                    if not source_pool:
                        print_ts(f"  No source examples for subtype {subtype}; skipping")
                        continue

                    added_total = 0
                    round_seed = args.seed + (0 if split_name == "train" else 1)
                    attempt = 0

                    while added_total < needed:
                        attempt += 1
                        print_ts(
                            f"  Round {attempt}: generating augmentations for subtype={subtype} "
                            f"(need {needed - added_total} more)"
                        )

                        new_indices, aug_stats = materialize_subarray_augmentations(
                            base_dataset=base_dataset,
                            source_indices=list(source_pool),
                            seen_signatures=seen_signatures,
                            test_signatures=test_signatures,
                            test_signatures_by_idx=test_signatures_by_idx,
                            test_token_sets=test_token_sets,
                            inverted_index=inverted_index,
                            seed=round_seed,
                            mode="enumerate",
                            prob=args.augment_subarrays_prob,
                            min_spacers=min_spacers,
                            max_per_array=max_per_array,
                            split_name=f"{split_name}_balance_{subtype}_r{attempt}",
                            use_diversity=not args.augment_subarrays_enumerate_fast,
                            similarity_metric=similarity_metric or "jaccard",
                            min_distance=args.aug_similarity_min_distance,
                            target_additions=needed - added_total,
                            balance_per_array=True,
                        )

                        if not new_indices:
                            print_ts(f"  Round {attempt}: no augmentations produced; stopped after adding {added_total}/{needed}")
                            break

                        split_indices.extend(new_indices)
                        added_total += len(new_indices)
                        counts[subtype] = counts.get(subtype, 0) + len(new_indices)
                        round_seed += 2
                        print_ts(f"  Round {attempt}: added {len(new_indices)} examples (total: {added_total}/{needed})")

                    print_ts(f"  Subtype={subtype} balancing complete: added {added_total}/{needed}, final_count={counts.get(subtype, 0)}")

    # Handle RC in "after" and "initial_only" modes
    if rc_mode == "after":
        print_ts("Augmentation: reverse-complement mode=after (augment first, then add RC of all augmented arrays)")
        train_rc_indices, train_rc_stats = materialize_reverse_complement_augmentation(
            base_dataset=base_dataset,
            source_indices=list(train_indices),
            test_signatures=test_signatures,
            test_signatures_by_idx=test_signatures_by_idx,
            test_token_sets=test_token_sets,
            inverted_index=inverted_index,
            similarity_metric=similarity_metric or "jaccard",
            min_distance=args.aug_similarity_min_distance,
        )
        val_rc_indices, val_rc_stats = materialize_reverse_complement_augmentation(
            base_dataset=base_dataset,
            source_indices=list(val_indices),
            test_signatures=test_signatures,
            test_signatures_by_idx=test_signatures_by_idx,
            test_token_sets=test_token_sets,
            inverted_index=inverted_index,
            similarity_metric=similarity_metric or "jaccard",
            min_distance=args.aug_similarity_min_distance,
        )
        train_indices = list(train_indices) + train_rc_indices
        val_indices = list(val_indices) + val_rc_indices
        print_ts(
            "Augmentation: reverse-complement duplication summary "
            f"train_added={train_rc_stats['added']} train_blocked={train_rc_stats['blocked_similarity']} "
            f"val_added={val_rc_stats['added']} val_blocked={val_rc_stats['blocked_similarity']}"
        )
    elif rc_mode == "initial_only":
        print_ts("Augmentation: reverse-complement mode=initial_only (augment first, then add RC only of initial arrays)")
        if initial_train_indices is not None:
            train_rc_indices, train_rc_stats = materialize_reverse_complement_augmentation(
                base_dataset=base_dataset,
                source_indices=initial_train_indices,
                test_signatures=test_signatures,
                test_signatures_by_idx=test_signatures_by_idx,
                test_token_sets=test_token_sets,
                inverted_index=inverted_index,
                similarity_metric=similarity_metric or "jaccard",
                min_distance=args.aug_similarity_min_distance,
            )
            train_indices = list(train_indices) + train_rc_indices
            print_ts(
                f"Augmentation: reverse-complement (initial_only, train) added={train_rc_stats['added']} "
                f"blocked={train_rc_stats['blocked_similarity']}"
            )
        if initial_val_indices is not None:
            val_rc_indices, val_rc_stats = materialize_reverse_complement_augmentation(
                base_dataset=base_dataset,
                source_indices=initial_val_indices,
                test_signatures=test_signatures,
                test_signatures_by_idx=test_signatures_by_idx,
                test_token_sets=test_token_sets,
                inverted_index=inverted_index,
                similarity_metric=similarity_metric or "jaccard",
                min_distance=args.aug_similarity_min_distance,
            )
            val_indices = list(val_indices) + val_rc_indices
            print_ts(
                f"Augmentation: reverse-complement (initial_only, val) added={val_rc_stats['added']} "
                f"blocked={val_rc_stats['blocked_similarity']}"
            )

    if augmentation_started_at is not None:
        augmentation_elapsed = time.perf_counter() - augmentation_started_at

    # Create datasets and dataloaders
    train_dataset = DirectionTorchDataset(base_dataset, train_indices, vocab, augment_fn=augment_fn)
    val_dataset = DirectionTorchDataset(base_dataset, val_indices, vocab, augment_fn=augment_fn)
    test_dataset = DirectionTorchDataset(base_dataset, test_indices, vocab) if test_indices else None

    print(
        f"Split sizes: train={len(train_dataset)}, val={len(val_dataset)}, "
        f"test={(len(test_dataset) if test_dataset else 0)}"
    )

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
            f"Train/validation split contains only a single class. "
            f"train={dict(train_label_counts)} val={dict(val_label_counts)}"
        )

    # Weighted sampling
    train_weights = None
    val_weights = None
    if getattr(args, "weighted_sampling", False):
        print("Weighted sampling enabled; computing per-sample weights...")

        if args.weighted_sampling_by == "cas_subtype":
            def _key_fn(i: int) -> str:
                return (base_dataset.records[i].cas_subtype or "Unknown").strip() or "Unknown"
        else:
            def _key_fn(i: int) -> str:
                return str(int(base_dataset.records[i].label))

        def _counts_for(indices: list[int]) -> Counter:
            c = Counter()
            for i in indices:
                if i < base_len:
                    c[_key_fn(i)] += 1
            if not c:
                for i in indices:
                    c[_key_fn(i)] += 1
            return c

        train_counts = _counts_for(train_indices)
        val_counts = _counts_for(val_indices)

        def _make_weights(indices: list[int], counts: Counter) -> list[float]:
            majority = max(counts.values()) if counts else 1
            alpha = float(getattr(args, "weighted_sampling_alpha", 1.0))
            max_w = float(getattr(args, "weighted_sampling_max_weight", 10.0))
            weight_map: dict[str, float] = {}
            for k, v in counts.items():
                ratio = majority / max(1, v)
                weight = min(max_w, ratio ** alpha)
                weight_map[k] = float(weight)

            weights: list[float] = []
            for i in indices:
                key = _key_fn(i)
                weights.append(weight_map.get(key, 1.0))

            if weights:
                mean_w = float(sum(weights)) / len(weights)
                if mean_w > 0:
                    weights = [w / mean_w for w in weights]
            return weights

        train_weights = _make_weights(train_indices, train_counts)
        val_weights = _make_weights(val_indices, val_counts)
        print(
            f"Weighted sampling: train weights mean={sum(train_weights)/len(train_weights):.3f} "
            f"max={max(train_weights):.3f} | "
            f"val weights mean={sum(val_weights)/len(val_weights):.3f} max={max(val_weights):.3f}"
        )

    train_loader = build_dataloader(
        train_dataset, batch_size=args.batch_size, shuffle=not bool(train_weights), weights=train_weights
    )
    val_loader = build_dataloader(
        val_dataset, batch_size=args.batch_size, shuffle=not bool(val_weights), weights=val_weights
    )
    test_loader = (
        build_dataloader(test_dataset, batch_size=args.batch_size, shuffle=False)
        if test_dataset
        else None
    )

    # Build model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    max_spacers_in_dataset = max((len(ex.spacers) for ex in base_dataset.records), default=64)
    model = build_model(
        vocab_size=len(vocab),
        include_flanks=args.include_flanks,
        max_spacers=max_spacers_in_dataset,
        dropout=args.dropout,
        positional_encoding=args.positional_encoding,
        pooling_strategy=args.pooling_strategy,
        activation=args.activation,
    ).to(device)
    print("Model architecture:")
    print(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    # Training loop
    best_val_loss = float("inf")
    patience_counter = 0
    best_model_state = None
    train_losses = []
    val_losses = []
    test_losses = []

    training_started_at = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        val_metrics = evaluate(model, val_loader, loss_fn, device)
        val_loss = val_metrics["loss"]
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if getattr(args, "plot_test_curve", False) and test_loader is not None:
            test_metrics_epoch = evaluate(model, test_loader, loss_fn, device)
            test_loss = test_metrics_epoch.get("loss", float("nan"))
            test_losses.append(test_loss)
        else:
            test_loss = float("nan")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = {k: v.cpu() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1

        es_marker = (
            " (BEST)"
            if patience_counter == 0
            else (" (STOP)" if patience_counter >= args.early_stopping_patience else "")
        )
        timestamp = datetime.now().strftime('%H:%M:%S')
        print(
            f"[{timestamp}] epoch={epoch} train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"test_loss={test_loss:.4f}{es_marker} val_accuracy={val_metrics['accuracy']:.4f} "
            f"val_f1={val_metrics['f1']:.4f} val_pos_rate={val_metrics['positive_rate_true']:.4f} "
            f"val_pred_pos_rate={val_metrics['positive_rate_pred']:.4f} "
            f"val_majority_baseline_acc={val_metrics['majority_baseline_accuracy']:.4f}"
        )

        if patience_counter >= args.early_stopping_patience:
            print(f"Early stopping: validation loss did not improve for {args.early_stopping_patience} epochs.")
            break

    training_elapsed = time.perf_counter() - training_started_at

    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print("Restored best model (lowest validation loss).")

    # Plot training curves
    if plt is not None:
        fig, ax = plt.subplots(figsize=(12, 7))
        epochs_range = range(1, len(train_losses) + 1)
        ax.plot(epochs_range, train_losses, marker='o', label='Train Loss', linewidth=2)
        ax.plot(epochs_range, val_losses, marker='s', label='Val Loss', linewidth=2)
        if getattr(args, "plot_test_curve", False) and test_loader is not None and len(test_losses) > 0:
            test_plot = [
                test_losses[i] if i < len(test_losses) else float('nan')
                for i in range(len(train_losses))
            ]
            ax.plot(epochs_range, test_plot, marker='^', label='Test Loss', linewidth=2)
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Loss', fontsize=12)
        ax.set_title('Training vs Validation Loss', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)

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
        ax.text(
            0.98, 0.97, params_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
            family='monospace'
        )

        output_path = Path("/tmp/training_curves.png")
        fig.savefig(str(output_path), dpi=100, bbox_inches='tight')
        print(f"Training curves saved to {output_path}")
        plt.close(fig)
    else:
        print("matplotlib not available; skipping training curves visualization.")

    # Test evaluation
    if test_loader is not None:
        test_metrics = evaluate(model, test_loader, loss_fn, device)
        auc_text = (
            f"{test_metrics['auc']:.4f}" if not (test_metrics['auc'] != test_metrics['auc']) else "nan"
        )
        aupr_text = (
            f"{test_metrics['aupr']:.4f}" if not (test_metrics['aupr'] != test_metrics['aupr']) else "nan"
        )
        print(
            f"test_loss={test_metrics['loss']:.4f} test_accuracy={test_metrics['accuracy']:.4f} "
            f"auc={auc_text} aupr={aupr_text} precision={test_metrics['precision']:.4f} "
            f"recall={test_metrics['recall']:.4f} f1={test_metrics['f1']:.4f}"
        )

        # Confusion matrix
        if plt is not None:
            try:
                import numpy as np
                from sklearn.metrics import confusion_matrix
                from matplotlib.colors import LinearSegmentedColormap

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

                y_pred = (np.array(all_probs) >= 0.5).astype(int)
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

                output_path = Path("/tmp/confusion_matrix.png")
                fig.savefig(str(output_path), dpi=100, bbox_inches='tight')
                print(f"Confusion matrix saved to {output_path}")
                plt.close(fig)
            except Exception as e:
                print(f"Could not generate confusion matrix: {e}")

        # Per-subtype metrics
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

    total_elapsed = augmentation_elapsed + training_elapsed
    print_ts(
        f"Timing summary: augmentation_time={augmentation_elapsed:.2f}s "
        f"training_time={training_elapsed:.2f}s total_time={total_elapsed:.2f}s"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
