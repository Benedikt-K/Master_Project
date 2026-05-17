from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    import torch
except ModuleNotFoundError:
    torch = None

try:
    import optuna
    from optuna.exceptions import TrialPruned
except ModuleNotFoundError:
    optuna = None
    TrialPruned = RuntimeError

from .dataset import DirectionJsonlDataset, build_vocab_from_jsonl
from .model import DirectionTransformerConfig, build_model
from .data import (
    DirectionTorchDataset,
    build_dataloader,
    split_groups,
    split_dev_pool_by_mode,
    stratified_holdout_by_mode,
)
from .augmentation import (
    make_subarray_augment_fn,
    make_subarray_augment_fn_with_similarity_filter,
    build_test_similarity_index,
    materialize_subarray_augmentations,
    example_signature,
)
from .training import (
    evaluate,
    train_one_epoch,
)
from .tokenizers.cnn_tokenizer import CNNTokenizer, CNNTokConfig


def _require_dependencies() -> None:
    if torch is None:
        raise ModuleNotFoundError("PyTorch is required to run direction_learning.hpo")
    if optuna is None:
        raise ModuleNotFoundError(
            "Optuna is required to run direction_learning.hpo. Install it with `pip install optuna`."
        )


def _timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _print(message: str) -> None:
    print(f"[{_timestamp()}] {message}")


def _choose_num_heads(transformer_dim: int, trial: Any) -> int:
    #valid_heads = [head for head in (1, 2, 4, 8) if transformer_dim % head == 0]
    valid_heads = [head for head in (8, 8) if transformer_dim % head == 0]
    if not valid_heads:
        return 1
    return trial.suggest_categorical("num_heads", valid_heads)


def _build_sampler(args: argparse.Namespace) -> Any:
    if args.sampler == "random":
        return optuna.samplers.RandomSampler(seed=args.seed)
    return optuna.samplers.TPESampler(
        seed=args.seed,
        n_startup_trials=args.tpe_startup_trials,
        n_ei_candidates=args.tpe_ei_candidates,
        multivariate=args.tpe_multivariate,
    )


def _sample_config(
    trial: Any,
    vocab_size: int,
    max_spacers: int,
    include_flanks: bool,
    forced_spacer_dim: int | None = None,
    forced_use_cls: bool | None = None,
) -> DirectionTransformerConfig:
    #positional_encoding = trial.suggest_categorical("positional_encoding", ["absolute", "alibi", "rope"])
    positional_encoding = trial.suggest_categorical("positional_encoding", ["alibi", "rope"])
    #pooling_strategy = trial.suggest_categorical("pooling_strategy", ["mean", "max", "attention", "learnable"])
    pooling_strategy = trial.suggest_categorical("pooling_strategy", ["mean", "attention"])
    #token_dim = trial.suggest_categorical("token_dim", [32, 48, 64, 96, 128])
    token_dim = trial.suggest_categorical("token_dim", [32, 48])
    #spacer_dim = trial.suggest_categorical("spacer_dim", [64, 96, 128, 160, 192, 256])
    spacer_dim = (
        int(forced_spacer_dim)
        if forced_spacer_dim is not None
        else trial.suggest_categorical("spacer_dim", [192, 256, 512])
    )
    #transformer_dim = trial.suggest_categorical("transformer_dim", [64, 96, 128, 160, 192, 256])
    transformer_dim = trial.suggest_categorical("transformer_dim", [192, 256, 512])
    num_heads = _choose_num_heads(transformer_dim, trial)
    #num_heads = 8
    #num_layers = trial.suggest_int("num_layers", 1, 6)
    num_layers = trial.suggest_int("num_layers", 1, 3)
    #dropout = trial.suggest_float("dropout", 0.0, 0.5)
    dropout = trial.suggest_float("dropout", 0.01, 0.10)
    feedforward_multiplier = trial.suggest_categorical("feedforward_multiplier", [2, 4, 6])
    feedforward_dim = transformer_dim * feedforward_multiplier
    #activation = trial.suggest_categorical("activation", ["gelu", "relu"])
    activation = trial.suggest_categorical("activation", ["gelu"])

    # If caller did not force CLS on/off, let Optuna sample it per-trial.
    if forced_use_cls is None:
        use_cls = trial.suggest_categorical("use_cls_token", [False, True])
    else:
        use_cls = bool(forced_use_cls)

    return DirectionTransformerConfig(
        vocab_size=vocab_size,
        token_dim=token_dim,
        spacer_dim=spacer_dim,
        transformer_dim=transformer_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        feedforward_dim=feedforward_dim,
        dropout=dropout,
        activation=activation,
        max_spacers=max_spacers,
        include_flanks=include_flanks,
        positional_encoding=positional_encoding,
        pooling_strategy=pooling_strategy,
        use_cls_token=bool(use_cls),
    )


def _sample_tokenizer_settings(trial: Any, args: argparse.Namespace, vocab_size: int) -> tuple[str, Any | None, int | None]:
    """Sample tokenizer settings for one trial.

    Returns:
        tokenizer name, optional CNN tokenizer instance, optional CNN output dim.
    """
    if args.tokenizer == "auto":
        tokenizer_name = trial.suggest_categorical("tokenizer", ["default", "cnn"])
    else:
        tokenizer_name = args.tokenizer

    if tokenizer_name != "cnn":
        return tokenizer_name, None, None

    cnn_output_dim = trial.suggest_categorical("cnn_output_dim", [64, 96, 128, 160, 192, 256])
    cnn_embed_dim = trial.suggest_categorical("cnn_embed_dim", [8, 16, 32])
    cnn_filters = trial.suggest_categorical("cnn_filters", [32, 64, 96, 128])
    cnn_kernels_str = trial.suggest_categorical(
        "cnn_kernels",
        ["3,5", "3,5,7", "3,5,9", "5,9", "3,7"],
    )
    cnn_pooling = trial.suggest_categorical("cnn_pooling", ["max", "avg"])
    cnn_activation = trial.suggest_categorical("cnn_activation", ["relu", "gelu"])

    cnn_kernels = [int(k.strip()) for k in cnn_kernels_str.split(",") if k.strip()]
    cnn_cfg = CNNTokConfig(
        output_dim=cnn_output_dim,
        embed_dim=cnn_embed_dim,
        filters=cnn_filters,
        kernels=cnn_kernels,
        pooling=cnn_pooling,
        activation=cnn_activation,
    )
    return tokenizer_name, CNNTokenizer(cnn_cfg, vocab_size=vocab_size), cnn_output_dim


def _build_loaders(dataset: DirectionJsonlDataset, vocab: dict[str, int], train_indices: list[int], val_indices: list[int], batch_size: int):
    train_dataset = DirectionJsonlDataset(dataset.jsonl_path, include_flanks=dataset.include_flanks)
    val_dataset = DirectionJsonlDataset(dataset.jsonl_path, include_flanks=dataset.include_flanks)
    train_loader = build_dataloader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = build_dataloader(val_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader


def run_study(args: argparse.Namespace) -> int:
    _require_dependencies()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    jsonl_path = Path(args.jsonl)
    if not jsonl_path.exists():
        raise FileNotFoundError(f"JSONL dataset not found: {jsonl_path}")

    dataset = DirectionJsonlDataset(jsonl_path, include_flanks=args.include_flanks)
    vocab = build_vocab_from_jsonl(jsonl_path)
    base_len = len(dataset.records)
    if base_len == 0:
        raise ValueError("Dataset is empty")

    stratify_mode = args.stratify_by
    use_explicit_test_holdout = args.test_size and 0.0 < args.test_size < 1.0
    if args.test_size and not (0.0 < args.test_size < 1.0):
        raise ValueError("test_size must be in (0.0, 1.0)")

    if use_explicit_test_holdout:
        dev_indices, test_indices = stratified_holdout_by_mode(
            dataset.records,
            seed=args.seed,
            holdout_fraction=args.test_size,
            stratify_mode=stratify_mode,
        )
        train_indices, val_indices = split_dev_pool_by_mode(
            dataset.records,
            pool_indices=dev_indices,
            seed=args.seed,
            stratify_mode=stratify_mode,
        )
    else:
        splits = split_groups(dataset.records, seed=args.seed, train_fraction=args.train_fraction, val_fraction=args.val_fraction)
        train_indices = splits["train"]
        val_indices = splits["val"]
        test_indices = splits["test"]

    _print(
        f"Loaded dataset={jsonl_path} records={base_len} train={len(train_indices)} val={len(val_indices)} test={len(test_indices)} stratify_by={stratify_mode}"
    )

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    loss_fn = torch.nn.BCEWithLogitsLoss()

    def objective(trial: Any) -> float:
        trial_seed = args.seed + trial.number * 17
        random.seed(trial_seed)
        torch.manual_seed(trial_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(trial_seed)

        # Work on local copies of split indices so we can materialize augmentations
        # per-trial without mutating the outer-scope indices used for final retrain.
        train_idx = list(train_indices)
        val_idx = list(val_indices)
        test_idx = list(test_indices)

        tokenizer_name, cnn_tok_trial, cnn_output_dim = _sample_tokenizer_settings(
            trial,
            args,
            vocab_size=len(vocab),
        )
        # If CLI explicitly set --enable_cls_token then force CLS on for all trials; otherwise allow HPO to sample it.
        forced_use_cls_val = True if getattr(args, "enable_cls_token", False) else None
        config = _sample_config(
            trial,
            vocab_size=len(vocab),
            max_spacers=max((len(ex.spacers) for ex in dataset.records), default=64),
            include_flanks=args.include_flanks,
            forced_spacer_dim=cnn_output_dim,
            forced_use_cls=forced_use_cls_val,
        )
        model = build_model(
            vocab_size=config.vocab_size,
            include_flanks=config.include_flanks,
            max_spacers=config.max_spacers,
            dropout=config.dropout,
            token_dim=config.token_dim,
            spacer_dim=config.spacer_dim,
            transformer_dim=config.transformer_dim,
            num_heads=config.num_heads,
            num_layers=config.num_layers,
            feedforward_dim=config.feedforward_dim,
            activation=config.activation,
            positional_encoding=config.positional_encoding,
            pooling_strategy=config.pooling_strategy,
            use_cls_token=config.use_cls_token,
        ).to(device)
        # Persist whether this trial uses the CLS token for auditing
        try:
            trial.set_user_attr("use_cls_token", bool(config.use_cls_token))
        except Exception:
            pass

        #optimizer_name = trial.suggest_categorical("optimizer", ["adamw", "adam", "sgd"])
        optimizer_name = trial.suggest_categorical("optimizer", ["adamw", "adam"])
        #lr = trial.suggest_float("lr", 1e-5, 5e-3, log=True)
        lr = trial.suggest_float("lr", 1e-5, 5e-3, log=True)
        #weight_decay = trial.suggest_float("weight_decay", 1e-7, 1e-3, log=True)
        weight_decay = trial.suggest_float("weight_decay", 7e-5, 2e-4, log=True)
        batch_size = trial.suggest_categorical("batch_size", [64])
        if optimizer_name == "sgd":
            momentum = trial.suggest_float("momentum", 0.0, 0.99)

        # Optionally configure augmentation settings and build augment_fn.
        # If augmentation flags were provided on the CLI, use those values directly.
        augment_fn = None
        if args.augment_subarrays:
            augment_subarrays = True
            aug_mode = args.augment_subarrays_mode
            if args.augment_subarrays_max_per_array is None:
                aug_max_per_array = trial.suggest_categorical("augment_subarrays_max_per_array", [0, 2, 5])
            else:
                aug_max_per_array = int(args.augment_subarrays_max_per_array)
            aug_prob = 0.0 if aug_max_per_array <= 0 else float(args.augment_subarrays_prob)
            # record fixed augmentation settings in the trial for traceability
            trial.set_user_attr("augment_subarrays_max_per_array", int(aug_max_per_array))
            trial.set_user_attr("augment_subarrays_mode", aug_mode)
            trial.set_user_attr("augment_subarrays_prob", float(aug_prob))
            trial.set_user_attr("augment_subarrays_used", aug_max_per_array > 0)
            print("max aug setting", aug_max_per_array)
            use_similarity = bool(args.augment_use_similarity)
            if use_similarity:
                test_token_sets, inverted_index = build_test_similarity_index(dataset.records, test_idx)
                # also expose exact test signatures for the materializer
                test_signatures = {example_signature(dataset.records[idx]) for idx in test_idx} if test_idx else None
                test_signatures_by_idx = {idx: example_signature(dataset.records[idx]) for idx in test_idx} if test_idx else None
                aug_similarity = args.aug_similarity
                aug_min_distance = float(args.aug_similarity_min_distance)
                # If enumerate mode requested, materialize enumerated augmentations so
                # `max_per_array` is enforced; otherwise build an on-the-fly augment_fn.
                if aug_mode != "random":
                    seen_signatures = {example_signature(example) for example in dataset.records}
                    train_new_indices, train_aug_stats = materialize_subarray_augmentations(
                        base_dataset=dataset,
                        source_indices=list(train_idx),
                        seen_signatures=seen_signatures,
                        test_signatures=test_signatures,
                        test_signatures_by_idx=test_signatures_by_idx,
                        test_token_sets=test_token_sets,
                        inverted_index=inverted_index,
                        seed=trial_seed,
                        mode=aug_mode,
                        prob=aug_prob,
                        min_spacers=2,
                        max_per_array=int(aug_max_per_array),
                        split_name="train",
                        use_diversity=True,
                        similarity_metric=aug_similarity or "jaccard",
                        min_distance=aug_min_distance,
                    )
                    train_idx = list(train_idx) + train_new_indices
                    val_new_indices, val_aug_stats = materialize_subarray_augmentations(
                        base_dataset=dataset,
                        source_indices=list(val_idx),
                        seen_signatures=seen_signatures,
                        test_signatures=test_signatures,
                        test_signatures_by_idx=test_signatures_by_idx,
                        test_token_sets=test_token_sets,
                        inverted_index=inverted_index,
                        seed=trial_seed + 1,
                        mode=aug_mode,
                        prob=aug_prob,
                        min_spacers=2,
                        max_per_array=int(aug_max_per_array),
                        split_name="val",
                        use_diversity=True,
                        similarity_metric=aug_similarity or "jaccard",
                        min_distance=aug_min_distance,
                    )
                    val_idx = list(val_idx) + val_new_indices
                    augment_fn = None
                else:
                    augment_fn = make_subarray_augment_fn_with_similarity_filter(
                        prob=aug_prob,
                        seed=trial_seed,
                        test_token_sets=test_token_sets,
                        inverted_index=inverted_index,
                        similarity_metric=aug_similarity,
                        min_distance=aug_min_distance,
                    )
            else:
                if aug_mode != "random":
                    # enumerate without similarity checks
                    seen_signatures = {example_signature(example) for example in dataset.records}
                    train_new_indices, train_aug_stats = materialize_subarray_augmentations(
                        base_dataset=dataset,
                        source_indices=list(train_idx),
                        seen_signatures=seen_signatures,
                        test_signatures=None,
                        test_signatures_by_idx=None,
                        test_token_sets=None,
                        inverted_index=None,
                        seed=trial_seed,
                        mode=aug_mode,
                        prob=aug_prob,
                        min_spacers=2,
                        max_per_array=int(aug_max_per_array),
                        split_name="train",
                        use_diversity=True,
                        similarity_metric="jaccard",
                        min_distance=0.0,
                    )
                    train_idx = list(train_idx) + train_new_indices
                    val_new_indices, val_aug_stats = materialize_subarray_augmentations(
                        base_dataset=dataset,
                        source_indices=list(val_idx),
                        seen_signatures=seen_signatures,
                        test_signatures=None,
                        test_signatures_by_idx=None,
                        test_token_sets=None,
                        inverted_index=None,
                        seed=trial_seed + 1,
                        mode=aug_mode,
                        prob=aug_prob,
                        min_spacers=2,
                        max_per_array=int(aug_max_per_array),
                        split_name="val",
                        use_diversity=True,
                        similarity_metric="jaccard",
                        min_distance=0.0,
                    )
                    val_idx = list(val_idx) + val_new_indices
                    augment_fn = None
                else:
                    augment_fn = make_subarray_augment_fn(prob=aug_prob, seed=trial_seed)
        else:
            augment_subarrays = trial.suggest_categorical("augment_subarrays", [False, True])
            if augment_subarrays:
                aug_mode = trial.suggest_categorical("augment_subarrays_mode", ["random", "enumerate"])
                if args.augment_subarrays_max_per_array is None:
                    aug_max_per_array = trial.suggest_categorical("augment_subarrays_max_per_array", [0, 2, 5])
                else:
                    aug_max_per_array = int(args.augment_subarrays_max_per_array)
                aug_prob = 0.0 if aug_max_per_array <= 0 else trial.suggest_float("augment_subarrays_prob", 0.1, 1.0)
                # record fixed augmentation settings
                trial.set_user_attr("augment_subarrays_max_per_array", int(aug_max_per_array))
                trial.set_user_attr("augment_subarrays_mode", aug_mode)
                trial.set_user_attr("augment_subarrays_prob", float(aug_prob))
                trial.set_user_attr("augment_subarrays_used", aug_max_per_array > 0)
                use_similarity = trial.suggest_categorical("augment_use_similarity", [True])
                if use_similarity:
                    test_token_sets, inverted_index = build_test_similarity_index(dataset.records, test_idx)
                    aug_similarity = trial.suggest_categorical("aug_similarity", ["jaccard"])
                    aug_min_distance = trial.suggest_float("aug_similarity_min_distance", 0.5, 1.0)
                    # if enumerate mode, materialize augmented examples so cap is enforced
                    if aug_mode != "random":
                        test_signatures = {example_signature(dataset.records[idx]) for idx in test_idx} if test_idx else None
                        test_signatures_by_idx = {idx: example_signature(dataset.records[idx]) for idx in test_idx} if test_idx else None
                        seen_signatures = {example_signature(example) for example in dataset.records}
                        train_new_indices, train_aug_stats = materialize_subarray_augmentations(
                            base_dataset=dataset,
                            source_indices=list(train_idx),
                            seen_signatures=seen_signatures,
                            test_signatures=test_signatures,
                            test_signatures_by_idx=test_signatures_by_idx,
                            test_token_sets=test_token_sets,
                            inverted_index=inverted_index,
                            seed=trial_seed,
                            mode=aug_mode,
                            prob=aug_prob,
                            min_spacers=2,
                            max_per_array=int(aug_max_per_array),
                            split_name="train",
                            use_diversity=True,
                            similarity_metric=aug_similarity or "jaccard",
                            min_distance=aug_min_distance,
                        )
                        train_idx = list(train_idx) + train_new_indices
                        val_new_indices, val_aug_stats = materialize_subarray_augmentations(
                            base_dataset=dataset,
                            source_indices=list(val_idx),
                            seen_signatures=seen_signatures,
                            test_signatures=test_signatures,
                            test_signatures_by_idx=test_signatures_by_idx,
                            test_token_sets=test_token_sets,
                            inverted_index=inverted_index,
                            seed=trial_seed + 1,
                            mode=aug_mode,
                            prob=aug_prob,
                            min_spacers=2,
                            max_per_array=int(aug_max_per_array),
                            split_name="val",
                            use_diversity=True,
                            similarity_metric=aug_similarity or "jaccard",
                            min_distance=aug_min_distance,
                        )
                        val_idx = list(val_idx) + val_new_indices
                        augment_fn = None
                    else:
                        augment_fn = make_subarray_augment_fn_with_similarity_filter(
                            prob=aug_prob,
                            seed=trial_seed,
                            test_token_sets=test_token_sets,
                            inverted_index=inverted_index,
                            similarity_metric=aug_similarity,
                            min_distance=aug_min_distance,
                        )
                else:
                    if aug_mode != "random":
                        seen_signatures = {example_signature(example) for example in dataset.records}
                        train_new_indices, train_aug_stats = materialize_subarray_augmentations(
                            base_dataset=dataset,
                            source_indices=list(train_idx),
                            seen_signatures=seen_signatures,
                            test_signatures=None,
                            test_signatures_by_idx=None,
                            test_token_sets=None,
                            inverted_index=None,
                            seed=trial_seed,
                            mode=aug_mode,
                            prob=aug_prob,
                            min_spacers=2,
                            max_per_array=int(aug_max_per_array),
                            split_name="train",
                            use_diversity=True,
                            similarity_metric="jaccard",
                            min_distance=0.0,
                        )
                        train_idx = list(train_idx) + train_new_indices
                        val_new_indices, val_aug_stats = materialize_subarray_augmentations(
                            base_dataset=dataset,
                            source_indices=list(val_idx),
                            seen_signatures=seen_signatures,
                            test_signatures=None,
                            test_signatures_by_idx=None,
                            test_token_sets=None,
                            inverted_index=None,
                            seed=trial_seed + 1,
                            mode=aug_mode,
                            prob=aug_prob,
                            min_spacers=2,
                            max_per_array=int(aug_max_per_array),
                            split_name="val",
                            use_diversity=True,
                            similarity_metric="jaccard",
                            min_distance=0.0,
                        )
                        val_idx = list(val_idx) + val_new_indices
                        augment_fn = None
                    else:
                        augment_fn = make_subarray_augment_fn(prob=aug_prob, seed=trial_seed)

        train_loader_trial = build_dataloader(
            DirectionTorchDataset(
                dataset,
                train_idx,
                vocab,
                augment_fn=augment_fn,
                exclude_repeats=args.exclude_repeats,
                tokenizer=tokenizer_name,
                cnn_tokenizer=cnn_tok_trial,
            ),
            batch_size=batch_size,
            shuffle=True,
        )
        val_loader_trial = build_dataloader(
            DirectionTorchDataset(
                dataset,
                val_idx,
                vocab,
                augment_fn=augment_fn,
                exclude_repeats=args.exclude_repeats,
                tokenizer=tokenizer_name,
                cnn_tokenizer=cnn_tok_trial,
            ),
            batch_size=batch_size,
            shuffle=False,
        )
        test_loader_trial = (
            build_dataloader(
                DirectionTorchDataset(
                    dataset,
                    test_idx,
                    vocab,
                    exclude_repeats=args.exclude_repeats,
                    tokenizer=tokenizer_name,
                    cnn_tokenizer=cnn_tok_trial,
                ),
                batch_size=batch_size,
                shuffle=False,
            )
            if test_idx
            else None
        )

        if optimizer_name == "adamw":
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_name == "adam":
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        else:
            optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)

        best_val_loss = float("inf")
        best_val_metrics = None
        best_model_state = None
        patience_counter = 0
        for epoch in range(1, args.max_epochs + 1):
            train_one_epoch(model, train_loader_trial, optimizer, loss_fn, device)
            metrics = evaluate(model, val_loader_trial, loss_fn, device)
            val_loss = float(metrics["loss"])
            trial.report(val_loss, step=epoch)
            if trial.should_prune():
                raise TrialPruned()
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_val_metrics = metrics
                best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
            if patience_counter >= args.early_stopping_patience:
                break

        # Restore best model and evaluate on test set
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
        
        best_test_metrics = None
        if test_loader_trial is not None:
            best_test_metrics = evaluate(model, test_loader_trial, loss_fn, device)
        
        # Store metrics in trial for later retrieval
        trial.set_user_attr("best_val_loss", best_val_loss)
        if best_val_metrics:
            for key, value in best_val_metrics.items():
                if isinstance(value, (int, float)):
                    trial.set_user_attr(f"val_{key}", float(value))
        if best_test_metrics:
            for key, value in best_test_metrics.items():
                if isinstance(value, (int, float)):
                    trial.set_user_attr(f"test_{key}", float(value))

        return best_val_loss

    study = optuna.create_study(
        study_name=args.study_name,
        storage=args.storage,
        load_if_exists=bool(args.storage),
        direction="minimize",
        sampler=_build_sampler(args),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=args.pruner_warmup_steps),
    )

    timeout = args.timeout if args.timeout and args.timeout > 0 else None
    _print(f"Starting Optuna study name={args.study_name!r} trials={args.n_trials} timeout={timeout}")
    study.optimize(objective, n_trials=args.n_trials, timeout=timeout)

    best_trial = study.best_trial
    best_config = {
        "objective": "val_loss",
        "best_value": best_trial.value,
        "params": best_trial.params,
        "fixed": {
            "jsonl": str(jsonl_path),
            "seed": args.seed,
            "include_flanks": args.include_flanks,
            "reverse_complement_mode": args.reverse_complement_mode,
            "train_fraction": args.train_fraction,
            "val_fraction": args.val_fraction,
            "tokenizer": args.tokenizer,
        },
    }

    results_dir = Path(args.output_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    best_config_path = results_dir / "best_config.json"
    study_path = results_dir / "study_summary.json"
    trials_path = results_dir / "trials.jsonl"

    best_config_path.write_text(json.dumps(best_config, indent=2, sort_keys=True) + "\n")
    study_path.write_text(
        json.dumps(
            {
                "study_name": args.study_name,
                "best_trial": best_trial.number,
                "best_value": best_trial.value,
                "n_trials": len(study.trials),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    with trials_path.open("w") as handle:
        for trial in study.trials:
            trial_data = {
                "number": trial.number,
                "state": str(trial.state),
                "value": trial.value,
                "params": trial.params,
            }
            # Include all user attributes (metrics) in the trial data
            if trial.user_attrs:
                trial_data["metrics"] = trial.user_attrs
            handle.write(json.dumps(trial_data, sort_keys=True) + "\n")

    _print(f"Best trial={best_trial.number} val_loss={best_trial.value:.6f}")
    _print(f"Best config saved to {best_config_path}")
    _print(f"Trials saved to {trials_path}")
    # Save best trial metrics
    best_metrics_path = results_dir / "best_metrics.json"
    best_metrics = {
        "trial": best_trial.number,
        "val_loss": float(best_trial.value),
    }
    if best_trial.user_attrs:
        best_metrics.update(best_trial.user_attrs)
    best_metrics_path.write_text(json.dumps(best_metrics, indent=2, sort_keys=True) + "\n")
    _print(f"Best metrics saved to {best_metrics_path}")


    if args.final_retrain:
        best_params = best_trial.params
        best_tokenizer = best_params.get("tokenizer", (args.tokenizer if args.tokenizer != "auto" else "default"))
        best_cnn_tok = None
        forced_spacer_dim = None
        if best_tokenizer == "cnn":
            best_cnn_output_dim = int(best_params.get("cnn_output_dim", 128))
            best_cnn_embed_dim = int(best_params.get("cnn_embed_dim", 8))
            best_cnn_filters = int(best_params.get("cnn_filters", 64))
            best_cnn_kernels_str = str(best_params.get("cnn_kernels", "3,5,7"))
            best_cnn_pooling = str(best_params.get("cnn_pooling", "max"))
            best_cnn_activation = str(best_params.get("cnn_activation", "relu"))
            best_cnn_kernels = [int(k.strip()) for k in best_cnn_kernels_str.split(",") if k.strip()]
            best_cnn_cfg = CNNTokConfig(
                output_dim=best_cnn_output_dim,
                embed_dim=best_cnn_embed_dim,
                filters=best_cnn_filters,
                kernels=best_cnn_kernels,
                pooling=best_cnn_pooling,
                activation=best_cnn_activation,
            )
            best_cnn_tok = CNNTokenizer(best_cnn_cfg, vocab_size=len(vocab))
            forced_spacer_dim = best_cnn_output_dim
        best_config_obj = DirectionTransformerConfig(
            vocab_size=len(vocab),
            token_dim=int(best_params["token_dim"]),
            spacer_dim=int(forced_spacer_dim) if forced_spacer_dim is not None else int(best_params["spacer_dim"]),
            transformer_dim=int(best_params["transformer_dim"]),
            num_heads=int(best_params["num_heads"]),
            num_layers=int(best_params["num_layers"]),
            feedforward_dim=int(best_params["transformer_dim"]) * int(best_params["feedforward_multiplier"]),
            dropout=float(best_params["dropout"]),
            max_spacers=max((len(ex.spacers) for ex in dataset.records), default=64),
            include_flanks=args.include_flanks,
            activation=str(best_params["activation"]),
            positional_encoding=str(best_params["positional_encoding"]),
            pooling_strategy=str(best_params["pooling_strategy"]),
            use_cls_token=bool(best_params.get("use_cls_token", args.enable_cls_token)),
        )
        model = build_model(
            vocab_size=best_config_obj.vocab_size,
            include_flanks=best_config_obj.include_flanks,
            max_spacers=best_config_obj.max_spacers,
            dropout=best_config_obj.dropout,
            token_dim=best_config_obj.token_dim,
            spacer_dim=best_config_obj.spacer_dim,
            transformer_dim=best_config_obj.transformer_dim,
            num_heads=best_config_obj.num_heads,
            num_layers=best_config_obj.num_layers,
            feedforward_dim=best_config_obj.feedforward_dim,
            activation=best_config_obj.activation,
            positional_encoding=best_config_obj.positional_encoding,
            pooling_strategy=best_config_obj.pooling_strategy,
            use_cls_token=bool(best_config_obj.use_cls_token),
        ).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=best_params["lr"], weight_decay=best_params["weight_decay"])
        retrain_train_indices = train_indices + val_indices
        retrain_train_loader = build_dataloader(
            DirectionTorchDataset(
                dataset,
                retrain_train_indices,
                vocab,
                exclude_repeats=args.exclude_repeats,
                tokenizer=best_tokenizer,
                cnn_tokenizer=best_cnn_tok,
            ),
            batch_size=best_params["batch_size"],
            shuffle=True,
        )
        retrain_test_loader = (
            build_dataloader(
                DirectionTorchDataset(
                    dataset,
                    test_indices,
                    vocab,
                    exclude_repeats=args.exclude_repeats,
                    tokenizer=best_tokenizer,
                    cnn_tokenizer=best_cnn_tok,
                ),
                batch_size=best_params["batch_size"],
                shuffle=False,
            )
            if test_indices
            else None
        )
        retrain_val_loader = build_dataloader(
            DirectionTorchDataset(
                dataset,
                val_indices,
                vocab,
                exclude_repeats=args.exclude_repeats,
                tokenizer=best_tokenizer,
                cnn_tokenizer=best_cnn_tok,
            ),
            batch_size=best_params["batch_size"],
            shuffle=False,
        )
        for _ in range(args.final_epochs):
            train_one_epoch(model, retrain_train_loader, optimizer, loss_fn, device)
        final_metrics = evaluate(model, retrain_test_loader, loss_fn, device) if retrain_test_loader is not None else evaluate(model, retrain_val_loader, loss_fn, device)
        (results_dir / "final_metrics.json").write_text(json.dumps(final_metrics, indent=2, sort_keys=True) + "\n")
        _print(f"Final retrain metrics written to {results_dir / 'final_metrics.json'}")

    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Hyperparameter optimization for direction_learning models")
    parser.add_argument("--jsonl", required=True, help="Path to the training JSONL dataset")
    parser.add_argument("--study_name", default="direction_hpo", help="Optuna study name")
    parser.add_argument("--storage", default="", help="Optuna storage URL, e.g. sqlite:///direction_hpo.db")
    parser.add_argument("--output_dir", default="direction_hpo_runs", help="Directory for study artifacts")
    parser.add_argument("--n_trials", type=int, default=20, help="Number of Optuna trials")
    parser.add_argument("--timeout", type=int, default=0, help="Optional time limit in seconds")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", default="", help="Torch device override")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for the initial split metadata")
    parser.add_argument(
        "--sampler",
        type=str,
        default="tpe",
        choices=["tpe", "random"],
        help="Optuna sampler to use. 'random' is more exploratory; 'tpe' adapts toward promising regions.",
    )
    parser.add_argument(
        "--tpe_startup_trials",
        type=int,
        default=10,
        help="Number of initial random TPE trials before it starts exploiting promising regions.",
    )
    parser.add_argument(
        "--tpe_ei_candidates",
        type=int,
        default=24,
        help="Number of candidate points sampled internally by TPE per trial.",
    )
    parser.add_argument(
        "--tpe_multivariate",
        action="store_true",
        help="Use multivariate TPE to model parameter interactions more directly.",
    )
    parser.add_argument(
        "--stratify_by",
        type=str,
        default="cas_subtype",
        choices=["label", "cas_subtype"],
        help="Stratification method: 'label' (balanced classes) or 'cas_subtype' (CRISPR type). Default: cas_subtype.",
    )
    parser.add_argument(
        "--test_size",
        "--test_within_train_fraction",
        dest="test_size",
        type=float,
        default=0.2,
        help="Fraction of the full dataset to hold out as the final test set (default 0.2 for 20%%).",
    )
    parser.add_argument("--train_fraction", type=float, default=0.64, help="Training fraction (only used if test_size=0; legacy behavior)")
    parser.add_argument("--val_fraction", type=float, default=0.16, help="Validation fraction (only used if test_size=0; legacy behavior)")
    parser.add_argument("--include_flanks", action="store_true", help="Include flank tokens in the model/search")
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="default",
        choices=["default", "cnn", "auto"],
        help="Tokenizer for HPO trials: 'default', 'cnn', or 'auto' (Optuna chooses per trial).",
    )
    parser.add_argument("--exclude_repeats", action="store_true", help="Exclude repeat tokens from encoding; train on spacers only (ablation study).")
    parser.add_argument(
        "--positional_encoding",
        type=str,
        default="absolute",
        choices=["absolute", "alibi", "rope"],
        help="Positional encoding to use for spacer order (default: absolute).",
    )
    parser.add_argument(
        "--reverse_complement_mode",
        type=str,
        default="none",
        choices=["none", "before", "after", "initial_only"],
        help=(
            "When and how to apply reverse-complement augmentation to train/val (test always stays untouched):\\n"
            "  none: Do not apply reverse-complement augmentation (default).\\n"
            "  before: Add reverse complements before subarray augmentation; augment all including RC examples.\\n"
            "  after: Apply subarray augmentation first, then add reverse complements of all resulting arrays.\\n"
            "  initial_only: Apply subarray augmentation, then add reverse complements only of the initial (non-augmented) arrays."
        ),
    )
    parser.add_argument("--max_epochs", type=int, default=30, help="Maximum epochs per trial")
    parser.add_argument("--early_stopping_patience", type=int, default=5, help="Early stopping patience per trial")
    parser.add_argument("--pruner_warmup_steps", type=int, default=2, help="Median pruner warmup steps")
    parser.add_argument("--final_retrain", action="store_true", help="Retrain the best config after the search")
    parser.add_argument("--final_epochs", type=int, default=20, help="Epochs for the optional final retrain")
    # Optional: force augmentation settings for all trials (overrides Optuna sampling)
    parser.add_argument("--augment_subarrays", action="store_true", help="Force subarray augmentation for all trials (overrides HPO sampling)")
    parser.add_argument("--augment_subarrays_mode", type=str, default="random", choices=["random", "enumerate"], help="Augmentation mode when augmentation is forced")
    parser.add_argument("--augment_subarrays_prob", type=float, default=0.5, help="Probability used by the augmenter when augmentation is forced")
    parser.add_argument("--augment_subarrays_max_per_array", type=int, default=None, help="Max augmentations per array when set; omit to search 0/2/5 in HPO")
    parser.add_argument("--augment_use_similarity", action="store_true", help="When forcing augmentation, use similarity-based filtering against the test set")
    parser.add_argument("--aug_similarity", type=str, default="jaccard", choices=["jaccard", "overlap"], help="Similarity metric to use when similarity filtering is enabled")
    parser.add_argument("--aug_similarity_min_distance", type=float, default=0.5, help="Minimum distance to test set required to accept augmented candidate (0.0-1.0)")
    parser.add_argument("--enable_cls_token", action="store_true", help="Enable learned CLS token prepended to spacer sequence (default: off)")
    return parser


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()
    if not args.storage:
        args.storage = None
    return run_study(args)


if __name__ == "__main__":
    raise SystemExit(main())
