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
)
from .training import (
    evaluate,
    train_one_epoch,
)


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
    valid_heads = [head for head in (8, 16) if transformer_dim % head == 0]
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


def _sample_config(trial: Any, vocab_size: int, max_spacers: int, include_flanks: bool) -> DirectionTransformerConfig:
    #positional_encoding = trial.suggest_categorical("positional_encoding", ["absolute", "alibi", "rope"])
    positional_encoding = trial.suggest_categorical("positional_encoding", ["alibi", "rope"])
    #pooling_strategy = trial.suggest_categorical("pooling_strategy", ["mean", "max", "attention", "learnable"])
    pooling_strategy = trial.suggest_categorical("pooling_strategy", ["mean", "attention", "learnable"])
    #token_dim = trial.suggest_categorical("token_dim", [32, 48, 64, 96, 128])
    token_dim = trial.suggest_categorical("token_dim", [32, 48])
    #spacer_dim = trial.suggest_categorical("spacer_dim", [64, 96, 128, 160, 192, 256])
    spacer_dim = trial.suggest_categorical("spacer_dim", [192, 256, 512])
    #transformer_dim = trial.suggest_categorical("transformer_dim", [64, 96, 128, 160, 192, 256])
    transformer_dim = trial.suggest_categorical("transformer_dim", [192, 256, 512])
    num_heads = _choose_num_heads(transformer_dim, trial)
    #num_layers = trial.suggest_int("num_layers", 1, 6)
    num_layers = trial.suggest_int("num_layers", 1, 3)
    #dropout = trial.suggest_float("dropout", 0.0, 0.5)
    dropout = trial.suggest_float("dropout", 0.01, 0.10)
    feedforward_multiplier = trial.suggest_categorical("feedforward_multiplier", [2, 4, 6])
    feedforward_dim = transformer_dim * feedforward_multiplier
    #activation = trial.suggest_categorical("activation", ["gelu", "relu"])
    activation = trial.suggest_categorical("activation", ["gelu"])

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
    )


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

    train_dataset = DirectionTorchDataset(dataset, train_indices, vocab, exclude_repeats=args.exclude_repeats)
    val_dataset = DirectionTorchDataset(dataset, val_indices, vocab, exclude_repeats=args.exclude_repeats)
    test_dataset = DirectionTorchDataset(dataset, test_indices, vocab, exclude_repeats=args.exclude_repeats) if test_indices else None

    train_loader = build_dataloader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = build_dataloader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = build_dataloader(test_dataset, batch_size=args.batch_size, shuffle=False) if test_dataset else None

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    loss_fn = torch.nn.BCEWithLogitsLoss()

    def objective(trial: Any) -> float:
        trial_seed = args.seed + trial.number * 17
        random.seed(trial_seed)
        torch.manual_seed(trial_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(trial_seed)

        config = _sample_config(trial, vocab_size=len(vocab), max_spacers=max((len(ex.spacers) for ex in dataset.records), default=64), include_flanks=args.include_flanks)
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
        ).to(device)

        #optimizer_name = trial.suggest_categorical("optimizer", ["adamw", "adam", "sgd"])
        optimizer_name = trial.suggest_categorical("optimizer", ["adamw", "adam"])
        #lr = trial.suggest_float("lr", 1e-5, 5e-3, log=True)
        lr = trial.suggest_float("lr", 1e-5, 5e-3, log=True)
        #weight_decay = trial.suggest_float("weight_decay", 1e-7, 1e-3, log=True)
        weight_decay = trial.suggest_float("weight_decay", 7e-5, 2e-4, log=True)
        batch_size = trial.suggest_categorical("batch_size", [8, 16, 32, 64])
        if optimizer_name == "sgd":
            momentum = trial.suggest_float("momentum", 0.0, 0.99)

        # Optionally sample augmentation settings and build augment_fn.
        # If augmentation flags were provided on the CLI, use those values (override sampling).
        augment_fn = None
        if args.augment_subarrays:
            augment_subarrays = True
            aug_mode = args.augment_subarrays_mode
            aug_prob = float(args.augment_subarrays_prob)
            aug_max_per_array = int(args.augment_subarrays_max_per_array)
            use_similarity = bool(args.augment_use_similarity)
            if use_similarity:
                test_token_sets, inverted_index = build_test_similarity_index(dataset.records, test_indices)
                aug_similarity = args.aug_similarity
                aug_min_distance = float(args.aug_similarity_min_distance)
                augment_fn = make_subarray_augment_fn_with_similarity_filter(
                    prob=aug_prob,
                    seed=trial_seed,
                    test_token_sets=test_token_sets,
                    inverted_index=inverted_index,
                    similarity_metric=aug_similarity,
                    min_distance=aug_min_distance,
                )
            else:
                augment_fn = make_subarray_augment_fn(prob=aug_prob, seed=trial_seed)
        else:
            augment_subarrays = trial.suggest_categorical("augment_subarrays", [False, True])
            if augment_subarrays:
                aug_mode = trial.suggest_categorical("augment_subarrays_mode", ["random", "enumerate"])
                aug_prob = trial.suggest_float("augment_subarrays_prob", 0.1, 1.0)
                aug_max_per_array = trial.suggest_categorical("augment_subarrays_max_per_array", [0, 2, 5, 10, 20])
                use_similarity = trial.suggest_categorical("augment_use_similarity", [False, True])
                if use_similarity:
                    test_token_sets, inverted_index = build_test_similarity_index(dataset.records, test_indices)
                    aug_similarity = trial.suggest_categorical("aug_similarity", ["jaccard", "overlap"])
                    aug_min_distance = trial.suggest_float("aug_similarity_min_distance", 0.0, 1.0)
                    augment_fn = make_subarray_augment_fn_with_similarity_filter(
                        prob=aug_prob,
                        seed=trial_seed,
                        test_token_sets=test_token_sets,
                        inverted_index=inverted_index,
                        similarity_metric=aug_similarity,
                        min_distance=aug_min_distance,
                    )
                else:
                    augment_fn = make_subarray_augment_fn(prob=aug_prob, seed=trial_seed)

        train_loader_trial = build_dataloader(DirectionTorchDataset(dataset, train_indices, vocab, augment_fn=augment_fn, exclude_repeats=args.exclude_repeats), batch_size=batch_size, shuffle=True)
        val_loader_trial = build_dataloader(DirectionTorchDataset(dataset, val_indices, vocab, augment_fn=augment_fn, exclude_repeats=args.exclude_repeats), batch_size=batch_size, shuffle=False)

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
        if test_loader is not None:
            best_test_metrics = evaluate(model, test_loader, loss_fn, device)
        
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
        best_config_obj = DirectionTransformerConfig(
            vocab_size=len(vocab),
            token_dim=int(best_params["token_dim"]),
            spacer_dim=int(best_params["spacer_dim"]),
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
        ).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=best_params["lr"], weight_decay=best_params["weight_decay"])
        retrain_train_indices = train_indices + val_indices
        retrain_train_loader = build_dataloader(DirectionTorchDataset(dataset, retrain_train_indices, vocab, exclude_repeats=args.exclude_repeats), batch_size=best_params["batch_size"], shuffle=True)
        retrain_test_loader = build_dataloader(test_dataset, batch_size=best_params["batch_size"], shuffle=False) if test_dataset else None
        for _ in range(args.final_epochs):
            train_one_epoch(model, retrain_train_loader, optimizer, loss_fn, device)
        final_metrics = evaluate(model, retrain_test_loader, loss_fn, device) if retrain_test_loader is not None else evaluate(model, val_loader, loss_fn, device)
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
    parser.add_argument("--augment_subarrays_max_per_array", type=int, default=2, help="Max augmentations per array when augmentation is forced")
    parser.add_argument("--augment_use_similarity", action="store_true", help="When forcing augmentation, use similarity-based filtering against the test set")
    parser.add_argument("--aug_similarity", type=str, default="jaccard", choices=["jaccard", "overlap"], help="Similarity metric to use when similarity filtering is enabled")
    parser.add_argument("--aug_similarity_min_distance", type=float, default=0.5, help="Minimum distance to test set required to accept augmented candidate (0.0-1.0)")
    return parser


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()
    if not args.storage:
        args.storage = None
    return run_study(args)


if __name__ == "__main__":
    raise SystemExit(main())
