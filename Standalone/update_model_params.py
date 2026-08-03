#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
from pathlib import Path


REQUIRED_TOKENIZER_FILES = [
    "tokenizer.json",
    "tokenizer_config.json",
]

OPTIONAL_USEFUL_FILES = [
    "tokenizer.py",
    "dna_config.json",
]


def _has_model_weights(model_dir: Path) -> bool:
    return (
        (model_dir / "model.safetensors").exists()
        or (model_dir / "pytorch_model.bin").exists()
        or any(model_dir.glob("model-*.safetensors"))
        or any(model_dir.glob("pytorch_model-*.bin"))
    )


def _validate_source_dir(source_dir: Path) -> None:
    if not source_dir.exists() or not source_dir.is_dir():
        raise FileNotFoundError(f"Source directory not found: {source_dir}")

    if not (source_dir / "config.json").exists():
        raise FileNotFoundError(f"Missing required file in source: {source_dir / 'config.json'}")

    if not _has_model_weights(source_dir):
        raise FileNotFoundError(
            "Missing model weights in source. Expected one of: model.safetensors, "
            "pytorch_model.bin, model-*.safetensors, pytorch_model-*.bin"
        )

    for filename in REQUIRED_TOKENIZER_FILES:
        path = source_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"Missing required tokenizer file in source: {path}")


def _copy_tree(source_dir: Path, target_dir: Path, clean_target: bool) -> None:
    if clean_target and target_dir.exists():
        shutil.rmtree(target_dir)

    target_dir.mkdir(parents=True, exist_ok=True)

    for item in source_dir.iterdir():
        if item.name == "__pycache__":
            continue

        dest = target_dir / item.name

        if item.is_dir():
            shutil.copytree(item, dest, dirs_exist_ok=True)
        else:
            shutil.copy2(item, dest)


def parse_args() -> argparse.Namespace:
    default_target = (Path(__file__).resolve().parent / "model_params").resolve()

    parser = argparse.ArgumentParser(
        description=(
            "Replace Standalone/model_params with a freshly saved full model folder "
            "from a new training run."
        )
    )
    parser.add_argument(
        "--source_dir",
        required=True,
        help=(
            "Path to your newly saved full model directory (must contain config, weights, "
            "and tokenizer files)."
        ),
    )
    parser.add_argument(
        "--target_dir",
        default=str(default_target),
        help="Destination model folder to update (default: Standalone/model_params).",
    )
    parser.add_argument(
        "--no_clean",
        action="store_true",
        help=(
            "Do not remove the target directory before copying. "
            "By default, target is fully replaced to avoid stale files."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    source_dir = Path(args.source_dir).expanduser().resolve()
    target_dir = Path(args.target_dir).expanduser().resolve()
    clean_target = not args.no_clean

    _validate_source_dir(source_dir)

    print(f"Source model: {source_dir}")
    print(f"Target model: {target_dir}")
    print(f"Clean target before copy: {clean_target}")

    _copy_tree(source_dir, target_dir, clean_target=clean_target)

    missing_optional = [
        name for name in OPTIONAL_USEFUL_FILES if not (target_dir / name).exists()
    ]

    print("Model update complete.")
    if missing_optional:
        print("Optional files not present in target (this may be fine):")
        for name in missing_optional:
            print(f"  - {name}")

    print("You can now run: python Standalone/predict_direction.py --input_file Standalone/test.jsonl")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
