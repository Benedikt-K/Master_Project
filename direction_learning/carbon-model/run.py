from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
import numpy as np
from collections import Counter
from contextlib import nullcontext
from pathlib import Path
from importlib import import_module
from typing import Any
from dataclasses import replace
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    matthews_corrcoef, roc_auc_score, confusion_matrix,
)

from direction_learning.augmentation import (
	build_test_similarity_index,
	example_signature,
	materialize_subarray_augmentations,
)

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
	sys.path.insert(0, str(ROOT))

try:
	import torch
	from torch import nn
	from torch.utils.data import DataLoader, Dataset
except ModuleNotFoundError:
	torch = None
	nn = None
	DataLoader = object
	Dataset = object

AutoModelForSequenceClassification = None
AutoTokenizer = None
get_linear_schedule_with_warmup = None
LoraConfig = None
TaskType = None
get_peft_model = None
prepare_model_for_kbit_training = None

from direction_learning.data import split_dev_pool_by_mode, stratified_holdout_by_mode
from direction_learning.dataset import DirectionExample, DirectionJsonlDataset

DNA_SEPARATOR = "NNNNNN"
DEFAULT_MODEL_ID = "HuggingFaceBio/Carbon-500M"
AUGMENT_SPACER_DELETION_SIMILARITY_METRIC = "jaccard"
AUGMENT_SPACER_DELETION_MIN_DISTANCE = 0.7


def _require_runtime() -> None:
	if torch is None or nn is None:
		raise ModuleNotFoundError(
			"PyTorch is required for Carbon finetuning. Install torch before running this script."
		)
	global AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
	if AutoTokenizer is None or AutoModelForSequenceClassification is None or get_linear_schedule_with_warmup is None:
		transformers = import_module("transformers")
		AutoTokenizer = transformers.AutoTokenizer
		AutoModelForSequenceClassification = transformers.AutoModelForSequenceClassification
		get_linear_schedule_with_warmup = transformers.get_linear_schedule_with_warmup
	if AutoTokenizer is None or AutoModelForSequenceClassification is None:
		raise ModuleNotFoundError(
			"transformers is required for Carbon finetuning. Install transformers before running this script."
		)


def _require_peft_runtime() -> None:
	global LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
	if LoraConfig is None or TaskType is None or get_peft_model is None or prepare_model_for_kbit_training is None:
		try:
			peft = import_module("peft")
		except ModuleNotFoundError as exc:
			raise ModuleNotFoundError(
				"peft is required when using LoRA. Install it with `pip install peft`."
			) from exc
		LoraConfig = peft.LoraConfig
		TaskType = peft.TaskType
		get_peft_model = peft.get_peft_model
		prepare_model_for_kbit_training = peft.prepare_model_for_kbit_training


def _normalize_dna(sequence: str) -> str:
	sequence = sequence.upper().replace("U", "T")
	cleaned: list[str] = []
	for char in sequence:
		if char.isspace():
			continue
		cleaned.append(char if char in {"A", "C", "G", "T"} else "N")
	return "".join(cleaned)


def _interleave_segments(repeats: list[str], spacers: list[str]) -> list[str]:
	segments: list[str] = []
	max_length = max(len(repeats), len(spacers))
	for index in range(max_length):
		if index < len(repeats):
			segments.append(_normalize_dna(repeats[index]))
		if index < len(spacers):
			segments.append(_normalize_dna(spacers[index]))
	return [segment for segment in segments if segment]


def build_carbon_sequence(example: DirectionExample, include_flanks: bool, sequence_mode: str) -> str:
	pieces: list[str] = []

	if include_flanks and example.left_flank:
		pieces.append(_normalize_dna(example.left_flank))

	if sequence_mode == "spacers_only":
		pieces.extend(_normalize_dna(spacer) for spacer in example.spacers if spacer)
	else:
		pieces.extend(_interleave_segments(example.repeats, example.spacers))

	if include_flanks and example.right_flank:
		pieces.append(_normalize_dna(example.right_flank))

	core = DNA_SEPARATOR.join(piece for piece in pieces if piece)
	return f"<dna>{core}</dna>"


class CarbonDirectionDataset(Dataset if Dataset is not object else object):
	def __init__(
		self,
		examples: list[DirectionExample],
		tokenizer: Any,
		max_length: int,
		include_flanks: bool,
		sequence_mode: str,
	):
		self.examples = examples
		self.tokenizer = tokenizer
		self.max_length = max_length
		self.include_flanks = include_flanks
		self.sequence_mode = sequence_mode

	def __len__(self) -> int:
		return len(self.examples)

	def __getitem__(self, index: int) -> dict[str, Any]:
		example = self.examples[index]
		text = build_carbon_sequence(example, include_flanks=self.include_flanks, sequence_mode=self.sequence_mode)
		encoded = self.tokenizer(
			text,
			add_special_tokens=False,
			truncation=True,
			max_length=self.max_length,
		)
		encoded["labels"] = int(example.label)
		encoded["array_name"] = example.array_name
		encoded["group_name"] = example.group_name
		encoded["cas_subtype"] = example.cas_subtype
		return encoded


class CarbonBatchCollator:
	def __init__(self, tokenizer: Any):
		self.tokenizer = tokenizer

	def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
		labels = torch.tensor([feature["labels"] for feature in features], dtype=torch.long)
		model_features = [
			{key: value for key, value in feature.items() if key in {"input_ids", "attention_mask"}}
			for feature in features
		]
		batch = self.tokenizer.pad(model_features, padding=True, return_tensors="pt")
		batch["labels"] = labels
		return batch


def _truncate_indices(indices: list[int], limit: int | None) -> list[int]:
	if limit is None or limit <= 0:
		return indices
	return indices[:limit]


def _resolve_jsonl_path(value: str) -> Path:
	path = Path(value)
	if path.exists():
		return path

	candidates = [
		Path("/tmp") / value,
		Path("/tmp/direction_no_aug.jsonl"),
		Path("/tmp/direction_training_dataset_full.jsonl"),
		Path("/tmp/direction_training_dataset_test.jsonl"),
	]
	for candidate in candidates:
		if candidate.exists():
			return candidate

	matches = sorted(Path("/tmp").glob("*direction*dataset*.jsonl"))
	if matches:
		return matches[0]

	raise FileNotFoundError(
		f"Could not find dataset JSONL at {value!r} or in /tmp. "
		"Pass --jsonl /absolute/path/to/file.jsonl explicitly."
	)


def _safe_divide(numerator: float, denominator: float) -> float:
	return numerator / denominator if denominator else 0.0


def _binary_auc(labels: list[int], scores: list[float]) -> float:
	positives = sum(labels)
	negatives = len(labels) - positives
	if positives == 0 or negatives == 0:
		return float("nan")

	ranked = sorted(zip(scores, labels), key=lambda item: item[0])
	rank_sum = 0.0
	index = 0
	rank = 1
	while index < len(ranked):
		end = index + 1
		while end < len(ranked) and ranked[end][0] == ranked[index][0]:
			end += 1
		average_rank = (rank + (rank + (end - index) - 1)) / 2.0
		positives_in_group = sum(label for _, label in ranked[index:end])
		rank_sum += positives_in_group * average_rank
		rank += end - index
		index = end

	return (rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


def _classification_metrics(labels, scores, threshold: float = 0.5) -> dict[str, float]:
    predictions = [1 if s >= threshold else 0 for s in scores]

    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()

    try:
        auroc = roc_auc_score(labels, scores)
    except ValueError:
        auroc = float("nan")  # only one class present in this split/batch

    return {
        "accuracy": accuracy_score(labels, predictions),
        "precision": precision_score(labels, predictions, zero_division=0),
        "recall": recall_score(labels, predictions, zero_division=0),
        "f1": f1_score(labels, predictions, zero_division=0),
        "mcc": matthews_corrcoef(labels, predictions),
        "auroc": auroc,
        "tp": float(tp), "tn": float(tn), "fp": float(fp), "fn": float(fn),
    }


def _print_split_summary(name: str, examples: list[DirectionExample]) -> None:
	label_counts = Counter(example.label for example in examples)
	subtype_counts = Counter((example.cas_subtype or "Unknown") for example in examples)
	print(
		f"{name}: {len(examples)} examples | "
		f"Forward={label_counts.get(1, 0)} Reverse={label_counts.get(0, 0)} | "
		f"Top subtypes={dict(subtype_counts.most_common(5))}"
	)


def _build_splits(
	examples: list[DirectionExample],
	seed: int,
	test_fraction: float,
	stratify_mode: str,
) -> dict[str, list[int]]:
	dev_indices, test_indices = stratified_holdout_by_mode(
		examples,
		seed=seed,
		holdout_fraction=test_fraction,
		stratify_mode=stratify_mode,
	)
	train_indices, val_indices = split_dev_pool_by_mode(
		examples,
		pool_indices=dev_indices,
		seed=seed,
		stratify_mode=stratify_mode,
	)

	# --- sanity check: verify splits don't leak indices into each other ---
	train_set, val_set, test_set = set(train_indices), set(val_indices), set(test_indices)
	train_val_overlap = train_set & val_set
	train_test_overlap = train_set & test_set
	val_test_overlap = val_set & test_set
 
	print(
		f"[split check] sizes: train={len(train_indices)} val={len(val_indices)} test={len(test_indices)} "
		f"total={len(train_indices) + len(val_indices) + len(test_indices)} | dataset size={len(examples)}"
	)
	print(
		f"[split check] duplicates within a split: "
		f"train={len(train_indices) - len(train_set)} val={len(val_indices) - len(val_set)} test={len(test_indices) - len(test_set)}"
	)
	print(
		f"[split check] overlap counts: train&val={len(train_val_overlap)} "
		f"train&test={len(train_test_overlap)} val&test={len(val_test_overlap)}"
	)
	if train_val_overlap or train_test_overlap or val_test_overlap:
		print(f"[split check] OVERLAPPING INDICES train&val={sorted(train_val_overlap)[:20]} "
			  f"train&test={sorted(train_test_overlap)[:20]} val&test={sorted(val_test_overlap)[:20]}")
		raise ValueError("Index leakage detected between splits — train/val/test sets are not disjoint.")
 
	all_covered = train_set | val_set | test_set
	missing = set(range(len(examples))) - all_covered
	if missing:
		print(f"[split check] WARNING: {len(missing)} dataset indices not assigned to any split (e.g. {sorted(missing)[:10]})")
	# --- end sanity check ---

	return {"train": train_indices, "val": val_indices, "test": test_indices}


def _build_label_weights(labels: list[int]) -> Any:
	counts = Counter(labels)
	total = max(len(labels), 1)
	weights = []
	for label in (0, 1):
		count = counts.get(label, 0)
		if count == 0:
			weights.append(1.0)
		else:
			weights.append(total / (2.0 * count))
	return torch.tensor(weights, dtype=torch.float32)


def _freeze_backbone(model: Any) -> None:
	for name, parameter in model.named_parameters():
		if any(tag in name for tag in ("classifier", "score", "pre_classifier", "classification_head")):
			parameter.requires_grad = True
		else:
			parameter.requires_grad = False


def _batch_to_device(batch: dict[str, Any], device: Any) -> dict[str, Any]:
	moved: dict[str, Any] = {}
	for key, value in batch.items():
		if isinstance(value, torch.Tensor):
			moved[key] = value.to(device)
		else:
			moved[key] = value
	return moved


def _evaluate_model(
	model: Any,
	loader: Any,
	device: Any,
	loss_fn: Any,
	use_bfloat16: bool,
) -> dict[str, float]:
	model.eval()
	all_labels: list[int] = []
	all_scores: list[float] = []
	total_loss = 0.0
	total_examples = 0

	autocast_context = torch.autocast("cuda", dtype=torch.bfloat16) if use_bfloat16 else nullcontext()
	with torch.no_grad():
		for batch in loader:
			batch = _batch_to_device(batch, device)
			labels = batch["labels"]
			with autocast_context:
				outputs = model(input_ids=batch.get("input_ids"), attention_mask=batch.get("attention_mask"))
				logits = outputs.logits
				loss = loss_fn(logits, labels)
			probabilities = torch.softmax(logits.float(), dim=-1)[:, 1]
			all_labels.extend(labels.detach().cpu().tolist())
			all_scores.extend(probabilities.detach().cpu().tolist())
			total_loss += float(loss.item()) * len(labels)
			total_examples += len(labels)

	metrics = _classification_metrics(all_labels, all_scores)
	metrics["loss"] = _safe_divide(total_loss, total_examples)
	return metrics


def _format_metrics(metrics: dict[str, float]) -> str:
	ordered = ["loss", "accuracy", "precision", "recall", "f1", "mcc", "auroc"]
	parts = []
	for key in ordered:
		value = metrics.get(key, float("nan"))
		if math.isnan(value):
			parts.append(f"{key}=nan")
		else:
			parts.append(f"{key}={value:.4f}")
	return " | ".join(parts)


def _cuda_memory_hint(device: Any) -> str:
	if device.type != "cuda" or not torch.cuda.is_available():
		return ""
	free_bytes, total_bytes = torch.cuda.mem_get_info(device)
	free_gb = free_bytes / (1024 ** 3)
	total_gb = total_bytes / (1024 ** 3)
	return f"CUDA free memory: {free_gb:.2f} GiB / {total_gb:.2f} GiB"


def _parse_csv_modules(value: str) -> list[str]:
	modules = [item.strip() for item in value.split(",") if item.strip()]
	return modules


def _resolve_lora_target_modules(model: Any, preset: str, explicit_modules: str) -> list[str] | str:
	if explicit_modules:
		modules = _parse_csv_modules(explicit_modules)
		if not modules:
			raise ValueError("--lora_target_modules was provided but no module names were parsed.")
		return modules

	if preset == "qv":
		return ["q_proj", "v_proj"]
	if preset == "qkv":
		return ["q_proj", "k_proj", "v_proj"]
	if preset == "attn":
		return ["q_proj", "k_proj", "v_proj", "o_proj"]
	if preset == "mlp":
		return ["gate_proj", "up_proj", "down_proj"]
	if preset == "attn_mlp":
		return ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
	if preset == "all":
		available = []
		for name, module in model.named_modules():
			if hasattr(module, "weight") and any(marker in name for marker in ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")):
				available.append(name.split(".")[-1])
		if available:
			return sorted(set(available))
		return ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
	raise ValueError(f"Unknown LoRA preset: {preset}")


def _apply_lora(
	model: Any,
	*,
	task_type: str,
	preset: str,
	explicit_modules: str,
	r: int,
	alpha: int,
	dropout: float,
	bias: str,
	use_kbit_training: bool,
) -> Any:
	if get_peft_model is None or LoraConfig is None or TaskType is None or prepare_model_for_kbit_training is None:
		_require_peft_runtime()
	if get_peft_model is None or LoraConfig is None or TaskType is None or prepare_model_for_kbit_training is None:
		raise ModuleNotFoundError(
			"peft is required for LoRA support. Install it with `pip install peft`."
		)

	target_modules = _resolve_lora_target_modules(model, preset=preset, explicit_modules=explicit_modules)
	if use_kbit_training:
		model = prepare_model_for_kbit_training(model)

	config = LoraConfig(
		task_type=getattr(TaskType, task_type),
		r=r,
		lora_alpha=alpha,
		lora_dropout=dropout,
		bias=bias,
		target_modules=target_modules,
	)
	model = get_peft_model(model, config)
	return model


def main() -> int:
	parser = argparse.ArgumentParser(description="Finetune Carbon-500M for CRISPR array direction prediction.")
	parser.add_argument("--jsonl", default="/tmp/direction_no_aug.jsonl")
	parser.add_argument("--output_dir", default="direction_learning/carbon-model/outputs/carbon-500m-direction")
	parser.add_argument("--model_id", default=DEFAULT_MODEL_ID)
	parser.add_argument("--sequence_mode", choices=["interleaved", "spacers_only"], default="interleaved")
	parser.add_argument("--include_flanks", action="store_true")
	parser.add_argument("--stratify_mode", choices=["label", "cas_subtype", "cas_subtype_and_label"], default="cas_subtype_and_label")
	parser.add_argument("--test_fraction", type=float, default=0.15)
	parser.add_argument("--max_length", type=int, default=512)
	parser.add_argument("--batch_size", type=int, default=2)
	parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
	parser.add_argument("--epochs", type=int, default=3)
	parser.add_argument("--learning_rate", type=float, default=2e-5)
	parser.add_argument("--weight_decay", type=float, default=0.05)
	parser.add_argument("--warmup_ratio", type=float, default=0.06)
	parser.add_argument("--seed", type=int, default=42)
	parser.add_argument("--num_workers", type=int, default=0)
	parser.add_argument("--max_train_examples", type=int, default=0)
	parser.add_argument("--max_val_examples", type=int, default=0)
	parser.add_argument("--max_test_examples", type=int, default=0)
	parser.add_argument(
		"--augment_spacer_deletion",
		action="store_true",
		help="If set, materialize spacer-deletion augmentations for the training split.",
	)
	parser.add_argument(
		"--augment_spacer_deletion_count",
		type=int,
		default=5,
		help="Number of spacer-deletion augmentations to add when --augment_spacer_deletion is set (default: 5).",
	)
	parser.add_argument("--freeze_backbone", action="store_true")
	parser.add_argument("--use_lora", action="store_true", help="Attach LoRA adapters to the Carbon backbone.")
	parser.add_argument("--lora_task_type", choices=["SEQ_CLS", "CAUSAL_LM"], default="SEQ_CLS")
	parser.add_argument("--lora_r", type=int, default=8, help="LoRA rank; lower uses less VRAM.")
	parser.add_argument("--lora_alpha", type=int, default=16, help="LoRA scaling factor.")
	parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout.")
	parser.add_argument("--lora_bias", choices=["none", "all", "lora_only"], default="none")
	parser.add_argument("--lora_preset", choices=["qv", "qkv", "attn", "mlp", "attn_mlp", "all"], default="qv", help="Predefined target-module set for LoRA.")
	parser.add_argument("--lora_target_modules", default="", help="Comma-separated module names; overrides --lora_preset.")
	parser.add_argument("--lora_kbit_training", action="store_true", help="Prepare the model for k-bit training when using quantized loading.")
	parser.add_argument("--load_in_8bit", action="store_true", help="Load the backbone in 8-bit if supported by your environment.")
	parser.add_argument("--load_in_4bit", action="store_true", help="Load the backbone in 4-bit if supported by your environment.")
	parser.add_argument("--lora_only_train", action="store_true", help="Freeze the backbone and train only LoRA/head parameters.")
	parser.add_argument("--gradient_checkpointing", action="store_true")
	parser.add_argument("--cpu", action="store_true")
	parser.add_argument("--bf16", action="store_true", help="Use bfloat16 on CUDA when available.")
	parser.add_argument("--dry_run", action="store_true", help="Load the model and run a single forward pass, then exit.")
	parser.add_argument("--shuffle_labels", action="store_true", help="Randomly shuffle labels as a control to test for memorization.")

	args = parser.parse_args()
	_require_runtime()

	random.seed(args.seed)
	torch.manual_seed(args.seed)
	if torch.cuda.is_available():
		torch.cuda.manual_seed_all(args.seed)

	device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
	use_bfloat16 = bool(args.bf16 and device.type == "cuda")

	output_dir = Path(args.output_dir)
	output_dir.mkdir(parents=True, exist_ok=True)

	dataset_path = _resolve_jsonl_path(args.jsonl)
	print(f"Loading dataset: {dataset_path}")
	dataset = DirectionJsonlDataset(dataset_path, include_flanks=args.include_flanks)
	if len(dataset) == 0:
		raise ValueError("The dataset is empty.")
	print(dataset.records[0])

	
		# test for memorization due to foundation model having seen all data and only remembering
	if args.shuffle_labels:
		# --- MEMORIZATION TEST: randomize labels ---
		labels = [r.label for r in dataset.records]
		print(f"Before shuffle - first 10 labels: {labels[:10]}")
		print(f"Before shuffle - label distribution: {sum(labels)} forward, {len(labels)-sum(labels)} reverse")
 
		labels = [r.label for r in dataset.records]
		random.shuffle(labels)
		dataset.records = [replace(r, label=new_label) for r, new_label in zip(dataset.records, labels)]
 
		labels_after = [r.label for r in dataset.records]
		print(f"After shuffle  - first 10 labels: {labels_after[:10]}")
		print(f"After shuffle  - label distribution: {sum(labels_after)} forward, {len(labels_after)-sum(labels_after)} reverse")
 
		print(f"Shuffled labels randomly. Sample check: {dataset.records[0].label}, {dataset.records[1].label}")
 
		mismatches = sum(
			1 for r in dataset.records
			if (r.label == 1) != (r.evor_direction == "Forward")
		)
		print(f"Label/direction mismatches: {mismatches}/{len(dataset.records)} (should be ~50% if shuffled correctly)")
		# ------------------------------------------

	

	splits = _build_splits(dataset.records, seed=args.seed, test_fraction=args.test_fraction, stratify_mode=args.stratify_mode)
	train_indices = _truncate_indices(splits["train"], args.max_train_examples)
	val_indices = _truncate_indices(splits["val"], args.max_val_examples)
	test_indices = _truncate_indices(splits["test"], args.max_test_examples)

	train_examples = [dataset.records[index] for index in train_indices]
	val_examples = [dataset.records[index] for index in val_indices]
	test_examples = [dataset.records[index] for index in test_indices]

	print("Split summary:")
	_print_split_summary("train", train_examples)
	_print_split_summary("val", val_examples)
	_print_split_summary("test", test_examples)

	augment_spacer_deletion_count = max(0, int(args.augment_spacer_deletion_count))
	if args.augment_spacer_deletion:
		if augment_spacer_deletion_count <= 0:
			print("Spacer deletion augmentation requested, but the augmentation count is <= 0; skipping.")
		else:
			test_signatures = None
			test_signatures_by_idx = None
			test_token_sets = None
			inverted_index = None
			if test_indices:
				test_signatures = {example_signature(dataset.records[index]) for index in test_indices}
				test_signatures_by_idx = {
					index: example_signature(dataset.records[index])
					for index in test_indices
				}
				test_token_sets, inverted_index = build_test_similarity_index(
					dataset.records,
					list(test_indices),
				)
			else:
				print("Spacer deletion augmentation requested, but no test split is present; similarity filtering will be skipped.")

			seen_signatures = {example_signature(example) for example in dataset.records}
			train_new_indices, train_aug_stats = materialize_subarray_augmentations(
				base_dataset=dataset,
				source_indices=list(train_indices),
				seen_signatures=seen_signatures,
				test_signatures=test_signatures,
				test_signatures_by_idx=test_signatures_by_idx,
				test_token_sets=test_token_sets,
				inverted_index=inverted_index,
				seed=args.seed,
				mode="enumerate",
				prob=1.0,
				min_spacers=2,
				max_per_array=augment_spacer_deletion_count,
				split_name="train",
				use_diversity=True,
				similarity_metric=AUGMENT_SPACER_DELETION_SIMILARITY_METRIC,
				min_distance=AUGMENT_SPACER_DELETION_MIN_DISTANCE,
				target_additions=augment_spacer_deletion_count,
				balance_per_array=True,
			)
			train_indices = list(train_indices) + train_new_indices
			train_examples = [dataset.records[index] for index in train_indices]
			print(
				"Spacer deletion augmentation summary: "
				f"requested={augment_spacer_deletion_count} added={train_aug_stats['added']} "
				f"blocked_overlap={train_aug_stats['blocked_overlap']} "
				f"blocked_similarity={train_aug_stats['blocked_similarity']} "
				f"skipped_short={train_aug_stats['skipped_short']}"
			)

	tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
	if tokenizer.pad_token is None:
		if tokenizer.eos_token is not None:
			tokenizer.pad_token = tokenizer.eos_token
		elif tokenizer.unk_token is not None:
			tokenizer.pad_token = tokenizer.unk_token
		else:
			raise ValueError("Carbon tokenizer does not define a pad, eos, or unk token.")
	tokenizer.padding_side = "right"

	model_kwargs: dict[str, Any] = {
		"num_labels": 2,
		"id2label": {0: "Reverse", 1: "Forward"},
		"label2id": {"Reverse": 0, "Forward": 1},
		"trust_remote_code": True,
		"ignore_mismatched_sizes": True,
	}
	if args.load_in_4bit and args.load_in_8bit:
		raise ValueError("Choose only one of --load_in_4bit or --load_in_8bit.")
	if args.load_in_4bit:
		model_kwargs["load_in_4bit"] = True
	if args.load_in_8bit:
		model_kwargs["load_in_8bit"] = True
	if use_bfloat16:
		model_kwargs["dtype"] = torch.bfloat16

	model = AutoModelForSequenceClassification.from_pretrained(args.model_id, **model_kwargs)
	model.config.pad_token_id = tokenizer.pad_token_id
	if hasattr(model.config, "use_cache"):
		model.config.use_cache = False
	if args.gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
		model.gradient_checkpointing_enable()
	if args.freeze_backbone or args.lora_only_train:
		_freeze_backbone(model)
	if args.use_lora:
		model = _apply_lora(
			model,
			task_type=args.lora_task_type,
			preset=args.lora_preset,
			explicit_modules=args.lora_target_modules,
			r=args.lora_r,
			alpha=args.lora_alpha,
			dropout=args.lora_dropout,
			bias=args.lora_bias,
			use_kbit_training=args.lora_kbit_training,
		)
		if args.lora_only_train:
			for name, parameter in model.named_parameters():
				if "lora_" not in name and "classifier" not in name and "score" not in name:
					parameter.requires_grad = False

	model.to(device)

	# output params to see if lora is working
	total = sum(p.numel() for p in model.parameters())
	trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

	print(f"Trainable params: {trainable:,}")
	print(f"Total params: {total:,}")
	print(f"Trainable %: {100*trainable/total:.4f}%")

	train_dataset = CarbonDirectionDataset(
		train_examples,
		tokenizer=tokenizer,
		max_length=args.max_length,
		include_flanks=args.include_flanks,
		sequence_mode=args.sequence_mode,
	)
	val_dataset = CarbonDirectionDataset(
		val_examples,
		tokenizer=tokenizer,
		max_length=args.max_length,
		include_flanks=args.include_flanks,
		sequence_mode=args.sequence_mode,
	)
	test_dataset = CarbonDirectionDataset(
		test_examples,
		tokenizer=tokenizer,
		max_length=args.max_length,
		include_flanks=args.include_flanks,
		sequence_mode=args.sequence_mode,
	)

	collator = CarbonBatchCollator(tokenizer)
	train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=collator)
	val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collator)
	test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collator)

	train_labels = [example.label for example in train_examples]
	class_weights = _build_label_weights(train_labels).to(device)
	loss_fn = nn.CrossEntropyLoss(weight=class_weights)

	trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
	if not trainable_parameters:
		raise ValueError("No trainable parameters remain after applying the current fine-tuning settings.")

	optimizer = torch.optim.AdamW(trainable_parameters, lr=args.learning_rate, weight_decay=args.weight_decay)
	total_steps = max(1, math.ceil(len(train_loader) / max(1, args.gradient_accumulation_steps)) * args.epochs)
	warmup_steps = max(0, round(total_steps * args.warmup_ratio))
	scheduler = None
	if get_linear_schedule_with_warmup is not None:
		scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

	print(f"Training Carbon-500M from {args.model_id}")
	print(f"Device: {device} | bf16={use_bfloat16} | max_length={args.max_length} | batch_size={args.batch_size}")
	print(f"LoRA: enabled={args.use_lora} | r={args.lora_r} | alpha={args.lora_alpha} | dropout={args.lora_dropout} | preset={args.lora_preset} | target_modules={args.lora_target_modules or 'preset'}")
	if device.type == "cuda":
		hint = _cuda_memory_hint(device)
		if hint:
			print(hint)
	print(f"Class weights: {[float(x) for x in class_weights.detach().cpu().tolist()]}")

	if args.dry_run:
		try:
			sample_batch = next(iter(train_loader))
			sample_batch = _batch_to_device(sample_batch, device)
			autocast_context = torch.autocast("cuda", dtype=torch.bfloat16) if use_bfloat16 else nullcontext()
			with torch.no_grad(), autocast_context:
				outputs = model(input_ids=sample_batch.get("input_ids"), attention_mask=sample_batch.get("attention_mask"))
			print(f"Dry run logits shape: {tuple(outputs.logits.shape)}")
			return 0
		except torch.cuda.OutOfMemoryError as exc:
			raise RuntimeError(
				"Carbon dry_run ran out of CUDA memory. Try one of: --cpu, --max_length 256, --batch_size 1, "
				"--freeze_backbone, or free GPU memory before rerunning."
			) from exc

	best_val_metric = float("-inf")
	best_val_metrics: dict[str, float] | None = None
	best_epoch = 0
	autocast_context = torch.autocast("cuda", dtype=torch.bfloat16) if use_bfloat16 else nullcontext()

	for epoch in range(1, args.epochs + 1):
		model.train()
		optimizer.zero_grad(set_to_none=True)
		running_loss = 0.0
		batches_seen = 0
		epoch_start = time.time()

		try:
			for step, batch in enumerate(train_loader, start=1):
				batch = _batch_to_device(batch, device)
				with autocast_context:
					outputs = model(input_ids=batch.get("input_ids"), attention_mask=batch.get("attention_mask"))
					loss = loss_fn(outputs.logits, batch["labels"]) / max(1, args.gradient_accumulation_steps)

				loss.backward()
				running_loss += float(loss.item()) * max(1, args.gradient_accumulation_steps)
				batches_seen += 1

				if step % args.gradient_accumulation_steps == 0 or step == len(train_loader):
					torch.nn.utils.clip_grad_norm_(trainable_parameters, max_norm=1.0)
					optimizer.step()
					if scheduler is not None:
						scheduler.step()
					optimizer.zero_grad(set_to_none=True)
		except torch.cuda.OutOfMemoryError as exc:
			raise RuntimeError(
				"Carbon training ran out of CUDA memory. Try --max_length 256, --batch_size 1, --freeze_backbone, "
				"or --cpu. If other GPU jobs are running, free memory and rerun."
			) from exc

		train_loss = _safe_divide(running_loss, batches_seen)
		val_metrics = _evaluate_model(model, val_loader, device, loss_fn, use_bfloat16)
		current_metric = val_metrics["mcc"]
		if math.isnan(current_metric):
			current_metric = val_metrics["f1"]
		
		epoch_end = time.time()
		epoch_duration = epoch_end - epoch_start
		epoch_duration_minutes = int(epoch_duration // 60)
		epoch_duration_seconds = int(epoch_duration % 60)

		print(f"Epoch {epoch:02d} | time={epoch_duration_minutes:.2f}min:{epoch_duration_seconds:.2f}s | train_loss={train_loss:.4f} | val={_format_metrics(val_metrics)}")

		if current_metric > best_val_metric:
			best_val_metric = current_metric
			best_val_metrics = dict(val_metrics)
			best_epoch = epoch
			model.save_pretrained(output_dir)
			tokenizer.save_pretrained(output_dir)
			with (output_dir / "best_metrics.json").open("w") as fh:
				json.dump(
					{
						"epoch": epoch,
						"metric_name": "mcc",
						"metric_value": current_metric,
						"metrics": best_val_metrics,
						"args": vars(args),
					},
					fh,
					indent=2,
					sort_keys=True,
				)

	if best_val_metrics is None:
		best_val_metrics = _evaluate_model(model, val_loader, device, loss_fn, use_bfloat16)

	test_metrics = _evaluate_model(model, test_loader, device, loss_fn, use_bfloat16)
	print(f"Best epoch: {best_epoch} | best_val={_format_metrics(best_val_metrics)}")
	print(f"Test: {_format_metrics(test_metrics)}")

	with (output_dir / "training_summary.json").open("w") as fh:
		json.dump(
			{
				"args": vars(args),
				"best_epoch": best_epoch,
				"best_validation": best_val_metrics,
				"test": test_metrics,
				"train_examples": len(train_examples),
				"val_examples": len(val_examples),
				"test_examples": len(test_examples),
			},
			fh,
			indent=2,
			sort_keys=True,
		)

	return 0


if __name__ == "__main__":
	raise SystemExit(main())
