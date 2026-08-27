from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
import numpy as np
from collections import Counter, defaultdict
from contextlib import nullcontext
from dataclasses import asdict, replace
from pathlib import Path
from importlib import import_module
from typing import Any
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    matthews_corrcoef, roc_auc_score, confusion_matrix,
)

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
	sys.path.insert(0, str(ROOT))

from direction_learning.augmentation import (
	build_test_similarity_index,
	example_signature,
	materialize_subarray_augmentations,
)

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

from direction_learning.data import print_split_overlap_report, split_dev_pool_by_mode, stratified_holdout_by_mode
from direction_learning.dataset import DirectionExample, DirectionJsonlDataset
from direction_learning.tokenization import reverse_complement

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
	subtype_counts_sorted = dict(sorted(subtype_counts.items(), key=lambda item: (-item[1], item[0])))
	print(
		f"{name}: {len(examples)} examples | "
		f"Forward={label_counts.get(1, 0)} Reverse={label_counts.get(0, 0)} | "
		f"Subtypes={subtype_counts_sorted}"
	)


def _print_split_group_names(name: str, examples: list[DirectionExample], indices: list[int], *, max_names: int = 30) -> None:
	group_names = sorted({_example_group_name(examples[index], index) for index in indices})
	if not group_names:
		print(f"{name} groups: none")
		return

	display_names = group_names[:max_names]
	if len(group_names) > max_names:
		display_names.append(f"... (+{len(group_names) - max_names} more)")

	print(f"{name} groups ({len(group_names)}): {', '.join(display_names)}")


def _print_group_overlap_report(examples: list[DirectionExample], split_indices: dict[str, list[int]]) -> None:
	group_to_splits: dict[str, set[str]] = defaultdict(set)
	for split_name, indices in split_indices.items():
		for index in indices:
			group_to_splits[_example_group_name(examples[index], index)].add(split_name)

	overlaps = sorted(
		(group_name, sorted(split_names))
		for group_name, split_names in group_to_splits.items()
		if len(split_names) > 1
	)
	if not overlaps:
		print("[overlap] no group names appear in multiple splits")
		return

	print(f"[overlap] {len(overlaps)} group names appear in multiple splits:")
	for group_name, split_names in overlaps[:30]:
		print(f"[overlap] {group_name}: {', '.join(split_names)}")
	if len(overlaps) > 30:
		print(f"[overlap] ... (+{len(overlaps) - 30} more)")


def _example_group_name(example: DirectionExample, fallback_index: int) -> str:
	group_name = (example.group_name or "").strip()
	if group_name:
		return group_name
	return f"ungrouped_{fallback_index}"


def _split_group_anchor_name(example: DirectionExample, fallback_index: int) -> str:
	"""
	Return the group anchor used by --split_group.

	For cluster-style names like:
	  orig_<parent>__sub_0002__g_000325
	we collapse to the shared parent key:
	  orig_<parent>
	so all sub-clusters from the same original large group are kept together.
	"""
	group_name = _example_group_name(example, fallback_index)
	if group_name.startswith("orig_") and "__sub_" in group_name:
		return group_name.split("__sub_", 1)[0]
	return group_name


def _build_group_components(examples: list[DirectionExample]) -> list[list[int]]:
	"""Build connected components that keep identical groups and signatures together."""
	n = len(examples)
	parent = list(range(n))

	def find(index: int) -> int:
		while parent[index] != index:
			parent[index] = parent[parent[index]]
			index = parent[index]
		return index

	def union(left: int, right: int) -> None:
		left_root = find(left)
		right_root = find(right)
		if left_root != right_root:
			parent[right_root] = left_root

	first_by_group: dict[str, int] = {}
	first_by_signature: dict[tuple[tuple[str, ...], tuple[str, ...]], int] = {}
	for index, example in enumerate(examples):
		group_name = _split_group_anchor_name(example, index)
		previous = first_by_group.get(group_name)
		if previous is None:
			first_by_group[group_name] = index
		else:
			union(index, previous)

		signature = _canonical_array_signature(example)
		previous = first_by_signature.get(signature)
		if previous is None:
			first_by_signature[signature] = index
		else:
			union(index, previous)

	components: dict[int, list[int]] = {}
	for index in range(n):
		components.setdefault(find(index), []).append(index)

	return [sorted(indices) for indices in components.values()]


def _component_stratum_key(example: DirectionExample, stratify_mode: str) -> Any:
	subtype = (example.cas_subtype or "Unknown").strip() or "Unknown"
	label = int(example.label)
	if stratify_mode == "label":
		return label
	if stratify_mode == "cas_subtype":
		return subtype
	return (subtype, label)


def _split_components_by_mode(
	examples: list[DirectionExample],
	components: list[list[int]],
	seed: int,
	right_fraction: float,
	stratify_mode: str,
	size_aware: bool = False,
) -> tuple[list[list[int]], list[list[int]]]:
	"""Split connected components into left/right partitions while keeping each component intact."""
	if not (0.0 <= right_fraction < 1.0):
		raise ValueError("right_fraction must be in [0.0, 1.0)")

	rng = random.Random(seed)
	strata_groups: dict[Any, list[list[int]]] = {}
	for component in components:
		representative = examples[component[0]]
		key = _component_stratum_key(representative, stratify_mode)
		strata_groups.setdefault(key, []).append(component)

	left_components: list[list[int]] = []
	right_components: list[list[int]] = []
	for key in sorted(strata_groups.keys(), key=str):
		groups = list(strata_groups[key])
		rng.shuffle(groups)
		n_groups = len(groups)
		if n_groups == 1:
			n_right = 0 if right_fraction == 0.0 else 1
		elif size_aware:
			total_examples = sum(len(component) for component in groups)
			target_examples = total_examples * right_fraction
			best_cutoff = 1 if right_fraction > 0.0 else 0
			best_distance = float("inf")
			best_right_examples = float("inf")
			cumulative_examples = 0
			for cutoff in range(1, n_groups):
				cumulative_examples += len(groups[cutoff - 1])
				distance = abs(cumulative_examples - target_examples)
				if distance < best_distance or (distance == best_distance and cumulative_examples < best_right_examples):
					best_distance = distance
					best_right_examples = cumulative_examples
					best_cutoff = cutoff
			n_right = best_cutoff if right_fraction > 0.0 else 0
		else:
			n_right = min(n_groups - 1, max(1, round(n_groups * right_fraction)))
		for component in groups[:n_right]:
			right_components.append(component)
		for component in groups[n_right:]:
			left_components.append(component)

	return left_components, right_components


def _flatten_components(components: list[list[int]]) -> list[int]:
	indices = [index for component in components for index in component]
	return sorted(indices)


def _build_splits_once(
	examples: list[DirectionExample],
	seed: int,
	test_fraction: float,
	stratify_mode: str,
	split_group: bool,
	size_aware: bool = False,
) -> dict[str, list[int]]:
	if split_group:
		components = _build_group_components(examples)
		dev_components, test_components = _split_components_by_mode(
			examples,
			components,
			seed=seed,
			right_fraction=test_fraction,
			stratify_mode=stratify_mode,
			size_aware=size_aware,
		)
		train_components, val_components = _split_components_by_mode(
			examples,
			dev_components,
			seed=seed,
			right_fraction=0.20,
			stratify_mode=stratify_mode,
			size_aware=size_aware,
		)
		train_indices = _flatten_components(train_components)
		val_indices = _flatten_components(val_components)
		test_indices = _flatten_components(test_components)
	else:
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

	return {"train": train_indices, "val": val_indices, "test": test_indices}


def _build_example_k_subarray_index(
	examples: list[DirectionExample],
	k: int,
) -> list[set[tuple[str, ...]]]:
	per_example: list[set[tuple[str, ...]]] = []
	for example in examples:
		spacers = tuple(_normalize_dna_sequence(sequence) for sequence in example.spacers if sequence)
		current: set[tuple[str, ...]] = set()
		if len(spacers) >= k:
			for start in range(len(spacers) - k + 1):
				current.add(_canonical_sequence_tuple(spacers[start:start + k]))
		per_example.append(current)
	return per_example


def _subarray_leakage_fraction_from_index(
	reference_indices: list[int],
	query_indices: list[int],
	per_example_subarrays: list[set[tuple[str, ...]]],
) -> tuple[float, int]:
	if not query_indices:
		return 0.0, 0

	reference_subarrays: set[tuple[str, ...]] = set()
	for index in reference_indices:
		reference_subarrays.update(per_example_subarrays[index])

	leak_count = 0
	for index in query_indices:
		query_subarrays = per_example_subarrays[index]
		if query_subarrays and any(subarray in reference_subarrays for subarray in query_subarrays):
			leak_count += 1

	return leak_count / len(query_indices), leak_count


def _split_candidate_objective(
	candidate: dict[str, list[int]],
	*,
	n_examples: int,
	target_test_fraction: float,
	optimize_target: str,
	per_example_subarrays: list[set[tuple[str, ...]]],
	size_aware: bool = False,
	size_weight: float = 25.0,
) -> dict[str, float]:
	train_indices = candidate["train"]
	val_indices = candidate["val"]
	test_indices = candidate["test"]
	train_all_indices = train_indices + val_indices

	test_leak_fraction, test_leak_count = _subarray_leakage_fraction_from_index(
		train_indices,
		test_indices,
		per_example_subarrays,
	)
	val_leak_fraction, val_leak_count = _subarray_leakage_fraction_from_index(
		train_indices,
		val_indices,
		per_example_subarrays,
	)
	val_test_leak_fraction, val_test_leak_count = _subarray_leakage_fraction_from_index(
		val_indices,
		test_indices,
		per_example_subarrays,
	)
	train_all_test_leak_fraction, train_all_test_leak_count = _subarray_leakage_fraction_from_index(
		train_all_indices,
		test_indices,
		per_example_subarrays,
	)

	observed_test_fraction = _safe_divide(len(test_indices), n_examples)
	target_val_fraction = (1.0 - target_test_fraction) * 0.20
	observed_val_fraction = _safe_divide(len(val_indices), n_examples)
	test_size_penalty = abs(observed_test_fraction - target_test_fraction)
	val_size_penalty = abs(observed_val_fraction - target_val_fraction)
	size_penalty = test_size_penalty + val_size_penalty

	worst_query_leak = max(test_leak_fraction, val_leak_fraction)
	leak_balance_penalty = abs(test_leak_fraction - val_leak_fraction)
	effective_size_weight = size_weight if size_aware else 0.05

	if optimize_target == "test":
		objective = test_leak_fraction + (effective_size_weight * (test_size_penalty if size_aware else size_penalty))
	elif optimize_target == "train_all_test":
		objective = train_all_test_leak_fraction + (effective_size_weight * (test_size_penalty if size_aware else size_penalty))
	elif optimize_target in {"all_pairs", "all_pais", "all-pairs"}:
		pair_max = max(test_leak_fraction, val_leak_fraction, val_test_leak_fraction)
		pair_min = min(test_leak_fraction, val_leak_fraction, val_test_leak_fraction)
		pair_balance_penalty = pair_max - pair_min
		objective = pair_max + (0.5 * pair_balance_penalty) + (effective_size_weight * (test_size_penalty if size_aware else size_penalty))
	elif optimize_target == "both":
		objective = worst_query_leak + (0.5 * leak_balance_penalty) + (effective_size_weight * (test_size_penalty if size_aware else size_penalty))
	else:
		raise ValueError(f"Unknown split optimization target: {optimize_target}")

	return {
		"objective": objective,
		"test_leak_fraction": test_leak_fraction,
		"test_leak_count": float(test_leak_count),
		"val_leak_fraction": val_leak_fraction,
		"val_leak_count": float(val_leak_count),
		"val_test_leak_fraction": val_test_leak_fraction,
		"val_test_leak_count": float(val_test_leak_count),
		"train_all_test_leak_fraction": train_all_test_leak_fraction,
		"train_all_test_leak_count": float(train_all_test_leak_count),
		"worst_query_leak": worst_query_leak,
		"leak_balance_penalty": leak_balance_penalty,
		"test_size_penalty": test_size_penalty,
		"val_size_penalty": val_size_penalty,
		"size_penalty": size_penalty,
	}


def _build_reference_similarity_indexes(examples: list[DirectionExample]) -> dict[str, Any]:
	full_signatures: set[tuple[tuple[str, ...], tuple[str, ...]]] = set()
	reference_spacer_tokens: set[str] = set()
	reference_subarrays: dict[int, set[tuple[str, ...]]] = {k: set() for k in (2, 3, 4, 5)}
	reference_kmers: set[str] = set()

	for example in examples:
		full_signatures.add(_canonical_array_signature(example))
		for spacer in example.spacers:
			if spacer:
				reference_spacer_tokens.add(_canonical_dna_sequence(spacer))

		spacers = tuple(_normalize_dna_sequence(sequence) for sequence in example.spacers if sequence)
		for k, k_subarrays in reference_subarrays.items():
			if len(spacers) < k:
				continue
			for start in range(len(spacers) - k + 1):
				k_subarrays.add(_canonical_sequence_tuple(spacers[start:start + k]))

		core_sequence = _array_core_sequence(example)
		if len(core_sequence) >= 31:
			for start in range(len(core_sequence) - 31 + 1):
				reference_kmers.add(_canonical_dna_sequence(core_sequence[start:start + 31]))

	return {
		"raw_count": len(examples),
		"full_signatures": full_signatures,
		"spacer_tokens": reference_spacer_tokens,
		"subarrays": reference_subarrays,
		"kmers": reference_kmers,
	}


def _normalize_dna_sequence(sequence: str) -> str:
	return _normalize_dna(sequence)


def _canonical_dna_sequence(sequence: str) -> str:
	normalized = _normalize_dna_sequence(sequence)
	reverse_complemented = _normalize_dna_sequence(reverse_complement(normalized))
	return min(normalized, reverse_complemented)


def _canonical_sequence_tuple(sequences: list[str] | tuple[str, ...]) -> tuple[str, ...]:
	normalized = tuple(_normalize_dna_sequence(sequence) for sequence in sequences)
	reverse_complemented = tuple(
		_normalize_dna_sequence(reverse_complement(sequence)) for sequence in reversed(normalized)
	)
	return min(normalized, reverse_complemented)


def _array_core_sequence(example: DirectionExample) -> str:
	pieces: list[str] = []
	max_length = max(len(example.repeats), len(example.spacers))
	for index in range(max_length):
		if index < len(example.repeats):
			pieces.append(_normalize_dna_sequence(example.repeats[index]))
		if index < len(example.spacers):
			pieces.append(_normalize_dna_sequence(example.spacers[index]))
	return "".join(piece for piece in pieces if piece)


def _canonical_array_signature(example: DirectionExample) -> tuple[tuple[str, ...], tuple[str, ...]]:
	spacers = tuple(_normalize_dna_sequence(sequence) for sequence in example.spacers)
	repeats = tuple(_normalize_dna_sequence(sequence) for sequence in example.repeats)
	reverse_complemented = (
		tuple(_normalize_dna_sequence(reverse_complement(sequence)) for sequence in reversed(spacers)),
		tuple(_normalize_dna_sequence(reverse_complement(sequence)) for sequence in reversed(repeats)),
	)
	return min((spacers, repeats), reverse_complemented)


def _format_percent(value: float) -> str:
	return f"{value * 100:.2f}%"


def _format_ratio(numerator: int, denominator: int) -> str:
	return f"{numerator} / {denominator}  ({_format_percent(numerator / denominator if denominator else 0.0)})"


def _format_decimal_percentage(value: float) -> str:
	return f"{value * 100:.1f}%"


def _bucket_coverage_fraction(value: float) -> str:
	if value == 0.0:
		return "0"
	if value < 0.25:
		return "(0,0.25]"
	if value < 0.5:
		return "(0.25,0.5]"
	if value < 0.75:
		return "(0.5,0.75]"
	if value < 1.0:
		return "(0.75,1.0)"
	return "1.0"


def _sample_multi_repeat_records(examples: list[DirectionExample], sample_size: int, seed: int) -> list[DirectionExample]:
	multi_repeat_examples = [example for example in examples if len(example.repeats) >= 2]
	if not multi_repeat_examples:
		return []
	if sample_size <= 0 or len(multi_repeat_examples) <= sample_size:
		return list(multi_repeat_examples)
	rng = random.Random(seed)
	return rng.sample(multi_repeat_examples, sample_size)


def _analyze_parse_verification(examples: list[DirectionExample], sample_size: int, seed: int) -> dict[str, Any]:
	sampled_examples = _sample_multi_repeat_records(examples, sample_size, seed)
	repeat_modal_fractions: list[float] = []
	spacer_distinct_ratios: list[float] = []
	for example in sampled_examples:
		repeats = [_normalize_dna_sequence(sequence) for sequence in example.repeats if sequence]
		spacers = [_normalize_dna_sequence(sequence) for sequence in example.spacers if sequence]
		if repeats:
			repeat_counts = Counter(repeats)
			repeat_modal_fractions.append(max(repeat_counts.values()) / len(repeats))
		if spacers:
			spacer_distinct_ratios.append(len(set(spacers)) / len(spacers))

	records_with_consensus = sum(1 for fraction in repeat_modal_fractions if fraction >= 0.80)
	records_with_identical_repeats = sum(1 for fraction in repeat_modal_fractions if math.isclose(fraction, 1.0, rel_tol=0.0, abs_tol=1e-12))
	return {
		"sampled_multi_repeat_records": len(sampled_examples),
		"mean_repeat_modal_fraction": (sum(repeat_modal_fractions) / len(repeat_modal_fractions)) if repeat_modal_fractions else 0.0,
		"records_with_ge_80_percent_repeats_on_single_consensus": _safe_divide(records_with_consensus, len(repeat_modal_fractions)),
		"records_with_all_repeats_byte_identical": _safe_divide(records_with_identical_repeats, len(repeat_modal_fractions)),
		"mean_spacer_distinct_ratio": (sum(spacer_distinct_ratios) / len(spacer_distinct_ratios)) if spacer_distinct_ratios else 0.0,
	}


def _analyze_query_against_reference(
	reference_name: str,
	reference_indexes: dict[str, Any],
	query_name: str,
	query_examples: list[DirectionExample],
	seed: int,
) -> dict[str, Any]:
	reference_full_signatures = reference_indexes["full_signatures"]
	reference_spacer_tokens = reference_indexes["spacer_tokens"]
	reference_subarrays = reference_indexes["subarrays"]
	reference_kmers = reference_indexes["kmers"]

	full_matches = 0
	coverage_fractions: list[float] = []
	coverage_histogram = Counter()
	k_subarray_counts = {k: 0 for k in (2, 3, 4, 5)}
	kmer_matches = 0

	for example in query_examples:
		if _canonical_array_signature(example) in reference_full_signatures:
			full_matches += 1

		spacers = [_normalize_dna_sequence(sequence) for sequence in example.spacers if sequence]
		if spacers:
			covered = sum(1 for spacer in spacers if _canonical_dna_sequence(spacer) in reference_spacer_tokens)
			coverage_fraction = covered / len(spacers)
		else:
			coverage_fraction = 0.0
		coverage_fractions.append(coverage_fraction)
		coverage_histogram[_bucket_coverage_fraction(coverage_fraction)] += 1

		for k in (2, 3, 4, 5):
			if len(spacers) < k:
				continue
			if any(_canonical_sequence_tuple(spacers[start:start + k]) in reference_subarrays[k] for start in range(len(spacers) - k + 1)):
				k_subarray_counts[k] += 1

		core_sequence = _array_core_sequence(example)
		if len(core_sequence) >= 31:
			if any(_canonical_dna_sequence(core_sequence[start:start + 31]) in reference_kmers for start in range(len(core_sequence) - 31 + 1)):
				kmer_matches += 1

	parse_stats = _analyze_parse_verification(query_examples, sample_size=1000, seed=seed)
	return {
		"reference_name": reference_name,
		"reference_size": reference_indexes["raw_count"],
		"query_name": query_name,
		"query_size": len(query_examples),
		"parse_stats": parse_stats,
		"exact_full_array_matches": full_matches,
		"coverage_histogram": dict(coverage_histogram),
		"coverage_100_count": sum(1 for fraction in coverage_fractions if math.isclose(fraction, 1.0, rel_tol=0.0, abs_tol=1e-12)),
		"k_subarray_counts": k_subarray_counts,
		"kmer_matches": kmer_matches,
	}


def _write_split_artifacts(
	output_dir: Path,
	dataset_path: Path,
	seed: int,
	chunk_size: int,
	examples: list[DirectionExample],
	train_examples: list[DirectionExample],
	val_examples: list[DirectionExample],
	test_examples: list[DirectionExample],
	train_indices: list[int],
	val_indices: list[int],
	test_indices: list[int],
) -> Path:
	split_dir = output_dir / "splits"
	split_dir.mkdir(parents=True, exist_ok=True)

	def _build_split_group_name_map(indices: list[int]) -> dict[int, str]:
		groups: dict[str, list[int]] = defaultdict(list)
		for index in indices:
			groups[_example_group_name(examples[index], index)].append(index)

		name_by_index: dict[int, str] = {}
		for group_name, group_indices in groups.items():
			ordered = sorted(group_indices)
			if chunk_size <= 0 or len(ordered) <= chunk_size:
				for index in ordered:
					name_by_index[index] = group_name
				continue

			for part_index, start in enumerate(range(0, len(ordered), chunk_size)):
				chunk_name = f"{group_name}__part{part_index:03d}"
				for index in ordered[start:start + chunk_size]:
					name_by_index[index] = chunk_name
		return name_by_index

	train_group_names = _build_split_group_name_map(train_indices)
	val_group_names = _build_split_group_name_map(val_indices)
	test_group_names = _build_split_group_name_map(test_indices)
	train_all_group_names = dict(train_group_names)
	train_all_group_names.update(val_group_names)

	def _write_split_jsonl(path: Path, indices: list[int], examples: list[DirectionExample], group_name_map: dict[int, str]) -> None:
		with path.open("w") as fh:
			for index, example in zip(indices, examples):
				fh.write(
					json.dumps(
						{
							"index": index,
							"group_name": example.group_name,
							"split_group_name": group_name_map.get(index),
							"example": asdict(example),
						},
						sort_keys=True,
					)
					+ "\n"
				)

	_write_split_jsonl(split_dir / "train.jsonl", train_indices, train_examples, train_group_names)
	_write_split_jsonl(split_dir / "val.jsonl", val_indices, val_examples, val_group_names)
	_write_split_jsonl(split_dir / "train_all.jsonl", train_indices + val_indices, train_examples + val_examples, train_all_group_names)
	_write_split_jsonl(split_dir / "test.jsonl", test_indices, test_examples, test_group_names)

	manifest = {
		"dataset_path": str(dataset_path),
		"seed": seed,
		"split_group_chunk_size": chunk_size,
		"split_dir": str(split_dir),
		"splits": {
			"train": {"count": len(train_examples), "indices": train_indices, "path": str(split_dir / "train.jsonl")},
			"val": {"count": len(val_examples), "indices": val_indices, "path": str(split_dir / "val.jsonl")},
			"train_all": {"count": len(train_examples) + len(val_examples), "indices": train_indices + val_indices, "path": str(split_dir / "train_all.jsonl")},
			"test": {"count": len(test_examples), "indices": test_indices, "path": str(split_dir / "test.jsonl")},
		},
	}
	manifest["split_groups"] = {
		"train": sorted(set(train_group_names.values())),
		"val": sorted(set(val_group_names.values())),
		"train_all": sorted(set(train_all_group_names.values())),
		"test": sorted(set(test_group_names.values())),
	}
	manifest_path = split_dir / "split_manifest.json"
	with manifest_path.open("w") as fh:
		json.dump(manifest, fh, indent=2, sort_keys=True)
	return manifest_path


def _build_split_similarity_report(
	dataset_path: Path,
	seed: int,
	train_examples: list[DirectionExample],
	val_examples: list[DirectionExample],
	test_examples: list[DirectionExample],
) -> dict[str, Any]:
	train_all_examples = train_examples + val_examples
	train_reference_indexes = _build_reference_similarity_indexes(train_examples)
	train_all_reference_indexes = _build_reference_similarity_indexes(train_all_examples)
	sections = [
		_analyze_query_against_reference("train", train_reference_indexes, "val", val_examples, seed),
		_analyze_query_against_reference("train", train_reference_indexes, "test", test_examples, seed),
		_analyze_query_against_reference("train_all", train_all_reference_indexes, "test", test_examples, seed),
	]
	return {
		"dataset_path": str(dataset_path),
		"seed": seed,
		"sections": sections,
	}


def _save_similarity_report(output_dir: Path, report: dict[str, Any]) -> None:
	text_path = output_dir / "split_similarity_report.txt"
	json_path = output_dir / "split_similarity_report.json"
	with json_path.open("w") as fh:
		json.dump(report, fh, indent=2, sort_keys=True)

	lines: list[str] = []
	for section in report["sections"]:
		lines.append(f"{section['reference_name']} arrays (reference): {section['reference_size']}")
		lines.append(f"{section['query_name']} arrays (queried):    {section['query_size']}")
		parse_stats = section["parse_stats"]
		lines.append("parse verification (even-index=repeats, odd-index=spacers):")
		lines.append(f"    sampled multi-repeat records: {parse_stats['sampled_multi_repeat_records']}")
		lines.append(f"    mean repeat modal-fraction (repeats matching the record's consensus): {_format_decimal_percentage(parse_stats['mean_repeat_modal_fraction'])}")
		lines.append(f"    records with >=80% repeats on a single consensus: {_format_decimal_percentage(parse_stats['records_with_ge_80_percent_repeats_on_single_consensus'])}")
		lines.append(f"    records with all repeats byte-identical: {_format_decimal_percentage(parse_stats['records_with_all_repeats_byte_identical'])}")
		lines.append(f"    mean spacer distinct-ratio (unique spacers / spacers): {_format_decimal_percentage(parse_stats['mean_spacer_distinct_ratio'])}  (expected ~100%)")
		lines.append("")
		lines.append("Check 1 - Exact full-array leakage (fwd or RC identical to a reference array):")
		lines.append(f"    {_format_ratio(section['exact_full_array_matches'], section['query_size'])}")
		lines.append("")
		lines.append("Check 2 - Single-spacer overlap (fraction of query spacers seen in reference, fwd/RC):")
		lines.append("    coverage-fraction histogram:")
		for bucket in ["0", "(0,0.25]", "(0.25,0.5]", "(0.5,0.75]", "(0.75,1.0)", "1.0"]:
			count = section["coverage_histogram"].get(bucket, 0)
			lines.append(f"        {bucket:<14} {count:>4}  ({_format_percent(count / section['query_size'] if section['query_size'] else 0.0)})")
		lines.append(f"    {section['query_name']} arrays with 100% spacer coverage in reference: {_format_ratio(section['coverage_100_count'], section['query_size'])}")
		lines.append("")
		lines.append("Check 3 - Contiguous sub-array leakage (PRIMARY; fwd or RC):")
		lines.append("    k    query arrays sharing a contiguous k-spacer sub-array with reference")
		for k in (2, 3, 4, 5):
			count = section["k_subarray_counts"][k]
			lines.append(f"    {k:<4}{count:>6} / {section['query_size']:<4} ({_format_percent(count / section['query_size'] if section['query_size'] else 0.0)})")
		lines.append("")
		lines.append("Check 4 - Nucleotide k-mer leakage (>= 31 bp exact stretch, fwd or RC):")
		lines.append(f"    {_format_ratio(section['kmer_matches'], section['query_size'])}")
		lines.append("")

	text = "\n".join(lines).rstrip() + "\n"
	with text_path.open("w") as fh:
		fh.write(text)
	print(text)


def _build_splits(
	examples: list[DirectionExample],
	seed: int,
	test_fraction: float,
	stratify_mode: str,
	split_group: bool,
	split_optimize_trials: int,
	split_optimize_k: int,
	split_optimize_target: str,
	split_size_aware: bool = False,
	split_size_weight: float = 25.0,
) -> dict[str, list[int]]:
	if split_optimize_k < 2:
		raise ValueError("--split_optimize_k must be >= 2")
	if split_optimize_target not in {"both", "test", "train_all_test", "all_pairs", "all_pais", "all-pairs"}:
		raise ValueError("--split_optimize_target must be one of: both, test, train_all_test, all_pairs, all-pais, all_pais")

	trials = max(1, int(split_optimize_trials))
	if trials == 1:
		splits = _build_splits_once(
			examples,
			seed=seed,
			test_fraction=test_fraction,
			stratify_mode=stratify_mode,
			split_group=split_group,
			size_aware=split_size_aware,
		)
	else:
		per_example_subarrays = _build_example_k_subarray_index(examples, k=split_optimize_k)
		best_splits: dict[str, list[int]] | None = None
		best_stats: dict[str, float] | None = None
		best_seed = seed
		for offset in range(trials):
			trial_seed = seed + offset
			candidate = _build_splits_once(
				examples,
				seed=trial_seed,
				test_fraction=test_fraction,
				stratify_mode=stratify_mode,
				split_group=split_group,
				size_aware=split_size_aware,
			)
			candidate_stats = _split_candidate_objective(
				candidate,
				n_examples=len(examples),
				target_test_fraction=test_fraction,
				optimize_target=split_optimize_target,
				per_example_subarrays=per_example_subarrays,
				size_aware=split_size_aware,
				size_weight=split_size_weight,
			)
			if best_stats is None or candidate_stats["objective"] < best_stats["objective"]:
				best_splits = candidate
				best_stats = candidate_stats
				best_seed = trial_seed

		if best_splits is None or best_stats is None:
			raise RuntimeError("split optimization failed to produce a candidate split")
		splits = best_splits
		print(
			"[split optimize] "
			f"trials={trials} k={split_optimize_k} target={split_optimize_target} selected_seed={best_seed} "
			f"objective={best_stats['objective']:.4f} "
			f"worst_k{split_optimize_k}={best_stats['worst_query_leak']:.4f} "
			f"test_k{split_optimize_k}={best_stats['test_leak_fraction']:.4f} "
			f"train_all->test_k{split_optimize_k}={best_stats['train_all_test_leak_fraction']:.4f} "
			f"val_k{split_optimize_k}={best_stats['val_leak_fraction']:.4f} "
			f"val->test_k{split_optimize_k}={best_stats['val_test_leak_fraction']:.4f} "
			f"balance={best_stats['leak_balance_penalty']:.4f} "
			f"size_penalty={best_stats['size_penalty']:.4f}"
		)

	train_indices = splits["train"]
	val_indices = splits["val"]
	test_indices = splits["test"]
	print_split_overlap_report(train_indices, val_indices, test_indices, examples)

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
	metrics, _, _ = _evaluate_model_with_outputs(model, loader, device, loss_fn, use_bfloat16)
	return metrics


def _evaluate_model_with_outputs(
	model: Any,
	loader: Any,
	device: Any,
	loss_fn: Any,
	use_bfloat16: bool,
) -> tuple[dict[str, float], list[int], list[int]]:
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
	predictions = [1 if score >= 0.5 else 0 for score in all_scores]
	return metrics, all_labels, predictions


def _build_subtype_accuracy_report(
	examples: list[DirectionExample],
	labels: list[int],
	predictions: list[int],
) -> dict[str, dict[str, float | int]]:
	if len(examples) != len(labels) or len(labels) != len(predictions):
		raise ValueError(
			"Subtype analysis expects equal lengths for examples, labels, and predictions "
			f"(got {len(examples)}, {len(labels)}, {len(predictions)})."
		)

	stats: dict[str, dict[str, int]] = defaultdict(lambda: {
		"n": 0,
		"correct": 0,
		"forward_total": 0,
		"reverse_total": 0,
	})

	for example, label, prediction in zip(examples, labels, predictions):
		subtype = (example.cas_subtype or "Unknown").strip() or "Unknown"
		entry = stats[subtype]
		entry["n"] += 1
		entry["correct"] += int(label == prediction)
		if int(label) == 1:
			entry["forward_total"] += 1
		else:
			entry["reverse_total"] += 1

	report: dict[str, dict[str, float | int]] = {}
	for subtype in sorted(stats):
		entry = stats[subtype]
		n = int(entry["n"])
		correct = int(entry["correct"])
		report[subtype] = {
			"n": n,
			"correct": correct,
			"accuracy": _safe_divide(correct, n),
			"forward_total": int(entry["forward_total"]),
			"reverse_total": int(entry["reverse_total"]),
		}

	return report


def _print_subtype_accuracy_report(
	title: str,
	report: dict[str, dict[str, float | int]],
) -> None:
	print(title)
	if not report:
		print("  No subtype rows available.")
		return

	# Show larger subtypes first for more stable per-subtype estimates.
	sorted_rows = sorted(
		report.items(),
		key=lambda item: (int(item[1]["n"]), float(item[1]["accuracy"])),
		reverse=True,
	)
	for subtype, row in sorted_rows:
		n = int(row["n"])
		correct = int(row["correct"])
		accuracy = float(row["accuracy"])
		forward_total = int(row["forward_total"])
		reverse_total = int(row["reverse_total"])
		print(
			f"  {subtype}: acc={accuracy:.4f} ({correct}/{n}) "
			f"| Forward={forward_total} Reverse={reverse_total}"
		)


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


def _update_early_stopping_state(
	current_metric: float,
	best_metric: float,
	patience_counter: int,
	patience: int,
	min_delta: float,
) -> tuple[float, int, bool]:
	"""Return the updated best metric, patience counter, and whether training should stop early."""
	if current_metric > best_metric + min_delta:
		return current_metric, 0, False
	updated_patience = patience_counter + 1
	should_stop = patience > 0 and updated_patience >= patience
	return best_metric, updated_patience, should_stop


def main() -> int:
	parser = argparse.ArgumentParser(description="Finetune Carbon-500M for CRISPR array direction prediction.")
	parser.add_argument("--jsonl", default="/tmp/direction_no_aug.jsonl")
	parser.add_argument("--output_dir", default="direction_learning/carbon-model/outputs/carbon-500m-direction")
	parser.add_argument("--model_id", default=DEFAULT_MODEL_ID)
	parser.add_argument("--sequence_mode", choices=["interleaved", "spacers_only"], default="interleaved")
	parser.add_argument("--include_flanks", action="store_true")
	parser.add_argument("--stratify_mode", choices=["label", "cas_subtype", "cas_subtype_and_label"], default="cas_subtype_and_label")
	parser.add_argument(
		"--split_group",
		action="store_true",
		help="Use group-aware connected-component splitting to keep related arrays, including reverse complements, in the same split.",
	)
	parser.add_argument(
		"--split_size_aware",
		action="store_true",
		help="Use example-count-aware group splitting and stronger size penalties to keep the test fraction closer to --test_fraction.",
	)
	parser.add_argument(
		"--split_size_weight",
		type=float,
		default=25.0,
		help="Size penalty weight used when --split_size_aware is enabled. Lower values let overlap matter more.",
	)
	parser.add_argument(
		"--split_optimize_trials",
		type=int,
		default=1,
		help="Try multiple candidate split seeds and keep the one with minimal train->query contiguous k-spacer overlap.",
	)
	parser.add_argument(
		"--split_optimize_k",
		type=int,
		default=3,
		help="k for contiguous spacer sub-array overlap objective used by --split_optimize_trials.",
	)
	parser.add_argument(
		"--split_optimize_target",
		choices=["both", "test", "train_all_test", "all_pairs", "all-pairs", "all_pais"],
		default="both",
		help="Optimize split search for both val+test leakage balance (default), test leakage only, train_all->test leakage (train+val as reference), or all split pairs (all_pairs / all-pairs / all_pais).",
	)
	parser.add_argument("--test_fraction", type=float, default=0.15)
	parser.add_argument("--max_length", type=int, default=512)
	parser.add_argument("--batch_size", type=int, default=2)
	parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
	parser.add_argument("--epochs", type=int, default=3)
	parser.add_argument("--early_stopping_patience", type=int, default=0, help="Stop training if validation does not improve for this many epochs. 0 disables early stopping.")
	parser.add_argument("--early_stopping_min_delta", type=float, default=0.0, help="Minimum validation improvement required to count as an improvement for early stopping.")
	parser.add_argument("--learning_rate", type=float, default=2e-5)
	parser.add_argument("--weight_decay", type=float, default=0.05)
	parser.add_argument("--warmup_ratio", type=float, default=0.06)
	parser.add_argument("--seed", type=int, default=42)
	parser.add_argument("--num_workers", type=int, default=0)
	parser.add_argument("--max_train_examples", type=int, default=0)
	parser.add_argument("--max_val_examples", type=int, default=0)
	parser.add_argument("--max_test_examples", type=int, default=0)
	parser.add_argument(
		"--split_group_chunk_size",
		type=int,
		default=250,
		help="Optional subgroup size for split artifact naming; chunks stay in the same split and are written as group__partNNN.",
	)
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

	

	splits = _build_splits(
		dataset.records,
		seed=args.seed,
		test_fraction=args.test_fraction,
		stratify_mode=args.stratify_mode,
		split_group=args.split_group,
		split_optimize_trials=args.split_optimize_trials,
		split_optimize_k=args.split_optimize_k,
		split_optimize_target=args.split_optimize_target,
		split_size_aware=args.split_size_aware,
		split_size_weight=args.split_size_weight,
	)
	train_indices = _truncate_indices(splits["train"], args.max_train_examples)
	val_indices = _truncate_indices(splits["val"], args.max_val_examples)
	test_indices = _truncate_indices(splits["test"], args.max_test_examples)

	train_examples = [dataset.records[index] for index in train_indices]
	val_examples = [dataset.records[index] for index in val_indices]
	test_examples = [dataset.records[index] for index in test_indices]

	manifest_path = _write_split_artifacts(
		output_dir=output_dir,
		dataset_path=dataset_path,
		seed=args.seed,
		chunk_size=args.split_group_chunk_size,
		examples=dataset.records,
		train_examples=train_examples,
		val_examples=val_examples,
		test_examples=test_examples,
		train_indices=train_indices,
		val_indices=val_indices,
		test_indices=test_indices,
	)
	print(f"Saved split artifacts to {manifest_path.parent}")

	similarity_report = _build_split_similarity_report(
		dataset_path=dataset_path,
		seed=args.seed,
		train_examples=train_examples,
		val_examples=val_examples,
		test_examples=test_examples,
	)
	_save_similarity_report(output_dir, similarity_report)

	print("Split summary:")
	_print_split_summary("train", train_examples)
	_print_split_summary("val", val_examples)
	_print_split_summary("test", test_examples)
	_print_split_group_names("train", dataset.records, train_indices)
	_print_split_group_names("val", dataset.records, val_indices)
	_print_split_group_names("test", dataset.records, test_indices)
	_print_group_overlap_report(
		dataset.records,
		{"train": train_indices, "val": val_indices, "test": test_indices},
	)

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
				target_additions=augment_spacer_deletion_count * len(train_indices),
				balance_per_array=True,
			)
			train_indices = list(train_indices) + train_new_indices
			train_examples = [dataset.records[index] for index in train_indices]
			print(
				"Spacer deletion augmentation summary: "
				f"requested={augment_spacer_deletion_count} per array added={train_aug_stats['added']} "
				f"blocked_overlap={train_aug_stats['blocked_overlap']} "
				f"blocked_similarity={train_aug_stats['blocked_similarity']} "
				f"skipped_short={train_aug_stats['skipped_short']}"
			)

	# Augment validation set with same augmentation count
	if args.augment_spacer_deletion and augment_spacer_deletion_count > 0 and val_indices:
		seen_signatures = {example_signature(example) for example in dataset.records}
		val_new_indices, val_aug_stats = materialize_subarray_augmentations(
			base_dataset=dataset,
			source_indices=list(val_indices),
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
			split_name="val",
			use_diversity=True,
			similarity_metric=AUGMENT_SPACER_DELETION_SIMILARITY_METRIC,
			min_distance=AUGMENT_SPACER_DELETION_MIN_DISTANCE,
			target_additions=augment_spacer_deletion_count * len(val_indices),
			balance_per_array=True,
		)
		val_indices = list(val_indices) + val_new_indices
		val_examples = [dataset.records[index] for index in val_indices]
		print(
			"Spacer deletion augmentation (validation) summary: "
			f"requested={augment_spacer_deletion_count} per array added={val_aug_stats['added']} "
			f"blocked_overlap={val_aug_stats['blocked_overlap']} "
			f"blocked_similarity={val_aug_stats['blocked_similarity']} "
			f"skipped_short={val_aug_stats['skipped_short']}"
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
	early_stopping_patience = max(0, int(args.early_stopping_patience))
	early_stopping_min_delta = float(args.early_stopping_min_delta)
	patience_counter = 0
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

		if current_metric > best_val_metric + early_stopping_min_delta:
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
			patience_counter = 0
		else:
			patience_counter += 1
			if early_stopping_patience > 0 and patience_counter >= early_stopping_patience:
				print(
					f"Early stopping triggered after epoch {epoch}: validation metric {current_metric:.6f} did not improve by at least {early_stopping_min_delta:.6f} for {early_stopping_patience} consecutive epochs."
				)
				break

	if best_val_metrics is None:
		best_val_metrics = _evaluate_model(model, val_loader, device, loss_fn, use_bfloat16)

	test_metrics, test_labels, test_predictions = _evaluate_model_with_outputs(
		model,
		test_loader,
		device,
		loss_fn,
		use_bfloat16,
	)
	test_subtype_accuracy = _build_subtype_accuracy_report(test_examples, test_labels, test_predictions)
	print(f"Best epoch: {best_epoch} | best_val={_format_metrics(best_val_metrics)}")
	print(f"Test: {_format_metrics(test_metrics)}")
	_print_subtype_accuracy_report("Test subtype accuracy:", test_subtype_accuracy)

	with (output_dir / "training_summary.json").open("w") as fh:
		json.dump(
			{
				"args": vars(args),
				"best_epoch": best_epoch,
				"best_validation": best_val_metrics,
				"test": test_metrics,
				"test_subtype_accuracy": test_subtype_accuracy,
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
