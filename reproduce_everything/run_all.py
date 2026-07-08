from __future__ import annotations

import csv
import json
import math
import random
import sys
import time
from collections import Counter
from contextlib import nullcontext
from dataclasses import dataclass, replace
from importlib import import_module
from pathlib import Path
from typing import Any

try:
	import torch
	from torch import nn
	from torch.utils.data import DataLoader, Dataset
except ModuleNotFoundError:
	torch = None
	nn = None
	DataLoader = object
	Dataset = object

try:
	from sklearn.metrics import (
		accuracy_score,
		confusion_matrix,
		f1_score,
		matthews_corrcoef,
		precision_score,
		recall_score,
		roc_auc_score,
	)
except ModuleNotFoundError:
	raise


REPO_ROOT = Path(__file__).resolve().parents[1]
# standard paths generated from the scripts where they are located on my computer,
# change then to whereever they are on yours
CURATED_TSV = REPO_ROOT / "output_dataset_10k" / "curated_dataset.tsv"
SOURCE_LOOKUP_TSV = REPO_ROOT / "output_dataset_10k" / "intermediate_array_cas_annotations.tsv"
# if using my lookup tables, then you can use these paths instead
#CURATED_TSV = REPO_ROOT / "reproduce_everything" / "output_dataset_10k" / "curated_dataset.tsv"
#SOURCE_LOOKUP_TSV = REPO_ROOT / "reproduce_everything" / "output_dataset_10k" / "intermediate_array_cas_annotations.tsv"

JSONL_OUT = Path("/tmp/direction_dataset.jsonl")
OUTPUT_DIR = REPO_ROOT / "reproduce_everything" / "outputs" / "carbon-500m-direction"

DEFAULT_MODEL_ID = "HuggingFaceBio/Carbon-500M"
DNA_SEPARATOR = "NNNNNN"
DNA_COMPLEMENT = str.maketrans("ACGTacgtNn", "TGCAtgcaNn")

# here we filter for only the samples where evor and crispr direction agree
PREPARE_REQUIRE_AGREE = True
PREPARE_ALLOW_NOT_COMPARABLE = False
PREPARE_COLLAPSE_DUPLICATES = True


# These are the parameters I used for my final performances, if you want to try others change them here
# in my "normal" code they are passed as CLI arguments, but for less code, i hardcoded them here
TRAIN_SEQUENCE_MODE = "interleaved"
TRAIN_INCLUDE_FLANKS = False
TRAIN_STRATIFY_MODE = "cas_subtype_and_label"
TRAIN_TEST_FRACTION = 0.15
TRAIN_MAX_LENGTH = 256
TRAIN_BATCH_SIZE = 1
TRAIN_GRADIENT_ACCUMULATION_STEPS = 16
TRAIN_EPOCHS = 5
TRAIN_LEARNING_RATE = 5e-4
TRAIN_WEIGHT_DECAY = 0.05
TRAIN_WARMUP_RATIO = 0.06
TRAIN_SEED = 42
TRAIN_NUM_WORKERS = 0
TRAIN_BF16 = True
TRAIN_USE_LORA = True
TRAIN_LORA_ONLY_TRAIN = True
TRAIN_LORA_R = 4
TRAIN_LORA_ALPHA = 8
TRAIN_LORA_DROPOUT = 0.05
TRAIN_LORA_BIAS = "none"
TRAIN_LORA_PRESET = "qv"
TRAIN_LORA_TASK_TYPE = "SEQ_CLS"

AutoModelForSequenceClassification = None
AutoTokenizer = None
get_linear_schedule_with_warmup = None
LoraConfig = None
TaskType = None
get_peft_model = None
prepare_model_for_kbit_training = None


"""
check if torch and transformers are available, if not raise an error
"""
def _require_runtime() -> None:
	if torch is None or nn is None:
		raise ModuleNotFoundError("PyTorch is required for Carbon finetuning. Install torch first.")
	global AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
	if AutoTokenizer is None or AutoModelForSequenceClassification is None or get_linear_schedule_with_warmup is None:
		transformers = import_module("transformers")
		AutoTokenizer = transformers.AutoTokenizer
		AutoModelForSequenceClassification = transformers.AutoModelForSequenceClassification
		get_linear_schedule_with_warmup = transformers.get_linear_schedule_with_warmup


def _require_peft_runtime() -> None:
	global LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
	if LoraConfig is None or TaskType is None or get_peft_model is None or prepare_model_for_kbit_training is None:
		try:
			peft = import_module("peft")
		except ModuleNotFoundError as exc:
			raise ModuleNotFoundError("peft is required when using LoRA. Install it with `pip install peft`.") from exc
		LoraConfig = peft.LoraConfig
		TaskType = peft.TaskType
		get_peft_model = peft.get_peft_model
		prepare_model_for_kbit_training = peft.prepare_model_for_kbit_training


"""
DNA preparation functions
"""
def _normalize_dna(sequence: str) -> str:
	sequence = sequence.upper().replace("U", "T")
	cleaned: list[str] = []
	for char in sequence:
		if char.isspace():
			continue
		cleaned.append(char if char in {"A", "C", "G", "T"} else "N")
	return "".join(cleaned)


def reverse_complement(sequence: str) -> str:
	return sequence.translate(DNA_COMPLEMENT)[::-1]


def reverse_complement_array(record: dict[str, Any]) -> dict[str, Any]:
	spacers_rc = [reverse_complement(seq) for seq in reversed(record["spacers"])]
	repeats_rc = [reverse_complement(seq) for seq in reversed(record["repeats"])]
	flipped = dict(record)
	flipped["spacers"] = spacers_rc
	flipped["repeats"] = repeats_rc
	flipped["left_flank"] = reverse_complement(record["right_flank"])
	flipped["right_flank"] = reverse_complement(record["left_flank"])
	flipped["orientation_variant"] = "reverse_complement"
	flipped["label"] = 1 - int(record["label"])
	flipped["source_variant"] = record["orientation_variant"]
	flipped["evor_direction"] = "Reverse" if record["evor_direction"] == "Forward" else "Forward"
	return flipped


"""
Convert direction string to label
"""
def direction_to_label(evor_direction: str) -> int:
	direction = str(evor_direction).strip()
	if direction == "Forward":
		return 1
	if direction == "Reverse":
		return 0
	raise ValueError(f"Unsupported evor_direction for labeling: {evor_direction!r}")

"""
Data loading
"""
def load_curated_rows(curated_tsv: Path) -> list[dict[str, str]]:
	with open(curated_tsv, newline="") as fh:
		return list(csv.DictReader(fh, delimiter="\t"))


def load_source_lookup(intermediate_tsv: Path) -> dict[str, str]:
	if not intermediate_tsv.exists():
		return {}
	lookup: dict[str, str] = {}
	with open(intermediate_tsv, newline="") as fh:
		reader = csv.DictReader(fh, delimiter="\t")
		for row in reader:
			array_name = row.get("array_name", "")
			source_json = row.get("source_json", "")
			if array_name and source_json and array_name not in lookup:
				lookup[array_name] = source_json
	return lookup


def load_result_json(json_path: Path) -> dict[str, Any]:
	with open(json_path) as fh:
		return json.load(fh)

"""
extract info from data
"""
def extract_array_record(result_json: dict[str, Any], array_name: str) -> dict[str, Any]:
	for seq in result_json.get("Sequences", []):
		for crispr in seq.get("Crisprs", []):
			if crispr.get("Name", "") == array_name:
				return {
					"sequence_id": seq.get("Id", ""),
					"sequence_version": seq.get("Version", seq.get("Id", "")),
					"sequence_description": seq.get("Description", ""),
					"sequence_length": seq.get("Length", 0),
					"crispr": crispr,
				}
	raise KeyError(f"Array {array_name!r} not found in result.json")


def extract_region_sequences(crispr: dict[str, Any]) -> dict[str, Any]:
	spacers: list[str] = []
	repeats: list[str] = []
	left_flank = ""
	right_flank = ""
	for region in crispr.get("Regions", []):
		region_type = str(region.get("Type", "")).upper()
		sequence = str(region.get("Sequence", ""))
		if region_type == "SPACER":
			spacers.append(sequence)
		elif region_type == "DR":
			repeats.append(sequence)
		elif region_type == "LEFTFLANK":
			left_flank = sequence
		elif region_type == "RIGHTFLANK":
			right_flank = sequence
	return {"spacers": spacers, "repeats": repeats, "left_flank": left_flank, "right_flank": right_flank}


"""
build one data point
"""
# TODO delete all non essentail ones
def build_example(row: dict[str, str], include_flanks: bool) -> dict[str, Any]:
	json_path = Path(row["source_json"])
	result_json = load_result_json(json_path)
	matched = extract_array_record(result_json, row["array_name"])
	crispr = matched["crispr"]
	region_data = extract_region_sequences(crispr)
	record: dict[str, Any] = {
		"array_name": row["array_name"],
		"genome_id": row["genome_id"],
		"genome_version": row["genome_version"],
		"group_name": row["group_name"],
		"agreement": row["agreement"],
		"evor_direction": row["evor_direction"],
		"crispr_direction": row["crispr_direction"],
		"potential_orientation": row["potential_orientation"],
		"evidence_level": int(row["evidence_level"]),
		"source_json": row["source_json"],
		"sequence_id": matched["sequence_id"],
		"sequence_version": matched["sequence_version"],
		"sequence_description": matched["sequence_description"],
		"sequence_length": matched["sequence_length"],
		"array_start": int(row["array_start"]),
		"array_end": int(row["array_end"]),
		"array_length": int(row["array_length"]),
		"dr_consensus": row["dr_consensus"],
		"dr_length": int(row["dr_length"]),
		"repeat_id": row["repeat_id"],
		"n_spacers": int(row["n_spacers"]),
		"cas_subtype": row.get("cas_subtype", ""),
		"spacers": region_data["spacers"],
		"repeats": region_data["repeats"],
		"label": direction_to_label(row["evor_direction"]),
		"orientation_variant": "native",
		"source_variant": "native",
	}
	if include_flanks:
		record["left_flank"] = region_data["left_flank"]
		record["right_flank"] = region_data["right_flank"]
	else:
		record["left_flank"] = ""
		record["right_flank"] = ""
	return record


"""
print preparation summary
"""
def summarize_collapse(records: list[dict[str, Any]]) -> tuple[dict[str, int], dict[str, int], dict[str, int], list[dict[str, Any]]]:
	signature_to_records: dict[tuple[str, ...], list[dict[str, Any]]] = {}
	for record in records:
		signature = tuple(record["spacers"])
		signature_to_records.setdefault(signature, []).append(record)

	pre_counts = Counter(str(record.get("cas_subtype", "") or "Unknown") for record in records)
	post_counts = Counter()
	removed_counts = Counter()
	cross_subtype_groups: list[dict[str, Any]] = []

	for group_records in signature_to_records.values():
		canonical = group_records[0]
		canonical_subtype = str(canonical.get("cas_subtype", "") or "Unknown")
		post_counts[canonical_subtype] += 1
		for removed_record in group_records[1:]:
			removed_subtype = str(removed_record.get("cas_subtype", "") or "Unknown")
			removed_counts[removed_subtype] += 1
		subtype_counts = Counter(str(record.get("cas_subtype", "") or "Unknown") for record in group_records)
		if len(subtype_counts) > 1:
			cross_subtype_groups.append(
				{
					"collapsed_total": len(group_records) - 1,
					"canonical_subtype": canonical_subtype,
					"subtype_counts": dict(sorted(subtype_counts.items(), key=lambda kv: kv[0])),
				}
			)

	return dict(sorted(pre_counts.items(), key=lambda kv: kv[0])), dict(sorted(post_counts.items(), key=lambda kv: kv[0])), dict(sorted(removed_counts.items(), key=lambda kv: kv[0])), cross_subtype_groups


"""
deduplication checking
"""
def audit_rc_pairing(records: list[dict[str, Any]]) -> None:
	"""
	check for accidental native RC pairs in the dataset (both orientations already present pre-augmentation)
	"""
	sig_set = {tuple(r["spacers"]) for r in records}
	seen_pairs: set[frozenset] = set()
	examples: list[tuple[str, Any]] = []
	for record in records:
		spacers = tuple(record["spacers"])
		if not spacers:
			continue
		rc = tuple(reverse_complement(s) for s in reversed(spacers))
		if rc in sig_set and rc != spacers:
			pair_key = frozenset((spacers, rc))
			if pair_key not in seen_pairs:
				seen_pairs.add(pair_key)
				if len(examples) < 5:
					examples.append((record.get("array_name", ""), record.get("label")))
	print(f"Accidental native RC pairs found (both orientations already present pre-augmentation): {len(seen_pairs)}")
	if examples:
		print(f"  Sample array_names involved: {examples}")


"""
delete duplicates and print summary
"""
def collapse_records(records: list[dict[str, Any]], audit_rc: bool = True) -> list[dict[str, Any]]:
	pre_counts, post_counts, removed_counts, cross_subtype_groups = summarize_collapse(records)
	signature_to_record: dict[tuple[str, ...], dict[str, Any]] = {}
	for record in records:
		sig = tuple(record["spacers"])
		if sig not in signature_to_record:
			signature_to_record[sig] = record
	collapsed = list(signature_to_record.values())
	total_collapsed = sum(removed_counts.values())
	print(f"Collapsed {total_collapsed} duplicate records (by exact spacer signature)")
	print(f"Subtype counts before collapse: {pre_counts}")
	print(f"Subtype counts after collapse:  {post_counts}")
	print(f"Subtype counts removed by collapse: {removed_counts}")
	print(f"Collapsed signatures spanning multiple CRISPR types: {len(cross_subtype_groups)}")
	if cross_subtype_groups:
		print("Examples of cross-subtype collapsed signatures:")
		for entry in cross_subtype_groups[:10]:
			print(f"  canonical={entry['canonical_subtype']} collapsed_total={entry['collapsed_total']} subtypes={entry['subtype_counts']}")
	if audit_rc:
		audit_rc_pairing(records)
	return collapsed


"""
save dataset
"""
def write_jsonl(records: list[dict[str, Any]], out_path: Path) -> None:
	out_path.parent.mkdir(parents=True, exist_ok=True)
	with open(out_path, "w") as out_fh:
		for record in records:
			json.dump(record, out_fh)
			out_fh.write("\n")


"""
sample used for training
"""
@dataclass(frozen=True)
class DirectionExample:
	array_name: str
	group_name: str
	agreement: str
	evor_direction: str
	label: int
	orientation_variant: str
	source_variant: str
	spacers: list[str]
	repeats: list[str]
	cas_subtype: str = ""
	left_flank: str = ""
	right_flank: str = ""
	source_json: str = ""


"""
loading dataset and extracting datapoints 
(in "normal" code dataset creation and training are seperate, hence the loading function)
"""
def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
	records: list[dict[str, Any]] = []
	with open(path) as fh:
		for line in fh:
			line = line.strip()
			if line:
				records.append(json.loads(line))
	return records


def load_direction_examples(jsonl_path: str | Path) -> list[DirectionExample]:
	raw_records = load_jsonl(jsonl_path)
	records: list[DirectionExample] = []
	for raw in raw_records:
		records.append(
			DirectionExample(
				array_name=str(raw.get("array_name", "")),
				group_name=str(raw.get("group_name", "")),
				agreement=str(raw.get("agreement", "")),
				evor_direction=str(raw.get("evor_direction", "")),
				label=int(raw.get("label", 0)),
				orientation_variant=str(raw.get("orientation_variant", "native")),
				source_variant=str(raw.get("source_variant", "native")),
				spacers=list(raw.get("spacers", [])),
				repeats=list(raw.get("repeats", [])),
				cas_subtype=str(raw.get("cas_subtype", "")),
				left_flank=str(raw.get("left_flank", "")),
				right_flank=str(raw.get("right_flank", "")),
				source_json=str(raw.get("source_json", "")),
			)
		)
	return records


"""
make sure dataset is the right format for carbon input with <dna> tags and interleaved repeats and spacers
(repeats are not used, they are a leftover option from original code, to use if i wanted to)
"""
def build_carbon_sequence(example: DirectionExample) -> str:
	pieces: list[str] = []
	if example.spacers and example.repeats:
		for repeat, spacer in zip(example.repeats, example.spacers):
			pieces.append(_normalize_dna(repeat))
			pieces.append(_normalize_dna(spacer))
		if len(example.repeats) > len(example.spacers):
			pieces.append(_normalize_dna(example.repeats[-1]))
	else:
		pieces.extend(_normalize_dna(spacer) for spacer in example.spacers if spacer)
	core = DNA_SEPARATOR.join(piece for piece in pieces if piece)
	return f"<dna>{core}</dna>"


"""
build connected components of examples that share the same signature (spacers and repeats)
"""
def build_signature_components(examples: list[DirectionExample]) -> dict[int, list[int]]:
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


"""
splitting into train/val/test
-> splitting based on subtype and orientation label is used
"""
# TODO check which of these are actually used, some are leftovers from original code
def stratified_holdout_by_mode(
	examples: list[DirectionExample],
	seed: int,
	holdout_fraction: float,
	stratify_mode: str,
) -> tuple[list[int], list[int]]:
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


"""
decide on what splitting to do -> with the parameters provided, 
the splitting is done by subtype and label, so only the last else is used here
--> stratified_split_by_cas_subtype() and stratified_train_test_and_val_by_label()
	are not used in this codepath
"""
def split_dev_pool_by_mode(
	examples: list[DirectionExample],
	pool_indices: list[int],
	seed: int,
	stratify_mode: str,
) -> tuple[list[int], list[int]]:
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
		splits = stratified_train_test_and_val_by_cas_subtype_and_label(pool_examples, seed=seed, train_test_fraction=0.8)
		train_indices = [pool_indices[i] for i in splits["train_test"]]
		val_indices = [pool_indices[i] for i in splits["val"]]
	return sorted(train_indices), sorted(val_indices)


"""
make sure that no indices are shared between train/val/test and print a report of the split
"""
def _print_split_overlap_report(
	train_indices: list[int],
	val_indices: list[int],
	test_indices: list[int],
	examples: list[DirectionExample],
	max_examples: int = 10,
) -> None:
	all_sets = {
		"train": set(train_indices),
		"val": set(val_indices),
		"test": set(test_indices),
	}
	counts = {
		name: len(indices)
		for name, indices in (("train", train_indices), ("val", val_indices), ("test", test_indices))
	}
	overlaps: list[tuple[str, str, set[int]]] = []
	for left_name, left_indices in all_sets.items():
		for right_name, right_indices in all_sets.items():
			if left_name >= right_name:
				continue
			shared = left_indices & right_indices
			if shared:
				overlaps.append((left_name, right_name, shared))

	print("Split overlap report:")
	print(f"  train examples: {counts['train']}")
	print(f"  val examples:   {counts['val']}")
	print(f"  test examples:  {counts['test']}")
	if not overlaps:
		print("  No index overlap detected between train/val/test splits.")
		return

	print("  Overlaps detected:")
	for left_name, right_name, shared in overlaps:
		shared_list = sorted(shared)
		print(f"    {left_name} ∩ {right_name}: {len(shared_list)} shared indices")
		for index in shared_list[:max_examples]:
			example = examples[index]
			print(
				f"      idx={index} array_name={example.array_name!r} "
				f"group_name={example.group_name!r} label={example.label} "
				f"cas_subtype={example.cas_subtype!r}"
			)
		if len(shared_list) > max_examples:
			print(f"      ... {len(shared_list) - max_examples} more")


"""
splitting based on subtype and orientation label
"""
def stratified_train_test_and_val_by_cas_subtype_and_label(
	examples: list[DirectionExample], seed: int = 13, train_test_fraction: float = 0.8
) -> dict[str, list[int]]:
	if not (0.0 < train_test_fraction < 1.0):
		raise ValueError("train_test_fraction must be between 0 and 1")
	rng = random.Random(seed)
	strata_indices: dict[tuple[str, int], list[int]] = {}
	for idx, example in enumerate(examples):
		subtype = (example.cas_subtype or "Unknown").strip() or "Unknown"
		label = int(example.label)
		strata_indices.setdefault((subtype, label), []).append(idx)
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


"""
splitting based on subtype
not used in this codepath, but left here for completeness
"""
def stratified_split_by_cas_subtype(
	examples: list[DirectionExample],
	seed: int = 13,
	train_fraction: float = 0.8,
	test_fraction: float = 0.1,
) -> dict[str, list[int]]:
	if train_fraction + test_fraction > 1.0:
		raise ValueError("train_fraction + test_fraction must be <= 1.0")
	rng = random.Random(seed)
	components = build_signature_components(examples)
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
		for comp in shuffled[:n_train]:
			train_indices.extend(comp)
		for comp in shuffled[n_train:n_train + n_test]:
			test_indices.extend(comp)
		for comp in shuffled[n_train + n_test:]:
			val_indices.extend(comp)
	return {"train": train_indices, "val": val_indices, "test": test_indices}


"""
splitting based on orientation label
not used in this codepath, but left here for completeness
"""
def stratified_train_test_and_val_by_label(
	examples: list[DirectionExample], seed: int = 13, train_test_fraction: float = 0.8
) -> dict[str, list[int]]:
	if not (0.0 < train_test_fraction < 1.0):
		raise ValueError("train_test_fraction must be between 0 and 1")
	rng = random.Random(seed)
	components = build_signature_components(examples)
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


"""
safe divide for metrics calculation
"""
def _safe_divide(numerator: float, denominator: float) -> float:
	return numerator / denominator if denominator else 0.0


"""
calculate metrics
done by importing from sklearn, so there is no way I did an error in the calculation of them
"""
def _classification_metrics(labels: list[int], scores: list[float], threshold: float = 0.5) -> dict[str, float]:
	predictions = [1 if score >= threshold else 0 for score in scores]
	tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
	try:
		auroc = roc_auc_score(labels, scores)
	except ValueError:
		auroc = float("nan")
	return {
		"accuracy": accuracy_score(labels, predictions),
		"precision": precision_score(labels, predictions, zero_division=0),
		"recall": recall_score(labels, predictions, zero_division=0),
		"f1": f1_score(labels, predictions, zero_division=0),
		"mcc": matthews_corrcoef(labels, predictions),
		"auroc": auroc,
		"tp": float(tp),
		"tn": float(tn),
		"fp": float(fp),
		"fn": float(fn),
	}


"""
format metrics for printing
"""
def _format_metrics(metrics: dict[str, float]) -> str:
	ordered = ["loss", "accuracy", "precision", "recall", "f1", "mcc", "auroc"]
	parts = []
	for key in ordered:
		value = metrics.get(key, float("nan"))
		parts.append(f"{key}=nan" if math.isnan(value) else f"{key}={value:.4f}")
	return " | ".join(parts)


"""
print how much cuda memory is available, if using cuda
"""
def _cuda_memory_hint(device: Any) -> str:
	if device.type != "cuda" or not torch.cuda.is_available():
		return ""
	free_bytes, total_bytes = torch.cuda.mem_get_info(device)
	return f"CUDA free memory: {free_bytes / (1024 ** 3):.2f} GiB / {total_bytes / (1024 ** 3):.2f} GiB"


"""
oprions for fine tuning, including LoRA
"""
def _build_label_weights(labels: list[int]) -> Any:
	counts = Counter(labels)
	total = max(len(labels), 1)
	weights = []
	for label in (0, 1):
		count = counts.get(label, 0)
		weights.append(1.0 if count == 0 else total / (2.0 * count))
	return torch.tensor(weights, dtype=torch.float32)


"""
freeze backbone, only train classifier head
-> used when not doing LoRA
"""
def _freeze_backbone(model: Any) -> None:
	for name, parameter in model.named_parameters():
		if any(tag in name for tag in ("classifier", "score", "pre_classifier", "classification_head")):
			parameter.requires_grad = True
		else:
			parameter.requires_grad = False


def _batch_to_device(batch: dict[str, Any], device: Any) -> dict[str, Any]:
	moved: dict[str, Any] = {}
	for key, value in batch.items():
		moved[key] = value.to(device) if isinstance(value, torch.Tensor) else value
	return moved


"""
LORA functions
"""
def _resolve_lora_target_modules() -> list[str]:
	return ["q_proj", "v_proj"]


def _apply_lora(model: Any) -> Any:
	if get_peft_model is None or LoraConfig is None or TaskType is None or prepare_model_for_kbit_training is None:
		_require_peft_runtime()
	config = LoraConfig(
		task_type=getattr(TaskType, TRAIN_LORA_TASK_TYPE),
		r=TRAIN_LORA_R,
		lora_alpha=TRAIN_LORA_ALPHA,
		lora_dropout=TRAIN_LORA_DROPOUT,
		bias=TRAIN_LORA_BIAS,
		target_modules=_resolve_lora_target_modules(),
	)
	model = get_peft_model(model, config)
	return model


"""
print out summary of splitting
"""
def _print_split_summary(name: str, examples: list[DirectionExample]) -> None:
	label_counts = Counter(example.label for example in examples)
	subtype_counts = Counter((example.cas_subtype or "Unknown") for example in examples)
	print(
		f"{name}: {len(examples)} examples | "
		f"Forward={label_counts.get(1, 0)} Reverse={label_counts.get(0, 0)} | "
		f"Top subtypes={dict(subtype_counts.most_common(5))}"
	)


"""
carbon classes for dataset and batch collation
"""
class CarbonDirectionDataset(Dataset if Dataset is not object else object):
	def __init__(self, examples: list[DirectionExample], tokenizer: Any, max_length: int):
		self.examples = examples
		self.tokenizer = tokenizer
		self.max_length = max_length

	def __len__(self) -> int:
		return len(self.examples)

	def __getitem__(self, index: int) -> dict[str, Any]:
		example = self.examples[index]
		text = build_carbon_sequence(example)
		encoded = self.tokenizer(text, add_special_tokens=False, truncation=True, max_length=self.max_length)
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
		model_features = [{key: value for key, value in feature.items() if key in {"input_ids", "attention_mask"}} for feature in features]
		batch = self.tokenizer.pad(model_features, padding=True, return_tensors="pt")
		batch["labels"] = labels
		return batch


"""
evaluating the model
"""
def _evaluate_model(model: Any, loader: Any, device: Any, loss_fn: Any, use_bfloat16: bool) -> dict[str, float]:
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


"""
prepare dataset based on the curated TSV and source lookup TSV:
filter rows, build examples, collapse duplicates, and write to JSONL
"""
def prepare_dataset() -> None:
	rows = load_curated_rows(CURATED_TSV)
	source_lookup = load_source_lookup(SOURCE_LOOKUP_TSV)
	filtered_rows: list[dict[str, str]] = []
	for row in rows:
		agreement = str(row.get("agreement", "")).strip().lower()
		evor_direction = str(row.get("evor_direction", "")).strip()
		if PREPARE_REQUIRE_AGREE and agreement != "agree":
			if not (PREPARE_ALLOW_NOT_COMPARABLE and agreement == "not_comparable"):
				continue
		if evor_direction not in {"Forward", "Reverse"}:
			continue
		if not row.get("source_json"):
			source_json = source_lookup.get(row.get("array_name", ""), "")
			if source_json:
				row = dict(row)
				row["source_json"] = source_json
			else:
				continue
		filtered_rows.append(row)

	pre_collapse_label_counts = Counter(row["evor_direction"] for row in filtered_rows)
	native_records: list[dict[str, Any]] = []
	for row in filtered_rows:
		try:
			native_records.append(build_example(row, include_flanks=False))
		except Exception as exc:
			print(f"WARNING: skipping {row.get('array_name', '<unknown>')}: {exc}", file=sys.stderr)

	print("=== Collapsing NATIVE records before RC augmentation ===")
	native_records = collapse_records(native_records)
	all_records: list[dict[str, Any]] = []
	for base_record in native_records:
		all_records.append(base_record)
		all_records.append(reverse_complement_array(base_record))
	if PREPARE_COLLAPSE_DUPLICATES:
		all_records = collapse_records(all_records, audit_rc=False)
	final_label_counts = Counter(record["evor_direction"] for record in all_records)
	write_jsonl(all_records, JSONL_OUT)
	print(f"Filtered rows (pre-collapse): {len(filtered_rows)}")
	print(f"Label counts (pre-collapse): {dict(pre_collapse_label_counts)}")
	print(f"Prepared records (post-collapse): {len(all_records)}")
	print(f"Label counts (post-collapse): {dict(final_label_counts)}")
	print(f"Wrote {len(all_records)} JSONL records to {JSONL_OUT}")


"""
actual carbon training code
"""
def train_carbon() -> None:
	_require_runtime()
	random.seed(TRAIN_SEED)
	torch.manual_seed(TRAIN_SEED)
	if torch.cuda.is_available():
		torch.cuda.manual_seed_all(TRAIN_SEED)

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	use_bfloat16 = bool(TRAIN_BF16 and device.type == "cuda")
	output_dir = OUTPUT_DIR
	output_dir.mkdir(parents=True, exist_ok=True)

	dataset = load_direction_examples(JSONL_OUT)
	if len(dataset) == 0:
		raise ValueError("The dataset is empty.")

	# The line above gives us the dev pool split; recover the held-out test set from the first split call.
	dev_indices, test_indices = stratified_holdout_by_mode(dataset, seed=TRAIN_SEED, holdout_fraction=TRAIN_TEST_FRACTION, stratify_mode=TRAIN_STRATIFY_MODE)
	train_indices, val_indices = split_dev_pool_by_mode(dataset, pool_indices=dev_indices, seed=TRAIN_SEED, stratify_mode=TRAIN_STRATIFY_MODE)
	_print_split_overlap_report(train_indices, val_indices, test_indices, dataset)

	train_examples = [dataset[index] for index in train_indices]
	val_examples = [dataset[index] for index in val_indices]
	test_examples = [dataset[index] for index in test_indices]

	print("Split summary:")
	_print_split_summary("train", train_examples)
	_print_split_summary("val", val_examples)
	_print_split_summary("test", test_examples)

	tokenizer = AutoTokenizer.from_pretrained(DEFAULT_MODEL_ID, trust_remote_code=True)
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
	if use_bfloat16:
		model_kwargs["dtype"] = torch.bfloat16
	model = AutoModelForSequenceClassification.from_pretrained(DEFAULT_MODEL_ID, **model_kwargs)
	model.config.pad_token_id = tokenizer.pad_token_id
	if hasattr(model.config, "use_cache"):
		model.config.use_cache = False
	_freeze_backbone(model)
	if TRAIN_USE_LORA:
		model = _apply_lora(model)
		if TRAIN_LORA_ONLY_TRAIN:
			for name, parameter in model.named_parameters():
				if "lora_" not in name and "classifier" not in name and "score" not in name:
					parameter.requires_grad = False
	model.to(device)

	total = sum(parameter.numel() for parameter in model.parameters())
	trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
	print(f"Trainable params: {trainable:,}")
	print(f"Total params: {total:,}")
	print(f"Trainable %: {100 * trainable / total:.4f}%")

	train_dataset = CarbonDirectionDataset(train_examples, tokenizer=tokenizer, max_length=TRAIN_MAX_LENGTH)
	val_dataset = CarbonDirectionDataset(val_examples, tokenizer=tokenizer, max_length=TRAIN_MAX_LENGTH)
	test_dataset = CarbonDirectionDataset(test_examples, tokenizer=tokenizer, max_length=TRAIN_MAX_LENGTH)
	collator = CarbonBatchCollator(tokenizer)
	train_loader = DataLoader(train_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True, num_workers=TRAIN_NUM_WORKERS, collate_fn=collator)
	val_loader = DataLoader(val_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=False, num_workers=TRAIN_NUM_WORKERS, collate_fn=collator)
	test_loader = DataLoader(test_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=False, num_workers=TRAIN_NUM_WORKERS, collate_fn=collator)

	class_weights = _build_label_weights([example.label for example in train_examples]).to(device)
	loss_fn = nn.CrossEntropyLoss(weight=class_weights)
	trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
	if not trainable_parameters:
		raise ValueError("No trainable parameters remain after applying the current fine-tuning settings.")

	optimizer = torch.optim.AdamW(trainable_parameters, lr=TRAIN_LEARNING_RATE, weight_decay=TRAIN_WEIGHT_DECAY)
	total_steps = max(1, math.ceil(len(train_loader) / max(1, TRAIN_GRADIENT_ACCUMULATION_STEPS)) * TRAIN_EPOCHS)
	warmup_steps = max(0, round(total_steps * TRAIN_WARMUP_RATIO))
	scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

	print(f"Training Carbon-500M from {DEFAULT_MODEL_ID}")
	print(f"Device: {device} | bf16={use_bfloat16} | max_length={TRAIN_MAX_LENGTH} | batch_size={TRAIN_BATCH_SIZE}")
	print(f"LoRA: enabled={TRAIN_USE_LORA} | r={TRAIN_LORA_R} | alpha={TRAIN_LORA_ALPHA} | preset={TRAIN_LORA_PRESET}")
	if device.type == "cuda":
		hint = _cuda_memory_hint(device)
		if hint:
			print(hint)
	print(f"Class weights: {[float(x) for x in class_weights.detach().cpu().tolist()]}")

	best_val_metric = float("-inf")
	best_val_metrics: dict[str, float] | None = None
	best_epoch = 0
	autocast_context = torch.autocast("cuda", dtype=torch.bfloat16) if use_bfloat16 else nullcontext()

	for epoch in range(1, TRAIN_EPOCHS + 1):
		model.train()
		optimizer.zero_grad(set_to_none=True)
		running_loss = 0.0
		batches_seen = 0
		epoch_start = time.time()
		for step, batch in enumerate(train_loader, start=1):
			batch = _batch_to_device(batch, device)
			with autocast_context:
				outputs = model(input_ids=batch.get("input_ids"), attention_mask=batch.get("attention_mask"))
				loss = loss_fn(outputs.logits, batch["labels"]) / max(1, TRAIN_GRADIENT_ACCUMULATION_STEPS)
			loss.backward()
			running_loss += float(loss.item()) * max(1, TRAIN_GRADIENT_ACCUMULATION_STEPS)
			batches_seen += 1
			if step % TRAIN_GRADIENT_ACCUMULATION_STEPS == 0 or step == len(train_loader):
				torch.nn.utils.clip_grad_norm_(trainable_parameters, max_norm=1.0)
				optimizer.step()
				scheduler.step()
				optimizer.zero_grad(set_to_none=True)

		train_loss = _safe_divide(running_loss, batches_seen)
		val_metrics = _evaluate_model(model, val_loader, device, loss_fn, use_bfloat16)
		current_metric = val_metrics["mcc"]
		if math.isnan(current_metric):
			current_metric = val_metrics["f1"]
		epoch_duration = time.time() - epoch_start
		print(
			f"Epoch {epoch:02d} | time={int(epoch_duration // 60):02d}m:{int(epoch_duration % 60):02d}s | "
			f"train_loss={train_loss:.4f} | val={_format_metrics(val_metrics)}"
		)

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


"""
main function, combine dataset preparation and training
"""
def main() -> int:
	prepare_dataset()
	train_carbon()
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
