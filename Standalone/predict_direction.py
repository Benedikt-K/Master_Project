#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

DNA_SEPARATOR = "NNNNNN"
FASTA_SUFFIXES = {".fa", ".fna", ".fasta"}
LOOKUP_DB_PATH = (Path(__file__).resolve().parent / "lookup" / "array_lookup_db.json").resolve()
LOOKUP_TOP_K = 5

# TODO add fasta support, when inputing array as one sequence.

def normalize_dna(sequence: str) -> str:
    sequence = sequence.upper().replace("U", "T")
    cleaned: list[str] = []
    for char in sequence:
        if char.isspace():
            continue
        cleaned.append(char if char in {"A", "C", "G", "T"} else "N")
    return "".join(cleaned)


def interleave_segments(repeats: list[str], spacers: list[str]) -> list[str]:
    segments: list[str] = []
    max_length = max(len(repeats), len(spacers))
    for index in range(max_length):
        if index < len(repeats):
            segments.append(normalize_dna(repeats[index]))
        if index < len(spacers):
            segments.append(normalize_dna(spacers[index]))
    return [segment for segment in segments if segment]


def build_carbon_sequence(
    repeats: list[str],
    spacers: list[str],
) -> str:
    pieces: list[str] = []

    pieces.extend(interleave_segments(repeats, spacers))

    core = DNA_SEPARATOR.join(piece for piece in pieces if piece)
    return f"<dna>{core}</dna>"


def load_input_json(path: Path) -> dict[str, Any] | list[dict[str, Any]]:
    if path.suffix.lower() in FASTA_SUFFIXES:
        return load_input_objects(path)

    text = path.read_text().strip()
    if not text:
        raise ValueError(f"Input file is empty: {path}")

    # First try regular JSON (supports pretty-printed multi-line files).
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Fallback: JSONL, use the first non-empty line.
    for line in text.splitlines():
        line = line.strip()
        if line:
            return json.loads(line)
    raise ValueError(f"No non-empty JSON line found in: {path}")


def load_input_objects(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() in FASTA_SUFFIXES:
        return load_fasta_objects(path)

    text = path.read_text().strip()
    if not text:
        raise ValueError(f"Input file is empty: {path}")

    # Prefer parsing as a single JSON payload first.
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return parsed
        if isinstance(parsed, dict) and isinstance(parsed.get("examples"), list):
            return parsed["examples"]
        if isinstance(parsed, dict):
            return [parsed]
        raise ValueError(f"Unsupported JSON root type in: {path}")
    except json.JSONDecodeError:
        pass

    # Fallback to JSONL.
    objects: list[dict[str, Any]] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(
                f"JSONL row {line_number} in {path} is not an object (got {type(row).__name__})."
            )
        objects.append(row)

    if not objects:
        raise ValueError(f"No non-empty JSON object rows found in: {path}")
    return objects


def load_fasta_objects(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    current_name: str | None = None
    current_parts: list[str] = []

    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if current_name is not None and current_parts:
                records.append(
                    {
                        "array_name": current_name,
                        "group_name": current_name,
                        "cas_subtype": "",
                        "repeats": [],
                        "spacers": [normalize_dna(part) for part in current_parts if part],
                        "left_flank": "",
                        "right_flank": "",
                        "label": None,
                    }
                )
            current_name = line[1:].strip() or f"seq_{len(records) + 1}"
            current_parts = []
        else:
            current_parts.append(line)

    if current_name is not None and current_parts:
        records.append(
            {
                "array_name": current_name,
                "group_name": current_name,
                "cas_subtype": "",
                "repeats": [],
                "spacers": [normalize_dna(part) for part in current_parts if part],
                "left_flank": "",
                "right_flank": "",
                "label": None,
            }
        )

    if not records:
        raise ValueError(f"No FASTA records found in: {path}")
    return records


def extract_example(data: dict[str, Any]) -> dict[str, Any]:
    if isinstance(data.get("example"), dict):
        data = data["example"]

    repeats = data.get("repeats")
    spacers = data.get("spacers")

    if not isinstance(repeats, list) or not all(isinstance(x, str) for x in repeats):
        raise ValueError("Input must contain 'repeats' as a list of strings.")
    if not isinstance(spacers, list) or not all(isinstance(x, str) for x in spacers):
        raise ValueError("Input must contain 'spacers' as a list of strings.")
    if len(repeats) == 0 and len(spacers) == 0:
        raise ValueError("Input has no sequence content: both repeats and spacers are empty.")

    return {
        "repeats": repeats,
        "spacers": spacers,
        "left_flank": str(data.get("left_flank", "") or ""),
        "right_flank": str(data.get("right_flank", "") or ""),
        "array_name": str(data.get("array_name", "")),
        "group_name": str(data.get("group_name", "")),
        "cas_subtype": str(data.get("cas_subtype", "")),
        "label": data.get("label", None),
    }


def extract_example_from_ccf(
    data: dict[str, Any],
    sequence_index: int,
    crispr_index: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    sequences = data.get("Sequences")
    if not isinstance(sequences, list) or not sequences:
        raise ValueError("CCF file is missing a non-empty 'Sequences' list.")

    if sequence_index < 0 or sequence_index >= len(sequences):
        raise ValueError(
            f"--ccf_sequence_index {sequence_index} is out of range (available: 0..{len(sequences)-1})."
        )

    seq_obj = sequences[sequence_index]
    crisprs = seq_obj.get("Crisprs")
    if not isinstance(crisprs, list) or not crisprs:
        raise ValueError("Selected CCF sequence has no 'Crisprs' entries.")

    if crispr_index < 0 or crispr_index >= len(crisprs):
        raise ValueError(
            f"--ccf_crispr_index {crispr_index} is out of range (available: 0..{len(crisprs)-1})."
        )

    crispr = crisprs[crispr_index]
    regions = crispr.get("Regions")
    if not isinstance(regions, list) or not regions:
        raise ValueError("Selected CCF Crispr entry has no 'Regions' list.")

    repeats: list[str] = []
    spacers: list[str] = []
    left_flank = ""
    right_flank = ""

    for region in regions:
        region_type = str(region.get("Type", "")).strip().upper()
        sequence = str(region.get("Sequence", "") or "")
        if not sequence:
            continue
        if region_type == "DR":
            repeats.append(sequence)
        elif region_type == "SPACER":
            spacers.append(sequence)
        elif region_type == "LEFTFLANK" and not left_flank:
            left_flank = sequence
        elif region_type == "RIGHTFLANK" and not right_flank:
            right_flank = sequence

    # Fallback when DR records are absent but DR consensus is present.
    if not repeats:
        dr_consensus = str(crispr.get("DR_Consensus", "") or "")
        if dr_consensus:
            repeats = [dr_consensus] * max(1, len(spacers) + 1)

    if len(repeats) == 0 and len(spacers) == 0:
        raise ValueError("Selected CCF array has no usable DR/Spacer sequences.")

    array_name = str(crispr.get("Name") or f"ccf_seq{sequence_index}_crispr{crispr_index}")
    group_name = array_name

    example = {
        "repeats": repeats,
        "spacers": spacers,
        "left_flank": left_flank,
        "right_flank": right_flank,
        "array_name": array_name,
        "group_name": group_name,
        "cas_subtype": str(crispr.get("Cas_subtype", "") or ""),
        "label": None,
    }

    ccf_meta = {
        "sequence_id": str(seq_obj.get("Id", "")),
        "sequence_version": str(seq_obj.get("Version", "")),
        "crispr_name": str(crispr.get("Name", "")),
        "ccf_potential_orientation": str(crispr.get("Potential_Orientation", "")),
        "ccf_direction": str(crispr.get("CRISPRDirection", "")),
        "evidence_level": crispr.get("Evidence_Level", None),
        "start": crispr.get("Start", None),
        "end": crispr.get("End", None),
        "repeat_id": crispr.get("Repeat_ID", None),
        "dr_consensus": crispr.get("DR_Consensus", None),
        "ccf_sequence_index": sequence_index,
        "ccf_crispr_index": crispr_index,
    }

    return example, ccf_meta


def extract_examples_from_ccf(data: dict[str, Any]) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    sequences = data.get("Sequences")
    if not isinstance(sequences, list) or not sequences:
        raise ValueError("CCF file is missing a non-empty 'Sequences' list.")

    examples: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for sequence_index, seq_obj in enumerate(sequences):
        crisprs = seq_obj.get("Crisprs")
        if not isinstance(crisprs, list):
            continue
        for crispr_index in range(len(crisprs)):
            try:
                ex, ccf_meta = extract_example_from_ccf(data, sequence_index, crispr_index)
            except ValueError:
                continue
            if not ex.get("array_name"):
                ex["array_name"] = f"ccf_seq{sequence_index}_crispr{crispr_index}"
            examples.append((ex, ccf_meta))

    if not examples:
        raise ValueError("No usable CRISPR arrays found in CCF input.")
    return examples


def _iter_input_files(input_dir: Path) -> list[Path]:
    if not input_dir.exists() or not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    blocked = {
        "prediction_result.json",
        "prediction_results.json",
        "prediction_summary.json",
    }
    files = [
        p
        for p in sorted(input_dir.iterdir())
        if p.is_file()
        and p.suffix.lower() in {".json", ".jsonl", *FASTA_SUFFIXES}
        and p.name not in blocked
    ]
    if not files:
        raise ValueError(f"No .json, .jsonl, .fa, .fna, or .fasta files found in directory: {input_dir}")
    return files


def _resolve_batch_output_paths(
    *,
    base_dir: Path,
    result_file_arg: str,
    summary_file_arg: str,
) -> tuple[Path, Path]:
    results_path = (
        Path(result_file_arg).expanduser().resolve()
        if result_file_arg.strip()
        else (base_dir / "prediction_results.json").resolve()
    )
    summary_path = (
        Path(summary_file_arg).expanduser().resolve()
        if summary_file_arg.strip()
        else (base_dir / "prediction_summary.json").resolve()
    )
    return results_path, summary_path


def _build_batch_summary(
    *,
    records: list[dict[str, Any]],
    total_files: int,
    input_mode: str,
) -> dict[str, Any]:
    successes = [r for r in records if r.get("status") == "ok"]
    failures = [r for r in records if r.get("status") == "error"]
    labels = Counter(r.get("predicted_label", "") for r in successes)

    confidences: list[float] = []
    for row in successes:
        label = row.get("predicted_label")
        if label == "Forward":
            confidences.append(float(row.get("prob_forward", 0.0)))
        elif label == "Reverse":
            confidences.append(float(row.get("prob_reverse", 0.0)))

    avg_forward = (
        sum(float(r.get("prob_forward", 0.0)) for r in successes) / len(successes)
        if successes
        else 0.0
    )
    avg_reverse = (
        sum(float(r.get("prob_reverse", 0.0)) for r in successes) / len(successes)
        if successes
        else 0.0
    )

    return {
        "input_mode": input_mode,
        "total_files": total_files,
        "total_arrays": len(records),
        "successful_predictions": len(successes),
        "failed_predictions": len(failures),
        "label_counts": dict(labels),
        "avg_prob_forward": avg_forward,
        "avg_prob_reverse": avg_reverse,
        "avg_confidence": (sum(confidences) / len(confidences)) if confidences else 0.0,
        "max_confidence": max(confidences) if confidences else 0.0,
        "min_confidence": min(confidences) if confidences else 0.0,
    }


def _reverse_complement(sequence: str) -> str:
    trans = str.maketrans("ACGTN", "TGCAN")
    return normalize_dna(sequence).translate(trans)[::-1]


def _canonical_spacer_token(sequence: str) -> str:
    normalized = normalize_dna(sequence)
    rc = _reverse_complement(normalized)
    return min(normalized, rc)


def _canonical_sequence_tuple(sequences: list[str] | tuple[str, ...]) -> tuple[str, ...]:
    normalized = tuple(normalize_dna(sequence) for sequence in sequences if sequence)
    reverse_complemented = tuple(_reverse_complement(sequence) for sequence in reversed(normalized))
    return min(normalized, reverse_complemented)


def _array_core_sequence_from_lists(repeats: list[str], spacers: list[str]) -> str:
    pieces: list[str] = []
    max_length = max(len(repeats), len(spacers))
    for index in range(max_length):
        if index < len(repeats):
            pieces.append(normalize_dna(repeats[index]))
        if index < len(spacers):
            pieces.append(normalize_dna(spacers[index]))
    return "".join(piece for piece in pieces if piece)


def _canonical_31mers(sequence: str) -> set[str]:
    if len(sequence) < 31:
        return set()
    return {
        _canonical_spacer_token(sequence[start:start + 31])
        for start in range(len(sequence) - 31 + 1)
    }


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


def _canonical_array_signature(repeats: list[str], spacers: list[str]) -> tuple[tuple[str, ...], tuple[str, ...]]:
    repeats_norm = tuple(normalize_dna(sequence) for sequence in repeats if sequence)
    spacers_norm = tuple(normalize_dna(sequence) for sequence in spacers if sequence)
    forward = (repeats_norm, spacers_norm)

    repeats_rc = tuple(_reverse_complement(sequence) for sequence in reversed(repeats_norm))
    spacers_rc = tuple(_reverse_complement(sequence) for sequence in reversed(spacers_norm))
    reverse = (repeats_rc, spacers_rc)
    return min(forward, reverse)


def _extract_lookup_examples_from_jsonl(path: Path, split: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, raw_line in enumerate(path.read_text().splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        payload = json.loads(line)
        if isinstance(payload.get("example"), dict):
            payload = payload["example"]

        repeats = payload.get("repeats")
        spacers = payload.get("spacers")
        if not isinstance(repeats, list) or not all(isinstance(x, str) for x in repeats):
            continue
        if not isinstance(spacers, list) or not all(isinstance(x, str) for x in spacers):
            continue

        rows.append(
            {
                "split": split,
                "array_name": str(payload.get("array_name", f"{split}_{line_number}")),
                "repeats": [normalize_dna(x) for x in repeats if x],
                "spacers": [normalize_dna(x) for x in spacers if x],
            }
        )
    return rows


def _build_lookup_db(
    *,
    train_jsonl: Path | None,
    val_jsonl: Path | None,
    output_path: Path,
) -> dict[str, Any]:
    entries: list[dict[str, Any]] = []
    if train_jsonl is not None:
        entries.extend(_extract_lookup_examples_from_jsonl(train_jsonl, split="train"))
    if val_jsonl is not None:
        entries.extend(_extract_lookup_examples_from_jsonl(val_jsonl, split="val"))

    db = {
        "version": 1,
        "sources": {
            "train_jsonl": str(train_jsonl) if train_jsonl is not None else "",
            "val_jsonl": str(val_jsonl) if val_jsonl is not None else "",
        },
        "entries": entries,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(db, indent=2, sort_keys=True))
    return db


def _load_lookup_db(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    entries = payload.get("entries", [])
    if not isinstance(entries, list):
        raise ValueError(f"Lookup DB has invalid 'entries' format: {path}")

    signature_map: dict[tuple[tuple[str, ...], tuple[str, ...]], list[int]] = {}
    spacer_sets: list[set[str]] = []
    cleaned_entries: list[dict[str, Any]] = []

    for raw_entry in entries:
        if not isinstance(raw_entry, dict):
            continue
        repeats = raw_entry.get("repeats")
        spacers = raw_entry.get("spacers")
        split = str(raw_entry.get("split", "unknown"))
        if not isinstance(repeats, list) or not all(isinstance(x, str) for x in repeats):
            continue
        if not isinstance(spacers, list) or not all(isinstance(x, str) for x in spacers):
            continue

        entry = {
            "split": split,
            "array_name": str(raw_entry.get("array_name", "")),
            "repeats": [normalize_dna(x) for x in repeats if x],
            "spacers": [normalize_dna(x) for x in spacers if x],
        }
        cleaned_entries.append(entry)

        index = len(cleaned_entries) - 1
        signature = _canonical_array_signature(entry["repeats"], entry["spacers"])
        signature_map.setdefault(signature, []).append(index)
        spacer_sets.append({_canonical_spacer_token(x) for x in entry["spacers"] if x})

    return {
        "path": str(path),
        "entries": cleaned_entries,
        "signature_map": signature_map,
        "spacer_sets": spacer_sets,
    }


def _lookup_array_against_db(
    ex: dict[str, Any],
    lookup_state: dict[str, Any],
    *,
    top_k: int,
) -> dict[str, Any]:
    query_signature = _canonical_array_signature(ex["repeats"], ex["spacers"])
    matches = lookup_state["signature_map"].get(query_signature, [])
    entries = lookup_state["entries"]

    query_spacers = {_canonical_spacer_token(x) for x in ex["spacers"] if x}
    query_spacer_list = [normalize_dna(x) for x in ex["spacers"] if x]
    query_k_subarrays: dict[int, set[tuple[str, ...]]] = {}
    for k in (2, 3, 4, 5):
        if len(query_spacer_list) < k:
            query_k_subarrays[k] = set()
            continue
        query_k_subarrays[k] = {
            _canonical_sequence_tuple(query_spacer_list[start:start + k])
            for start in range(len(query_spacer_list) - k + 1)
        }
    query_core_sequence = _array_core_sequence_from_lists(ex["repeats"], ex["spacers"])
    query_31mers = _canonical_31mers(query_core_sequence)

    scored: list[tuple[float, float, int, int]] = []
    split_stats: dict[str, dict[str, Any]] = {}
    for index, candidate_spacers in enumerate(lookup_state["spacer_sets"]):
        union = query_spacers | candidate_spacers
        if not union:
            jaccard = 0.0
        else:
            jaccard = len(query_spacers & candidate_spacers) / len(union)
        shared = len(query_spacers & candidate_spacers)
        containment = (shared / len(query_spacers)) if query_spacers else 0.0
        scored.append((jaccard, containment, shared, index))

        split_name = entries[index].get("split", "unknown")
        current = split_stats.get(split_name)
        if current is None:
            split_stats[split_name] = {
                "count": 1,
                "sum_jaccard": jaccard,
                "sum_containment": containment,
                "best_jaccard": jaccard,
                "best_containment": containment,
                "best_shared_spacers": shared,
                "best_array_name": entries[index].get("array_name", ""),
                "coverage_histogram": Counter({_bucket_coverage_fraction(containment): 1}),
                "coverage_100_count": 1 if containment == 1.0 else 0,
                "k_overlap_counts": {2: 0, 3: 0, 4: 0, 5: 0},
                "kmer31_overlap_count": 0,
            }
        else:
            current["count"] += 1
            current["sum_jaccard"] += jaccard
            current["sum_containment"] += containment
            current["coverage_histogram"][_bucket_coverage_fraction(containment)] += 1
            if containment == 1.0:
                current["coverage_100_count"] += 1
            is_better = (
                jaccard > current["best_jaccard"]
                or (jaccard == current["best_jaccard"] and containment > current["best_containment"])
                or (
                    jaccard == current["best_jaccard"]
                    and containment == current["best_containment"]
                    and shared > current["best_shared_spacers"]
                )
            )
            if is_better:
                current["best_jaccard"] = jaccard
                current["best_containment"] = containment
                current["best_shared_spacers"] = shared
                current["best_array_name"] = entries[index].get("array_name", "")

        candidate_spacer_list = entries[index].get("spacers", [])
        for k in (2, 3, 4, 5):
            if not query_k_subarrays[k] or len(candidate_spacer_list) < k:
                continue
            has_k_overlap = any(
                _canonical_sequence_tuple(candidate_spacer_list[start:start + k]) in query_k_subarrays[k]
                for start in range(len(candidate_spacer_list) - k + 1)
            )
            if has_k_overlap:
                split_stats[split_name]["k_overlap_counts"][k] += 1

        if query_31mers:
            candidate_core = _array_core_sequence_from_lists(entries[index].get("repeats", []), candidate_spacer_list)
            candidate_31mers = _canonical_31mers(candidate_core)
            if query_31mers & candidate_31mers:
                split_stats[split_name]["kmer31_overlap_count"] += 1

    split_similarity = {
        split_name: {
            "count": int(values["count"]),
            "best_jaccard": float(values["best_jaccard"]),
            "best_containment": float(values["best_containment"]),
            "best_shared_spacers": int(values["best_shared_spacers"]),
            "best_array_name": str(values["best_array_name"]),
            "mean_jaccard": float(values["sum_jaccard"] / values["count"]),
            "mean_containment": float(values["sum_containment"] / values["count"]),
            "coverage_histogram": {
                bucket: int(values["coverage_histogram"].get(bucket, 0))
                for bucket in ["0", "(0,0.25]", "(0.25,0.5]", "(0.5,0.75]", "(0.75,1.0)", "1.0"]
            },
            "coverage_100_count": int(values["coverage_100_count"]),
            "k_overlap_counts": {str(k): int(values["k_overlap_counts"][k]) for k in (2, 3, 4, 5)},
            "k_overlap_flags": {str(k): bool(values["k_overlap_counts"][k] > 0) for k in (2, 3, 4, 5)},
            "kmer31_overlap_count": int(values["kmer31_overlap_count"]),
            "kmer31_overlap_flag": bool(values["kmer31_overlap_count"] > 0),
        }
        for split_name, values in split_stats.items()
    }

    if matches:
        split_counts = Counter(entries[index]["split"] for index in matches)
        return {
            "db_path": lookup_state["path"],
            "exact_match": True,
            "match_count": len(matches),
            "split_counts": dict(split_counts),
            "split_similarity": split_similarity,
            "matched_arrays": [
                {
                    "array_name": entries[index]["array_name"],
                    "split": entries[index]["split"],
                }
                for index in matches[: min(len(matches), 10)]
            ],
        }

    scored.sort(key=lambda item: (item[0], item[1], item[2]), reverse=True)
    top_hits = []
    for jaccard, containment, shared, index in scored[: max(1, top_k)]:
        entry = entries[index]
        top_hits.append(
            {
                "array_name": entry["array_name"],
                "split": entry["split"],
                "spacer_jaccard": jaccard,
                "query_spacer_containment": containment,
                "shared_spacers": shared,
                "candidate_spacer_count": len(entry["spacers"]),
            }
        )

    return {
        "db_path": lookup_state["path"],
        "exact_match": False,
        "split_similarity": split_similarity,
        "top_similar": top_hits,
    }


def _full_model_exists(model_dir: Path) -> bool:
    config_exists = (model_dir / "config.json").exists()
    weights_exist = (
        (model_dir / "model.safetensors").exists()
        or (model_dir / "pytorch_model.bin").exists()
        or len(list(model_dir.glob("model-*.safetensors"))) > 0
        or len(list(model_dir.glob("pytorch_model-*.bin"))) > 0
    )
    return config_exists and weights_exist


def _base_cache_dir(model_dir: Path) -> Path:
    return model_dir / "base_model_cache"


def _copy_base_model_files(source_dir: Path, target_dir: Path) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)

    names_to_copy = {
        "config.json",
        "generation_config.json",
        "model.safetensors",
        "model.safetensors.index.json",
        "pytorch_model.bin",
        "pytorch_model.bin.index.json",
    }

    for path in source_dir.iterdir():
        if path.is_file() and (
            path.name in names_to_copy
            or path.name.startswith("model-")
            or path.name.startswith("pytorch_model-")
        ):
            destination = target_dir / path.name
            shutil.copy2(path, destination)


def _adapter_exists(model_dir: Path) -> bool:
    adapter_config_exists = (model_dir / "adapter_config.json").exists()
    adapter_weights_exist = (
        (model_dir / "adapter_model.safetensors").exists()
        or (model_dir / "adapter_model.bin").exists()
    )
    return adapter_config_exists and adapter_weights_exist


def _read_adapter_base_model(model_dir: Path) -> str | None:
    adapter_config_path = model_dir / "adapter_config.json"
    if not adapter_config_path.exists():
        return None

    try:
        payload = json.loads(adapter_config_path.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {adapter_config_path}: {exc}") from exc

    base_model_name = payload.get("base_model_name_or_path")
    if isinstance(base_model_name, str) and base_model_name.strip():
        return base_model_name.strip()
    return None


def _ensure_base_model_cached(
    model_dir: Path,
    *,
    allow_downloads: bool,
    local_files_only: bool,
) -> str:
    cache_dir = _base_cache_dir(model_dir)
    if _full_model_exists(cache_dir):
        return str(cache_dir)

    # Backward compatibility: previous versions stored base weights directly in model_dir.
    # If adapter files are also present there, move/copy base-only files to cache_dir so
    # we can load base and adapter separately and avoid duplicate adapter application.
    if _full_model_exists(model_dir):
        _copy_base_model_files(model_dir, cache_dir)
        return str(cache_dir)

    base_model_name = _read_adapter_base_model(model_dir)
    if not base_model_name:
        raise FileNotFoundError(
            f"No full model weights found in {model_dir}, and adapter_config.json is missing or "
            "does not define base_model_name_or_path."
        )

    if not allow_downloads:
        raise FileNotFoundError(
            f"No full model weights found in {model_dir}. To fetch the base model ({base_model_name}) "
            "on first run, pass --allow_downloads."
        )

    print(f"No full model cache found in {cache_dir}. Downloading base model '{base_model_name}'...")

    tokenizer = AutoTokenizer.from_pretrained(
        base_model_name,
        trust_remote_code=True,
        local_files_only=local_files_only,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        base_model_name,
        trust_remote_code=True,
        local_files_only=local_files_only,
    )

    cache_dir.mkdir(parents=True, exist_ok=True)
    # Tokenizer stays in model_dir; base model weights/config are stored in cache_dir.
    tokenizer.save_pretrained(str(model_dir))
    model.save_pretrained(str(cache_dir))

    print(f"Saved base model cache to: {cache_dir}")
    return str(cache_dir)


def load_model_and_tokenizer(
    model_dir: Path,
    device: torch.device,
    *,
    allow_downloads: bool,
    local_files_only: bool,
):
    base_model_source = _ensure_base_model_cached(
        model_dir,
        allow_downloads=allow_downloads,
        local_files_only=local_files_only,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        str(model_dir),
        trust_remote_code=True,
        local_files_only=local_files_only,
    )
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        elif tokenizer.unk_token is not None:
            tokenizer.pad_token = tokenizer.unk_token
        else:
            raise ValueError("Tokenizer has no pad/eos/unk token.")

    model = AutoModelForSequenceClassification.from_pretrained(
        str(base_model_source),
        trust_remote_code=True,
        local_files_only=local_files_only,
    )

    lora_applied = False
    if _adapter_exists(model_dir):
        try:
            from peft import PeftModel
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "LoRA adapter files were found in model_dir, but the 'peft' package is not installed. "
                "Install it with `pip install peft`."
            ) from exc

        model = PeftModel.from_pretrained(
            model,
            str(model_dir),
            local_files_only=local_files_only,
        )
        lora_applied = True
        print(f"Applied LoRA adapter from: {model_dir}")

    model.config.pad_token_id = tokenizer.pad_token_id
    model.to(device)
    model.eval()
    model_id = str(base_model_source)
    if lora_applied:
        model_id = f"{base_model_source} + LoRA({model_dir})"
    return model, tokenizer, model_id


def predict(
    model: Any,
    tokenizer: Any,
    sequence: str,
    max_length: int,
    device: torch.device,
) -> dict[str, Any]:
    encoded = tokenizer(
        sequence,
        add_special_tokens=False,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    encoded = {k: v.to(device) for k, v in encoded.items()}

    with torch.no_grad():
        logits = model(**encoded).logits
        probs = torch.softmax(logits, dim=-1)[0]

    pred_id = int(torch.argmax(probs).item())
    labels = {0: "Reverse", 1: "Forward"}
    return {
        "predicted_label_id": pred_id,
        "predicted_label": labels.get(pred_id, str(pred_id)),
        "prob_reverse": float(probs[0].item()),
        "prob_forward": float(probs[1].item()),
        "token_count": int(encoded["input_ids"].shape[-1]),
    }


def default_model_dir() -> Path:
    return (Path(__file__).resolve().parent / "model_params").resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predict CRISPR array direction using the packaged standalone Carbon model."
    )
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--input_file",
        help="Path to JSON/JSONL input file or a FASTA file (.fa/.fna/.fasta).",
    )
    input_group.add_argument(
        "--input_dir",
        help="Path to a folder with multiple JSON/JSONL files.",
    )
    parser.add_argument(
        "--predict_all",
        action="store_true",
        help="Predict all arrays found in file(s). Without this flag, uses the first array per file.",
    )
    parser.add_argument(
        "--ccf",
        action="store_true",
        help="Interpret --input_file as CRISPRCasFinder result.json and extract one Crispr entry for prediction.",
    )
    parser.add_argument(
        "--ccf_sequence_index",
        type=int,
        default=0,
        help="When --ccf is used: index in top-level Sequences list (default: 0).",
    )
    parser.add_argument(
        "--ccf_crispr_index",
        type=int,
        default=0,
        help="When --ccf is used: index in selected sequence's Crisprs list (default: 0).",
    )
    parser.add_argument(
        "--model_dir",
        default=str(default_model_dir()),
        help="Path to standalone full model folder (default: Standalone/model_params).",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=256,
        help="Tokenizer truncation length.",
    )
    device_group = parser.add_mutually_exclusive_group()
    device_group.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU inference.",
    )
    device_group.add_argument(
        "--gpu",
        action="store_true",
        help="Force GPU inference (errors if CUDA is unavailable).",
    )
    parser.add_argument(
        "--result_file",
        default="",
        help=(
            "Path to write result JSON. Single mode default: alongside input as prediction_result.json. "
            "Batch mode default: base directory as prediction_results.json."
        ),
    )
    parser.add_argument(
        "--summary_file",
        default="",
        help="Batch mode only: path to write summary JSON (default: base directory/prediction_summary.json).",
    )
    parser.add_argument(
        "--allow_downloads",
        action="store_true",
        help="Allow online downloads from Hugging Face Hub if local files are missing.",
    )
    parser.add_argument(
        "--lookup",
        action="store_true",
        help="Enable lookup against the bundled train/val DB (disabled by default for faster inference).",
    )
    return parser.parse_args()


def _format_probability(value: float) -> str:
    return f"{value:.3f} ({value * 100:.1f}%)"


def _print_text_summary(
    *,
    input_path: Path,
    ex: dict[str, Any],
    result: dict[str, Any],
    lookup: dict[str, Any] | None,
) -> None:
    predicted = result["predicted_label"]
    confidence = result["prob_reverse"] if predicted == "Reverse" else result["prob_forward"]
    opposite = "Forward" if predicted == "Reverse" else "Reverse"
    opposite_prob = result["prob_forward"] if predicted == "Reverse" else result["prob_reverse"]

    print(f"=============================================================")
    print(f"Predicted direction is \"{predicted}\" with probability {_format_probability(confidence)}.")
    print(f"Alternative direction \"{opposite}\" has probability {_format_probability(opposite_prob)}.")
    print(
        f"Input summary: repeats={len(ex['repeats'])}, spacers={len(ex['spacers'])}, "
        f"tokens={result['token_count']}"
    )
    if ex.get("array_name"):
        print(f"Array name: {ex['array_name']}")
    if ex.get("cas_subtype"):
        print(f"CAS subtype: {ex['cas_subtype']}")
    if lookup is not None:
        split_similarity = lookup.get("split_similarity", {})
        if isinstance(split_similarity, dict) and split_similarity:
            print("Lookup similarity by split:")
            for split_name in sorted(split_similarity.keys()):
                stats = split_similarity.get(split_name, {})
                if not isinstance(stats, dict):
                    continue
                count = max(1, int(stats.get("count", 0)))
                print(
                    f"  - {split_name}: n={int(stats.get('count', 0))}, "
                    f"best_jaccard={float(stats.get('best_jaccard', 0.0)):.3f}, "
                    f"best_containment={float(stats.get('best_containment', 0.0)):.3f}, "
                    f"mean_jaccard={float(stats.get('mean_jaccard', 0.0)):.3f}, "
                    f"best_array={str(stats.get('best_array_name', ''))}"
                )
                histogram = stats.get("coverage_histogram", {})
                if isinstance(histogram, dict):
                    buckets = ["0", "(0,0.25]", "(0.25,0.5]", "(0.5,0.75]", "(0.75,1.0)", "1.0"]
                    parts = [
                        f"{bucket}:{int(histogram.get(bucket, 0))}/{count}"
                        for bucket in buckets
                    ]
                    print(f"    spacer coverage histogram: {' | '.join(parts)}")
                k_counts = stats.get("k_overlap_counts", {})
                k_flags = stats.get("k_overlap_flags", {})
                if isinstance(k_counts, dict) and isinstance(k_flags, dict):
                    k_parts = [
                        f"k={k}:{int(k_counts.get(str(k), 0))}/{count} ({'yes' if k_flags.get(str(k), False) else 'no'})"
                        for k in (2, 3, 4, 5)
                    ]
                    print(f"    contiguous k-spacer overlap: {' | '.join(k_parts)}")
                kmer_count = int(stats.get("kmer31_overlap_count", 0))
                kmer_flag = bool(stats.get("kmer31_overlap_flag", False))
                print(
                    f"    31-mer overlap: {kmer_count}/{count} candidate arrays "
                    f"(flag={'yes' if kmer_flag else 'no'})"
                )
        if lookup.get("exact_match"):
            split_counts = lookup.get("split_counts", {})
            train_count = int(split_counts.get("train", 0))
            val_count = int(split_counts.get("val", 0))
            print(
                "Lookup: exact match found in train/val database "
                f"(train={train_count}, val={val_count})."
            )
        else:
            top_hits = lookup.get("top_similar", [])
            if top_hits:
                best = top_hits[0]
                print(
                    "Lookup: no exact train/val match found. "
                    f"Most similar: {best.get('array_name', '')} "
                    f"[{best.get('split', 'unknown')}] with spacer_jaccard={float(best.get('spacer_jaccard', 0.0)):.3f}."
                )
    print(f"Input file: {input_path}")


def _resolve_result_path(input_path: Path, user_value: str) -> Path:
    if user_value.strip():
        return Path(user_value).expanduser().resolve()
    return (input_path.parent / "prediction_result.json").resolve()


def main() -> int:
    args = parse_args()

    lookup_state: dict[str, Any] | None = None
    if args.lookup:
        lookup_db_path = LOOKUP_DB_PATH
        if not lookup_db_path.exists():
            raise FileNotFoundError(
                f"Bundled lookup DB not found: {lookup_db_path}. Reinstall the Standalone bundle with lookup assets."
            )
        lookup_state = _load_lookup_db(lookup_db_path)
        print(f"Loaded lookup DB with {len(lookup_state['entries'])} arrays: {lookup_db_path}")

    local_files_only = not args.allow_downloads
    if local_files_only:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    input_path = Path(args.input_file).expanduser().resolve() if args.input_file else None
    input_dir = Path(args.input_dir).expanduser().resolve() if args.input_dir else None
    model_dir = Path(args.model_dir).expanduser().resolve()

    if input_path is not None and not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    batch_mode = bool(args.predict_all or input_dir is not None)

    if input_dir is not None:
        input_files = _iter_input_files(input_dir)
        batch_base_dir = input_dir
    else:
        assert input_path is not None
        input_files = [input_path]
        batch_base_dir = input_path.parent

    if args.gpu:
        if not torch.cuda.is_available():
            raise RuntimeError("--gpu was requested, but CUDA is not available on this machine.")
        device = torch.device("cuda")
    elif args.cpu:
        device = torch.device("cpu")
    else:
        # Default behavior: prefer GPU when available, otherwise fall back to CPU.
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, tokenizer, base_model_id = load_model_and_tokenizer(
        model_dir,
        device,
        allow_downloads=args.allow_downloads,
        local_files_only=local_files_only,
    )

    records: list[dict[str, Any]] = []
    for current_file in input_files:
        try:
            raw_data = load_input_json(current_file)
            if args.ccf:
                if batch_mode:
                    pairs = extract_examples_from_ccf(raw_data)
                    if not args.predict_all:
                        pairs = pairs[:1]
                else:
                    ex, ccf_meta = extract_example_from_ccf(
                        raw_data,
                        sequence_index=args.ccf_sequence_index,
                        crispr_index=args.ccf_crispr_index,
                    )
                    if not ex.get("array_name"):
                        ex["array_name"] = f"ccf_seq{args.ccf_sequence_index}_crispr{args.ccf_crispr_index}"
                    pairs = [(ex, ccf_meta)]
            else:
                objects = load_input_objects(current_file)
                if not args.predict_all:
                    objects = objects[:1]
                pairs = []
                for index, obj in enumerate(objects):
                    ex = extract_example(obj)
                    pairs.append(
                        (
                            ex,
                            {
                                "source_index": index,
                            },
                        )
                    )

            for index, (ex, source_meta) in enumerate(pairs):
                try:
                    sequence = build_carbon_sequence(
                        repeats=ex["repeats"],
                        spacers=ex["spacers"],
                    )
                    result = predict(model, tokenizer, sequence, args.max_length, device)
                    lookup_report = None
                    if lookup_state is not None:
                        lookup_report = _lookup_array_against_db(
                            ex,
                            lookup_state,
                            top_k=LOOKUP_TOP_K,
                        )

                    row: dict[str, Any] = {
                        "status": "ok",
                        "input_mode": "ccf" if args.ccf else "standard",
                        "input_file": str(current_file),
                        "model_dir": str(model_dir),
                        "base_model": base_model_id,
                        "device": str(device),
                        "array_index_in_file": index,
                        "array_name": ex["array_name"],
                        "group_name": ex["group_name"],
                        "cas_subtype": ex["cas_subtype"],
                        "predicted_label": result["predicted_label"],
                        "predicted_label_id": result["predicted_label_id"],
                        "prob_reverse": result["prob_reverse"],
                        "prob_forward": result["prob_forward"],
                        "token_count": result["token_count"],
                    }
                    if lookup_report is not None:
                        row["lookup"] = lookup_report
                    if ex["label"] is not None:
                        row["input_label"] = ex["label"]
                    if source_meta:
                        row["source"] = source_meta
                    records.append(row)

                    if not batch_mode:
                        _print_text_summary(
                            input_path=current_file,
                            ex=ex,
                            result=result,
                            lookup=lookup_report,
                        )
                except Exception as exc:  # noqa: BLE001
                    records.append(
                        {
                            "status": "error",
                            "input_mode": "ccf" if args.ccf else "standard",
                            "input_file": str(current_file),
                            "array_index_in_file": index,
                            "error": str(exc),
                        }
                    )
        except Exception as exc:  # noqa: BLE001
            records.append(
                {
                    "status": "error",
                    "input_mode": "ccf" if args.ccf else "standard",
                    "input_file": str(current_file),
                    "array_index_in_file": None,
                    "error": str(exc),
                }
            )

    if not records:
        raise RuntimeError("No predictions were produced.")

    if not batch_mode and len(records) == 1 and records[0].get("status") == "ok":
        single_output = records[0]
        assert input_path is not None
        result_path = _resolve_result_path(input_path, args.result_file)
        result_path.parent.mkdir(parents=True, exist_ok=True)
        with result_path.open("w") as fh:
            json.dump(single_output, fh, indent=2, sort_keys=True)
        print(f"Saved result JSON to: {result_path}")
        return 0

    results_path, summary_path = _resolve_batch_output_paths(
        base_dir=batch_base_dir,
        result_file_arg=args.result_file,
        summary_file_arg=args.summary_file,
    )
    results_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    summary = _build_batch_summary(
        records=records,
        total_files=len(input_files),
        input_mode="ccf" if args.ccf else "standard",
    )

    with results_path.open("w") as fh:
        json.dump({"results": records}, fh, indent=2, sort_keys=True)
    with summary_path.open("w") as fh:
        json.dump(summary, fh, indent=2, sort_keys=True)

    print(
        "Batch prediction complete: "
        f"files={summary['total_files']}, arrays={summary['total_arrays']}, "
        f"ok={summary['successful_predictions']}, failed={summary['failed_predictions']}."
    )
    print(f"Saved detailed results to: {results_path}")
    print(f"Saved summary to: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
