"""
prepare_direction_dataset.py
============================
Builds a sequence-oriented training dataset for CRISPR-evOr direction prediction.

The initial slice keeps only arrays where CRISPRCasFinder and evOr agree on the
orientation signal, then emits one positive example in the native orientation
and one negative example as the reverse-complemented array.

Input:
  - output_dataset/curated_dataset.tsv from generate_dataset.py
  - CRISPRCasFinder result.json files referenced by source_json

Output:
  - JSONL records with ordered spacer sequences, repeat consensus, optional
    flanks, labels, split keys, and provenance metadata.

Usage:
  python prepare_direction_dataset.py \
      --curated_tsv output_dataset/curated_dataset.tsv \
      --out_jsonl output_dataset/direction_training_dataset.jsonl

Optional:
  --include_flanks   Keep left/right flanks in the emitted examples.
  --no_augmentation  Emit only the native-orientation sample.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

DNA_COMPLEMENT = str.maketrans("ACGTacgtNn", "TGCAtgcaNn")


def direction_to_label(evor_direction: str) -> int:
    """Map evOr direction string to binary label.

    Forward -> 1, Reverse -> 0.
    """
    direction = str(evor_direction).strip()
    if direction == "Forward":
        return 1
    if direction == "Reverse":
        return 0
    raise ValueError(f"Unsupported evor_direction for labeling: {evor_direction!r}")


def reverse_complement(seq: str) -> str:
    return seq.translate(DNA_COMPLEMENT)[::-1]


def load_curated_rows(curated_tsv: Path) -> list[dict[str, str]]:
    with open(curated_tsv, newline="") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        return list(reader)


def load_source_lookup(intermediate_tsv: Path) -> dict[str, str]:
    if not intermediate_tsv.exists():
        return {}
    with open(intermediate_tsv, newline="") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        lookup: dict[str, str] = {}
        for row in reader:
            array_name = row.get("array_name", "")
            source_json = row.get("source_json", "")
            if array_name and source_json and array_name not in lookup:
                lookup[array_name] = source_json
        return lookup


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def load_result_json(json_path: Path) -> dict[str, Any]:
    with open(json_path) as fh:
        return json.load(fh)


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
    regions = crispr.get("Regions", [])
    spacers: list[str] = []
    repeats: list[str] = []
    left_flank = ""
    right_flank = ""

    for region in regions:
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

    return {
        "spacers": spacers,
        "repeats": repeats,
        "left_flank": left_flank,
        "right_flank": right_flank,
    }


def reverse_complement_array(record: dict[str, Any]) -> dict[str, Any]:
    spacers_rc = [reverse_complement(seq) for seq in reversed(record["spacers"])]
    repeats_rc = [reverse_complement(seq) for seq in reversed(record["repeats"])]

    flipped = dict(record)
    flipped["spacers"] = spacers_rc
    flipped["repeats"] = repeats_rc
    flipped["left_flank"] = reverse_complement(record["right_flank"])
    flipped["right_flank"] = reverse_complement(record["left_flank"])
    flipped["orientation_variant"] = "reverse_complement"
    # RC example must carry the opposite class of the native orientation.
    flipped["label"] = 1 - int(record["label"])
    flipped["source_variant"] = record["orientation_variant"]
    return flipped


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
        "cas_subtype": row.get("cas_subtype", ""),  # For stratified splitting
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


def summarize_collapse(records: list[dict[str, Any]]) -> tuple[
    dict[str, int],
    dict[str, int],
    dict[str, int],
    list[dict[str, Any]],
]:
    """Summarize duplicate collapse impact by cas_subtype."""
    signature_to_records: dict[tuple[tuple[str, ...], tuple[str, ...]], list[dict[str, Any]]] = {}
    for record in records:
        signature = (tuple(record["spacers"]), tuple(record["repeats"]))
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--curated_tsv",
        default="output_dataset/curated_dataset.tsv",
        help="Path to curated_dataset.tsv produced by generate_dataset.py",
    )
    parser.add_argument(
        "--source_lookup_tsv",
        default="output_dataset/intermediate_array_cas_annotations.tsv",
        help="Optional lookup table with source_json for each array_name",
    )
    parser.add_argument(
        "--out_jsonl",
        default="output_dataset/direction_training_dataset.jsonl",
        help="Output JSONL file for model training",
    )
    parser.add_argument(
        "--include_flanks",
        action="store_true",
        help="Include left/right flanks in the output records",
    )
    parser.add_argument(
        "--no_augmentation",
        action="store_true",
        help="Emit only the native-orientation example per array",
    )
    parser.add_argument(
        "--require_agree",
        action="store_true",
        default=True,
        help="Keep only rows where agreement == agree (default: enabled)",
    )
    parser.add_argument(
        "--allow_not_comparable",
        action="store_true",
        help="Also keep rows whose agreement is not comparable",
    )
    parser.add_argument(
        "--collapse_duplicates",
        action="store_true",
        help="Collapse exact spacer/repeat duplicates into single canonical records",
    )
    args = parser.parse_args()

    curated_tsv = Path(args.curated_tsv)
    source_lookup_tsv = Path(args.source_lookup_tsv)
    out_jsonl = Path(args.out_jsonl)
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)

    rows = load_curated_rows(curated_tsv)
    source_lookup = load_source_lookup(source_lookup_tsv)
    filtered_rows: list[dict[str, str]] = []
    for row in rows:
        agreement = str(row.get("agreement", "")).strip().lower()
        evor_direction = str(row.get("evor_direction", "")).strip()
        if args.require_agree and agreement != "agree":
            if not (args.allow_not_comparable and agreement == "not_comparable"):
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
    written = 0
    skipped = 0

    # Build all records first
    all_records: list[dict[str, Any]] = []
    for row in filtered_rows:
        try:
            base_record = build_example(row, include_flanks=args.include_flanks)
            all_records.append(base_record)

            if not args.no_augmentation:
                rc_record = reverse_complement_array(base_record)
                all_records.append(rc_record)
        except Exception as exc:
            skipped += 1
            print(f"WARNING: skipping {row.get('array_name', '<unknown>')}: {exc}", file=sys.stderr)

    # Optionally collapse duplicates by (spacers, repeats) signature
    if args.collapse_duplicates:
        pre_counts, post_counts, removed_counts, cross_subtype_groups = summarize_collapse(all_records)

        signature_to_record: dict[tuple, dict[str, Any]] = {}
        for record in all_records:
            sig = (tuple(record["spacers"]), tuple(record["repeats"]))
            if sig not in signature_to_record:
                signature_to_record[sig] = record
        all_records = list(signature_to_record.values())

        total_collapsed = sum(removed_counts.values())
        print(f"Collapsed {total_collapsed} duplicate records (by spacer/repeat signature)")
        print(f"Subtype counts before collapse: {pre_counts}")
        print(f"Subtype counts after collapse:  {post_counts}")
        print(f"Subtype counts removed by collapse: {removed_counts}")
        print(f"Collapsed signatures spanning multiple CRISPR types: {len(cross_subtype_groups)}")
        if cross_subtype_groups:
            print("Examples of cross-subtype collapsed signatures:")
            for entry in cross_subtype_groups[:10]:
                print(
                    f"  canonical={entry['canonical_subtype']} collapsed_total={entry['collapsed_total']} "
                    f"subtypes={entry['subtype_counts']}"
                )

    final_label_counts = Counter(record["evor_direction"] for record in all_records)

    # Write records to JSONL
    with open(out_jsonl, "w") as out_fh:
        for record in all_records:
            json.dump(record, out_fh)
            out_fh.write("\n")
            written += 1

    print(f"Filtered rows (pre-collapse): {len(filtered_rows)}")
    print(f"Label counts (pre-collapse): {dict(pre_collapse_label_counts)}")
    print(f"Prepared records (post-collapse): {len(all_records)}")
    print(f"Label counts (post-collapse): {dict(final_label_counts)}")
    print(f"Wrote {written} JSONL records to {out_jsonl}")
    print(f"Skipped arrays: {skipped}")
    print(f"Include flanks: {parse_bool(args.include_flanks)}")
    print(f"Augmentation enabled: {not args.no_augmentation}")
    print(f"Collapse duplicates: {args.collapse_duplicates}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
