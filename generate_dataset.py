"""
generate_dataset.py
=======================
Builds a comprehensive, ML-ready CRISPR dataset from CRISPRCasFinder result folders
and a CRISPR-evOr cluster CSV.

Pipeline:
  1. Walk all Result_*/ subdirectories, parse each result.json
  2. For every CRISPR array, find the nearest Cas cassette by genomic distance
  3. Assign Cas type from that cassette → intermediate annotated file
  4. Join with the evOr cluster CSV (agreement filter: keep only "agree")
  5. Write final ML-ready dataset

Usage:
  python build_crispr_dataset.py \
      --results_dir  /path/to/folder/with/Result_*_dirs \
      --cluster_csv  ccf_vs_out-sp_style_clusters_detail.csv \
      --out_dir      ./output_dataset

Outputs:
  output_dataset/intermediate_array_cas_annotations.tsv   - every array + nearest Cas type
  output_dataset/curated_dataset.tsv                      - filtered, ML-ready rows
    output_dataset/low_evidence_analysis.tsv                - optional analysis of evidence < 4 arrays
    output_dataset/evidence4_all_cas_links.tsv             - optional: all Cas cassettes within distance window
  output_dataset/build_report.txt                         - summary statistics
"""

import os
import json
import csv
import re
import argparse
from pathlib import Path
from collections import defaultdict, Counter


# Print each problematic raw label once during parsing, then summarise counts later.
SEEN_UNKNOWN_TYPE_RAW = set()
SEEN_UNKNOWN_SUBTYPE_RAW = set()
SEEN_COARSE_TYPE_RAW = set()

# ──────────────────────────────────────────────────────────────────────────────
# Type normalisation
# CRISPRCasFinder writes types like "NC_000913.3_CAS_Class1-Subtype-I-E_1"
# We extract the canonical subtype (e.g. "I-E") and the class ("Class1"/"Class2")
# ──────────────────────────────────────────────────────────────────────────────

# All 46 subtypes from Makarova et al. 2025 (Nature Microbiology)
KNOWN_SUBTYPES = {
    # Class 1 – Type I
    "I-A", "I-B", "I-B1", "I-B2", "I-C", "I-D", "I-E", "I-E2", "I-E3",
    "I-E4", "I-E5", "I-F", "I-F1", "I-F2", "I-F3", "I-F4", "I-G", "I-H",
    # Class 1 – Type III
    "III-A", "III-A2", "III-B", "III-C", "III-D", "III-E", "III-F",
    "III-G", "III-H", "III-I",
    # Class 1 – Type IV
    "IV-A", "IV-A2", "IV-A3", "IV-B", "IV-C", "IV-D", "IV-E",
    # Class 1 – Type VII
    "VII",
    # Class 2 – Type II
    "II-A", "II-B", "II-C", "II-C2", "II-D",
    # Class 2 – Type V
    "V-A", "V-A2", "V-B", "V-B1", "V-B2", "V-B3",
    "V-C", "V-D", "V-E", "V-F", "V-F1", "V-F2", "V-F3", "V-F4",
    "V-G", "V-H", "V-I", "V-J", "V-K", "V-L", "V-M", "V-N", "V-O", "V-P",
    # Class 2 – Type VI
    "VI-A", "VI-B", "VI-B1", "VI-B2", "VI-B3", "VI-C", "VI-D", "VI-E", "VI-F",
}

# Type → Class mapping
TYPE_TO_CLASS = {
    "I": "Class1", "III": "Class1", "IV": "Class1", "VII": "Class1",
    "II": "Class2", "V": "Class2", "VI": "Class2",
}

COARSE_TYPES = set(TYPE_TO_CLASS.keys())


def parse_cas_type(raw_type_string: str) -> dict:
    """
    Extract normalised class, type, and subtype from a CRISPRCasFinder Cas type string.

    Examples:
      "NC_000913.3_CAS_Class1-Subtype-I-E_1"  → class=Class1, type=I,  subtype=I-E
      "NC_000913.3_CAS_Class2-Subtype-V-A_2"  → class=Class2, type=V,  subtype=V-A
      "NC_000913.3_CAS_Class1-Type-IV_1"       → class=Class1, type=IV, subtype=IV
    """
    result = {"class": "Unknown", "type": "Unknown", "subtype": "Unknown",
              "raw": raw_type_string, "raw_canonical": "", "parse_status": "ok"}

    # Canonicalise away accession-specific prefixes/suffixes so diagnostics aggregate
    # by label shape, e.g. "NZ_..._CAS_Class1-Subtype-IV-E_1" -> "Class1-Subtype-IV-E".
    canonical = raw_type_string if raw_type_string else ""
    if "_CAS_" in canonical:
        canonical = canonical.split("_CAS_", 1)[1]
    canonical = re.sub(r"_\d+$", "", canonical)
    result["raw_canonical"] = canonical if canonical else "<EMPTY>"

    # Extract class
    cls_match = re.search(r"Class([12])", canonical, re.IGNORECASE)
    if cls_match:
        result["class"] = f"Class{cls_match.group(1)}"

    # Try to find subtype (e.g. I-E, V-A, III-B, IV-A3 …)
    # CRISPRCasFinder uses both "Subtype-X-Y" and just "Type-X" patterns
    sub_match = re.search(
        r"(?:Subtype|Type)[_-]?([IVX]+(?:-[A-Z0-9]+)?)",
        canonical,
        re.IGNORECASE
    )
    if sub_match:
        st = sub_match.group(1).upper()
        result["subtype"] = st
        result["type"] = st.split("-")[0]
    else:
        # fallback: just a type without subtype letter
        type_match = re.search(
            r"(?:Subtype|Type)[_-]([IVX]+)",
            canonical, re.IGNORECASE
        )
        if type_match:
            t = type_match.group(1).upper()
            result["type"] = t
            result["subtype"] = t   # no finer resolution available

    # Normalise class from type if not already found
    if result["class"] == "Unknown" and result["type"] != "Unknown":
        result["class"] = TYPE_TO_CLASS.get(result["type"], "Unknown")

    has_known_type = result["type"] in TYPE_TO_CLASS

    subtype_is_known = result["subtype"] in KNOWN_SUBTYPES
    subtype_is_coarse = result["subtype"] in COARSE_TYPES

    if not subtype_is_known and not subtype_is_coarse:
        # Keep type-level assignment even if subtype is not recognised.
        result["subtype"] = "Unknown"

    raw_display = result["raw_canonical"]

    is_coarse_type_only = bool(
        re.search(r"Type[_-][IVX]+(?:$|[_-])", canonical, re.IGNORECASE)
    ) and not bool(re.search(r"Subtype", canonical, re.IGNORECASE))

    # Type-level unknown: parser could not map raw string to a known type.
    if not has_known_type:
        result["parse_status"] = "unknown_type"
        if raw_display not in SEEN_UNKNOWN_TYPE_RAW:
            print(f"UNPARSED TYPE (mapped to Unknown type): {raw_display}")
            SEEN_UNKNOWN_TYPE_RAW.add(raw_display)
    # Coarse type-only labels are accepted at subtype level as coarse tokens (I/III/IV...).
    elif subtype_is_coarse and is_coarse_type_only:
        result["parse_status"] = "coarse_type_only"
        if raw_display not in SEEN_COARSE_TYPE_RAW:
            print(
                "COARSE TYPE ONLY "
                f"(kept type={result['type']}, subtype={result['subtype']}): {raw_display}"
            )
            SEEN_COARSE_TYPE_RAW.add(raw_display)
    # Subtype-level unknown: type is known, subtype was not recognised.
    elif result["subtype"] == "Unknown":
        result["parse_status"] = "unknown_subtype"
        if raw_display not in SEEN_UNKNOWN_SUBTYPE_RAW:
            print(
                "UNRECOGNISED SUBTYPE "
                f"(kept type={result['type']}, subtype=Unknown): {raw_display}"
            )
            SEEN_UNKNOWN_SUBTYPE_RAW.add(raw_display)

    return result


def genomic_distance(array_start: int, array_end: int,
                     cas_start: int, cas_end: int) -> int:
    """
    Minimum base-pair distance between an array and a Cas cassette.
    Returns 0 if they overlap.
    """
    if array_end < cas_start:
        return cas_start - array_end
    if cas_end < array_start:
        return array_start - cas_end
    return 0   # overlap


def find_nearest_cas(array_start, array_end, cas_list):
    """
    Given a list of Cas cassette dicts (each with Start, End, parsed_type),
    return the one with the smallest genomic distance to the array.
    Returns (nearest_cas_dict, distance) or (None, None) if cas_list is empty.
    """
    if not cas_list:
        return None, None
    best, best_dist = None, float("inf")
    for cas in cas_list:
        d = genomic_distance(array_start, array_end, cas["Start"], cas["End"])
        if d < best_dist:
            best_dist = d
            best = cas
    return best, best_dist


# ──────────────────────────────────────────────────────────────────────────────
# Core parsing
# ──────────────────────────────────────────────────────────────────────────────

def parse_result_json(json_path: Path,
                      global_counter: Counter,
                      unknown_type_counter: Counter | None = None,
                      unknown_subtype_counter: Counter | None = None,
                      coarse_type_counter: Counter | None = None,
                      include_all_cas_links: bool = False,
                      all_cas_min_evidence: int = 4,
                      all_cas_max_distance_bp: int = 0) -> tuple[list[dict], list[dict]]:
    """
    Parse one CRISPRCasFinder result.json.
    Returns a list of annotated array dicts.
    """
    with open(json_path) as fh:
        data = json.load(fh)

    rows = []
    all_cas_links = []
    for seq in data.get("Sequences", []):
        seq_id   = seq.get("Id", "")
        seq_ver  = seq.get("Version", seq_id)
        seq_desc = seq.get("Description", "")
        seq_len  = seq.get("Length", 0)

        # Parse and enrich all Cas cassettes in this sequence
        cas_list = []
        for cas in seq.get("Cas", []):
            raw_type = cas.get("Type", "")
            global_counter[raw_type] += 1

            parsed = parse_cas_type(raw_type)
            canonical_raw = parsed["raw_canonical"]
            if unknown_type_counter is not None and parsed["parse_status"] == "unknown_type":
                unknown_type_counter[canonical_raw] += 1
            if unknown_subtype_counter is not None and parsed["parse_status"] == "unknown_subtype":
                unknown_subtype_counter[canonical_raw] += 1
            if coarse_type_counter is not None and parsed["parse_status"] == "coarse_type_only":
                coarse_type_counter[canonical_raw] += 1
            parsed["Start"] = cas.get("Start", 0)
            parsed["End"]   = cas.get("End", 0)
            parsed["n_genes"] = len(cas.get("Genes", []))
            cas_list.append(parsed)

        for array in seq.get("Crisprs", []):
            ev = array.get("Evidence_Level", 0)
            a_start = array.get("Start", 0)
            a_end   = array.get("End", 0)
            nearest_cas, dist = find_nearest_cas(a_start, a_end, cas_list)

            row = {
                # ── array identity ───────────────────────────────────────────
                "array_name":           array.get("Name", ""),
                "genome_id":            seq_id,
                "genome_version":       seq_ver,
                "genome_description":   seq_desc,
                "genome_length":        seq_len,
                # ── array coordinates ────────────────────────────────────────
                "array_start":          a_start,
                "array_end":            a_end,
                "array_length":         a_end - a_start,
                # ── array properties ─────────────────────────────────────────
                "evidence_level":       ev,
                "repeat_id":            array.get("Repeat_ID", "Unknown"),
                "dr_consensus":         array.get("DR_Consensus", ""),
                "dr_length":            array.get("DR_Length", 0),
                "n_spacers":            array.get("Spacers", 0),
                "potential_orientation":array.get("Potential_Orientation", "ND"),
                "crispr_direction":     array.get("CRISPRDirection", "ND"),
                "conservation_drs":     array.get("Conservation_DRs", None),
                "conservation_spacers": array.get("Conservation_Spacers", None),
                # ── Cas cassette ─────────────────────────────────────────────
                "n_cas_cassettes_in_genome": len(cas_list),
                "nearest_cas_start":    nearest_cas["Start"]   if nearest_cas else None,
                "nearest_cas_end":      nearest_cas["End"]     if nearest_cas else None,
                "nearest_cas_raw_type": nearest_cas["raw"]     if nearest_cas else None,
                "nearest_cas_raw_type_canonical": nearest_cas["raw_canonical"] if nearest_cas else None,
                "nearest_cas_parse_status": nearest_cas["parse_status"] if nearest_cas else "no_cas",
                "cas_class":            nearest_cas["class"]   if nearest_cas else "NoCas",
                "cas_type":             nearest_cas["type"]    if nearest_cas else "NoCas",
                "cas_subtype":          nearest_cas["subtype"] if nearest_cas else "NoCas",
                "cas_n_genes":          nearest_cas["n_genes"] if nearest_cas else 0,
                "distance_to_cas_bp":   dist,
                # ── source file ──────────────────────────────────────────────
                "source_json":          str(json_path),
            }
            rows.append(row)

            if include_all_cas_links and ev >= all_cas_min_evidence and cas_list:
                candidate_links = []
                for cas in cas_list:
                    cas_dist = genomic_distance(a_start, a_end, cas["Start"], cas["End"])
                    if all_cas_max_distance_bp > 0 and cas_dist > all_cas_max_distance_bp:
                        continue
                    candidate_links.append({
                        "array_name": row["array_name"],
                        "genome_id": seq_id,
                        "evidence_level": ev,
                        "array_start": a_start,
                        "array_end": a_end,
                        "nearest_cas_raw_type": row["nearest_cas_raw_type"],
                        "nearest_cas_raw_type_canonical": row["nearest_cas_raw_type_canonical"],
                        "cas_start": cas["Start"],
                        "cas_end": cas["End"],
                        "distance_to_cas_bp": cas_dist,
                        "cas_class": cas["class"],
                        "cas_type": cas["type"],
                        "cas_subtype": cas["subtype"],
                        "cas_raw_type": cas["raw"],
                        "cas_n_genes": cas["n_genes"],
                        "source_json": str(json_path),
                    })

                candidate_links.sort(
                    key=lambda x: (
                        x["distance_to_cas_bp"],
                        x["cas_start"],
                        x["cas_end"],
                        x["cas_raw_type"],
                    )
                )
                for rank, link in enumerate(candidate_links, start=1):
                    link["cas_rank_by_distance"] = rank
                    all_cas_links.append(link)

    return rows, all_cas_links


# ──────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--results_dir", required=True,
        help="Folder containing all Result_*/ subdirectories",
    )
    parser.add_argument(
        "--cluster_csv", required=True,
        help="ccf_vs_out-sp_style_clusters_detail.csv (or similar)",
    )
    parser.add_argument(
        "--out_dir", default="./output_dataset",
        help="Where to write output files (default: ./output_dataset)",
    )
    parser.add_argument(
        "--max_distance_bp", type=int, default=50000,
        help="Max bp distance to nearest Cas cassette to keep a pair "
             "(default: 15000). Use 0 to keep all distances.",
    )
    parser.add_argument(
        "--min_spacers", type=int, default=2,
        help="Minimum spacer count to keep an array (default: 2). "
             "Evidence-level-4 arrays already tend to have ≥3.",
    )
    parser.add_argument(
        "--require_ccf_agreement", action="store_true",
        help="Keep only arrays where evOr agrees with the CCF direction. "
             "Useful for a high-confidence subset, but usually not ideal "
             "if you want to train on all evOr outputs.",
    )
    parser.add_argument(
        "--analyze_other_evidence_levels", action="store_true",
        help="Also write a separate analysis file for arrays with evidence level < 4. "
             "These arrays are not added to the curated ML dataset.",
    )
    parser.add_argument(
        "--write_all_cas_links_for_evidence4", action="store_true",
        help="Write a second table linking each evidence-level-4 array to all Cas cassettes "
             "within the distance window (--max_distance_bp).",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    out_dir     = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Step 1: find all result.json files ───────────────────────────────────
    json_files = sorted(results_dir.rglob("result.json"))
    print(f"Found {len(json_files)} result.json files under {results_dir}")

    # ── Step 2: parse all JSONs → intermediate table ─────────────────────────
    global_raw_type_counter = Counter()
    unknown_type_counter = Counter()
    unknown_subtype_counter = Counter()
    coarse_type_counter = Counter()

    all_rows = []
    all_cas_links_rows = []
    parse_errors = []
    for jf in json_files:
        try:
            rows, links = parse_result_json(
                jf,
                global_raw_type_counter,
                unknown_type_counter=unknown_type_counter,
                unknown_subtype_counter=unknown_subtype_counter,
                coarse_type_counter=coarse_type_counter,
                include_all_cas_links=args.write_all_cas_links_for_evidence4,
                all_cas_min_evidence=4,
                all_cas_max_distance_bp=args.max_distance_bp,
            )
            all_rows.extend(rows)
            if args.write_all_cas_links_for_evidence4:
                all_cas_links_rows.extend(links)
        except Exception as exc:
            parse_errors.append((str(jf), str(exc)))
            print(f"  WARNING: could not parse {jf}: {exc}")

    print(f"Parsed {len(all_rows)} CRISPR arrays total "
          f"({len(parse_errors)} files with errors)")

    print("\n=== GLOBAL RAW CAS TYPE DISTRIBUTION ===")
    for k, v in global_raw_type_counter.most_common(20):
        print(f"{v:6d}  {k}")

    print("\n=== UNKNOWN CAS LABEL DIAGNOSTICS ===")
    print(f"Unknown TYPE assignments    : {sum(unknown_type_counter.values())}")
    print(f"Unknown SUBTYPE assignments : {sum(unknown_subtype_counter.values())}")
    print(f"Coarse TYPE-only labels     : {sum(coarse_type_counter.values())}")
    if unknown_type_counter:
        print("Top canonical labels causing Unknown type:")
        for raw, count in unknown_type_counter.most_common(20):
            print(f"  {count:6d}  {raw}")
    if unknown_subtype_counter:
        print("Top canonical labels causing Unknown subtype:")
        for raw, count in unknown_subtype_counter.most_common(20):
            print(f"  {count:6d}  {raw}")
    if coarse_type_counter:
        print("Top canonical coarse type-only labels:")
        for raw, count in coarse_type_counter.most_common(20):
            print(f"  {count:6d}  {raw}")
    # ── Step 3: write intermediate file (all evidence levels) ────────────────
    intermediate_path = out_dir / "intermediate_array_cas_annotations.tsv"
    intermediate_fields = [
        "array_name", "genome_id", "genome_version", "genome_description",
        "genome_length", "array_start", "array_end", "array_length",
        "evidence_level", "repeat_id", "dr_consensus", "dr_length",
        "n_spacers", "potential_orientation", "crispr_direction",
        "conservation_drs", "conservation_spacers",
        "n_cas_cassettes_in_genome",
        "nearest_cas_start", "nearest_cas_end", "nearest_cas_raw_type",
        "nearest_cas_raw_type_canonical",
        "nearest_cas_parse_status",
        "cas_class", "cas_type", "cas_subtype", "cas_n_genes",
        "distance_to_cas_bp", "source_json",
    ]
    with open(intermediate_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=intermediate_fields, delimiter="\t",
                                extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"Intermediate file written: {intermediate_path}")

    # ── Step 4: load evOr cluster CSV ────────────────────────────────────────
    cluster_map = {}   # array_name → {ccf_direction, evor_direction, group_name, agreement}
    with open(args.cluster_csv, newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            cluster_map[row["array_name"]] = row

    print(f"Loaded {len(cluster_map)} arrays from cluster CSV")

    # ── Step 5: apply filters and join ───────────────────────────────────────
    ml_rows        = []
    dropped_evlvl  = 0
    dropped_spacer = 0
    dropped_nocas  = 0
    dropped_dist   = 0
    dropped_nocsv  = 0
    dropped_agree  = 0

    for row in all_rows:
        # Filter 1: evidence level 4 only
        if row["evidence_level"] < 4:
            dropped_evlvl += 1
            continue

        # Filter 2: minimum spacers
        if row["n_spacers"] < args.min_spacers:
            dropped_spacer += 1
            continue

        # Filter 3: must have a paired Cas cassette
        if row["cas_type"] == "NoCas":
            dropped_nocas += 1
            continue

        # Filter 4: distance threshold
        if args.max_distance_bp > 0 and row["distance_to_cas_bp"] is not None:
            if row["distance_to_cas_bp"] > args.max_distance_bp:
                dropped_dist += 1
                continue

        # Filter 5: must be in the evOr CSV
        clu = cluster_map.get(row["array_name"])
        if clu is None:
            dropped_nocsv += 1
            continue

        # Filter 6: optionally require evOr / CCF direction agreement
        if args.require_ccf_agreement and clu["agreement"] != "agree":
            dropped_agree += 1
            continue

        # Build final ML row
        ml_row = dict(row)
        ml_row["evor_direction"]  = clu["evor_direction"]
        ml_row["group_name"]      = clu["group_name"]
        ml_row["agreement"]       = clu["agreement"]
        ml_rows.append(ml_row)

    low_evidence_rows = [row for row in all_rows if row["evidence_level"] < 4]

    print(f"\nFiltering summary:")
    print(f"  Dropped evidence_level < 4 : {dropped_evlvl}")
    print(f"  Dropped n_spacers < {args.min_spacers}       : {dropped_spacer}")
    print(f"  Dropped no Cas cassette    : {dropped_nocas}")
    print(f"  Dropped distance > {args.max_distance_bp} bp : {dropped_dist}")
    print(f"  Dropped not in cluster CSV : {dropped_nocsv}")
    print(f"  Dropped direction disagree : {dropped_agree}")
    if args.require_ccf_agreement:
        print(f"  CCF agreement filter      : enabled")
    else:
        print(f"  CCF agreement filter      : disabled")
    print(f"  ──────────────────────────────")
    print(f"  KEPT for ML dataset        : {len(ml_rows)}")

    if args.analyze_other_evidence_levels:
        low_evidence_path = out_dir / "low_evidence_analysis.tsv"
        low_evidence_fields = [
            "array_name", "genome_id", "genome_version", "genome_description",
            "genome_length", "array_start", "array_end", "array_length",
            "evidence_level", "repeat_id", "dr_consensus", "dr_length",
            "n_spacers", "potential_orientation", "crispr_direction",
            "conservation_drs", "conservation_spacers",
            "n_cas_cassettes_in_genome",
            "nearest_cas_start", "nearest_cas_end", "nearest_cas_raw_type",
            "nearest_cas_raw_type_canonical",
            "nearest_cas_parse_status",
            "cas_class", "cas_type", "cas_subtype", "cas_n_genes",
            "distance_to_cas_bp", "source_json",
        ]
        with open(low_evidence_path, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=low_evidence_fields, delimiter="\t",
                                    extrasaction="ignore")
            writer.writeheader()
            writer.writerows(low_evidence_rows)
        print(f"\nLow-evidence analysis written: {low_evidence_path}")

        low_counts = Counter(row["evidence_level"] for row in low_evidence_rows)
        low_class_counts = Counter(row["cas_class"] for row in low_evidence_rows)
        low_type_counts = Counter(row["cas_type"] for row in low_evidence_rows)
        low_subtype_counts = Counter(row["cas_subtype"] for row in low_evidence_rows)

        print("Low-evidence arrays by evidence level:")
        for evidence_level, count in sorted(low_counts.items()):
            print(f"  Evidence {evidence_level}: {count}")

        print("Low-evidence class distribution:")
        for key, value in sorted(low_class_counts.items(), key=lambda item: (-item[1], item[0])):
            print(f"  {key:15s}  {value:8d}")

        print("Low-evidence type distribution:")
        for key, value in sorted(low_type_counts.items(), key=lambda item: (-item[1], item[0])):
            print(f"  {key:15s}  {value:8d}")

        print("Low-evidence subtype distribution:")
        for key, value in sorted(low_subtype_counts.items(), key=lambda item: (-item[1], item[0])):
            known = "✓" if key in KNOWN_SUBTYPES else "?"
            print(f"  {known} {key:20s}  {value:8d}")

    if args.write_all_cas_links_for_evidence4:
        links_path = out_dir / "evidence4_all_cas_links.tsv"
        links_fields = [
            "array_name", "genome_id", "evidence_level",
            "array_start", "array_end",
            "nearest_cas_raw_type",
            "nearest_cas_raw_type_canonical",
            "cas_rank_by_distance",
            "cas_start", "cas_end", "distance_to_cas_bp",
            "cas_class", "cas_type", "cas_subtype", "cas_raw_type", "cas_n_genes",
            "source_json",
        ]
        with open(links_path, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=links_fields, delimiter="\t", extrasaction="ignore")
            writer.writeheader()
            writer.writerows(all_cas_links_rows)

        e4_total = sum(1 for row in all_rows if row["evidence_level"] >= 4)
        linked_arrays = {row["array_name"] for row in all_cas_links_rows}
        linked_count = len(linked_arrays)
        mean_links = round(len(all_cas_links_rows) / linked_count, 3) if linked_count else 0.0

        print(f"\nAll-cassette evidence-4 links written: {links_path}")
        print("Evidence-4 all-cassette link summary:")
        print(f"  Evidence-4 arrays total           : {e4_total}")
        print(f"  Evidence-4 arrays with >=1 link   : {linked_count}")
        print(f"  Total array-to-cassette links     : {len(all_cas_links_rows)}")
        print(f"  Mean links per linked array       : {mean_links}")

    # ── Step 6: write ML dataset ──────────────────────────────────────────────
    ml_fields = [
        # identifiers
        "array_name", "genome_id", "genome_version",
        # array features (potential ML input features)
        "array_start", "array_end", "array_length",
        "dr_length", "n_spacers", "dr_consensus",
        "potential_orientation", "crispr_direction", "evor_direction",
        "conservation_drs", "conservation_spacers",
        "repeat_id",
        # evOr cluster (the grouping your model should predict)
        "group_name",
        # Cas type labels (the ground-truth type labels)
        "cas_class", "cas_type", "cas_subtype",
        "distance_to_cas_bp",
        # metadata
        "evidence_level", "agreement",
    ]
    ml_path = out_dir / "curated_dataset.tsv"
    with open(ml_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=ml_fields, delimiter="\t",
                                extrasaction="ignore")
        writer.writeheader()
        writer.writerows(ml_rows)
    print(f"\nML dataset written: {ml_path}")

    # ── Step 7: build report ──────────────────────────────────────────────────
    subtype_counts = Counter(r["cas_subtype"] for r in ml_rows)
    type_counts    = Counter(r["cas_type"]    for r in ml_rows)
    class_counts   = Counter(r["cas_class"]   for r in ml_rows)
    group_counts   = Counter(r["group_name"]  for r in ml_rows)

    report_path = out_dir / "build_report.txt"
    with open(report_path, "w") as fh:
        fh.write("=" * 60 + "\n")
        fh.write("CRISPR Dataset Build Report\n")
        fh.write("=" * 60 + "\n\n")

        fh.write(f"Input result.json files processed : {len(json_files)}\n")
        fh.write(f"Files with parse errors           : {len(parse_errors)}\n")
        fh.write(f"Total arrays parsed               : {len(all_rows)}\n\n")

        fh.write("Filtering:\n")
        fh.write(f"  evidence_level < 4 dropped      : {dropped_evlvl}\n")
        fh.write(f"  n_spacers < {args.min_spacers} dropped           : {dropped_spacer}\n")
        fh.write(f"  no Cas cassette dropped          : {dropped_nocas}\n")
        fh.write(f"  distance > {args.max_distance_bp} bp dropped      : {dropped_dist}\n")
        fh.write(f"  not in cluster CSV dropped       : {dropped_nocsv}\n")
        fh.write(f"  direction disagree dropped       : {dropped_agree}\n")
        fh.write(f"  FINAL ML DATASET ROWS            : {len(ml_rows)}\n\n")

        fh.write("Class distribution:\n")
        for k, v in sorted(class_counts.items()):
            fh.write(f"  {k:15s}: {v}\n")

        fh.write("\nType distribution:\n")
        for k, v in sorted(type_counts.items()):
            fh.write(f"  {k:15s}: {v}\n")

        fh.write("\nSubtype distribution (from Makarova et al. 2025 classification):\n")
        for k, v in sorted(subtype_counts.items(), key=lambda x: -x[1]):
            known = "✓" if k in KNOWN_SUBTYPES else "?"
            fh.write(f"  {known} {k:20s}: {v}\n")

        fh.write(f"\nevOr group distribution (top 30):\n")
        for k, v in group_counts.most_common(30):
            fh.write(f"  {k:20s}: {v}\n")

        if parse_errors:
            fh.write("\nFiles with parse errors:\n")
            for path, err in parse_errors:
                fh.write(f"  {path}: {err}\n")

    print(f"Build report written: {report_path}")

    def print_distribution(title: str, counts: Counter):
        print(f"\n{title}")
        total = sum(counts.values()) or 1
        for key, value in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
            pct = 100.0 * value / total
            print(f"  {key:15s}  {value:8d}  ({pct:6.2f}%)")

    print_distribution("Class representation in final ML dataset:", class_counts)
    print_distribution("Type representation in final ML dataset:", type_counts)
    print_distribution("Subtype representation in final ML dataset:", subtype_counts)

    # Quick type coverage check
    covered = {r["cas_subtype"] for r in ml_rows} & KNOWN_SUBTYPES
    missing = KNOWN_SUBTYPES - covered
    print(f"\nSubtype coverage: {len(covered)}/{len(KNOWN_SUBTYPES)} known subtypes present")
    if missing:
        print(f"  Not represented (likely rare/new): {', '.join(sorted(missing))}")


if __name__ == "__main__":
    main()