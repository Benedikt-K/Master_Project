#!/usr/bin/env python3
"""
CRISPR Array Orientation Comparison Script
==========================================
Compares orientation predictions from:
  - CRISPRCasFinder (test_out/)
  - CRISPR-evOr across multiple clusterings

Usage:
    python compare-results.py [--base-dir BASE_DIR] [--out-dir OUT_DIR]

Defaults:
    --base-dir  .   (current working directory)
    --out-dir   ./crispr_comparison_results/
"""

import os
import re
import json
import ast
import csv
import argparse
import itertools
from pathlib import Path
from collections import defaultdict

import pandas as pd
from scipy import stats


# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

EVOL_TOOL_DIRS = [
    "out-sp_style_clusters",
]

SINGLETON_DIRS = {
    "out-cluster-size-50":       "cluster-size-50",
    "out-cluster-size-150":      "cluster-size-150",
    "out-cluster-size-300":      "cluster-size-300",
    "out-SP-cluster-smaller200": "SP-cluster-smaller200",
    "out-sp_style_clusters":     "sp_style_clusters",
}

CCF_TOOL_DIR = "test_out"

# Name of the SP-style clustering — needs flip-map correction
SP_STYLE_CLUSTERING = "out-sp_style_clusters"


# ─────────────────────────────────────────────────────────────────────────────
# FLIP-MAP HELPERS  (for SP-style clustering orientation correction)
# ─────────────────────────────────────────────────────────────────────────────

def load_flip_map(metadata_path: Path) -> dict[str, bool]:
    """
    Load cluster_metadata.json written by cluster_like_spacerplacer.py.
    Returns: array_id -> was_flipped (bool)
    """
    if not metadata_path.exists():
        print(f"  [WARN] SP-style metadata not found at {metadata_path}. "
              f"No flip correction will be applied.")
        return {}
    with open(metadata_path) as fh:
        meta = json.load(fh)
    flip_map: dict[str, bool] = {}
    for cluster_info in meta.values():
        for array_id, info in cluster_info.get("arrays", {}).items():
            flip_map[array_id] = bool(info.get("was_flipped", False))
    n_flipped = sum(flip_map.values())
    print(f"  [SP-style metadata] {len(flip_map)} arrays loaded, "
          f"{n_flipped} were flipped during SP-style clustering.")
    return flip_map


def flip_direction(direction: str) -> str:
    """Invert Forward<->Reverse. Leaves Unknown/Singleton unchanged."""
    if direction == "Forward":
        return "Reverse"
    if direction == "Reverse":
        return "Forward"
    return direction


# ─────────────────────────────────────────────────────────────────────────────
# NORMALISATION HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def normalise_ccf_direction(raw: str) -> str:
    """
    Convert CRISPRCasFinder direction strings to Forward / Reverse / Unknown.

    JSON values:  "+", "-", "ND", "", None
    TSV values:   "F [… Confidence: HIGH]", "R [… Confidence: MEDIUM]",
                  "NA [… Confidence: NA]", "Unknown"
    """
    if raw is None:
        return "Unknown"
    raw = str(raw).strip()
    if raw in ("+", "F") or raw.startswith("F "):
        return "Forward"
    if raw in ("-", "R") or raw.startswith("R "):
        return "Reverse"
    return "Unknown"


def normalise_evor_direction(raw: str) -> str:
    """CRISPR-evOr already uses 'Forward'/'Reverse'; just normalise."""
    if raw is None:
        return "Unknown"
    raw = str(raw).strip()
    if raw.lower() == "forward":
        return "Forward"
    if raw.lower() == "reverse":
        return "Reverse"
    return "Unknown"


def orientations_agree(a: str, b: str) -> str:
    """
    Return 'agree', 'disagree', or 'not_comparable'
    (when at least one prediction is Unknown).
    """
    if a == "Unknown" or b == "Unknown":
        return "not_comparable"
    return "agree" if a == b else "disagree"


# ─────────────────────────────────────────────────────────────────────────────
# CRISPRCASFINDER PARSERS
# ─────────────────────────────────────────────────────────────────────────────

def parse_ccf_json(json_path: Path) -> list[dict]:
    """
    Parse a CRISPRCasFinder result.json file.
    Returns only arrays with Evidence_Level == 4.
    """
    records = []
    with open(json_path) as fh:
        data = json.load(fh)

    for seq in data.get("Sequences", []):
        seq_id = seq.get("Id", "")
        for crispr in seq.get("Crisprs", []):
            ev = crispr.get("Evidence_Level", 0)
            if ev != 4:
                continue
            records.append({
                "array_name": crispr["Name"],
                "sequence_id": seq_id,
                "start": crispr["Start"],
                "end": crispr["End"],
                "spacers": crispr.get("Spacers", 0),
                "dr_consensus": crispr.get("DR_Consensus", ""),
                "evidence_level": ev,
                "ccf_direction_raw": crispr.get("CRISPRDirection", "ND"),
                "ccf_direction": normalise_ccf_direction(crispr.get("CRISPRDirection", "ND")),
                "source_file": str(json_path),
            })
    return records


def parse_ccf_tsv(tsv_path: Path) -> list[dict]:
    """
    Parse a CRISPRCasFinder Crisprs_REPORT.tsv file.
    Returns only arrays with Evidence_Level == 4.
    """
    records = []
    df = pd.read_csv(tsv_path, sep="\t", dtype=str)
    for _, row in df.iterrows():
        try:
            ev = int(float(row.get("Evidence_Level", 0)))
        except (ValueError, TypeError):
            ev = 0
        if ev != 4:
            continue

        raw_dir = row.get("CRISPRDirection", "ND")
        records.append({
            "array_name": row.get("CRISPR_Id", ""),
            "sequence_id": row.get("Sequence_basename", ""),
            "start": row.get("CRISPR_Start", ""),
            "end": row.get("CRISPR_End", ""),
            "spacers": row.get("Spacers_Nb", ""),
            "dr_consensus": row.get("Consensus_Repeat", ""),
            "evidence_level": ev,
            "ccf_direction_raw": raw_dir,
            "ccf_direction": normalise_ccf_direction(raw_dir),
            "source_file": str(tsv_path),
        })
    return records


def load_ccf_predictions(base_dir: Path) -> pd.DataFrame:
    """
    Walk test_out/ and collect all Evidence-Level-4 arrays.
    Prefers result.json; falls back to TSV/Crisprs_REPORT.tsv.
    De-duplicates by array_name (first occurrence wins).
    """
    tool_dir = base_dir / CCF_TOOL_DIR
    if not tool_dir.exists():
        raise FileNotFoundError(f"CRISPRCasFinder output directory not found: {tool_dir}")

    all_records = []
    seen_arrays = set()

    for result_folder in sorted(tool_dir.iterdir()):
        if not result_folder.is_dir():
            continue

        json_path = result_folder / "result.json"
        tsv_path  = result_folder / "TSV" / "Crisprs_REPORT.tsv"

        records = []
        if json_path.exists():
            try:
                records = parse_ccf_json(json_path)
            except Exception as e:
                print(f"  [WARN] Could not parse {json_path}: {e}")
        elif tsv_path.exists():
            try:
                records = parse_ccf_tsv(tsv_path)
            except Exception as e:
                print(f"  [WARN] Could not parse {tsv_path}: {e}")
        else:
            print(f"  [WARN] No usable result file in {result_folder}")
            continue

        for r in records:
            if r["array_name"] not in seen_arrays:
                seen_arrays.add(r["array_name"])
                all_records.append(r)

    print(f"[CRISPRCasFinder] Loaded {len(all_records)} Evidence-Level-4 arrays "
          f"from {tool_dir}")
    return pd.DataFrame(all_records)


# ─────────────────────────────────────────────────────────────────────────────
# CRISPR-evOr PARSERS
# ─────────────────────────────────────────────────────────────────────────────

def parse_evor_csv(csv_path: Path) -> list[dict]:
    """
    Parse a CRISPR-evOr 0_results.csv file.
    Returns one record per *array* contained in the cluster (the cluster
    prediction is broadcast to every member array).
    """
    records = []
    df = pd.read_csv(csv_path, dtype=str)
    if df.empty:
        return records

    for _, row in df.iterrows():
        group_name = row.get("name", "")
        raw_orientation = row.get("predicted orientation", "")
        orientation = normalise_evor_direction(raw_orientation)
        recommend_reverse = str(row.get("recommend reversing array", "False")).strip()

        # Parse the list of array names stored as a Python-repr string
        raw_names = row.get("array names", "[]")
        try:
            array_names = ast.literal_eval(raw_names)
        except Exception:
            # fallback: strip brackets and split on comma
            array_names = [
                n.strip().strip("'\"")
                for n in raw_names.strip("[]").split(",")
                if n.strip()
            ]

        for arr_name in array_names:
            arr_name = arr_name.strip()
            if not arr_name:
                continue
            records.append({
                "array_name": arr_name,
                "group_name": group_name,
                "evor_direction": orientation,
                "evor_direction_raw": raw_orientation,
                "recommend_reverse": recommend_reverse,
                "source_file": str(csv_path),
            })
    return records


def parse_fasta_ids(fa_path: Path) -> list[str]:
    """Return all sequence IDs from a FASTA file (singleton arrays)."""
    ids = []
    with open(fa_path) as fh:
        for line in fh:
            line = line.strip()
            if line.startswith(">"):
                ids.append(line[1:].split()[0])
    return ids


def load_evor_predictions(base_dir: Path, tool_dir_name: str,
                          flip_map: dict[str, bool] | None = None) -> pd.DataFrame:
    """
    Load CRISPR-evOr predictions for one clustering configuration.
    Includes singleton arrays (orientation = 'Singleton – no prediction').

    flip_map (optional): array_id -> was_flipped
      When provided (SP-style clustering), the evOr direction is treated as
      relative to the normalised input orientation. Arrays that were flipped
      during clustering have their direction inverted so that `evor_direction`
      always represents the *genomic* strand.
      The raw tool output is preserved in `evor_direction_tool`.
    """
    tool_dir = base_dir / tool_dir_name
    singleton_root = base_dir / SINGLETON_DIRS[tool_dir_name] / "singletons"

    records = []

    # ── clustered arrays ──────────────────────────────────────────────────────
    if tool_dir.exists():
        for group_dir in sorted(tool_dir.iterdir()):
            if not group_dir.is_dir():
                continue
            csv_path = group_dir / "0_results.csv"
            if not csv_path.exists():
                continue
            try:
                records.extend(parse_evor_csv(csv_path))
            except Exception as e:
                print(f"  [WARN] Could not parse {csv_path}: {e}")
    else:
        print(f"  [WARN] Directory not found: {tool_dir}")

    # ── singleton arrays ──────────────────────────────────────────────────────
    n_singletons = 0
    if singleton_root.exists():
        for fa_file in sorted(singleton_root.glob("*.fa")) or \
                       sorted(singleton_root.glob("*.fasta")):
            for seq_id in parse_fasta_ids(fa_file):
                seq_id = seq_id.strip()
                if not seq_id:
                    continue
                records.append({
                    "array_name": seq_id,
                    "group_name": "singleton",
                    "evor_direction": "Unknown", # is singleton has no prediction
                    "evor_direction_raw": "singleton",
                    "recommend_reverse": "N/A",
                    "source_file": str(fa_file),
                })
                n_singletons += 1
    else:
        print(f"  [WARN] Singleton directory not found: {singleton_root}")

    # ── apply flip correction if a flip_map was provided ─────────────────────
    n_corrected = 0
    for r in records:
        # always store what the tool actually said
        r["evor_direction_tool"] = r["evor_direction"]
        r["was_flipped"] = False

        if flip_map is not None:
            flipped = flip_map.get(r["array_name"], False)
            r["was_flipped"] = flipped
            if flipped and r["evor_direction"] in ("Forward", "Reverse"):
                r["evor_direction"] = flip_direction(r["evor_direction"])
                n_corrected += 1

    df = pd.DataFrame(records)

    msg = (f"[CRISPR-evOr / {tool_dir_name}] "
           f"Loaded {len(df) - n_singletons} clustered + {n_singletons} singleton arrays")
    if flip_map is not None:
        msg += f"  ({n_corrected} directions corrected for genomic strand)"
    print(msg)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# STATISTICS HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def compute_pair_stats(merged: pd.DataFrame, col_a: str, col_b: str,
                       label_a: str, label_b: str) -> dict:
    """
    Compute agreement statistics between two orientation columns.
    Excludes rows where either value is 'Unknown' or 'Singleton'.
    """
    comparable = merged[
        (~merged[col_a].isin(["Unknown"])) &
        (~merged[col_b].isin(["Unknown"]))
    ].copy()

    n_total   = len(merged)
    n_comp    = len(comparable)
    n_skip    = n_total - n_comp

    if n_total == 0:
        return {
            "label_a": label_a,
            "label_b": label_b,
            "n_total": 0,
            "n_comparable": 0,
            "n_skipped": 0,
            "n_agree": 0,
            "n_disagree": 0,
            "pct_agree": None,
            "pct_disagree": None,
            "pct_agree_effective": None,
            "coverage_pct": None,
            "pct_full_agree": None,
            "n_asymmetric": 0,
            "pct_asymmetric": None,
            "n_ccf_only": 0,
            "n_evor_only": 0,
            "pct_ccf_only": None,
            "pct_evor_only": None,
        }

    # asymmetry: one predicts, the other is Unknown
    asymmetric = (
        ((merged[col_a] == "Unknown") & (merged[col_b] != "Unknown")) |
        ((merged[col_a] != "Unknown") & (merged[col_b] == "Unknown"))
    ).sum()

    pct_asymmetric = round(100 * asymmetric / n_total, 2)    

    # directional asymmetry
    ccf_only = (
        (merged[col_a] != "Unknown") &
        (merged[col_b] == "Unknown")
    ).sum()

    evor_only = (
        (merged[col_a] == "Unknown") &
        (merged[col_b] != "Unknown")
    ).sum()

    pct_ccf_only = round(100 * ccf_only / n_total, 2)
    pct_evor_only = round(100 * evor_only / n_total, 2)

    if n_comp == 0:
        return {
            "label_a": label_a, "label_b": label_b,
            "n_total": n_total, "n_comparable": 0, "n_skipped": n_skip,
            "n_agree": 0, "n_disagree": 0,
            "pct_agree": None, "pct_disagree": None,
            "pct_agree_effective": 0.0,
            "coverage_pct": 0.0,
            "pct_full_agree": round(100 * (merged[col_a] == merged[col_b]).sum() / n_total, 2),
            "n_asymmetric": int(asymmetric),
            "pct_asymmetric": pct_asymmetric,
            "n_ccf_only": int(ccf_only),
            "n_evor_only": int(evor_only),
            "pct_ccf_only": pct_ccf_only,
            "pct_evor_only": pct_evor_only,
        }

    agree    = (comparable[col_a] == comparable[col_b]).sum()
    disagree = n_comp - agree

    # FULL agreement (including Unknown)
    full_agree = (merged[col_a] == merged[col_b]).sum()
    pct_full_agree = round(100 * full_agree / n_total, 2)

    return {
        "label_a":     label_a,
        "label_b":     label_b,
        "n_total":     n_total,
        "n_comparable": n_comp,
        "n_skipped":   n_skip,
        "n_agree":     int(agree),
        "n_disagree":  int(disagree),
        "pct_agree":   round(100 * agree / n_comp, 2),
        "pct_disagree": round(100 * disagree / n_comp, 2),

        # NEW metrics
        "pct_agree_effective": round(100 * agree / n_total, 2),
        "coverage_pct": round(100 * n_comp / n_total, 2),
        "pct_full_agree": pct_full_agree,
        "n_asymmetric": int(asymmetric),
        "pct_asymmetric": pct_asymmetric,

        # % of predictions that are unique to one tool (the other is Unknown)
        "n_ccf_only": int(ccf_only),
        "n_evor_only": int(evor_only),
        "pct_ccf_only": pct_ccf_only,
        "pct_evor_only": pct_evor_only,
    }


def compute_confidence_percentages(merged: pd.DataFrame, col_a: str, col_b: str) -> dict:
    """
    Compute overall confidence coverage per tool and jointly.

    A prediction is treated as confident when direction is Forward/Reverse
    (i.e., not Unknown).
    """
    n_total = len(merged)
    if n_total == 0:
        return {
            "n_total": 0,
            "n_confident_a": 0,
            "n_confident_b": 0,
            "n_confident_both": 0,
            "pct_confident_a": None,
            "pct_confident_b": None,
            "pct_confident_both": None,
        }

    confident_a = (merged[col_a] != "Unknown").sum()
    confident_b = (merged[col_b] != "Unknown").sum()
    confident_both = ((merged[col_a] != "Unknown") & (merged[col_b] != "Unknown")).sum()

    return {
        "n_total": int(n_total),
        "n_confident_a": int(confident_a),
        "n_confident_b": int(confident_b),
        "n_confident_both": int(confident_both),
        "pct_confident_a": round(100 * confident_a / n_total, 2),
        "pct_confident_b": round(100 * confident_b / n_total, 2),
        "pct_confident_both": round(100 * confident_both / n_total, 2),
    }


def compute_group_confidence(evor_df: pd.DataFrame) -> dict:
    """
    Group-level confidence score for SpacerPlacer/evOr output.

    Singletons are excluded by definition.
    A group is confidently predicted if its non-singleton direction is not Unknown.
    """
    if evor_df.empty or "group_name" not in evor_df.columns:
        return {
            "n_groups_non_singleton": 0,
            "n_groups_confident": 0,
            "pct_groups_confident": None,
        }

    non_singleton = evor_df[evor_df["group_name"] != "singleton"].copy()
    if non_singleton.empty:
        return {
            "n_groups_non_singleton": 0,
            "n_groups_confident": 0,
            "pct_groups_confident": None,
        }

    group_dir = non_singleton.groupby("group_name", dropna=True)["evor_direction"].first()
    n_groups = len(group_dir)
    n_conf = (group_dir != "Unknown").sum()

    return {
        "n_groups_non_singleton": int(n_groups),
        "n_groups_confident": int(n_conf),
        "pct_groups_confident": round(100 * n_conf / n_groups, 2),
    }


def direction_crosstab(merged: pd.DataFrame, col_a: str, col_b: str) -> pd.DataFrame:
    """Return a crosstab of the two direction columns."""
    return pd.crosstab(
        merged[col_a].fillna("Unknown"),
        merged[col_b].fillna("Unknown"),
        rownames=[col_a],
        colnames=[col_b],
    )


def resolve_default_cas_annotations(base_dir: Path) -> Path | None:
    """Pick a default Cas-annotation table if available."""
    candidates = [
        base_dir / "output_dataset" / "curated_dataset.tsv",
        base_dir / "output_dataset" / "intermediate_array_cas_annotations.tsv",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def load_cas_annotations(cas_path: Path) -> pd.DataFrame:
    """
    Load per-array Cas annotations.

    Expects at least: array_name, cas_type
    Optional: cas_subtype, cas_class
    """
    df = pd.read_csv(cas_path, sep="\t", dtype=str)
    needed = {"array_name", "cas_type"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(
            f"Cas annotation file {cas_path} missing required columns: {sorted(missing)}"
        )

    keep_cols = ["array_name", "cas_class", "cas_type", "cas_subtype"]
    cols = [c for c in keep_cols if c in df.columns]

    out = df[cols].copy()
    out["array_name"] = out["array_name"].astype(str).str.strip()
    out = out[out["array_name"] != ""]

    if "cas_type" in out.columns:
        out["cas_type"] = out["cas_type"].fillna("Unknown")

    # deterministic dedup in case array_name appears multiple times
    out = out.drop_duplicates(subset=["array_name"], keep="first")
    return out


def summarise_by_cas_type(
    merged: pd.DataFrame,
    col_a: str,
    col_b: str,
    label_a: str,
    label_b: str,
) -> pd.DataFrame:
    """
    Per-Cas-type performance summary to spot weakly recognised types.
    """
    if "cas_type" not in merged.columns:
        return pd.DataFrame()

    rows = []
    for cas_type, sub in merged.groupby("cas_type", dropna=False):
        cas_label = "Unknown" if pd.isna(cas_type) else str(cas_type)
        stats = compute_pair_stats(sub, col_a, col_b, label_a, label_b)
        conf = compute_confidence_percentages(sub, col_a, col_b)
        sub_no_sp_singletons = sub
        if "group_name" in sub.columns:
            sub_no_sp_singletons = sub[sub["group_name"] != "singleton"].copy()
        stats_no_sp_singletons = compute_pair_stats(
            sub_no_sp_singletons,
            col_a,
            col_b,
            label_a,
            f"{label_b} (no SP singletons)",
        )
        conf_no_sp_singletons = compute_confidence_percentages(
            sub_no_sp_singletons,
            col_a,
            col_b,
        )

        rows.append({
            "cas_type": cas_label,
            "n_total": stats["n_total"],
            "n_comparable": stats["n_comparable"],
            "coverage_pct": stats.get("coverage_pct"),
            "n_agree": stats["n_agree"],
            "n_disagree": stats["n_disagree"],
            "pct_agree": stats.get("pct_agree"),
            "pct_agree_effective": stats.get("pct_agree_effective"),
            "pct_asymmetric": stats.get("pct_asymmetric"),
            "pct_ccf_confident": conf.get("pct_confident_a"),
            "pct_sp_confident": conf.get("pct_confident_b"),
            "pct_both_confident": conf.get("pct_confident_both"),
            "pct_agree_no_sp_singletons": stats_no_sp_singletons.get("pct_agree"),
            "pct_agree_effective_no_sp_singletons": stats_no_sp_singletons.get("pct_agree_effective"),
            "pct_ccf_confident_no_sp_singletons": conf_no_sp_singletons.get("pct_confident_a"),
            "pct_sp_confident_no_sp_singletons": conf_no_sp_singletons.get("pct_confident_b"),
            "pct_both_confident_no_sp_singletons": conf_no_sp_singletons.get("pct_confident_both"),
        })

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    out = out.sort_values(["pct_agree_effective", "n_total"], ascending=[True, False])
    return out


def summarise_by_cas_subtype(
    merged: pd.DataFrame,
    col_a: str,
    col_b: str,
    label_a: str,
    label_b: str,
) -> pd.DataFrame:
    """
    Per-Cas-subtype performance summary to spot weakly recognised subtypes.
    """
    if "cas_subtype" not in merged.columns:
        return pd.DataFrame()

    rows = []
    for cas_subtype, sub in merged.groupby("cas_subtype", dropna=False):
        subtype_label = "Unknown" if pd.isna(cas_subtype) else str(cas_subtype)
        cas_type_label = "Unknown"
        if "cas_type" in sub.columns:
            cas_type_mode = sub["cas_type"].dropna()
            if not cas_type_mode.empty:
                cas_type_label = str(cas_type_mode.mode().iloc[0])

        stats = compute_pair_stats(sub, col_a, col_b, label_a, label_b)
        conf = compute_confidence_percentages(sub, col_a, col_b)
        sub_no_sp_singletons = sub
        if "group_name" in sub.columns:
            sub_no_sp_singletons = sub[sub["group_name"] != "singleton"].copy()
        stats_no_sp_singletons = compute_pair_stats(
            sub_no_sp_singletons,
            col_a,
            col_b,
            label_a,
            f"{label_b} (no SP singletons)",
        )
        conf_no_sp_singletons = compute_confidence_percentages(
            sub_no_sp_singletons,
            col_a,
            col_b,
        )

        rows.append({
            "cas_type": cas_type_label,
            "cas_subtype": subtype_label,
            "n_total": stats["n_total"],
            "n_comparable": stats["n_comparable"],
            "coverage_pct": stats.get("coverage_pct"),
            "n_agree": stats["n_agree"],
            "n_disagree": stats["n_disagree"],
            "pct_agree": stats.get("pct_agree"),
            "pct_agree_effective": stats.get("pct_agree_effective"),
            "pct_asymmetric": stats.get("pct_asymmetric"),
            "pct_ccf_confident": conf.get("pct_confident_a"),
            "pct_sp_confident": conf.get("pct_confident_b"),
            "pct_both_confident": conf.get("pct_confident_both"),
            "pct_agree_no_sp_singletons": stats_no_sp_singletons.get("pct_agree"),
            "pct_agree_effective_no_sp_singletons": stats_no_sp_singletons.get("pct_agree_effective"),
            "pct_ccf_confident_no_sp_singletons": conf_no_sp_singletons.get("pct_confident_a"),
            "pct_sp_confident_no_sp_singletons": conf_no_sp_singletons.get("pct_confident_b"),
            "pct_both_confident_no_sp_singletons": conf_no_sp_singletons.get("pct_confident_both"),
        })

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    out = out.sort_values(["pct_agree_effective", "n_total"], ascending=[True, False])
    return out


def compute_macro_average_from_subtypes(subtype_df: pd.DataFrame) -> dict:
    """
    Compute macro-averaged subtype agreement metrics.

    Only subtypes with at least one comparable prediction are included.
    This gives each subtype equal weight instead of letting large subtypes dominate.
    """
    if subtype_df.empty:
        return {
            "macro_avg_pct_agree": None,
            "macro_avg_pct_agree_effective": None,
            "macro_avg_pct_agree_no_sp_singletons": None,
            "macro_avg_pct_agree_effective_no_sp_singletons": None,
            "n_subtypes_macro": 0,
        }

    comparable = subtype_df[subtype_df["n_comparable"] > 0].copy()
    if comparable.empty:
        return {
            "macro_avg_pct_agree": None,
            "macro_avg_pct_agree_effective": None,
            "macro_avg_pct_agree_no_sp_singletons": None,
            "macro_avg_pct_agree_effective_no_sp_singletons": None,
            "n_subtypes_macro": 0,
        }

    def mean_or_none(series: pd.Series):
        series = series.dropna()
        return round(float(series.mean()), 2) if not series.empty else None

    return {
        "macro_avg_pct_agree": mean_or_none(comparable["pct_agree"]),
        "macro_avg_pct_agree_effective": mean_or_none(comparable["pct_agree_effective"]),
        "macro_avg_pct_agree_no_sp_singletons": mean_or_none(comparable.get("pct_agree_no_sp_singletons", pd.Series(dtype=float))),
        "macro_avg_pct_agree_effective_no_sp_singletons": mean_or_none(comparable.get("pct_agree_effective_no_sp_singletons", pd.Series(dtype=float))),
        "n_subtypes_macro": int(len(comparable)),
    }


# ─────────────────────────────────────────────────────────────────────────────
# MAIN COMPARISON LOGIC
# ─────────────────────────────────────────────────────────────────────────────

def run_comparisons(base_dir: Path, out_dir: Path,
                    sp_metadata_path: Path | None = None,
                    cas_annotations_path: Path | None = None):
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── 0. Load SP-style flip map (if available) ──────────────────────────────
    flip_map: dict[str, bool] = {}
    if sp_metadata_path is None:
        # Try the conventional default location
        sp_metadata_path = base_dir / "sp_style_clusters" / "cluster_metadata.json"
    if sp_metadata_path.exists():
        print("\n=== Loading SP-style flip map ===")
        flip_map = load_flip_map(sp_metadata_path)
    else:
        print(f"\n  [INFO] No SP-style metadata found at {sp_metadata_path}. "
              f"SP-style orientation will NOT be corrected for genomic strand.")

    # ── 1. Load CRISPRCasFinder ───────────────────────────────────────────────
    print("\n=== Loading CRISPRCasFinder predictions ===")
    ccf_df = load_ccf_predictions(base_dir)
    ccf_df.to_csv(out_dir / "ccf_predictions_level4.csv", index=False)
    print(f"  Saved to: {out_dir / 'ccf_predictions_level4.csv'}")

    # ── 1b. Load optional Cas-type annotations ────────────────────────────────
    cas_df = pd.DataFrame()
    cas_source = cas_annotations_path or resolve_default_cas_annotations(base_dir)
    if cas_source is not None and cas_source.exists():
        print("\n=== Loading Cas-type annotations ===")
        cas_df = load_cas_annotations(cas_source)
        print(f"  Loaded {len(cas_df)} annotated arrays from {cas_source}")
    else:
        print("\n  [INFO] No Cas annotation table found/provided. "
              "Type-stratified comparison will be skipped.")

    # ── 2. Load CRISPR-evOr for each clustering ───────────────────────────────
    print("\n=== Loading CRISPR-evOr predictions ===")
    evor_dfs: dict[str, pd.DataFrame] = {}
    for tool_name in EVOL_TOOL_DIRS:
        # Pass the flip_map only for the SP-style clustering
        fm = flip_map if tool_name == SP_STYLE_CLUSTERING else None
        df = load_evor_predictions(base_dir, tool_name, flip_map=fm)
        evor_dfs[tool_name] = df
        df.to_csv(out_dir / f"evor_{tool_name}_predictions.csv", index=False)
        print(f"  Saved to: {out_dir / f'evor_{tool_name}_predictions.csv'}")

    summary_rows: list[dict] = []
    detail_frames: dict[str, pd.DataFrame] = {}

    # ── 3. Compare CRISPRCasFinder vs each CRISPR-evOr clustering ────────────
    print("\n=== CRISPRCasFinder vs CRISPR-evOr comparisons ===")

    if ccf_df.empty:
        print("  [SKIP] No CRISPRCasFinder arrays found (Evidence Level 4).")
    else:
        for tool_name, evor_df in evor_dfs.items():
            if evor_df.empty:
                print(f"  [SKIP] {tool_name}: no predictions loaded.")
                continue

            merged = ccf_df[["array_name", "ccf_direction"]].merge(
                evor_df[["array_name", "evor_direction", "group_name"]],
                on="array_name",
                how="outer",
            )
            merged["ccf_direction"]  = merged["ccf_direction"].fillna("Unknown")
            merged["evor_direction"] = merged["evor_direction"].fillna("Unknown")
            merged["agreement"] = merged.apply(
                lambda r: orientations_agree(r["ccf_direction"], r["evor_direction"]),
                axis=1,
            )

            if not cas_df.empty:
                merged = merged.merge(cas_df, on="array_name", how="left")
                merged["cas_type"] = merged["cas_type"].fillna("Unknown")

            key = f"ccf_vs_{tool_name}"
            detail_frames[key] = merged
            merged.to_csv(out_dir / f"{key}_detail.csv", index=False)

            stats = compute_pair_stats(
                merged, "ccf_direction", "evor_direction",
                "CRISPRCasFinder", tool_name,
            )

            confidence_stats = compute_confidence_percentages(
                merged, "ccf_direction", "evor_direction"
            )
            group_stats = compute_group_confidence(evor_df)

            # Alternative score: remove SpacerPlacer singletons entirely before comparison
            merged_no_sp_singletons = merged[merged["group_name"] != "singleton"].copy()
            stats_no_sp_singletons = compute_pair_stats(
                merged_no_sp_singletons,
                "ccf_direction",
                "evor_direction",
                "CRISPRCasFinder",
                f"{tool_name} (no SP singletons)",
            )
            confidence_stats_no_sp_singletons = compute_confidence_percentages(
                merged_no_sp_singletons,
                "ccf_direction",
                "evor_direction",
            )

            stats.update({
                "n_ccf_confident": confidence_stats["n_confident_a"],
                "n_sp_confident": confidence_stats["n_confident_b"],
                "n_both_confident": confidence_stats["n_confident_both"],
                "pct_ccf_confident": confidence_stats["pct_confident_a"],
                "pct_sp_confident": confidence_stats["pct_confident_b"],
                "pct_both_confident": confidence_stats["pct_confident_both"],

                "n_groups_non_singleton": group_stats["n_groups_non_singleton"],
                "n_groups_confident": group_stats["n_groups_confident"],
                "pct_groups_confident": group_stats["pct_groups_confident"],

                "n_total_no_sp_singletons": stats_no_sp_singletons["n_total"],
                "n_comparable_no_sp_singletons": stats_no_sp_singletons["n_comparable"],
                "pct_agree_no_sp_singletons": stats_no_sp_singletons["pct_agree"],
                "pct_agree_effective_no_sp_singletons": stats_no_sp_singletons["pct_agree_effective"],
                "coverage_pct_no_sp_singletons": stats_no_sp_singletons["coverage_pct"],

                "pct_ccf_confident_no_sp_singletons": confidence_stats_no_sp_singletons["pct_confident_a"],
                "pct_sp_confident_no_sp_singletons": confidence_stats_no_sp_singletons["pct_confident_b"],
                "pct_both_confident_no_sp_singletons": confidence_stats_no_sp_singletons["pct_confident_both"],
            })

            summary_rows.append(stats)

            print(f"\n  [{tool_name}]")
            print(f"    Total arrays (union):   {stats['n_total']}")
            print(f"    Comparable predictions: {stats['n_comparable']}")
            print(f"    Skipped (unknown/singleton): {stats['n_skipped']}")
            print(f"    Coverage: {stats['coverage_pct']}%")
            print(
                f"    Confident directions: "
                f"CCF {stats['pct_ccf_confident']}% | "
                f"SpacerPlacer {stats['pct_sp_confident']}% | "
                f"Both {stats['pct_both_confident']}%"
            )
            print(
                f"    Group confidence (no singletons): "
                f"{stats['n_groups_confident']}/{stats['n_groups_non_singleton']} "
                f"({stats['pct_groups_confident']}%)"
            )
            print(
                f"    Confident directions (excluding SP singletons): "
                f"CCF {stats['pct_ccf_confident_no_sp_singletons']}% | "
                f"SpacerPlacer {stats['pct_sp_confident_no_sp_singletons']}% | "
                f"Both {stats['pct_both_confident_no_sp_singletons']}%"
            )

            if stats["n_comparable"]:
                print(f"    Agree:    {stats['n_agree']} ({stats['pct_agree']}%)")
                print(f"    Disagree: {stats['n_disagree']} ({stats['pct_disagree']}%)")
                print(f"    Effective agreement (all arrays): {stats['pct_agree_effective']}%")
                print(
                    f"    Effective agreement (excluding SP singletons): "
                    f"{stats['pct_agree_effective_no_sp_singletons']}%"
                )
                print(f"    Full agreement (incl. Unknown==Unknown): {stats['pct_full_agree']}%")
                print(f"    Asymmetric predictions (one Unknown): {stats['pct_asymmetric']}%")
                print(f"    CCF-only predictions:  {stats['pct_ccf_only']}%")
                print(f"    evOr-only predictions: {stats['pct_evor_only']}%")
                print("    Crosstab:")
                for line in direction_crosstab(merged, "ccf_direction", "evor_direction").to_string().splitlines():
                    print("      " + line)

            # Cas-type stratified summary for this tool comparison
            if "cas_type" in merged.columns:
                by_type = summarise_by_cas_type(
                    merged,
                    "ccf_direction",
                    "evor_direction",
                    "CRISPRCasFinder",
                    tool_name,
                )
                by_type_path = out_dir / f"{key}_by_cas_type.csv"
                by_type.to_csv(by_type_path, index=False)
                print(f"    Cas-type summary saved: {by_type_path}")
                if not by_type.empty:
                    worst = by_type.head(5)[["cas_type", "n_total", "pct_agree_effective", "pct_agree"]]
                    print("    Lowest-performing Cas types (top 5 by effective agreement):")
                    for line in worst.to_string(index=False).splitlines():
                        print("      " + line)

            # Cas-subtype stratified summary for this tool comparison
            subtype_summary = pd.DataFrame()
            if "cas_subtype" in merged.columns:
                by_subtype = summarise_by_cas_subtype(
                    merged,
                    "ccf_direction",
                    "evor_direction",
                    "CRISPRCasFinder",
                    tool_name,
                )
                by_subtype_path = out_dir / f"{key}_by_cas_subtype.csv"
                by_subtype.to_csv(by_subtype_path, index=False)
                print(f"    Cas-subtype summary saved: {by_subtype_path}")
                subtype_summary = compute_macro_average_from_subtypes(by_subtype)

                stats.update({
                    "n_subtypes_macro": subtype_summary["n_subtypes_macro"],
                    "macro_avg_pct_agree": subtype_summary["macro_avg_pct_agree"],
                    "macro_avg_pct_agree_effective": subtype_summary["macro_avg_pct_agree_effective"],
                    "macro_avg_pct_agree_no_sp_singletons": subtype_summary["macro_avg_pct_agree_no_sp_singletons"],
                    "macro_avg_pct_agree_effective_no_sp_singletons": subtype_summary["macro_avg_pct_agree_effective_no_sp_singletons"],
                })

                print(
                    f"    Macro-average over subtypes (n={subtype_summary['n_subtypes_macro']}): "
                    f"agree {subtype_summary['macro_avg_pct_agree']}% | "
                    f"effective {subtype_summary['macro_avg_pct_agree_effective']}% | "
                    f"agree no SP singletons {subtype_summary['macro_avg_pct_agree_no_sp_singletons']}% | "
                    f"effective no SP singletons {subtype_summary['macro_avg_pct_agree_effective_no_sp_singletons']}%"
                )

    # ── 4. Compare CRISPR-evOr clusterings against each other ─────────────────
    print("\n=== CRISPR-evOr inter-clustering stability ===")

    tool_pairs = list(itertools.combinations(EVOL_TOOL_DIRS, 2))
    for name_a, name_b in tool_pairs:
        df_a = evor_dfs.get(name_a, pd.DataFrame())
        df_b = evor_dfs.get(name_b, pd.DataFrame())

        if df_a.empty or df_b.empty:
            print(f"  [SKIP] {name_a} vs {name_b}: one or both datasets empty.")
            continue

        merged = df_a[["array_name", "evor_direction"]].rename(
            columns={"evor_direction": f"dir_{name_a}"}
        ).merge(
            df_b[["array_name", "evor_direction"]].rename(
                columns={"evor_direction": f"dir_{name_b}"}
            ),
            on="array_name",
            how="outer",
        )
        merged[f"dir_{name_a}"] = merged[f"dir_{name_a}"].fillna("Unknown")
        merged[f"dir_{name_b}"] = merged[f"dir_{name_b}"].fillna("Unknown")
        merged["agreement"] = merged.apply(
            lambda r: orientations_agree(r[f"dir_{name_a}"], r[f"dir_{name_b}"]),
            axis=1,
        )

        key = f"evor_{name_a}_vs_{name_b}"
        detail_frames[key] = merged
        merged.to_csv(out_dir / f"{key}_detail.csv", index=False)

        stats = compute_pair_stats(
            merged, f"dir_{name_a}", f"dir_{name_b}", name_a, name_b,
        )
        summary_rows.append(stats)

        print(f"\n  [{name_a}] vs [{name_b}]")
        print(f"    Total arrays (union):   {stats['n_total']}")
        print(f"    Comparable predictions: {stats['n_comparable']}")
        print(f"    Skipped (unknown/singleton): {stats['n_skipped']}")
        print(f"    Coverage: {stats['coverage_pct']}%")
        if stats["n_comparable"]:
            print(f"    Agree:    {stats['n_agree']} ({stats['pct_agree']}%)")
            print(f"    Disagree: {stats['n_disagree']} ({stats['pct_disagree']}%)")
            print(f"    Effective agreement (all arrays): {stats['pct_agree_effective']}%")
            print(f"    Asymmetric predictions (one Unknown): {stats['pct_asymmetric']}%")
            print(f"    CCF-only predictions:  {stats['pct_ccf_only']}%")
            print(f"    evOr-only predictions: {stats['pct_evor_only']}%")
            print("    Crosstab:")
            for line in direction_crosstab(merged, f"dir_{name_a}", f"dir_{name_b}").to_string().splitlines():
                print("      " + line)

    # ── 5. Write summary table ─────────────────────────────────────────────────
    summary_df = pd.DataFrame(summary_rows)
    summary_path = out_dir / "comparison_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"\n=== Summary table saved to {summary_path} ===")
    print("\n=== Summary (overall agreement) ===")
    overall_cols = [
        "label_a", "label_b",
        "n_total",
        "n_comparable",
        "coverage_pct",
        "pct_agree",
        "pct_agree_effective",
        "pct_agree_effective_no_sp_singletons",
        "pct_asymmetric",
    ]
    print(summary_df[overall_cols].to_string(index=False))

    print("\n=== Summary (confidence) ===")
    confidence_cols = [
        "label_a", "label_b",
        "pct_ccf_confident",
        "pct_sp_confident",
        "pct_both_confident",
        "pct_ccf_confident_no_sp_singletons",
        "pct_sp_confident_no_sp_singletons",
        "pct_both_confident_no_sp_singletons",
        "pct_groups_confident",
    ]
    print(summary_df[confidence_cols].to_string(index=False))

    macro_cols = [
        "label_a", "label_b",
        "n_subtypes_macro",
        "macro_avg_pct_agree",
        "macro_avg_pct_agree_effective",
        "macro_avg_pct_agree_no_sp_singletons",
        "macro_avg_pct_agree_effective_no_sp_singletons",
    ]
    macro_df = summary_df[macro_cols].dropna(subset=["n_subtypes_macro"], how="all")
    if not macro_df.empty:
        print("\n=== Summary (macro-average over subtypes) ===")
        print(macro_df.to_string(index=False))

    # ── 6. Write disagreement detail files ────────────────────────────────────
    print("\n=== Writing disagreement-only detail files ===")
    for key, df in detail_frames.items():
        disagree_df = df[df["agreement"] == "disagree"]
        if not disagree_df.empty:
            path = out_dir / f"{key}_disagreements.csv"
            disagree_df.to_csv(path, index=False)
            print(f"  {len(disagree_df)} disagreements → {path}")

    # ── 7. Cross-clustering stability heatmap data ─────────────────────────────
    _write_stability_matrix(evor_dfs, out_dir)

    print(f"\n✓ All outputs written to: {out_dir}")


def _write_stability_matrix(evor_dfs: dict[str, pd.DataFrame], out_dir: Path):
    """
    Write a pairwise % agreement matrix across all CRISPR-evOr clusterings
    as a CSV so it can be easily imported into R / matplotlib / etc.
    """
    tools = [t for t in EVOL_TOOL_DIRS if not evor_dfs.get(t, pd.DataFrame()).empty]
    matrix = pd.DataFrame(index=tools, columns=tools, dtype=float)

    for t in tools:
        matrix.loc[t, t] = 100.0

    for name_a, name_b in itertools.combinations(tools, 2):
        df_a = evor_dfs[name_a][["array_name", "evor_direction"]].rename(
            columns={"evor_direction": "dir_a"})
        df_b = evor_dfs[name_b][["array_name", "evor_direction"]].rename(
            columns={"evor_direction": "dir_b"})
        merged = df_a.merge(df_b, on="array_name", how="inner")
        comparable = merged[
            (~merged["dir_a"].isin(["Unknown", "Singleton"])) &
            (~merged["dir_b"].isin(["Unknown", "Singleton"]))
        ]
        if len(comparable) == 0:
            pct = float("nan")
        else:
            pct = round(100 * (comparable["dir_a"] == comparable["dir_b"]).mean(), 2)
        matrix.loc[name_a, name_b] = pct
        matrix.loc[name_b, name_a] = pct

    matrix.to_csv(out_dir / "evor_stability_matrix.csv")
    print(f"\n  Stability matrix saved → {out_dir / 'evor_stability_matrix.csv'}")
    print("\n  CRISPR-evOr pairwise % agreement (inner join, excludes unknown/singletons):")
    print(matrix.to_string())


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Compare CRISPR array orientation predictions from "
                    "CRISPRCasFinder and CRISPR-evOr."
    )
    parser.add_argument(
        "--base-dir", default=".",
        help="Root directory containing test_out/, out-cluster-size-50/, "
             "cluster-size-50/, etc. (default: current directory)",
    )
    parser.add_argument(
        "--out-dir", default="./crispr_comparison_results",
        help="Directory to write all output files (default: ./crispr_comparison_results/)",
    )
    parser.add_argument(
        "--sp-metadata", default=None,
        help="Path to cluster_metadata.json produced by cluster_like_spacerplacer.py. "
             "Used to correct SP-style evOr predictions back to genomic strand orientation. "
             "Default: <base-dir>/sp_style_clusters/cluster_metadata.json",
    )
    parser.add_argument(
        "--cas-annotations", default=None,
        help="Optional TSV with array_name and cas_type columns (e.g., "
             "output_dataset/curated_dataset.tsv or "
             "output_dataset/intermediate_array_cas_annotations.tsv). "
             "If omitted, compare-results.py auto-detects these defaults under <base-dir>/output_dataset/.",
    )
    args = parser.parse_args()

    base_dir = Path(args.base_dir).resolve()
    out_dir  = Path(args.out_dir).resolve()
    sp_meta  = Path(args.sp_metadata).resolve() if args.sp_metadata else None
    cas_ann  = Path(args.cas_annotations).resolve() if args.cas_annotations else None

    print(f"Base directory : {base_dir}")
    print(f"Output directory: {out_dir}")
    if sp_meta:
        print(f"SP metadata    : {sp_meta}")
    if cas_ann:
        print(f"Cas annotations: {cas_ann}")

    run_comparisons(
        base_dir,
        out_dir,
        sp_metadata_path=sp_meta,
        cas_annotations_path=cas_ann,
    )


if __name__ == "__main__":
    main()