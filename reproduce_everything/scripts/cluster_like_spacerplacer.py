#!/usr/bin/env python3
"""
cluster_like_spacerplacer.py
============================
Replicates SpacerPlacer's internal ccf_json clustering pipeline as a
standalone script, so the resulting clusters can be fed directly into
CRISPR-evOr (or any other tool expecting spacer_fasta format) while
preserving SpacerPlacer's orientation normalisation logic.

Pipeline (mirrors SpacerPlacer ccf_json steps 1-6):
  1. Extract arrays from all Result_*/result.json files
     (Evidence Level >= min_evidence_level)
  2. Group arrays by consensus DR (repeat) sequence
     → arrays of the same CRISPR type end up in the same group
  3. Within each repeat group, orient all arrays to a canonical
     "forward" direction using CRISPRCasFinder's CRISPRDirection:
       - CRISPRDirection == "+"  → keep as-is
       - CRISPRDirection == "-"  → reverse spacer order + reverse-
                                   complement each spacer sequence
       - CRISPRDirection == "ND" / unknown → keep as-is (treated as +)
  4. (Optional) Cluster spacers within a repeat group by Levenshtein
     distance so that near-identical spacers (e.g. 1 SNP apart) are
     treated as the same spacer. Controlled by --cluster-spacers and
     --max-distance.
  5. Cluster arrays by spacer overlap (greedy connected-component).
  6. Enforce a maximum cluster size (recursive splitting, then random
     partitioning as a fallback). Controlled by --max-size.
  6b. (Optional) Rescue singleton clusters by attaching them to an
        existing non-singleton cluster if enough of the singleton's
        spacers are already present in that cluster.
        Controlled by --singleton-rescue and
        --singleton-rescue-min-similarity.
    Optional controls:
      --singleton-rescue-max-size
      --singleton-rescue-allow-singleton-targets
      --singleton-rescue-max-iterations
  7. Write each cluster as a spacer_fasta .fa file.
     Singleton clusters go to <output_dir>/singletons/.
     A metadata JSON is also written for traceability.

Usage:
    python cluster_like_spacerplacer.py \\
        --input-dir  ./test_out \\
        --output-dir ./sp_style_clusters \\
        [--max-size 300] \\
        [--min-evidence 4] \\
        [--cluster-spacers] \\
        [--max-distance 1] \\
        [--singleton-rescue] \\
        [--singleton-rescue-min-similarity 0.5] \\
        [--singleton-rescue-max-size 300] \\
        [--singleton-rescue-allow-singleton-targets] \\
        [--singleton-rescue-max-iterations 10]

Output spacer_fasta format (compatible with SpacerPlacer / CRISPR-evOr):
    >array_id
    0, 1, 2, 3, ...
    (spacer indices are local to each cluster, new spacers on the LEFT)
"""

import os
import re
import json
import glob
import argparse
import time
import sys
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from collections import defaultdict


# ─────────────────────────────────────────────────────────────────────────────
# SEQUENCE UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

_COMPLEMENT = str.maketrans("ACGTacgtNn", "TGCAtgcaNn")


_DIST_WORKER_SEQS: list[str] | None = None
_DIST_WORKER_MAX_DISTANCE: int = 0


def _init_distance_worker(seqs: list[str], max_distance: int):
    global _DIST_WORKER_SEQS, _DIST_WORKER_MAX_DISTANCE
    _DIST_WORKER_SEQS = seqs
    _DIST_WORKER_MAX_DISTANCE = max_distance


def _distance_worker(task: tuple[int, int]) -> tuple[int, list[tuple[int, int]]]:
    """Compute matching spacer pairs for a range of i-indices."""
    start_i, end_i = task
    assert _DIST_WORKER_SEQS is not None
    seqs = _DIST_WORKER_SEQS
    max_distance = _DIST_WORKER_MAX_DISTANCE
    n = len(seqs)
    edges: list[tuple[int, int]] = []

    for i in range(start_i, end_i):
        si = seqs[i]
        for j in range(i + 1, n):
            if levenshtein(si, seqs[j]) <= max_distance:
                edges.append((i, j))

    return start_i, edges


def _format_duration(seconds: float) -> str:
    if seconds < 0:
        seconds = 0
    secs = int(seconds)
    h, rem = divmod(secs, 3600)
    m, s = divmod(rem, 60)
    if h > 0:
        return f"{h:d}h{m:02d}m{s:02d}s"
    if m > 0:
        return f"{m:d}m{s:02d}s"
    return f"{s:d}s"


def _ellipsize(text: str, max_len: int) -> str:
    if max_len <= 0:
        return ""
    if len(text) <= max_len:
        return text
    if max_len <= 3:
        return text[:max_len]
    return text[: max_len - 3] + "..."


def _print_progress_line(prefix: str,
                         current: int,
                         total: int,
                         start_time: float,
                         suffix: str = ""):
    total = max(total, 1)
    current = min(max(current, 0), total)
    pct = 100.0 * current / total
    elapsed = max(time.time() - start_time, 1e-9)
    rate = current / elapsed
    remaining = (total - current) / rate if rate > 0 else 0
    eta = _format_duration(remaining)

    term_width = shutil.get_terminal_size(fallback=(120, 24)).columns
    fixed = len(f" [] {pct:6.2f}% ({current}/{total}) ETA {eta}{suffix}")
    available = max(10, term_width - fixed - 2)
    bar_width = max(10, min(40, available))
    filled = int(bar_width * (current / total))
    bar = "#" * filled + "-" * (bar_width - filled)

    prefix_room = max(0, term_width - len(f" [{bar}] {pct:6.2f}% ({current}/{total}) ETA {eta}{suffix}") - 1)
    safe_prefix = _ellipsize(prefix, prefix_room)

    line = f"{safe_prefix} [{bar}] {pct:6.2f}% ({current}/{total}) ETA {eta}{suffix}"
    sys.stdout.write("\r\033[2K" + line)
    sys.stdout.flush()


def reverse_complement(seq: str) -> str:
    return seq.translate(_COMPLEMENT)[::-1]


def orient_array(spacers: list[str], direction: str) -> tuple[list[str], bool]:
    """
    Return (oriented_spacers, was_flipped).

    Rules (matching SpacerPlacer ccf_json behaviour):
      "+"  or unknown → keep as-is
      "-"             → reverse list AND reverse-complement each spacer
    """
    d = str(direction).strip()
    if d == "-":
        return [reverse_complement(s) for s in reversed(spacers)], True
    return spacers, False


def parse_cas_type(raw_type: str) -> str:
    """Extract coarse Cas type token (e.g. I, II, III, IV, V, VI, VII)."""
    if not raw_type:
        return "Unknown"

    canonical = raw_type
    if "_CAS_" in canonical:
        canonical = canonical.split("_CAS_", 1)[1]

    m = re.search(r"(?:Subtype|Type)[_-]?([IVX]+(?:-[A-Z0-9]+)?)", canonical, re.IGNORECASE)
    if not m:
        return "Unknown"

    subtype_or_type = m.group(1).upper()
    return subtype_or_type.split("-")[0]


def genomic_distance(a_start: int, a_end: int, b_start: int, b_end: int) -> int:
    """Minimum genomic distance between [a_start, a_end] and [b_start, b_end]."""
    if a_end < b_start:
        return b_start - a_end
    if b_end < a_start:
        return a_start - b_end
    return 0


def find_nearest_cas_type(array_start: int, array_end: int, cas_list: list[dict]) -> str:
    """Return the coarse Cas type of the nearest cassette for an array."""
    if not cas_list:
        return "Unknown"

    best_type = "Unknown"
    best_dist = float("inf")
    for cas in cas_list:
        d = genomic_distance(array_start, array_end, cas["Start"], cas["End"])
        if d < best_dist:
            best_dist = d
            best_type = cas["type"]
    return best_type


# ─────────────────────────────────────────────────────────────────────────────
# LEVENSHTEIN (for optional spacer clustering)
# ─────────────────────────────────────────────────────────────────────────────

def levenshtein(a: str, b: str) -> int:
    """Standard DP Levenshtein distance."""
    if a == b:
        return 0
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        curr = [i]
        for j, cb in enumerate(b, 1):
            curr.append(min(prev[j] + 1,
                            curr[j - 1] + 1,
                            prev[j - 1] + (ca != cb)))
        prev = curr
    return prev[-1]


def cluster_spacers_by_distance(spacer_to_idx: dict[str, int],
                                max_distance: int,
                                progress_prefix: str | None = None,
                                workers: int = 1) -> dict[int, int]:
    """
    Cluster spacer sequences by Levenshtein distance.
    Returns a mapping  old_index -> canonical_index  (the lowest index
    in each cluster becomes the representative, mirroring SpacerPlacer).

    Only spacers within the same repeat group are compared (caller passes
    only the relevant slice of spacer_to_idx).
    """
    seqs = list(spacer_to_idx.keys())
    idxs = [spacer_to_idx[s] for s in seqs]
    n = len(seqs)

    # Union-Find
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py

    if workers <= 1 or n < 2:
        start_time = time.time()
        last_update = 0.0

        for i in range(n):
            for j in range(i + 1, n):
                if levenshtein(seqs[i], seqs[j]) <= max_distance:
                    union(i, j)

            if progress_prefix and n > 1:
                now = time.time()
                if (now - last_update) >= 1.0 or i == n - 1:
                    _print_progress_line(
                        prefix=progress_prefix,
                        current=i + 1,
                        total=n,
                        start_time=start_time,
                        suffix=" (rows)",
                    )
                    last_update = now

        if progress_prefix and n > 1:
            sys.stdout.write("\n")
    else:
        # Split the i-axis into chunks so the slow groups can use multiple CPUs.
        chunk_size = max(1, min(64, n // (workers * 4) or 1))
        tasks = [(start, min(start + chunk_size, n))
                 for start in range(0, n, chunk_size)]
        start_time = time.time()

        print(f"{progress_prefix} using {workers} workers over {len(tasks)} chunks")
        done_chunks = 0
        with ProcessPoolExecutor(
            max_workers=workers,
            initializer=_init_distance_worker,
            initargs=(seqs, max_distance),
        ) as pool:
            futures = [pool.submit(_distance_worker, task) for task in tasks]
            for future in as_completed(futures):
                _, edges = future.result()
                for i, j in edges:
                    union(i, j)
                done_chunks += 1
                if progress_prefix:
                    _print_progress_line(
                        prefix=progress_prefix,
                        current=done_chunks,
                        total=len(tasks),
                        start_time=start_time,
                        suffix=" (chunks)",
                    )
        if progress_prefix:
            sys.stdout.write("\n")

    # Build component -> representative (lowest original index)
    comp_to_rep: dict[int, int] = {}
    for i in range(n):
        comp = find(i)
        if comp not in comp_to_rep or idxs[i] < comp_to_rep[comp]:
            comp_to_rep[comp] = idxs[i]

    return {idxs[i]: comp_to_rep[find(i)] for i in range(n)}


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — PARSE CCF JSON
# ─────────────────────────────────────────────────────────────────────────────

def parse_ccf_json(path: Path, min_evidence: int) -> list[dict]:
    """
        Parse one result.json and return a list of array dicts:
            array_id, spacers (DNA strings), repeat, direction, was_flipped, cas_type
    Only arrays with Evidence_Level >= min_evidence are kept.
    """
    records = []
    try:
        with open(path) as fh:
            data = json.load(fh)
    except Exception as e:
        print(f"  [WARN] Cannot parse {path}: {e}")
        return records

    for seq in data.get("Sequences", []):
        cas_list = []
        for cas in seq.get("Cas", []):
            cas_type = parse_cas_type(cas.get("Type", ""))
            cas_list.append({
                "type": cas_type,
                "Start": cas.get("Start", 0),
                "End": cas.get("End", 0),
            })

        for crispr in seq.get("Crisprs", []):
            if crispr.get("Evidence_Level", 0) < min_evidence:
                continue
            array_id = crispr.get("Name", "")
            if not array_id:
                continue
            spacers = [r["Sequence"] for r in crispr.get("Regions", [])
                       if r.get("Type") == "Spacer" and r.get("Sequence")]
            repeat = crispr.get("DR_Consensus", "")
            direction = crispr.get("CRISPRDirection", "ND")
            if not spacers or not repeat:
                continue

            array_start = crispr.get("Start", 0)
            array_end = crispr.get("End", 0)
            cas_type = find_nearest_cas_type(array_start, array_end, cas_list)

            # ── Step 3: orient using CRISPRDirection ──────────────────────
            oriented_spacers, was_flipped = orient_array(spacers, direction)

            records.append({
                "array_id":    array_id,
                "spacers":     oriented_spacers,
                "repeat":      repeat,
                "cas_type":    cas_type,
                "direction":   direction,
                "was_flipped": was_flipped,
                "source":      str(path),
            })
    return records


def load_all_arrays(input_dir: Path, min_evidence: int) -> list[dict]:
    """Walk input_dir for Result_*/result.json and extract all arrays."""
    pattern = str(input_dir / "Result_*" / "result.json")
    paths = sorted(glob.glob(pattern))
    if not paths:
        # Also try input_dir directly (single run)
        direct = input_dir / "result.json"
        if direct.exists():
            paths = [str(direct)]
    if not paths:
        raise FileNotFoundError(
            f"No result.json files found under {input_dir}")

    all_records = []
    seen = set()
    for p in paths:
        recs = parse_ccf_json(Path(p), min_evidence)
        for r in recs:
            if r["array_id"] not in seen:
                seen.add(r["array_id"])
                all_records.append(r)
        print(f"  {Path(p).parent.name}: {len(recs)} arrays "
              f"(level >= {min_evidence})")

    print(f"  Total: {len(all_records)} arrays "
          f"({sum(r['was_flipped'] for r in all_records)} flipped to + strand)")
    return all_records


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — GROUP BY REPEAT
# ─────────────────────────────────────────────────────────────────────────────

def group_by_repeat(records: list[dict], group_by_cas_type: bool = False) -> dict[str, list[dict]]:
    """
    Group arrays by consensus DR sequence.
    Optionally split each repeat group by nearest Cas type.
    """
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        key = r["repeat"]
        if group_by_cas_type:
            key = f"{r['repeat']}|CAS={r.get('cas_type', 'Unknown')}"
        groups[key].append(r)
    label = "repeat+Cas-type" if group_by_cas_type else "repeat"
    print(f"\n  {len(groups)} {label} groups "
          f"(sizes: min={min(len(v) for v in groups.values())}, "
          f"max={max(len(v) for v in groups.values())})")
    return dict(groups)


def preview_heaviest_groups(repeat_groups: dict[str, list[dict]],
                            top_n: int = 15):
    """
    Print a workload preview of the heaviest repeat groups before clustering.
    Ranked by estimated spacer comparison pairs (n*(n-1)/2), then arrays.
    """
    metrics = []
    for repeat, group_records in repeat_groups.items():
        n_arrays = len(group_records)
        unique_spacers = {
            s for rec in group_records for s in rec["spacers"]
        }
        n_unique_spacers = len(unique_spacers)
        est_pairs = (n_unique_spacers * (n_unique_spacers - 1)) // 2
        metrics.append({
            "repeat": repeat,
            "n_arrays": n_arrays,
            "n_unique_spacers": n_unique_spacers,
            "est_pairs": est_pairs,
        })

    metrics.sort(
        key=lambda x: (x["est_pairs"], x["n_arrays"], x["n_unique_spacers"]),
        reverse=True,
    )

    print(f"\n  Heaviest repeat groups by estimated spacer comparisons "
          f"(top {min(top_n, len(metrics))}):")
    for i, m in enumerate(metrics[:top_n], start=1):
        print(
            f"    {i:>2}. arrays={m['n_arrays']}, "
            f"unique_spacers={m['n_unique_spacers']}, "
            f"est_spacer_pairs={m['est_pairs']}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — SPACER INDEX + OPTIONAL LEVENSHTEIN CLUSTERING
# ─────────────────────────────────────────────────────────────────────────────

def build_spacer_index(records: list[dict],
                       do_cluster: bool,
                       max_distance: int,
                       progress_prefix: str | None = None,
                       workers: int = 1) -> dict[str, list[int]]:
    """
    Assign integer IDs to spacer sequences (within a repeat group).
    If do_cluster=True, near-identical spacers (≤ max_distance edits)
    get the same ID (lowest-index wins, mirroring SpacerPlacer).

    Returns: array_id -> list of spacer integer IDs
    """
    spacer_to_idx: dict[str, int] = {}
    counter = 0

    for r in records:
        for s in r["spacers"]:
            if s not in spacer_to_idx:
                spacer_to_idx[s] = counter
                counter += 1

    if do_cluster and max_distance > 0:
        remap = cluster_spacers_by_distance(
            spacer_to_idx,
            max_distance,
            progress_prefix=progress_prefix,
            workers=workers,
        )
    else:
        remap = {v: v for v in spacer_to_idx.values()}

    result = {}
    for r in records:
        result[r["array_id"]] = [remap[spacer_to_idx[s]] for s in r["spacers"]]
    return result


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — CLUSTER BY SPACER OVERLAP (greedy connected-component)
# ─────────────────────────────────────────────────────────────────────────────

def cluster_by_overlap(array_ids: list[str],
                       spacer_map: dict[str, list[int]]) -> list[list[str]]:
    """Greedy connected-component clustering on shared spacer IDs."""
    clusters: list[list[str]] = []
    for aid in array_ids:
        s = set(spacer_map[aid])
        placed = False
        for cluster in clusters:
            if any(s & set(spacer_map[eid]) for eid in cluster):
                cluster.append(aid)
                placed = True
                break
        if not placed:
            clusters.append([aid])
    return clusters


def rescue_singletons(clusters: list[list[str]],
                      spacer_map: dict[str, list[int]],
                      min_similarity: float,
                      max_size: int | None = None,
                      rescue_max_size: int | None = None,
                      allow_singleton_targets: bool = False,
                      max_iterations: int = 10) -> tuple[list[list[str]], int, int]:
    """
    Attach singleton clusters to the most similar eligible target cluster.

    Similarity is computed as:
      overlap(singleton, cluster_union) / len(singleton)

    This keeps the criterion local to the singleton and avoids penalizing
    large target clusters where Jaccard would become very small.
    """
    if not clusters:
        return clusters, 0, 0

    effective_max_size = rescue_max_size if rescue_max_size is not None else max_size

    updated_clusters = [list(c) for c in clusters]
    rescued = 0
    iteration = 0

    while True:
        iteration += 1
        changed = False

        cluster_spacer_unions: dict[int, set[int]] = {}
        for i, cluster in enumerate(updated_clusters):
            if not cluster:
                continue
            union_set: set[int] = set()
            for aid in cluster:
                union_set.update(spacer_map.get(aid, []))
            cluster_spacer_unions[i] = union_set

        for src_idx, cluster in enumerate(updated_clusters):
            if len(cluster) != 1:
                continue

            aid = cluster[0]
            singleton_spacers = set(spacer_map.get(aid, []))
            if not singleton_spacers:
                continue

            best_target = None
            best_similarity = -1.0

            for tgt_idx, tgt_cluster in enumerate(updated_clusters):
                if src_idx == tgt_idx or not tgt_cluster:
                    continue
                if not allow_singleton_targets and len(tgt_cluster) <= 1:
                    continue
                if effective_max_size is not None and len(tgt_cluster) >= effective_max_size:
                    continue

                overlap = len(singleton_spacers & cluster_spacer_unions[tgt_idx])
                similarity = overlap / len(singleton_spacers)

                if similarity > best_similarity:
                    best_similarity = similarity
                    best_target = tgt_idx

            if best_target is not None and best_similarity >= min_similarity:
                updated_clusters[best_target].append(aid)
                cluster_spacer_unions[best_target].update(singleton_spacers)
                updated_clusters[src_idx] = []
                rescued += 1
                changed = True

        updated_clusters = [c for c in updated_clusters if c]

        if not allow_singleton_targets:
            break
        if not changed:
            break
        if iteration >= max_iterations:
            break

    return updated_clusters, rescued, iteration


# ─────────────────────────────────────────────────────────────────────────────
# STEP 6 — ENFORCE MAX CLUSTER SIZE
# ─────────────────────────────────────────────────────────────────────────────

def enforce_max_size(clusters: list[list[str]],
                     spacer_map: dict[str, list[int]],
                     max_size: int,
                     depth: int = 0) -> list[list[str]]:
    """
    Recursively split clusters exceeding max_size.
    Falls back to sequential partitioning when greedy splitting stalls.
    """
    MAX_DEPTH = 5
    result = []
    for cluster in clusters:
        if len(cluster) <= max_size:
            result.append(cluster)
            continue
        if depth < MAX_DEPTH:
            sub = cluster_by_overlap(cluster, spacer_map)
            if max(len(s) for s in sub) < len(cluster):
                result.extend(enforce_max_size(sub, spacer_map,
                                               max_size, depth + 1))
                continue
        # Force partition
        n_parts = (len(cluster) // max_size) + 1
        size = len(cluster) // n_parts
        for i in range(n_parts):
            part = cluster[i * size: (i + 1) * size if i < n_parts - 1
                          else len(cluster)]
            if part:
                result.append(part)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# STEP 7 — ASSIGN LOCAL IDS AND WRITE spacer_fasta
# ─────────────────────────────────────────────────────────────────────────────

def assign_local_ids(cluster: list[str],
                     spacer_map: dict[str, list[int]]) -> dict[str, list[int]]:
    """Re-number spacer IDs to be contiguous starting from 0."""
    global_to_local: dict[int, int] = {}
    counter = 0
    result = {}
    for aid in cluster:
        local = []
        for gid in spacer_map[aid]:
            if gid not in global_to_local:
                global_to_local[gid] = counter
                counter += 1
            local.append(global_to_local[gid])
        result[aid] = local
    return result


def write_clusters(all_clusters: list[list[str]],
                   spacer_map: dict[str, list[int]],
                   flip_map: dict[str, bool],
                   cas_type_map: dict[str, str],
                   output_dir: Path):
    """
    Write each cluster as a spacer_fasta .fa file.
    Singletons go to <output_dir>/singletons/.
    Also writes a metadata JSON for traceability.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    singletons_dir = output_dir / "singletons"

    n_multi = 0
    n_single = 0
    metadata = {}

    for idx, cluster in enumerate(all_clusters):
        cluster_id = f"g_{idx}"
        local_ids = assign_local_ids(cluster, spacer_map)

        if len(cluster) == 1:
            singletons_dir.mkdir(exist_ok=True)
            fa_path = singletons_dir / f"{cluster_id}.fa"
            n_single += 1
        else:
            fa_path = output_dir / f"{cluster_id}.fa"
            n_multi += 1

        with open(fa_path, "w") as fh:
            for aid in cluster:
                ids_str = ", ".join(map(str, local_ids[aid]))
                fh.write(f">{aid}\n{ids_str}\n")

        metadata[cluster_id] = {
            "size": len(cluster),
            "arrays": {
                aid: {
                    "was_flipped": flip_map.get(aid, False),
                    "cas_type": cas_type_map.get(aid, "Unknown"),
                }
                for aid in cluster
            },
        }

    meta_path = output_dir / "cluster_metadata.json"
    with open(meta_path, "w") as fh:
        json.dump(metadata, fh, indent=2)

    print(f"\n  Multi-array clusters : {n_multi}")
    print(f"  Singleton clusters   : {n_single}")
    print(f"  Metadata written to  : {meta_path}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def run(input_dir: Path, output_dir: Path,
        min_evidence: int, max_size: int | None,
    do_cluster_spacers: bool, max_distance: int,
    group_by_cas_type: bool,
    singleton_rescue: bool,
    singleton_rescue_min_similarity: float,
    singleton_rescue_max_size: int | None,
    singleton_rescue_allow_singleton_targets: bool,
    singleton_rescue_max_iterations: int,
    workers: int):

    print("=" * 65)
    print("SpacerPlacer-style ccf_json Clustering")
    print("=" * 65)
    print(f"Using {workers} worker(s) for spacer-distance clustering")
    if singleton_rescue:
        print("Singleton rescue enabled: "
              f"min similarity = {singleton_rescue_min_similarity:.3f}")
        if singleton_rescue_max_size is not None:
            print(f"Singleton rescue max size override: {singleton_rescue_max_size}")
        if singleton_rescue_allow_singleton_targets:
            print("Singleton rescue will allow singleton-to-singleton merging")

    # ── Step 1: Extract & orient ──────────────────────────────────────────
    print(f"\n[1] Extracting arrays from {input_dir} ...")
    records = load_all_arrays(input_dir, min_evidence)
    if not records:
        print("No arrays found. Exiting.")
        return

    # Build a quick lookup for later
    flip_map = {r["array_id"]: r["was_flipped"] for r in records}
    cas_type_map = {r["array_id"]: r.get("cas_type", "Unknown") for r in records}

    # ── Step 2: Group by repeat ───────────────────────────────────────────
    print("\n[2] Grouping by consensus repeat ...")
    if group_by_cas_type:
        print("    Cas-type split is enabled (repeat + nearest Cas type).")
    repeat_groups = group_by_repeat(records, group_by_cas_type=group_by_cas_type)
    preview_heaviest_groups(repeat_groups)

    # ── Steps 3-6: Per repeat group ───────────────────────────────────────
    print("\n[3-6] Clustering within each repeat group ...")
    all_clusters: list[list[str]] = []
    global_spacer_map: dict[str, list[int]] = {}
    repeat_items = list(repeat_groups.items())
    repeat_start = time.time()
    total_rescued_singletons = 0
    total_rescue_iterations = 0

    for idx, (repeat, group_records) in enumerate(repeat_items, start=1):
        n_arrays = len(group_records)
        unique_spacers = {
            s for rec in group_records for s in rec["spacers"]
        }
        n_unique_spacers = len(unique_spacers)
        est_pairs = (n_unique_spacers * (n_unique_spacers - 1)) // 2

        print(
            f"\n  Group {idx}/{len(repeat_items)}: arrays={n_arrays}, "
            f"unique_spacers={n_unique_spacers}, "
            f"est_spacer_pairs={est_pairs}"
        )

        if do_cluster_spacers:
            spacer_progress_prefix = (
                f"    Spacer clustering group {idx}/{len(repeat_items)}"
            )
        else:
            spacer_progress_prefix = None

        # Step 4: build spacer index (+ optional Levenshtein clustering)
        spacer_map = build_spacer_index(
            group_records,
            do_cluster_spacers,
            max_distance,
            progress_prefix=spacer_progress_prefix,
            workers=workers,
        )
        global_spacer_map.update(spacer_map)

        # Step 5: cluster by overlap
        array_ids = [r["array_id"] for r in group_records]
        clusters = cluster_by_overlap(array_ids, spacer_map)

        # Step 6: enforce max size
        if max_size is not None:
            clusters = enforce_max_size(clusters, spacer_map, max_size)

        # Optional singleton rescue pass
        if singleton_rescue:
            clusters, rescued_here, rescue_iterations = rescue_singletons(
                clusters,
                spacer_map,
                min_similarity=singleton_rescue_min_similarity,
                max_size=max_size,
                rescue_max_size=singleton_rescue_max_size,
                allow_singleton_targets=singleton_rescue_allow_singleton_targets,
                max_iterations=singleton_rescue_max_iterations,
            )
            total_rescued_singletons += rescued_here
            total_rescue_iterations += rescue_iterations
            if rescued_here:
                print("    Rescued singletons in this group: "
                      f"{rescued_here} (iterations: {rescue_iterations})")

        all_clusters.extend(clusters)

        _print_progress_line(
            prefix="  Repeat groups",
            current=idx,
            total=len(repeat_items),
            start_time=repeat_start,
        )

    if repeat_items:
        sys.stdout.write("\n")

    print(f"\n  Total clusters (before singleton separation): "
          f"{len(all_clusters)}")
    if singleton_rescue:
        print(f"  Total rescued singletons: {total_rescued_singletons}")
        print(f"  Total rescue iterations: {total_rescue_iterations}")
    sizes = [len(c) for c in all_clusters]
    print(f"  Sizes — min: {min(sizes)}, max: {max(sizes)}, "
          f"mean: {sum(sizes)/len(sizes):.1f}, "
          f"median: {sorted(sizes)[len(sizes)//2]}")

    # ── Step 7: Write output ──────────────────────────────────────────────
    print(f"\n[7] Writing spacer_fasta output to {output_dir} ...")
    write_clusters(all_clusters, global_spacer_map, flip_map, cas_type_map, output_dir)

    # ── Summary of orientation normalisation ──────────────────────────────
    n_flipped = sum(flip_map.values())
    n_nd = sum(1 for r in records
               if str(r["direction"]).strip() not in ("+", "-"))
    print(f"\n  Orientation summary:")
    print(f"    Total arrays     : {len(records)}")
    print(f"    Flipped (was −)  : {n_flipped}")
    print(f"    Kept as-is (+)   : {len(records) - n_flipped - n_nd}")
    print(f"    Unknown (ND/etc) : {n_nd}  ← treated as +, not flipped")
    print(f"\n  NOTE: All arrays in the output are in the canonical")
    print(f"  'forward' orientation (new spacers on the LEFT).")
    print(f"  The CRISPR-evOr 'Forward' prediction for a cluster means")
    print(f"  the orientation SpacerPlacer normalised to — NOT the")
    print(f"  genomic strand. Use cluster_metadata.json to see which")
    print(f"  arrays were flipped during this step.")

    print("\n" + "=" * 65)
    print("Done.")
    print("=" * 65)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Replicate SpacerPlacer's ccf_json clustering pipeline "
                    "as a standalone script.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input-dir", required=True,
        help="Directory containing Result_*/result.json files "
             "(CRISPRCasFinder output).",
    )
    parser.add_argument(
        "--output-dir", required=True,
        help="Directory to write spacer_fasta cluster files.",
    )
    parser.add_argument(
        "--min-evidence", type=int, default=4,
        help="Minimum CCF evidence level to include an array.",
    )
    parser.add_argument(
        "--max-size", type=int, default=None,
        help="Maximum arrays per cluster. None = no limit.",
    )
    parser.add_argument(
        "--cluster-spacers", action="store_true",
        help="Cluster near-identical spacers by Levenshtein distance "
             "(mirrors SpacerPlacer --cluster_spacers option).",
    )
    parser.add_argument(
        "--max-distance", type=int, default=1,
        help="Maximum Levenshtein distance for spacer clustering "
             "(only used if --cluster-spacers is set).",
    )
    parser.add_argument(
        "--workers", type=int, default=max(1, (os.cpu_count() or 1) // 2),
        help="Number of worker processes for spacer-distance clustering.",
    )
    parser.add_argument(
        "--group-by-cas-type", action="store_true",
        help="Split repeat groups by nearest Cas type before overlap clustering.",
    )
    parser.add_argument(
        "--singleton-rescue", action="store_true",
        help="Attach singleton clusters to the most similar non-singleton cluster "
             "when the minimum similarity threshold is met.",
    )
    parser.add_argument(
        "--singleton-rescue-min-similarity", type=float, default=0.5,
        help="Minimum singleton-to-cluster similarity for singleton rescue "
             "(range: 0.0 to 1.0).",
    )
    parser.add_argument(
        "--singleton-rescue-max-size", type=int, default=None,
        help="Optional max cluster size cap used only during singleton rescue. "
             "Defaults to --max-size if not provided.",
    )
    parser.add_argument(
        "--singleton-rescue-allow-singleton-targets", action="store_true",
        help="Allow singleton-to-singleton and iterative singleton-chain merges "
             "during rescue.",
    )
    parser.add_argument(
        "--singleton-rescue-max-iterations", type=int, default=10,
        help="Maximum rescue iterations when singleton targets are allowed.",
    )
    args = parser.parse_args()

    if not 0.0 <= args.singleton_rescue_min_similarity <= 1.0:
        parser.error("--singleton-rescue-min-similarity must be between 0.0 and 1.0")
    if args.singleton_rescue_max_size is not None and args.singleton_rescue_max_size < 1:
        parser.error("--singleton-rescue-max-size must be at least 1")
    if args.singleton_rescue_max_iterations < 1:
        parser.error("--singleton-rescue-max-iterations must be at least 1")

    workers = max(1, args.workers)

    run(
        input_dir=Path(args.input_dir).resolve(),
        output_dir=Path(args.output_dir).resolve(),
        min_evidence=args.min_evidence,
        max_size=args.max_size,
        do_cluster_spacers=args.cluster_spacers,
        max_distance=args.max_distance,
        group_by_cas_type=args.group_by_cas_type,
        singleton_rescue=args.singleton_rescue,
        singleton_rescue_min_similarity=args.singleton_rescue_min_similarity,
        singleton_rescue_max_size=args.singleton_rescue_max_size,
        singleton_rescue_allow_singleton_targets=args.singleton_rescue_allow_singleton_targets,
        singleton_rescue_max_iterations=args.singleton_rescue_max_iterations,
        workers=workers,
    )


if __name__ == "__main__":
    main()