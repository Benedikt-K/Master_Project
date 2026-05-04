#!/usr/bin/env python3
"""
split_crispr_clusters.py

Splits oversized CRISPR cluster .fa files (g_0.fa, g_1.fa, ...) into
biologically coherent sub-clusters based on shared spacer/repeat type
similarity (Jaccard index), keeping the most similar sequences together.

Run this script from the directory containing your g_*.fa files.
Files with <= MAX_SIZE sequences are left untouched.
Files with >  MAX_SIZE sequences are replaced with g_N_0.fa, g_N_1.fa, ...

Usage:
    python split_crispr_clusters.py
    python split_crispr_clusters.py --max-size 300 --min-similarity 0.0
"""

import os
import re
import sys
import argparse
from collections import defaultdict


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
MAX_SIZE = 200          # maximum sequences per output file
MIN_SIM  = 0.0         # Jaccard threshold: only edges with Jaccard >= this AND > 0 are kept.
                        # 0.0 means "connect if any overlap at all" (most conservative/broadest grouping).
                        # Raise to e.g. 0.1 to require stronger similarity before grouping together.


# ---------------------------------------------------------------------------
# FASTA parsing
# ---------------------------------------------------------------------------
def parse_fa(path):
    """Return list of (header_str, type_ids_list) tuples."""
    records = []
    header = None
    with open(path) as fh:
        for line in fh:
            line = line.rstrip("\n")
            if line.startswith(">"):
                header = line[1:].strip()
            elif header is not None and line.strip():
                ids = [int(x.strip()) for x in line.split(",") if x.strip().lstrip('-').isdigit()]
                records.append((header, ids))
                header = None
    return records


def write_fa(path, records):
    with open(path, "w") as fh:
        for header, ids in records:
            fh.write(f">{header}\n")
            fh.write(", ".join(str(i) for i in ids) + "\n")


# ---------------------------------------------------------------------------
# Similarity
# ---------------------------------------------------------------------------
def jaccard(a_ids, b_ids):
    sa, sb = set(a_ids), set(b_ids)
    inter = len(sa & sb)
    if inter == 0:
        return 0.0
    return inter / len(sa | sb)


# ---------------------------------------------------------------------------
# Graph-based greedy community splitting
#
# Strategy:
#   1. Build a similarity graph (nodes = sequences, edges = Jaccard >= threshold).
#   2. Find connected components — sequences with ANY overlap stay in the
#      same component if threshold=0 (biologically: they share at least one
#      spacer/repeat type).
#   3. If a component is still > MAX_SIZE, split it further using a
#      greedy seed-expansion approach:
#        - Pick the node with the highest degree as seed.
#        - Greedily add its neighbours sorted by similarity (descending)
#          until the sub-cluster is full.
#        - Repeat with remaining nodes until all are assigned.
# ---------------------------------------------------------------------------

def connected_components(n, adj):
    """Union-Find connected components. adj: dict node -> set of nodes."""
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

    for u, neighbours in adj.items():
        for v in neighbours:
            union(u, v)

    comps = defaultdict(list)
    for i in range(n):
        comps[find(i)].append(i)
    return list(comps.values())


def greedy_seed_split(indices, sim_matrix, max_size):
    """
    Split a list of node indices into groups of <= max_size using
    greedy seed expansion on the similarity matrix (list of dicts).

    Bug fixes vs original:
      - Degree is recomputed against *remaining* nodes each round so stale
        counts from already-placed nodes don't bias seed selection.
      - Ties are broken by node index for reproducibility (sets are
        non-deterministic across runs).
    """
    remaining = set(indices)
    groups = []

    while remaining:
        if len(remaining) <= max_size:
            # Sort for deterministic output order
            groups.append(sorted(remaining))
            break

        # Recompute degree against current remaining nodes only (fix: was stale)
        degree = {i: sum(1 for j in remaining if i != j and sim_matrix[i].get(j, 0) > 0)
                  for i in remaining}

        # Seed: highest degree; break ties by node index for reproducibility
        seed = max(remaining, key=lambda x: (degree[x], -x))
        group = [seed]
        remaining.remove(seed)

        # Rank remaining by similarity to seed (fix: sort list, not set, for
        # determinism); secondary sort by node index to break ties
        candidates = sorted(remaining,
                            key=lambda x: (sim_matrix[seed].get(x, 0), -x),
                            reverse=True)
        for cand in candidates:
            if len(group) >= max_size:
                break
            group.append(cand)
            remaining.remove(cand)

        groups.append(group)

    return groups


def split_records(records, max_size, min_sim):
    """
    Given a list of records, return a list of groups (each group is a
    list of record indices), all of size <= max_size.
    """
    n = len(records)

    # Build pairwise Jaccard similarity (upper triangle) and adjacency
    print(f"    Computing {n*(n-1)//2} pairwise similarities …", flush=True)
    sim_matrix = [{} for _ in range(n)]  # list of dicts to save memory
    adj = defaultdict(set)

    for i in range(n):
        for j in range(i + 1, n):
            s = jaccard(records[i][1], records[j][1])
            if s > 0 and s >= min_sim:
                sim_matrix[i][j] = s
                sim_matrix[j][i] = s
                adj[i].add(j)
                adj[j].add(i)

    # Connected components
    components = connected_components(n, adj)

    # Split any component that is still too large
    final_groups = []
    for comp in components:
        if len(comp) <= max_size:
            final_groups.append(comp)
        else:
            print(f"    Component of size {len(comp)} > {max_size}, "
                  f"applying greedy seed split …", flush=True)
            # Build local sim_matrix slice (reuse global dict)
            sub_groups = greedy_seed_split(comp, sim_matrix, max_size)
            final_groups.extend(sub_groups)

    # Isolates (no edges, not yet in any component) are already handled
    # by connected_components (they form size-1 components).

    return final_groups


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Split oversized CRISPR cluster .fa files biologically.")
    parser.add_argument("--max-size", type=int, default=MAX_SIZE,
                        help=f"Max sequences per file (default: {MAX_SIZE})")
    parser.add_argument("--min-similarity", type=float, default=MIN_SIM,
                        help=f"Min Jaccard to form an edge (default: {MIN_SIM})")
    parser.add_argument("--dry-run", action="store_true",
                        help="Report what would be done without writing files")
    args = parser.parse_args()

    fa_files = sorted(
        f for f in os.listdir(".")
        if re.match(r"g_\d+\.fa$", f)
    )

    if not fa_files:
        print("No g_*.fa files found in the current directory. Exiting.")
        sys.exit(1)

    print(f"Found {len(fa_files)} cluster file(s). Max size = {args.max_size}.\n")

    for fa_file in fa_files:
        m = re.match(r"g_(\d+)\.fa$", fa_file)
        group_id = int(m.group(1))
        records = parse_fa(fa_file)
        n = len(records)

        if n <= args.max_size:
            print(f"  {fa_file}: {n} sequences — OK, no split needed.")
            continue

        print(f"  {fa_file}: {n} sequences — SPLITTING …")

        groups = split_records(records, args.max_size, args.min_similarity)
        print(f"    → {len(groups)} sub-clusters: "
              + ", ".join(str(len(g)) for g in groups))

        if args.dry_run:
            print("    [dry-run] No files written.")
            continue

        # Write sub-cluster files
        for sub_idx, group in enumerate(groups):
            out_name = f"g_{group_id}_{sub_idx}.fa"
            sub_records = [records[i] for i in group]
            write_fa(out_name, sub_records)
            print(f"    Written: {out_name}  ({len(sub_records)} sequences)")

        # Remove original oversized file
        os.remove(fa_file)
        print(f"    Removed original: {fa_file}")

    print("\nDone.")


if __name__ == "__main__":
    main()