#!/usr/bin/env python3
"""
Cluster CRISPR arrays based on spacer overlap, similar to SpacerPlacer.
Reads extracted arrays from extract-arrays.py and clusters them by:
1. Grouping arrays with the same consensus repeat
2. Clustering arrays within each repeat group by spacer overlap
3. Enforcing a maximum cluster size (with optional recursive splitting)

Output: Groups in spacer_fasta format (>array_id\nspacer_idx_1, spacer_idx_2, ...\n)
"""

import json
import os
from pathlib import Path
from collections import defaultdict


# ----------------------------
# PARAMETERS (tuneable)
# ----------------------------
MAX_CLUSTER_SIZE = 300  # Maximum arrays per cluster (set to None to disable)
INPUT_ARRAYS_FILE = "crisprcasfinder_arrays.txt"
INPUT_METADATA_FILE = "crisprcasfinder_arrays_metadata.json"
OUTPUT_DIR = "ccf_json_clustered_groups"


# ----------------------------
# 1. Parse extracted arrays
# ----------------------------
def parse_extracted_arrays(arrays_file):
    """
    Parse arrays extracted by extract-arrays.py.
    Returns dict: array_id -> spacer_indices (list of ints)
    """
    arrays = {}
    
    try:
        with open(arrays_file) as f:
            current_id = None
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if line.startswith('>'):
                    current_id = line[1:].strip()
                else:
                    # Parse spacer indices
                    spacer_indices = [int(x.strip()) for x in line.split(',')]
                    arrays[current_id] = spacer_indices
    except Exception as e:
        print(f"Error reading arrays file {arrays_file}: {e}")
        return {}
    
    return arrays


def parse_metadata(metadata_file):
    """
    Parse metadata (repeats and directions) from extract-arrays.py.
    Returns dict: array_id -> {repeat, direction}
    """
    if not os.path.exists(metadata_file):
        print(f"Warning: Metadata file {metadata_file} not found")
        return {}

    try:
        with open(metadata_file) as f:
            return json.load(f)
    except Exception as e:
        print(f"Error reading metadata file {metadata_file}: {e}")
        return {}


# ----------------------------
# 2. Group arrays by consensus repeat
# ----------------------------
def group_by_repeat(arrays, metadata):
    """
    Group arrays by their consensus repeat sequence.
    Returns dict: repeat_seq -> list of array_ids
    """
    repeat_groups = defaultdict(list)
    
    for array_id, spacer_indices in arrays.items():
        repeat_seq = metadata.get(array_id, {}).get("repeat", "UNKNOWN")
        repeat_groups[repeat_seq].append(array_id)
    
    # Log grouping results
    print(f"\nGrouped {len(arrays)} arrays by consensus repeat:")
    for repeat_seq, array_ids in sorted(repeat_groups.items(), key=lambda x: -len(x[1])):
        print(f"  Repeat {repeat_seq[:20]}...: {len(array_ids)} arrays")
    
    return repeat_groups


# ----------------------------
# 3. Cluster arrays by spacer overlap
# ----------------------------
def cluster_by_spacer_overlap(array_ids, arrays):
    """
    Cluster arrays within a repeat group by spacer overlap (greedy algorithm).
    Similar to SpacerPlacer's _cluster_groups_by_spacer_overlap.
    
    Returns list of clusters, where each cluster is a list of array_ids.
    """
    clusters = []
    
    for array_id in array_ids:
        spacers = set(arrays[array_id])
        found_cluster = False
        
        # Try to add to existing cluster
        for cluster in clusters:
            # Check if this array has spacer overlap with any array in the cluster
            for existing_id in cluster:
                existing_spacers = set(arrays[existing_id])
                if len(spacers & existing_spacers) > 0:
                    cluster.append(array_id)
                    found_cluster = True
                    break
            
            if found_cluster:
                break
        
        # Create new cluster if no overlap found
        if not found_cluster:
            clusters.append([array_id])
    
    return clusters


# ----------------------------
# 4. Enforce maximum cluster size
# ----------------------------
def enforce_max_size(clusters, arrays, max_size=None, depth=0):
    """
    Split clusters that exceed max_size.
    Uses greedy overlap clustering, but falls back to random partitioning
    if stuck in infinite recursion (highly interconnected clusters).
    
    If max_size is None, no splitting is performed.
    """
    if max_size is None:
        return clusters
    
    final_clusters = []
    MAX_RECURSION_DEPTH = 5  # Prevent infinite recursion
    
    for cluster in clusters:
        if len(cluster) <= max_size:
            final_clusters.append(cluster)
            continue
        
        # Try greedy overlap clustering first
        if depth < MAX_RECURSION_DEPTH:
            print(f"  Splitting large cluster with {len(cluster)} arrays (depth {depth})...")
            sub_clusters = cluster_by_spacer_overlap(cluster, arrays)
            
            # Check if splitting actually worked (clusters got smaller)
            max_subcluster_size = max(len(sc) for sc in sub_clusters) if sub_clusters else 0
            
            if max_subcluster_size < len(cluster):
                # Splitting worked, recurse
                final_clusters.extend(enforce_max_size(sub_clusters, arrays, max_size, depth + 1))
                continue
        
        # If greedy didn't work or max recursion reached, force random partition
        print(f"  Force-partitioning highly interconnected cluster with {len(cluster)} arrays...")
        num_partitions = (len(cluster) // max_size) + 1
        partition_size = len(cluster) // num_partitions
        
        for i in range(num_partitions):
            start_idx = i * partition_size
            if i == num_partitions - 1:
                end_idx = len(cluster)  # Last partition gets remainder
            else:
                end_idx = (i + 1) * partition_size
            
            partition = cluster[start_idx:end_idx]
            if partition:
                final_clusters.append(partition)
    
    return final_clusters


# ----------------------------
# 5. Assign spacer IDs within each cluster
# ----------------------------
def assign_spacer_ids(clusters, arrays):
    """
    Assign local spacer IDs (0, 1, 2, ...) within each cluster.
    Returns dict: cluster_id -> [(array_id, spacer_ids_list), ...]
    """
    cluster_results = {}
    
    for cluster_idx, cluster in enumerate(clusters):
        cluster_id = f"g_{cluster_idx}"
        
        # Build local spacer index for this cluster
        spacer_to_local_id = {}
        local_id_counter = 0
        
        arrays_with_ids = []
        for array_id in cluster:
            local_ids = []
            for global_spacer_idx in arrays[array_id]:
                if global_spacer_idx not in spacer_to_local_id:
                    spacer_to_local_id[global_spacer_idx] = local_id_counter
                    local_id_counter += 1
                local_ids.append(spacer_to_local_id[global_spacer_idx])
            
            arrays_with_ids.append((array_id, local_ids))
        
        cluster_results[cluster_id] = arrays_with_ids
    
    return cluster_results


# ----------------------------
# 6. Write output in spacer_fasta format
# ----------------------------
def write_spacer_fasta(cluster_results, output_dir):
    """
    Write clusters in spacer_fasta format compatible with SpacerPlacer.
    Each cluster becomes one .fa file.
    Singleton clusters (1 array) are written to a 'singletons' subdirectory.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    singletons_dir = output_path / "singletons"
    singletons_count = 0
    multi_count = 0
    
    print(f"\nWriting output to {output_dir}:")
    for cluster_id, arrays_with_ids in sorted(cluster_results.items()):
        # Determine output directory based on cluster size
        if len(arrays_with_ids) == 1:
            singletons_dir.mkdir(parents=True, exist_ok=True)
            out_dir = singletons_dir
            singletons_count += 1
        else:
            out_dir = output_path
            multi_count += 1
        
        fasta_file = out_dir / f"{cluster_id}.fa"
        
        with open(fasta_file, 'w') as f:
            for array_id, spacer_ids in arrays_with_ids:
                f.write(f">{array_id}\n")
                f.write(", ".join(map(str, spacer_ids)) + "\n")
        
        size_label = "singleton" if len(arrays_with_ids) == 1 else f"{len(arrays_with_ids)} arrays"
        print(f"  {cluster_id}.fa: {size_label}")
    
    # Save cluster statistics
    stats_file = output_path / "cluster_stats.json"
    stats = {
        "total_clusters": len(cluster_results),
        "singleton_clusters": singletons_count,
        "multi_array_clusters": multi_count,
        "arrays_per_cluster": {cid: len(arrs) for cid, arrs in cluster_results.items()}
    }
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"\nCluster statistics written to {stats_file}")
    print(f"  Singleton clusters: {singletons_count}")
    print(f"  Multi-array clusters: {multi_count}")


# ----------------------------
# 7. Main pipeline
# ----------------------------
def cluster_crispr_arrays(arrays_file, metadata_file, output_dir, max_cluster_size=None):
    """
    Main clustering pipeline.
    """
    print("=" * 60)
    print("SpacerPlacer-style CRISPR Array Clustering")
    print("=" * 60)
    
    # Parse input
    print("\n1. Parsing extracted arrays...")
    arrays = parse_extracted_arrays(arrays_file)
    metadata = parse_metadata(metadata_file)
    
    if not arrays:
        print("ERROR: No arrays found. Exiting.")
        return
    
    print(f"   Total arrays loaded: {len(arrays)}")
    
    # Group by repeat
    print("\n2. Grouping by consensus repeat...")
    repeat_groups = group_by_repeat(arrays, metadata)
    
    # Cluster by spacer overlap within each repeat group
    print("\n3. Clustering by spacer overlap...")
    final_clusters = []
    for repeat_seq, array_ids in repeat_groups.items():
        if len(array_ids) < 2:
            print(f"  Repeat {repeat_seq[:20]}...: only {len(array_ids)} array, skipping clustering")
            final_clusters.extend([[aid] for aid in array_ids])
            continue
        
        clusters = cluster_by_spacer_overlap(array_ids, arrays)
        print(f"  Repeat {repeat_seq[:20]}...: {len(array_ids)} arrays -> {len(clusters)} clusters")
        
        # Enforce max size within this repeat group
        if max_cluster_size:
            print(f"   Enforcing max cluster size of {max_cluster_size}...")
            clusters = enforce_max_size(clusters, arrays, max_cluster_size)
            print(f"   After splitting: {len(clusters)} clusters")
        
        final_clusters.extend(clusters)
    
    # Assign spacer IDs
    print(f"\n4. Assigning local spacer IDs...")
    cluster_results = assign_spacer_ids(final_clusters, arrays)
    
    # Write output
    print(f"\n5. Writing output...")
    write_spacer_fasta(cluster_results, output_dir)
    
    print("\n" + "=" * 60)
    print(f"Clustering complete! Created {len(cluster_results)} groups.")
    print("=" * 60)
    
    return cluster_results


# ----------------------------
# Main entry point
# ----------------------------
if __name__ == "__main__":
    import sys
    
    # Parse command line arguments
    arrays_file = sys.argv[1] if len(sys.argv) > 1 else INPUT_ARRAYS_FILE
    metadata_file = sys.argv[2] if len(sys.argv) > 2 else INPUT_METADATA_FILE
    output_dir = sys.argv[3] if len(sys.argv) > 3 else OUTPUT_DIR
    max_size = int(sys.argv[4]) if len(sys.argv) > 4 else MAX_CLUSTER_SIZE
    
    cluster_crispr_arrays(arrays_file, metadata_file, output_dir, max_cluster_size=max_size)
