import igraph as ig
import leidenalg
import numpy as np
from collections import defaultdict
import random
import json
from pathlib import Path

# ----------------------------
# PARAMETERS (tuneable)
# ----------------------------
K_NEIGHBORS = 20
JACCARD_THRESHOLD = 0.1
N_RUNS = 5
BASE_RESOLUTION = 1.0
MAX_CLUSTER_SIZE = 300
RANDOM_SEED = 42

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# Parameter variations for multiple clusterings
CLUSTERING_CONFIGS = [
    {"k": 15, "jaccard": 0.05, "resolution": 0.8, "name": "strict_overlap"},
    {"k": 20, "jaccard": 0.1, "resolution": 1.0, "name": "moderate_overlap"},
    {"k": 25, "jaccard": 0.15, "resolution": 1.2, "name": "loose_overlap"},
    {"k": 30, "jaccard": 0.2, "resolution": 1.5, "name": "very_loose_overlap"},
    {"k": 10, "jaccard": 0.05, "resolution": 0.5, "name": "conservative_strict"},
    {"k": 25, "jaccard": 0.05, "resolution": 2.0, "name": "aggressive_split"},
    {"k": 15, "jaccard": 0.2, "resolution": 0.8, "name": "balanced_loose"},
    {"k": 20, "jaccard": 0.1, "resolution": 0.6, "name": "moderate_merge"},
]


# ----------------------------
# 1. Parse CRISPRCasFinder
# ----------------------------
def parse_cf(filepath):
    """
    Parse arrays file in SpacerPlacer format:
    >array_id
    spacer_idx1, spacer_idx2, spacer_idx3, ...
    """
    arrays = []
    array_ids = []
    i = 0

    with open(filepath) as f:
        current_id = None
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            if line.startswith(">"):
                current_id = line[1:]  # Remove '>'
            else:
                # Parse spacer indices
                spacer_indices = [int(x.strip()) for x in line.split(",") if x.strip()]
                if current_id and spacer_indices:
                    # Create a set of spacer indices for Jaccard similarity
                    spacer_set = set(spacer_indices)
                    arrays.append((i, spacer_set))
                    array_ids.append(current_id)
                    i += 1

    return arrays, array_ids


# ----------------------------
# 2. Compute Jaccard similarity
# ----------------------------
def jaccard(a, b):
    inter = len(a & b)
    if inter == 0:
        return 0
    return inter / len(a | b)


# ----------------------------
# 3. Build sparse KNN graph
# ----------------------------
def build_graph(arrays, k_neighbors=K_NEIGHBORS, jaccard_thresh=JACCARD_THRESHOLD):
    n = len(arrays)
    edges = []
    weights = []

    # Pre-store spacer sets
    spacer_sets = {i: s for i, s in arrays}

    for i in range(n):
        sims = []

        for j in range(n):
            if i == j:
                continue

            sim = jaccard(spacer_sets[i], spacer_sets[j])
            if sim > 0:
                sims.append((j, sim))

        # keep top-K neighbors
        sims.sort(key=lambda x: x[1], reverse=True)
        sims = sims[:k_neighbors]

        for j, sim in sims:
            if sim >= jaccard_thresh:
                edges.append((i, j))
                weights.append(sim)

    g = ig.Graph(n=n, edges=edges, directed=False)
    g.es["weight"] = weights

    return g


# ----------------------------
# 4. Leiden clustering (multi-run consensus)
# ----------------------------
def leiden_runs(graph, resolution):
    memberships = []

    for run in range(N_RUNS):
        part = leidenalg.find_partition(
            graph,
            leidenalg.RBConfigurationVertexPartition,
            weights=graph.es["weight"],
            resolution_parameter=resolution,
            seed=RANDOM_SEED + run
        )
        memberships.append(part.membership)

    return memberships


def consensus_clustering(memberships):
    n = len(memberships[0])
    coassoc = np.zeros((n, n))

    for memb in memberships:
        for i in range(n):
            for j in range(n):
                if memb[i] == memb[j]:
                    coassoc[i, j] += 1

    coassoc /= len(memberships)

    # build graph from consensus matrix
    edges = []
    weights = []

    for i in range(n):
        for j in range(i + 1, n):
            if coassoc[i, j] > 0.5:
                edges.append((i, j))
                weights.append(coassoc[i, j])

    g = ig.Graph(n=n, edges=edges, directed=False)
    g.es["weight"] = weights

    final = leidenalg.find_partition(
        g,
        leidenalg.RBConfigurationVertexPartition,
        weights=g.es["weight"],
        resolution_parameter=1.0
    )

    clusters = defaultdict(list)
    for i, cid in enumerate(final.membership):
        clusters[cid].append(i)

    return list(clusters.values())


# ----------------------------
# 5. Recursive size control
# ----------------------------
def enforce_size(graph, clusters, resolution, k_neighbors=K_NEIGHBORS, depth=0):
    final = []

    for cl in clusters:
        if len(cl) <= MAX_CLUSTER_SIZE:
            final.append(cl)
            continue

        subg = graph.subgraph(cl)

        memberships = leiden_runs(
            subg,
            resolution + 0.5 + depth * 0.5
        )

        subclusters = consensus_clustering(memberships)

        mapped = []
        for sc in subclusters:
            mapped.append([cl[i] for i in sc])

        final.extend(enforce_size(graph, mapped, resolution, k_neighbors, depth + 1))

    return final


# ----------------------------
# 6. Pipeline
# ----------------------------
def cluster_crispr(filepath, k_neighbors=K_NEIGHBORS, jaccard_thresh=JACCARD_THRESHOLD, resolution=BASE_RESOLUTION):
    arrays, array_ids = parse_cf(filepath)

    print(f"Building graph (k={k_neighbors}, jaccard={jaccard_thresh})...")
    g = build_graph(arrays, k_neighbors=k_neighbors, jaccard_thresh=jaccard_thresh)

    print(f"Initial clustering (resolution={resolution})...")
    memberships = leiden_runs(g, resolution)
    clusters = consensus_clustering(memberships)

    print("Refining clusters...")
    clusters = enforce_size(g, clusters, resolution, k_neighbors=k_neighbors)

    return clusters, array_ids


# ----------------------------
# 7. Save output
# ----------------------------
# ----------------------------
# 7. Save output
# ----------------------------
def save_cluster_fasta(clusters, array_ids, arrays, outdir, singletons_dir=None):
    """
    Save each cluster as a separate .fa file in SpacerPlacer format.
    Clusters with <2 arrays are optionally saved to a separate directory.
    Format: >array_id
            spacer_index1, spacer_index2, spacer_index3, ...
    """
    output_path = Path(outdir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    singletons_path = None
    if singletons_dir:
        singletons_path = Path(singletons_dir)
        singletons_path.mkdir(parents=True, exist_ok=True)
    
    for cluster_idx, cluster_member_indices in enumerate(clusters):
        # Determine which directory to save to
        if len(cluster_member_indices) < 2 and singletons_path:
            save_dir = singletons_path
        else:
            save_dir = output_path
        
        fasta_path = save_dir / f"cluster_{cluster_idx}.fa"
        
        with open(fasta_path, "w") as f:
            for member_idx in cluster_member_indices:
                array_id = array_ids[member_idx]
                # Get the original spacer indices from arrays tuple (idx, spacer_set)
                spacer_set = arrays[member_idx][1]
                # Sort the indices to maintain order
                spacer_indices = sorted(list(spacer_set))
                
                # Write in SpacerPlacer format
                f.write(f">{array_id}\n")
                f.write(", ".join(map(str, spacer_indices)) + "\n")


def save_all_clusterings(filepath, output_dir="clusterings"):
    """
    Generate multiple clusterings with different parameters and save each.
    Creates output_dir with subdirectories for each clustering config.
    Each clustering config contains .fa files for each cluster (SpacerPlacer format).
    """
    # Parse input - arrays now contain spacer indices (not sequences)
    print("Parsing arrays...")
    arrays, array_ids = parse_cf(filepath)
    print(f"  Total arrays: {len(array_ids)}")
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    results_summary = []
    
    for config in CLUSTERING_CONFIGS:
        config_name = config["name"]
        print(f"\n{'='*60}")
        print(f"Running clustering: {config_name}")
        print(f"  Parameters: k={config['k']}, jaccard={config['jaccard']}, resolution={config['resolution']}")
        print(f"{'='*60}")
        
        # Run clustering
        clusters, _ = cluster_crispr(
            filepath,
            k_neighbors=config["k"],
            jaccard_thresh=config["jaccard"],
            resolution=config["resolution"]
        )
        
        # Create output directory for this config
        config_dir = output_path / config_name
        config_dir.mkdir(exist_ok=True)
        
        # Save clusters as .fa files (with separate singletons folder)
        fasta_dir = config_dir / "clusters"
        singletons_dir = config_dir / "singletons"
        save_cluster_fasta(clusters, array_ids, arrays, fasta_dir, singletons_dir)
        
        # Save config used
        config_file = config_dir / "config.json"
        with open(config_file, "w") as f:
            json.dump(config, f, indent=2)
        
        # Save summary stats
        cluster_sizes = [len(c) for c in clusters]
        multi_array_clusters = [s for s in cluster_sizes if s >= 2]
        singleton_count = sum(1 for s in cluster_sizes if s == 1)
        
        stats = {
            "config": config_name,
            "parameters": config,
            "total_clusters": len(clusters),
            "multi_array_clusters": len(multi_array_clusters),
            "singleton_clusters": singleton_count,
            "total_arrays": sum(cluster_sizes),
            "min_cluster_size": min(cluster_sizes),
            "max_cluster_size": max(cluster_sizes),
            "mean_cluster_size": float(np.mean(cluster_sizes)),
            "median_cluster_size": float(np.median(cluster_sizes)),
        }
        
        stats_file = config_dir / "stats.json"
        with open(stats_file, "w") as f:
            json.dump(stats, f, indent=2)
        
        results_summary.append(stats)
        
        print(f"  Output directory: {fasta_dir}")
        print(f"  Singletons directory: {singletons_dir}")
        print(f"  Multi-array clusters: {len(multi_array_clusters)}")
        print(f"  Singleton clusters: {singleton_count}")
        print(f"  Sizes (multi-array): min={min(multi_array_clusters) if multi_array_clusters else 'N/A'}, max={max(multi_array_clusters) if multi_array_clusters else 'N/A'}, median={np.median(multi_array_clusters) if multi_array_clusters else 'N/A':.1f}")
    
    # Save summary of all clusterings
    summary_file = output_path / "summary.json"
    with open(summary_file, "w") as f:
        json.dump(results_summary, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"All clusterings complete!")
    print(f"Results saved to: {output_path}")
    print(f"Summary: {summary_file}")
    print(f"{'='*60}")


# ----------------------------
# RUN
# ----------------------------
if __name__ == "__main__":
    import sys
    
    input_file = sys.argv[1] if len(sys.argv) > 1 else "crisprcasfinder_arrays.txt"
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "clusterings"
    
    if not Path(input_file).exists():
        print(f"Error: Input file '{input_file}' not found")
        print(f"Usage: python {sys.argv[0]} <input_arrays.txt> [output_dir]")
        sys.exit(1)
    
    save_all_clusterings(input_file, output_dir)