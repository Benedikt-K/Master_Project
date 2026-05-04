#!/usr/bin/env python3
"""
Extract CRISPR arrays from CRISPRCasFinder result.json output.
Reads all Result_*/result.json files and outputs arrays with their spacer indices.
"""

import os
import json
import glob
from pathlib import Path


def extract_spacers_from_result_json(result_json_path, min_evidence_level=4):
    """
    Extract arrays and spacer sequences from result.json file.
    Only includes arrays with evidence level >= min_evidence_level.
    Returns list of (array_id, spacer_sequences, repeat_sequence, direction) tuples.
    """
    arrays = []
    
    try:
        with open(result_json_path) as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                print(f"Warning: Could not parse JSON in {result_json_path}")
                return arrays
        
        # Iterate through sequences and their CRISPRs
        if "Sequences" not in data:
            return arrays
        
        for sequence in data["Sequences"]:
            if "Crisprs" not in sequence:
                continue
            
            for crispr in sequence["Crisprs"]:
                # Filter by evidence level
                evidence_level = crispr.get("Evidence_Level", 0)
                if evidence_level < min_evidence_level:
                    continue
                
                array_id = crispr.get("Name", "")
                if not array_id:
                    continue
                
                # Extract spacer sequences from Regions
                spacers = []
                if "Regions" in crispr:
                    for region in crispr["Regions"]:
                        if region.get("Type") == "Spacer":
                            spacers.append(region.get("Sequence", ""))
                
                # Get repeat sequence from DR_Consensus (the consensus repeat)
                repeat_seq = crispr.get("DR_Consensus", "")
                
                # Get direction from CRISPRDirection if available
                direction = crispr.get("CRISPRDirection", "?")
                
                if spacers and repeat_seq:
                    arrays.append((array_id, spacers, repeat_seq, direction))
    
    except Exception as e:
        print(f"Warning: Error reading {result_json_path}: {e}")
    
    return arrays


def extract_all_arrays(base_dir=".", output_file="crisprcasfinder_arrays.txt", min_evidence_level=4):
    """
    Extract all CRISPR arrays from all Result_* folders using result.json.
    Only includes arrays with evidence level >= min_evidence_level.
    Writes to output_file in SpacerPlacer format with spacer indices.
    Also builds and saves global spacer index and repeat information.
    """
    all_arrays = []
    spacer_to_idx = {}
    idx_counter = 0
    
    # Find all Result_* directories
    result_dirs = sorted(glob.glob(os.path.join(base_dir, "Result_*")))
    
    print(f"Found {len(result_dirs)} result directories")
    print(f"Filtering for evidence level >= {min_evidence_level}")
    
    for result_dir in result_dirs:
        result_json_path = os.path.join(result_dir, "result.json")
        
        if os.path.exists(result_json_path):
            arrays = extract_spacers_from_result_json(result_json_path, min_evidence_level=min_evidence_level)
            all_arrays.extend(arrays)
            result_dir_name = os.path.basename(result_dir)
            print(f"  {result_dir_name}: extracted {len(arrays)} arrays (level >= {min_evidence_level})")
    
    # Build global spacer index
    print("\nBuilding global spacer index...")
    array_spacer_indices = {}
    array_metadata = {}  # Store repeat sequences and directions
    
    for array_id, spacers, repeat_seq, direction in all_arrays:
        indices = []
        for spacer in spacers:
            if spacer not in spacer_to_idx:
                spacer_to_idx[spacer] = idx_counter
                idx_counter += 1
            indices.append(spacer_to_idx[spacer])
        array_spacer_indices[array_id] = indices
        array_metadata[array_id] = {
            "repeat": repeat_seq,
            "direction": direction
        }
    
    print(f"  Total unique spacers: {len(spacer_to_idx)}")
    
    # Write output in SpacerPlacer format
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w") as f:
        for array_id in sorted(array_spacer_indices.keys()):
            indices = array_spacer_indices[array_id]
            idx_str = ", ".join(map(str, indices))
            f.write(f">{array_id}\n{idx_str}\n")
    
    print(f"\nTotal arrays extracted: {len(all_arrays)}")
    print(f"Written to: {output_file}")
    
    # Save spacer index for reference
    spacer_index_file = output_path.parent / f"{output_path.stem}_spacer_index.json"
    with open(spacer_index_file, "w") as f:
        json.dump(spacer_to_idx, f, indent=2)
    print(f"Spacer index written to: {spacer_index_file}")
    
    # Save array metadata (repeats and directions)
    metadata_file = output_path.parent / f"{output_path.stem}_metadata.json"
    with open(metadata_file, "w") as f:
        json.dump(array_metadata, f, indent=2)
    print(f"Array metadata written to: {metadata_file}")
    
    return all_arrays, spacer_to_idx, array_spacer_indices


if __name__ == "__main__":
    import sys
    
    base_dir = sys.argv[1] if len(sys.argv) > 1 else "."
    output_file = sys.argv[2] if len(sys.argv) > 2 else "crisprcasfinder_arrays.txt"
    min_evidence = int(sys.argv[3]) if len(sys.argv) > 3 else 4
    
    extract_all_arrays(base_dir, output_file, min_evidence_level=min_evidence)
