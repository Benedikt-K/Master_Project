"""Diagnostic script to analyze predictiveness of CRISPR array features for direction prediction."""

import json
import numpy as np
from collections import Counter
from pathlib import Path

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
except ImportError:
    print("scikit-learn required. Install with: pip install scikit-learn")
    exit(1)


def load_jsonl(path: str) -> list[dict]:
    """Load JSONL dataset."""
    data = []
    with open(path) as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def extract_simple_features(examples: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """Extract simple statistical features from CRISPR arrays.
    
    Features:
    - Array length (number of spacers)
    - Mean spacer length
    - Mean repeat length
    - GC content of spacers
    - Repeat length variance
    - Spacer length variance
    """
    X = []
    y = []
    
    for ex in examples:
        spacers = ex.get('spacers', [])
        repeats = ex.get('repeats', [])
        label = ex.get('label', 0)
        
        if not spacers:
            continue
            
        spacer_lens = [len(s) for s in spacers]
        repeat_lens = [len(r) for r in repeats]
        
        # GC content
        spacer_seq = ''.join(spacers)
        gc_content = (spacer_seq.count('G') + spacer_seq.count('C')) / len(spacer_seq) if spacer_seq else 0
        
        # Features
        features = [
            len(spacers),                           # array length
            np.mean(spacer_lens),                   # mean spacer length
            np.std(spacer_lens) if len(spacer_lens) > 1 else 0,  # spacer length variance
            np.mean(repeat_lens) if repeat_lens else 0,          # mean repeat length
            np.std(repeat_lens) if len(repeat_lens) > 1 else 0,  # repeat length variance
            gc_content,                             # GC content
        ]
        
        X.append(features)
        y.append(label)
    
    return np.array(X), np.array(y)


def main():
    """Run diagnostic analysis."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze feature predictiveness for CRISPR direction prediction.")
    parser.add_argument("--jsonl", default="/tmp/direction_training_dataset_full.jsonl", help="Path to JSONL dataset.")
    args = parser.parse_args()
    
    print(f"Loading dataset from {args.jsonl}...")
    data = load_jsonl(args.jsonl)
    print(f"Loaded {len(data)} examples")
    
    # Label distribution
    label_counts = Counter(ex['label'] for ex in data)
    print(f"\nLabel distribution: {dict(label_counts)}")
    print(f"Class balance: {label_counts[1] / len(data):.1%} Forward, {label_counts[0] / len(data):.1%} Reverse")
    
    # Extract features
    print("\nExtracting simple features (array_len, spacer_len, repeat_len, gc_content, etc.)...")
    X, y = extract_simple_features(data)
    print(f"Feature matrix: {X.shape}")
    
    # Check feature statistics
    print("\nFeature Statistics:")
    print("="*60)
    feature_names = ['array_len', 'mean_spacer_len', 'spacer_len_std', 'mean_repeat_len', 'repeat_len_std', 'gc_content']
    for i, fname in enumerate(feature_names):
        col = X[:, i]
        print(f"  {fname:20s}: min={col.min():.4f}, max={col.max():.4f}, mean={col.mean():.4f}, std={col.std():.4f}")
    
    # Check if features actually differ between classes
    print("\nFeature Difference Between Classes (Forward vs Reverse):")
    print("="*60)
    X_forward = X[y == 1]
    X_reverse = X[y == 0]
    for i, fname in enumerate(feature_names):
        mean_fwd = X_forward[:, i].mean()
        mean_rev = X_reverse[:, i].mean()
        std_fwd = X_forward[:, i].std()
        std_rev = X_reverse[:, i].std()
        print(f"  {fname:20s}:")
        print(f"    Forward:  mean={mean_fwd:.4f}, std={std_fwd:.4f}")
        print(f"    Reverse:  mean={mean_rev:.4f}, std={std_rev:.4f}")
        print(f"    Difference: {abs(mean_fwd - mean_rev):.4f} (relative: {abs(mean_fwd - mean_rev) / (mean_fwd + mean_rev + 1e-6) * 100:.2f}%)")
    
    
    # Baseline model: logistic regression with 5-fold CV
    print("\n" + "="*60)
    print("BASELINE: Logistic Regression on Simple Features")
    print("="*60)
    
    lr = LogisticRegression(max_iter=1000)
    scores = cross_val_score(lr, X, y, cv=5, scoring='accuracy')
    print(f"5-fold CV accuracy: {scores.mean():.2%} ± {scores.std():.2%}")
    print(f"Per-fold scores: {[f'{s:.2%}' for s in scores]}")
    
    # Compare to majority baseline
    majority_baseline = max(np.bincount(y)) / len(y)
    print(f"Majority baseline: {majority_baseline:.2%}")
    
    if scores.mean() > majority_baseline + 0.05:
        print(f"✓ Task is learnable: LR outperforms majority baseline by {scores.mean() - majority_baseline:.1%}")
    else:
        print(f"✗ Task is hard: LR only {scores.mean() - majority_baseline:+.1%} vs majority baseline")
    
    # Feature importance (via coefficient magnitude)
    print("\n" + "="*60)
    print("Feature Importance (Logistic Regression Coefficients)")
    print("="*60)
    lr.fit(X, y)
    feature_names = ['array_len', 'mean_spacer_len', 'spacer_len_std', 'mean_repeat_len', 'repeat_len_std', 'gc_content']
    
    coef = lr.coef_[0]
    importance = np.abs(coef)
    sorted_idx = np.argsort(importance)[::-1]
    
    for idx in sorted_idx:
        print(f"  {feature_names[idx]:20s}: {coef[idx]:+.4f} (importance: {importance[idx]:.4f})")
    
    print("\n" + "="*60)
    print("RECOMMENDATIONS")
    print("="*60)
    
    if scores.mean() < 0.55:
        print("⚠ WARNING: Simple features barely beat random.")
        print("  The labels may not be predictive from sequence alone.")
        print("\nPOSSIBLE EXPLANATIONS:")
        print("  1. Direction is not determined by sequence statistics alone")
        print("     (need full sequence patterns, not just length/GC content)")
        print("  2. Labels are unreliable or mislabeled in the source data")
        print("  3. Missing domain-specific features (CRISPR type, organism, metadata)")
        print("\nNEXT STEPS:")
        print("  (1) Check source data reliability: are labels actually correct?")
        print("  (2) Look at a few examples: print some Forward vs Reverse arrays side-by-side")
        print("  (3) Check if direction info is even in the sequences or if it's external metadata")
        print("  (4) Consider: is 'direction' a genomic concept (which strand) rather than a sequence concept?")
    elif scores.mean() < 0.65:
        print("✓ Task is learnable but somewhat hard (~55-65%).")
        print("  Transformer accuracy ~62% is reasonable but room for improvement.")
        print("  Try: (1) Adding metadata features, (2) Better architectures, (3) Ensemble methods")
    else:
        print("✓ Task is clearly learnable (LR > 65%).")
        print("  Transformer should easily beat this if hyperparameters are tuned well.")
        print("  Check: (1) Model architecture, (2) Regularization settings, (3) Learning rate schedule")
    
    print("\n" + "="*60)
    print("SAMPLE DATA INSPECTION")
    print("="*60)
    print("\nLooking at a few Forward vs Reverse examples to spot patterns...\n")
    
    # Forward examples
    forward_examples = [ex for ex in data if ex['label'] == 1][:2]
    # Reverse examples
    reverse_examples = [ex for ex in data if ex['label'] == 0][:2]
    
    print("FORWARD ARRAYS (label=1):")
    for i, ex in enumerate(forward_examples, 1):
        print(f"  Example {i}:")
        print(f"    Spacers: {len(ex['spacers'])} total, lengths: {[len(s) for s in ex['spacers'][:3]]}...")
        print(f"    First spacer: {ex['spacers'][0]}")
        print(f"    First repeat:  {ex['repeats'][0]}")
    
    print("\nREVERSE ARRAYS (label=0):")
    for i, ex in enumerate(reverse_examples, 1):
        print(f"  Example {i}:")
        print(f"    Spacers: {len(ex['spacers'])} total, lengths: {[len(s) for s in ex['spacers'][:3]]}...")
        print(f"    First spacer: {ex['spacers'][0]}")
        print(f"    First repeat:  {ex['repeats'][0]}")
    
    print("\n CRITICAL QUESTION: Can you see a pattern between Forward and Reverse sequences?")
    print("   If NOT, direction may be EXTERNAL (genomic location/strand) not IN sequences themselves.")



if __name__ == "__main__":
    main()
