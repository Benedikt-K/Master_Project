#!/usr/bin/env python3
"""
Naive repeat-based classifier for direction prediction.
Tests if the model is just a glorified lookup table by classifying based on:
1. Exact repeat matches
2. Edit distance 1, 2, 3 matches (configurable)
"""

import json
import argparse
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Set
import sys
from dataclasses import dataclass
from tqdm import tqdm

# Try to import rapidfuzz for faster edit distance, fall back to standard if not available
try:
    from rapidfuzz.distance import Levenshtein
    RAPIDFUZZ_AVAILABLE = True
except ImportError:
    RAPIDFUZZ_AVAILABLE = False


@dataclass
class PredictionResult:
    """Result of a single prediction."""
    predicted_label: int
    methods_used: List[str]
    confidence: float
    matched_repeat: str = None
    edit_distance: int = None


def levenshtein_distance(s1: str, s2: str) -> int:
    """Calculate Levenshtein (edit) distance between two strings."""
    if RAPIDFUZZ_AVAILABLE:
        # Use rapidfuzz for much faster computation (~10-100x faster)
        return Levenshtein.distance(s1, s2)
    
    # Fallback to basic implementation
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    
    if len(s2) == 0:
        return len(s1)
    
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            # j+1 instead of j since previous_row and current_row are one character longer
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    
    return previous_row[-1]


class RepeatBasedClassifier:
    """
    Naive classifier that uses repeat matching for direction prediction.
    
    Strategy:
    1. During training: build a map of repeats -> label frequencies
    2. During prediction: for each spacer, find matching repeat at increasing edit distances
    3. Aggregate predictions across spacers
    """
    
    def __init__(self, max_edit_distance: int = 3):
        """
        Initialize the classifier.
        
        Args:
            max_edit_distance: Maximum edit distance to consider (1, 2, 3, etc.)
        """
        self.max_edit_distance = max_edit_distance
        # Map: repeat -> {label: count}
        self.repeat_to_labels: Dict[str, Counter] = defaultdict(Counter)
        # Store all training repeats for matching
        self.all_repeats: Set[str] = set()
        self.label_distribution: Counter = Counter()
        self.is_trained = False
    
    def train(self, train_file: Path):
        """
        Train on the training set.
        
        Args:
            train_file: Path to train.jsonl file
        """
        print(f"Training on {train_file}...")
        
        with open(train_file) as f:
            lines = f.readlines()
        
        for line in tqdm(lines, desc="Training", unit="example"):
            data = json.loads(line)
            example = data['example']
            label = example['label']
            repeats = example['repeats']
            
            # Record all repeats with their labels
            for repeat in repeats:
                self.repeat_to_labels[repeat][label] += 1
                self.all_repeats.add(repeat)
            
            self.label_distribution[label] += 1
        
        self.is_trained = True
        print(f"✓ Trained on {len(lines)} examples")
        print(f"✓ Found {len(self.all_repeats)} unique repeats")
        print(f"✓ Label distribution: {dict(self.label_distribution)}")
        print()
    
    def _extract_kmers(self, sequence: str, k: int = 3) -> Set[str]:
        """Extract k-mers from a sequence."""
        return {sequence[i:i+k] for i in range(len(sequence) - k + 1)} if len(sequence) >= k else set()
    
    def _find_best_match(self, query: str) -> Tuple[str, int, Dict]:
        """
        Find the best matching repeat for a query sequence using edit distance.
        Optimized with k-mer filtering and length-based pruning.
        
        Returns:
            (matched_repeat, edit_distance, label_counts)
        """
        # First try exact match
        if query in self.all_repeats:
            return query, 0, self.repeat_to_labels[query]
        
        query_len = len(query)
        query_kmers = self._extract_kmers(query, k=3)
        
        # Pre-filter candidates by k-mer overlap and length
        # Only keep repeats that share at least one k-mer with query
        candidates_by_distance = defaultdict(list)
        
        for repeat in self.all_repeats:
            repeat_len = len(repeat)
            
            # Strict length filtering: at edit distance d, length can differ by at most d
            if abs(query_len - repeat_len) > self.max_edit_distance:
                continue
            
            # K-mer based filtering: only check if they share at least one k-mer
            # Sequences with edit distance <= max_edit_distance likely share k-mers
            repeat_kmers = self._extract_kmers(repeat, k=3)
            if not (query_kmers & repeat_kmers):  # No shared k-mers
                continue
            
            # Compute edit distance
            dist = levenshtein_distance(query, repeat)
            if dist <= self.max_edit_distance:
                candidates_by_distance[dist].append(repeat)
        
        # Return the closest match
        for distance in range(self.max_edit_distance + 1):
            if distance in candidates_by_distance:
                candidates = candidates_by_distance[distance]
                if candidates:
                    # Pick the candidate with the most consistent label
                    best_repeat = max(candidates, 
                                    key=lambda r: max(self.repeat_to_labels[r].values()))
                    return best_repeat, distance, self.repeat_to_labels[best_repeat]
        
        return None, None, Counter()
    
    def predict_single(self, repeats: List[str]) -> PredictionResult:
        """
        Predict direction for a single example based on its repeats only.
        
        Uses majority voting across repeat->label matches.
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before prediction")
        
        if not repeats:
            # Default to most common label
            most_common_label = self.label_distribution.most_common(1)[0][0]
            return PredictionResult(
                predicted_label=most_common_label,
                methods_used=["no_repeats_default"],
                confidence=0.0
            )
        
        label_votes = Counter()
        matches = 0
        
        for repeat in repeats:
            # For exact match only mode (max_edit_distance=0), this is just a dict lookup
            if repeat in self.repeat_to_labels:
                repeat_labels = self.repeat_to_labels[repeat]
                best_label = repeat_labels.most_common(1)[0][0]
                weight = repeat_labels[best_label]
                label_votes[best_label] += weight
                matches += 1
            elif self.max_edit_distance > 0:
                # Only do expensive edit distance matching if needed
                matched_repeat, distance, repeat_labels = self._find_best_match(repeat)
                if matched_repeat is not None:
                    best_label = repeat_labels.most_common(1)[0][0]
                    weight = repeat_labels[best_label]
                    label_votes[best_label] += weight
                    matches += 1
        
        # Make final prediction
        if not label_votes:
            predicted_label = self.label_distribution.most_common(1)[0][0]
            confidence = 0.0
        else:
            predicted_label = label_votes.most_common(1)[0][0]
            total_votes = sum(label_votes.values())
            confidence = label_votes[predicted_label] / total_votes if total_votes > 0 else 0.0
        
        return PredictionResult(
            predicted_label=predicted_label,
            methods_used=[f"matched_{matches}/{len(repeats)}_repeats"],
            confidence=confidence
        )
    
    def evaluate(self, eval_file: Path, verbose: bool = False) -> Dict:
        """
        Evaluate the model on a dataset.
        
        Args:
            eval_file: Path to evaluation file (test.jsonl, val.jsonl, or custom)
            verbose: Whether to print per-example predictions
        
        Returns:
            Dictionary with metrics and per-example predictions
        """
        print(f"Evaluating on {eval_file.name}...")
        
        total = 0
        correct = 0
        label_correct = defaultdict(int)
        label_total = defaultdict(int)
        confidence_by_correct = {'correct': [], 'incorrect': []}
        predictions = []  # Store per-example predictions
        
        with open(eval_file) as f:
            lines = f.readlines()
        
        for line in tqdm(lines, desc="Evaluating", unit="example"):
            data = json.loads(line)
            
            # Handle both formats: {"example": {...}} and direct fields {...}
            if 'example' in data:
                example = data['example']
            else:
                example = data
            
            label = example.get('label')
            repeats = example.get('repeats')
            group_name = example.get('group_name', 'unknown')
            array_name = example.get('array_name', 'unknown')
            cas_subtype = example.get('cas_subtype', 'unknown')
            
            # Skip if missing required fields
            if label is None or repeats is None:
                continue
            
            prediction = self.predict_single(repeats)
            predicted_label = prediction.predicted_label
            
            total += 1
            label_total[label] += 1
            
            is_correct = predicted_label == label
            if is_correct:
                correct += 1
                label_correct[label] += 1
                confidence_by_correct['correct'].append(prediction.confidence)
            else:
                confidence_by_correct['incorrect'].append(prediction.confidence)
            
            # Store per-example prediction
            predictions.append({
                'array_name': array_name,
                'group_name': group_name,
                'cas_subtype': cas_subtype,
                'true_label': label,
                'predicted_label': predicted_label,
                'correct': is_correct,
                'confidence': prediction.confidence,
                'num_repeats': len(repeats),
                'methods': prediction.methods_used
            })
            
            if verbose and (total <= 10 or not is_correct):
                print(f"  Example {total}: True={label}, Pred={predicted_label}, "
                      f"Conf={prediction.confidence:.2f}, Methods={prediction.methods_used}")
        
        accuracy = correct / total if total > 0 else 0
        per_label_accuracy = {
            label: label_correct[label] / label_total[label] if label_total[label] > 0 else 0
            for label in label_total
        }
        
        results = {
            'total': total,
            'correct': correct,
            'accuracy': accuracy,
            'per_label_accuracy': per_label_accuracy,
            'avg_confidence_correct': (
                sum(confidence_by_correct['correct']) / len(confidence_by_correct['correct'])
                if confidence_by_correct['correct'] else 0
            ),
            'avg_confidence_incorrect': (
                sum(confidence_by_correct['incorrect']) / len(confidence_by_correct['incorrect'])
                if confidence_by_correct['incorrect'] else 0
            ),
            'predictions': predictions
        }
        
        return results


def print_results(results: Dict, split_name: str = "Evaluation"):
    """Pretty print evaluation results."""
    print(f"\n{'='*60}")
    print(f"{split_name} Results")
    print(f"{'='*60}")
    print(f"Accuracy: {results['accuracy']:.4f} ({results['correct']}/{results['total']})")
    
    if results['per_label_accuracy']:
        print("\nPer-label accuracy:")
        for label, acc in results['per_label_accuracy'].items():
            print(f"  Label {label}: {acc:.4f}")
    
    print(f"\nAverage confidence (correct): {results['avg_confidence_correct']:.4f}")
    print(f"Average confidence (incorrect): {results['avg_confidence_incorrect']:.4f}")
    print(f"{'='*60}\n")


def save_results_json(results: Dict, output_file: Path):
    """
    Save evaluation results to JSON file.
    
    Args:
        results: Results dictionary with predictions
        output_file: Path to save JSON file
    """
    # Make a copy to avoid modifying the original
    results_copy = results.copy()
    
    # Extract predictions for separate storage
    predictions = results_copy.pop('predictions', [])
    
    # Create output structure
    output_data = {
        'summary': {
            'total': results_copy['total'],
            'correct': results_copy['correct'],
            'accuracy': results_copy['accuracy'],
            'per_label_accuracy': results_copy['per_label_accuracy'],
            'avg_confidence_correct': results_copy['avg_confidence_correct'],
            'avg_confidence_incorrect': results_copy['avg_confidence_incorrect'],
        },
        'predictions': predictions
    }
    
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"  Saved results to {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Naive repeat-based classifier for direction prediction"
    )
    parser.add_argument(
        '--splits_dir',
        type=Path,
        default=Path(__file__).parent / 'outputs' / 'carbon-500m-direction' / 'splits',
        help='Directory containing train.jsonl, val.jsonl, test.jsonl'
    )
    parser.add_argument(
        '--max_edit_distance',
        type=int,
        default=3,
        help='Maximum edit distance to consider for repeat matching (0=exact only, 1-3 recommended)'
    )
    parser.add_argument(
        '--evaluate_on',
        type=Path,
        default=None,
        help='Path to custom evaluation file (replaces default test.jsonl)'
    )
    parser.add_argument(
        '--evaluate_also',
        type=Path,
        nargs='+',
        default=[],
        help='Additional evaluation files to evaluate on (in addition to val/test)'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Print per-example predictions'
    )
    parser.add_argument(
        '--exact_only',
        action='store_true',
        help='Only use exact repeat matches (much faster, baseline comparison)'
    )
    
    args = parser.parse_args()
    
    # Override edit distance if exact_only is set
    max_ed = 0 if args.exact_only else args.max_edit_distance
    
    # Ensure splits exist
    splits_dir = args.splits_dir
    if not splits_dir.exists():
        print(f"Error: Splits directory not found: {splits_dir}")
        sys.exit(1)
    
    train_file = splits_dir / 'train.jsonl'
    val_file = splits_dir / 'val.jsonl'
    test_file = splits_dir / 'test.jsonl'
    
    if not train_file.exists():
        print(f"Error: train.jsonl not found at {train_file}")
        sys.exit(1)
    
    print(f"Configuration:")
    print(f"  Splits dir: {splits_dir}")
    print(f"  Max edit distance: {max_ed}")
    print(f"  Mode: {'Exact match only' if args.exact_only else 'Edit distance matching'}")
    print(f"  Optimizations:")
    print(f"    - K-mer filtering: ✓ (pre-filter candidates by shared 3-mers)")
    print(f"    - Length pruning: ✓ (skip repeats too different in length)")
    if RAPIDFUZZ_AVAILABLE:
        print(f"    - Rapidfuzz: ✓ (10-100x faster edit distance)")
    else:
        print(f"    - Rapidfuzz: ✗ (install 'rapidfuzz' for 10-100x speedup)")
    print()
    
    # Create output directory if it doesn't exist
    output_dir = Path(splits_dir).parent / 'lookup-outputs'
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}\n")
    
    # Create and train model
    model = RepeatBasedClassifier(max_edit_distance=max_ed)
    model.train(train_file)
    
    # Evaluate on validation set
    if val_file.exists():
        val_results = model.evaluate(val_file, verbose=args.verbose)
        print_results(val_results, "Validation")
        # Save validation results
        val_output = output_dir / f'lookup-repeat-output-val.json'
        save_results_json(val_results, val_output)
    else:
        print(f"Warning: val.jsonl not found at {val_file}")
    
    # Evaluate on test set (or custom file)
    eval_file = args.evaluate_on if args.evaluate_on else test_file
    if eval_file.exists():
        test_results = model.evaluate(eval_file, verbose=args.verbose)
        split_name = eval_file.stem.title() if args.evaluate_on else "Test"
        print_results(test_results, split_name)
        # Save test results
        test_output_name = eval_file.stem if args.evaluate_on else 'test'
        test_output = output_dir / f'lookup-repeat-output-{test_output_name}.json'
        save_results_json(test_results, test_output)
    else:
        print(f"Error: Evaluation file not found at {eval_file}")
        sys.exit(1)
    
    # Evaluate on any additional files
    for extra_file in args.evaluate_also:
        extra_file = Path(extra_file)
        if extra_file.exists():
            extra_results = model.evaluate(extra_file, verbose=args.verbose)
            print_results(extra_results, f"Extra: {extra_file.stem.title()}")
            # Save extra results
            extra_output = output_dir / f'lookup-repeat-output-{extra_file.stem}.json'
            save_results_json(extra_results, extra_output)
        else:
            print(f"Warning: Extra evaluation file not found: {extra_file}")
    
    print(f"\n✓ All results saved to {output_dir}")


if __name__ == '__main__':
    main()
