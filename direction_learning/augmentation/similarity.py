"""Similarity detection and signature functions for data augmentation safeguards.

Provides utilities to prevent test set leakage during augmentation by computing
distances between candidates and held-out examples using token-based metrics
(Jaccard, overlap).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..dataset import DirectionExample


def _example_signature(example: DirectionExample) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Extract (spacers, repeats) tuple from example for exact matching."""
    return (tuple(example.spacers), tuple(example.repeats))


def _token_signature(example: DirectionExample) -> set[str]:
    """Return a lightweight token set for similarity checks.

    Prefixes keep spacer and repeat tokens distinct while still making set-based
    overlap computations cheap.
    """
    return {f"S:{spacer}" for spacer in example.spacers} | {f"R:{repeat}" for repeat in example.repeats}


def _build_test_similarity_index(
    records: list[DirectionExample],
    test_indices: list[int],
) -> tuple[dict[int, set[str]], dict[str, set[int]]]:
    """Build a compact inverted index over held-out test examples.

    The returned structures let us compute candidate-to-test similarity without
    comparing against every test record.
    
    Args:
        records: List of all DirectionExample records.
        test_indices: Indices of examples held out in test set.
        
    Returns:
        (test_token_sets, inverted_index) where:
        - test_token_sets: dict mapping test index -> token set
        - inverted_index: dict mapping token -> set of test indices containing it
    """
    test_token_sets: dict[int, set[str]] = {}
    inverted_index: dict[str, set[int]] = {}
    for idx in test_indices:
        token_set = _token_signature(records[idx])
        test_token_sets[idx] = token_set
        for token in token_set:
            inverted_index.setdefault(token, set()).add(idx)
    return test_token_sets, inverted_index


def _min_distance_to_test_set(
    candidate_tokens: set[str],
    test_token_sets: dict[int, set[str]],
    inverted_index: dict[str, set[int]],
    metric: str,
) -> float:
    """Compute the minimum distance from a candidate to any test example.

    Only test examples sharing at least one token with the candidate are checked,
    which keeps the runtime low while still being exact for the chosen metric.
    
    Args:
        candidate_tokens: Token set of candidate augmented example.
        test_token_sets: Dict from test index to token set.
        inverted_index: Dict from token to set of test indices.
        metric: Either "jaccard" or "overlap".
        
    Returns:
        Minimum distance (1.0 - similarity) to any test example; 1.0 if no test examples share tokens.
    """
    if not candidate_tokens or not test_token_sets:
        return 1.0

    candidate_test_indices: set[int] = set()
    for token in candidate_tokens:
        candidate_test_indices.update(inverted_index.get(token, set()))

    if not candidate_test_indices:
        return 1.0

    best_distance = 1.0
    candidate_size = len(candidate_tokens)
    for test_idx in candidate_test_indices:
        test_tokens = test_token_sets[test_idx]
        intersection = len(candidate_tokens & test_tokens)
        if metric == "overlap":
            denom = min(candidate_size, len(test_tokens))
            similarity = intersection / denom if denom > 0 else 0.0
        else:
            union = len(candidate_tokens | test_tokens)
            similarity = intersection / union if union > 0 else 0.0

        distance = 1.0 - similarity
        if distance < best_distance:
            best_distance = distance
            if best_distance <= 0.0:
                break

    return best_distance


def _candidate_passes_similarity_filter(
    candidate_example: DirectionExample,
    test_token_sets: dict[int, set[str]] | None,
    inverted_index: dict[str, set[int]] | None,
    metric: str,
    min_distance: float,
    test_signatures_by_idx: dict[int, tuple[tuple[str, ...], tuple[str, ...]]] | None = None,
) -> bool:
    """Return True if a candidate is far enough from the test set.
    
    Checks both token-based distance and exact contiguous subarray/superset relationships.
    
    Args:
        candidate_example: Augmented example to filter.
        test_token_sets: Dict from test index to token set (or None to skip filtering).
        inverted_index: Dict from token to test indices (or None to skip filtering).
        metric: "jaccard" or "overlap".
        min_distance: Minimum acceptable distance (0.0 to 1.0).
        test_signatures_by_idx: Optional dict from test index to (spacers, repeats) tuple
                                for exact contiguous subarray/superset checks.
        
    Returns:
        True if candidate passes the filter (safe to add); False if it fails (too similar to test).
    """
    if test_token_sets is None or inverted_index is None:
        return True

    candidate_tokens = _token_signature(candidate_example)

    # First, check contiguous subarray / superset relationships against nearby test examples
    if test_signatures_by_idx is not None and candidate_tokens:
        # compute candidate_test_indices (only those sharing any token)
        candidate_test_indices: set[int] = set()
        for token in candidate_tokens:
            candidate_test_indices.update(inverted_index.get(token, set()))

        if candidate_test_indices:
            cand_sp = tuple(candidate_example.spacers)
            cand_re = tuple(candidate_example.repeats)
            cand_len = len(cand_sp)

            for tidx in candidate_test_indices:
                tsig = test_signatures_by_idx.get(tidx)
                if not tsig:
                    continue
                test_sp, test_re = tsig
                # check if candidate is a contiguous subarray of test
                if cand_len <= len(test_sp):
                    for i in range(len(test_sp) - cand_len + 1):
                        if test_sp[i : i + cand_len] == cand_sp and test_re[i : i + cand_len] == cand_re:
                            return False
                # check if candidate is a contiguous superset (contains test)
                tst_len = len(test_sp)
                if tst_len <= cand_len:
                    for i in range(cand_len - tst_len + 1):
                        if cand_sp[i : i + tst_len] == test_sp and cand_re[i : i + tst_len] == test_re:
                            return False

    distance = _min_distance_to_test_set(candidate_tokens, test_token_sets, inverted_index, metric)
    return distance >= min_distance
