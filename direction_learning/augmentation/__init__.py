"""Augmentation module for CRISPR array data.

Provides augmentation strategies including reverse-complement duplication,
subarray deletion, and similarity-based test set safeguards.
"""

from .reverse_complement import (
    _materialize_reverse_complement_augmentation,
    _reverse_complement_example,
)
from .similarity import (
    _build_test_similarity_index,
    _candidate_passes_similarity_filter,
    _example_signature,
    _min_distance_to_test_set,
    _token_signature,
)
from .subarray import (
    _materialize_subarray_augmentations,
    _select_diverse_keep_sets,
    make_subarray_augment_fn,
    make_subarray_augment_fn_with_similarity_filter,
)

__all__ = [
    "example_signature",
    "token_signature",
    "build_test_similarity_index",
    "min_distance_to_test_set",
    "candidate_passes_similarity_filter",
    "reverse_complement_example",
    "materialize_reverse_complement_augmentation",
    "select_diverse_keep_sets",
    "materialize_subarray_augmentations",
    "make_subarray_augment_fn",
    "make_subarray_augment_fn_with_similarity_filter",
]

# Re-export with clean names (without leading underscores)
example_signature = _example_signature
token_signature = _token_signature
build_test_similarity_index = _build_test_similarity_index
min_distance_to_test_set = _min_distance_to_test_set
candidate_passes_similarity_filter = _candidate_passes_similarity_filter
reverse_complement_example = _reverse_complement_example
materialize_reverse_complement_augmentation = _materialize_reverse_complement_augmentation
select_diverse_keep_sets = _select_diverse_keep_sets
materialize_subarray_augmentations = _materialize_subarray_augmentations
