"""Utility functions for training and logging.

Provides environment checks, logging utilities, and helper functions for
data analysis and signature management.
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime
from typing import TYPE_CHECKING, Any

from .tokenization import normalize_dna, reverse_complement

try:
    import torch
except ModuleNotFoundError:
    torch = None

if TYPE_CHECKING:
    from .dataset import DirectionExample


def _require_torch() -> None:
    """Raise error if PyTorch is not installed."""
    if torch is None:
        raise ModuleNotFoundError(
            "PyTorch is required to run the direction training entrypoint. Install torch first."
        )


def _timestamp() -> str:
    """Return a human-readable wall-clock timestamp for log lines."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _print_ts(message: str) -> None:
    """Print a log line with a wall-clock timestamp prefix."""
    print(f"[{_timestamp()}] {message}")


def summarize_cas_subtypes(
    records: list[DirectionExample],
    indices: list[int],
) -> tuple[int, dict[str, int]]:
    """Return unique subtype count and per-subtype counts for a split.
    
    Args:
        records: List of all DirectionExample records.
        indices: Indices to summarize.
        
    Returns:
        (num_unique_subtypes, dict from subtype to count)
    """
    counts = Counter((records[i].cas_subtype or "Unknown") for i in indices)
    return len(counts), dict(sorted(counts.items(), key=lambda kv: kv[0]))


def _build_signature_components(examples: list[DirectionExample]) -> dict[int, list[int]]:
    """Group indices into components by canonical spacer/repeat signature.
    
    Uses union-find to group examples with identical arrays after DNA normalization,
    treating reverse-complement-equivalent arrays as the same signature.
    Useful for ensuring signature cohesion in train/val/test splits.
    
    Args:
        examples: List of DirectionExample records.
        
    Returns:
        Dict mapping component root -> list of indices in that component.
    """
    n = len(examples)
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra = find(a)
        rb = find(b)
        if ra != rb:
            parent[rb] = ra

    def canonical_array_signature(example: DirectionExample) -> tuple[tuple[str, ...], tuple[str, ...]]:
        spacers = tuple(normalize_dna(sequence) for sequence in example.spacers)
        repeats = tuple(normalize_dna(sequence) for sequence in example.repeats)
        reverse_complemented = (
            tuple(normalize_dna(reverse_complement(sequence)) for sequence in reversed(spacers)),
            tuple(normalize_dna(reverse_complement(sequence)) for sequence in reversed(repeats)),
        )
        return min((spacers, repeats), reverse_complemented)

    first_by_signature: dict[tuple[tuple[str, ...], tuple[str, ...]], int] = {}
    for idx, example in enumerate(examples):
        signature = canonical_array_signature(example)
        if signature in first_by_signature:
            union(idx, first_by_signature[signature])
        else:
            first_by_signature[signature] = idx

    components: dict[int, list[int]] = {}
    for idx in range(n):
        components.setdefault(find(idx), []).append(idx)
    return components


# Public aliases (without leading underscores)
require_torch = _require_torch
timestamp = _timestamp
print_ts = _print_ts
build_signature_components = _build_signature_components
