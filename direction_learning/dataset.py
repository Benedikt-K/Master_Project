from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .tokenization import DNA_VOCAB, encode_sequence, reverse_complement, tokenize_dna

LABEL_TO_ID = {"Forward": 1, "Reverse": 0}


@dataclass(frozen=True)
class DirectionExample:
    """Immutable container for a single CRISPR array direction data sample.
    
    Stores an ordered array of spacers and repeats along with metadata needed
    for transformer-based direction prediction. Includes agreement status of
    CCF and evOr, the predicted direction label of evOr, and optionally flanking sequences.
    """
    array_name: str
    group_name: str
    agreement: str
    evor_direction: str
    label: int
    orientation_variant: str
    source_variant: str
    spacers: list[str]
    repeats: list[str]
    cas_subtype: str = ""
    left_flank: str = ""
    right_flank: str = ""
    source_json: str = ""
    source_spacer_count: int = 0
    deleted_spacers: int = 0
    spacer_deletion_fraction: float = 0.0


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Load records from a JSONL file.
    
    Reads a JSONL file.
    Skips empty lines and returns a list of dictionaries.
    
    Args:
        path: Path to the JSONL file.
        
    Returns:
        list[dict[str, Any]]: List of parsed JSON objects.
        
    Raises:
        FileNotFoundError: If the file does not exist.
        json.JSONDecodeError: If a line is not valid JSON.
    """
    records: list[dict[str, Any]] = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def build_vocab_from_jsonl(path: str | Path) -> dict[str, int]:
    """Build vocabulary from all DNA sequences in a JSONL dataset.
    
    Scans all spacer, repeat, and flank sequences in the JSONL file,
    and creates a token-to-ID mapping. Starts with the standard DNA_VOCAB
    (PAD, CLS, SEP, A, C, G, T, N) and assigns new IDs to any additional
    unique "word" encountered.
    
    Args:
        path: Path to the JSONL dataset file.
        
    Returns:
        dict[str, int]: Vocabulary mapping from base characters to token IDs.
    """
    records = load_jsonl(path)
    vocab = dict(DNA_VOCAB)
    next_id = max(vocab.values()) + 1

    def add_sequence(sequence: str) -> None:
        nonlocal next_id
        for base in sequence.upper():
            if base not in vocab:
                vocab[base] = next_id
                next_id += 1

    for record in records:
        for spacer in record.get("spacers", []):
            add_sequence(spacer)
        for repeat in record.get("repeats", []):
            add_sequence(repeat)
        if record.get("left_flank"):
            add_sequence(record["left_flank"])
        if record.get("right_flank"):
            add_sequence(record["right_flank"])

    return vocab


class DirectionJsonlDataset:
    def __init__(self, jsonl_path: str | Path, include_flanks: bool = False):
        self.jsonl_path = Path(jsonl_path)
        self.include_flanks = include_flanks
        self.records = self._load_records()

    def _load_records(self) -> list[DirectionExample]:
        """Load and parse raw JSONL records into DirectionExample objects."""
        raw_records = load_jsonl(self.jsonl_path)
        records: list[DirectionExample] = []
        for raw in raw_records:
            records.append(
                DirectionExample(
                    array_name=str(raw.get("array_name", "")),
                    group_name=str(raw.get("group_name", "")),
                    agreement=str(raw.get("agreement", "")),
                    evor_direction=str(raw.get("evor_direction", "")),
                    label=int(raw.get("label", LABEL_TO_ID.get(str(raw.get("evor_direction", "")), -1))),
                    orientation_variant=str(raw.get("orientation_variant", "native")),
                    source_variant=str(raw.get("source_variant", "native")),
                    spacers=list(raw.get("spacers", [])),
                    repeats=list(raw.get("repeats", [])),
                    cas_subtype=str(raw.get("cas_subtype", "")),
                    left_flank=str(raw.get("left_flank", "")),
                    right_flank=str(raw.get("right_flank", "")),
                    source_json=str(raw.get("source_json", "")),
                    source_spacer_count=int(raw.get("source_spacer_count", 0)),
                    deleted_spacers=int(raw.get("deleted_spacers", 0)),
                    spacer_deletion_fraction=float(raw.get("spacer_deletion_fraction", 0.0)),
                )
            )
        return records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> DirectionExample:
        return self.records[index]

    def filter_by_group(self, allowed_groups: Iterable[str]) -> list[DirectionExample]:
        allowed = set(allowed_groups)
        return [record for record in self.records if record.group_name in allowed]

    def encoded_example(self, index: int, vocab: dict[str, int] | None = None) -> dict[str, Any]:
        example = self.records[index]
        return encode_example(example, vocab=vocab, include_flanks=self.include_flanks)


def encode_dna_sequence(sequence: str, vocab: dict[str, int]) -> list[int]:
    """Encode a single DNA sequence using the provided vocabulary.
    
    Maps each base in the sequence to its token ID from the vocabulary.
    Unknown bases default to 'N' token ID.
    
    Args:
        sequence: DNA sequence string.
        vocab: Token vocabulary mapping base characters to IDs.
        
    Returns:
        list[int]: List of token IDs corresponding to the sequence bases.
    """
    return [vocab.get(base.upper(), vocab["N"]) for base in sequence]


def encode_example(
    example: DirectionExample,
    vocab: dict[str, int] | None = None,
    include_flanks: bool = False,
    exclude_repeats: bool = False,
    tokenizer: str = "default",
    cnn_tokenizer: object | None = None,
) -> dict[str, Any]:
    """Encode a DirectionExample to token sequences and metadata.
    
    Tokenizes all spacers, repeats, and optionally flanks. Returns a
    dictionary with token lists, lengths, and metadata.
    
    Args:
        example: DirectionExample to encode.
        vocab: Token vocabulary (defaults to DNA_VOCAB if None).
        include_flanks: If True, include left/right flank tokens.
        exclude_repeats: If True, skip encoding repeats (spacer-only mode for ablation).
        
    Returns:
        dict[str, Any]: Dictionary with spacer_tokens, repeat_tokens,
            spacer_lengths, repeat_lengths, label, and metadata fields.
    """
    vocab = dict(DNA_VOCAB) if vocab is None else vocab

    # Default: per-base integer tokenization for spacers/repeats
    if tokenizer == "cnn" and cnn_tokenizer is not None:
        # CNN tokenizer returns per-sequence dense embeddings (list[list[float]])
        spacer_tokens = cnn_tokenizer.encode_sequences(example.spacers, vocab)
        repeat_tokens = [] if exclude_repeats else cnn_tokenizer.encode_sequences(example.repeats, vocab)
    else:
        spacer_tokens = [encode_dna_sequence(spacer, vocab) for spacer in example.spacers]
        repeat_tokens = [] if exclude_repeats else [encode_dna_sequence(repeat, vocab) for repeat in example.repeats]

    payload: dict[str, Any] = {
        "array_name": example.array_name,
        "group_name": example.group_name,
        "agreement": example.agreement,
        "evor_direction": example.evor_direction,
        "label": example.label,
        "orientation_variant": example.orientation_variant,
        "source_variant": example.source_variant,
        "spacer_tokens": spacer_tokens,
        "spacer_lengths": [len(token_row) for token_row in spacer_tokens],
        "repeat_tokens": repeat_tokens,
        "repeat_lengths": [len(token_row) for token_row in repeat_tokens],
        "n_spacers": len(spacer_tokens),
        "source_json": example.source_json,
    }

    if include_flanks:
        payload["left_flank_tokens"] = encode_dna_sequence(example.left_flank, vocab) if example.left_flank else []
        payload["right_flank_tokens"] = encode_dna_sequence(example.right_flank, vocab) if example.right_flank else []
    else:
        payload["left_flank_tokens"] = []
        payload["right_flank_tokens"] = []

    return payload


def collate_encoded_examples(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Collate a batch of encoded examples into padded tensor-ready format.
    
    Handles variable-length spacer arrays and repeat sequences by finding
    maximums, padding to uniform lengths, creating attention masks, and
    stacking metadata into lists suitable for PyTorch tensor conversion.
    
    Args:
        batch: List of encoded examples (dicts from encode_example).
        
    Returns:
        dict[str, Any]: Collated batch with spacer_tokens (3D list),
            spacer_mask (binary), repeat_tokens, labels, and metadata lists.
    """
    max_spacers = max((len(item["spacer_tokens"]) for item in batch), default=0)

    # Detect whether spacers are integer token sequences or dense embeddings
    first_spacer_sample = None
    for item in batch:
        if item["spacer_tokens"]:
            first_spacer_sample = item["spacer_tokens"][0]
            break

    is_embedding = False
    embedding_dim = 0
    if first_spacer_sample is not None and not isinstance(first_spacer_sample[0], int):
        is_embedding = True
        embedding_dim = len(first_spacer_sample)

    if not is_embedding:
        max_spacer_length = max((len(seq) for item in batch for seq in item["spacer_tokens"]), default=0)
        max_repeat_length = max((len(seq) for item in batch for seq in item["repeat_tokens"]), default=0)
    else:
        max_spacer_length = embedding_dim
        # repeats may be embeddings or token sequences; detect similarly
        first_repeat_sample = None
        for item in batch:
            if item["repeat_tokens"]:
                first_repeat_sample = item["repeat_tokens"][0]
                break
        if first_repeat_sample is not None and not isinstance(first_repeat_sample[0], int):
            max_repeat_length = len(first_repeat_sample)
        else:
            max_repeat_length = 0
    max_flank_length = max(
        (
            max(len(item["left_flank_tokens"]), len(item["right_flank_tokens"]))
            for item in batch
        ),
        default=0,
    )

    def pad_sequence(sequence: list[int], target_length: int) -> list[int]:
        return sequence + [DNA_VOCAB["PAD"]] * (target_length - len(sequence))

    if not is_embedding:
        spacer_tensor = [
            [pad_sequence(spacer, max_spacer_length) for spacer in item["spacer_tokens"]] +
            [[DNA_VOCAB["PAD"]] * max_spacer_length for _ in range(max_spacers - len(item["spacer_tokens"]))]
            for item in batch
        ]

        spacer_mask = [
            [1] * len(item["spacer_tokens"]) + [0] * (max_spacers - len(item["spacer_tokens"]))
            for item in batch
        ]
    else:
        spacer_tensor = [
            [spacer for spacer in item["spacer_tokens"]] +
            [[0.0] * max_spacer_length for _ in range(max_spacers - len(item["spacer_tokens"]))]
            for item in batch
        ]

        spacer_mask = [
            [1] * len(item["spacer_tokens"]) + [0] * (max_spacers - len(item["spacer_tokens"]))
            for item in batch
        ]

    # Ensure repeat_tensor is padded to the same outer dimension as spacers
    repeat_tensor = []
    for item in batch:
        repeats = item["repeat_tokens"]
        # Truncate repeats if there are more repeats than spacers to keep shapes consistent
        if len(repeats) > max_spacers:
            repeats = repeats[:max_spacers]
        if repeats and not isinstance(repeats[0][0], int):
            # repeats are embeddings
            padded_repeats = [rep for rep in repeats]
            padded_repeats += [[0.0] * max_repeat_length for _ in range(max_spacers - len(padded_repeats))]
        else:
            padded_repeats = [pad_sequence(repeat, max_repeat_length) for repeat in repeats]
            padded_repeats += [[DNA_VOCAB["PAD"]] * max_repeat_length for _ in range(max_spacers - len(padded_repeats))]
        repeat_tensor.append(padded_repeats)

    labels = [item["label"] for item in batch]

    return {
        "array_name": [item["array_name"] for item in batch],
        "group_name": [item["group_name"] for item in batch],
        "agreement": [item["agreement"] for item in batch],
        "evor_direction": [item["evor_direction"] for item in batch],
        "label": labels,
        "orientation_variant": [item["orientation_variant"] for item in batch],
        "source_variant": [item["source_variant"] for item in batch],
        "spacer_tokens": spacer_tensor,
        "spacer_mask": spacer_mask,
        "repeat_tokens": repeat_tensor,
        "left_flank_tokens": [pad_sequence(item["left_flank_tokens"], max_flank_length) for item in batch],
        "right_flank_tokens": [pad_sequence(item["right_flank_tokens"], max_flank_length) for item in batch],
        "source_json": [item["source_json"] for item in batch],
    }