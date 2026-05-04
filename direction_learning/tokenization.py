from __future__ import annotations

from dataclasses import dataclass

DNA_VOCAB = {
    "PAD": 0,
    "CLS": 1,
    "SEP": 2,
    "A": 3,
    "C": 4,
    "G": 5,
    "T": 6,
    "N": 7,
}

DNA_COMPLEMENT = str.maketrans("ACGTacgtNn", "TGCAtgcaNn")


def reverse_complement(sequence: str) -> str:
    """Compute reverse complement of a DNA sequence.
    
    Applies the DNA complement (A↔T, C↔G) translation and then reverses
    the sequence. Handles both uppercase and lowercase input.
    
    Args:
        sequence: DNA sequence string (may contain A, C, G, T, N).
        
    Returns:
        str: Reverse complement of the input sequence.
        
    Example:
        >>> reverse_complement("ATCG")
        'CGAT'
    """
    return sequence.translate(DNA_COMPLEMENT)[::-1]


def normalize_dna(sequence: str) -> str:
    """Normalize DNA sequence to uppercase and replace invalid bases with N.
    
    Converts all bases to uppercase and replaces any non-standard bases
    (anything not A, C, G, T) with N (ambiguous base). Useful for cleaning
    sequence input before tokenization.
    
    Args:
        sequence: DNA sequence string (may contain lowercase, gaps, etc).
        
    Returns:
        str: Normalized sequence with only A, C, G, T, N characters.
        
    Example:
        >>> normalize_dna("atcgX-N")
        'ATCGNN'
    """
    return "".join(base.upper() if base.upper() in {"A", "C", "G", "T"} else "N" for base in sequence)


def tokenize_dna(sequence: str) -> list[int]:
    """Convert DNA sequence to token IDs using DNA_VOCAB.
    
    First normalizes the sequence using normalize_dna(), then maps each
    base to its corresponding vocabulary ID. The vocabulary includes
    special tokens (PAD, CLS, SEP) and bases (A, C, G, T, N).
    
    Args:
        sequence: DNA sequence string.
        
    Returns:
        list[int]: List of vocabulary token IDs corresponding to each base.
        
    Example:
        >>> tokenize_dna("ACG")
        [3, 4, 5]  # A=3, C=4, G=5
    """
    normalized = normalize_dna(sequence)
    return [DNA_VOCAB.get(base, DNA_VOCAB["N"]) for base in normalized]


def detokenize_dna(token_ids: list[int]) -> str:
    """Convert token IDs back to DNA sequence string.
    
    Reverses tokenize_dna() by mapping token IDs back to their base
    characters. Skips special tokens (PAD, CLS, SEP), so the output
    is a pure DNA sequence.
    
    Args:
        token_ids: List of vocabulary token IDs.
        
    Returns:
        str: DNA sequence reconstructed from tokens.
        
    Example:
        >>> detokenize_dna([3, 4, 5])
        'ACG'
    """
    inverse = {value: key for key, value in DNA_VOCAB.items()}
    bases = []
    for token_id in token_ids:
        base = inverse.get(token_id, "N")
        if base in {"PAD", "CLS", "SEP"}:
            continue
        bases.append(base)
    return "".join(bases)


@dataclass(frozen=True)
class EncodedSequence:
    token_ids: list[int]
    length: int


def encode_sequence(sequence: str, add_cls: bool = False, add_sep: bool = False) -> EncodedSequence:
    """Encode a DNA sequence to tokens with optional CLS/SEP markers.
    
    Tokenizes the sequence and optionally prepends a CLS (classification)
    token or appends a SEP (separator) token. Returns an immutable
    EncodedSequence with both token IDs and total length.
    
    Args:
        sequence: DNA sequence string to encode.
        add_cls: If True, prepend CLS token (ID=1) to mark sequence start.
        add_sep: If True, append SEP token (ID=2) to mark sequence end.
        
    Returns:
        EncodedSequence: Frozen dataclass with token_ids (list[int]) and
            length (int, including special tokens if added).
            
    Example:
        >>> encode_sequence("ACG", add_cls=True, add_sep=True)
        EncodedSequence(token_ids=[1, 3, 4, 5, 2], length=5)
    """
    token_ids = tokenize_dna(sequence)
    if add_cls:
        token_ids = [DNA_VOCAB["CLS"]] + token_ids
    if add_sep:
        token_ids = token_ids + [DNA_VOCAB["SEP"]]
    return EncodedSequence(token_ids=token_ids, length=len(token_ids))
