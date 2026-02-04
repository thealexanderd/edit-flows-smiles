"""Corruption utilities for creating x_t from x1 by applying random edits."""

from typing import List, Set
import random

from .edit_distance import EditOp, EditType, apply_edit
from .tokenizer import BOS, EOS


def corrupt(
    x1_tokens_with_special: List[str],
    t: float,
    vocab: Set[str],
    rng: random.Random,
    p_del: float = 0.4,
    p_sub: float = 0.4,
    p_ins: float = 0.2,
    k_max: int = 10
) -> List[str]:
    """
    Create a corrupted version x_t from x1 by applying random edits.
    
    The corruption intensity is determined by t and the sequence length.
    
    Args:
        x1_tokens_with_special: Target token sequence [BOS, ..., EOS]
        t: Corruption level in (0, 1), higher = more corruption
        vocab: Set of tokens for sampling insertions/substitutions
        rng: Random number generator
        p_del: Probability of delete operation
        p_sub: Probability of substitute operation
        p_ins: Probability of insert operation (should sum to 1 with del/sub)
        k_max: Maximum number of edits to apply
        
    Returns:
        Corrupted token sequence x_t
    """
    # Normalize probabilities
    total = p_del + p_sub + p_ins
    p_del /= total
    p_sub /= total
    p_ins /= total
    
    # Compute corruption intensity
    L = len(x1_tokens_with_special) - 2  # Exclude BOS/EOS
    k = max(1, min(round(t * L), k_max))
    
    # Filter vocab to exclude special tokens
    vocab_list = [tok for tok in vocab if tok not in [BOS, EOS, "<PAD>", "<UNK>"]]
    if not vocab_list:
        # Fallback if vocab is empty
        vocab_list = ["C", "N", "O"]
    
    # Start with x1
    tokens = x1_tokens_with_special.copy()
    
    # Apply k random edits
    for _ in range(k):
        if len(tokens) <= 2:  # Only BOS/EOS left
            break
        
        # Choose operation type
        op_type = rng.choices(
            [EditType.DEL, EditType.SUB, EditType.INS],
            weights=[p_del, p_sub, p_ins],
            k=1
        )[0]
        
        if op_type == EditType.DEL:
            # Delete a random interior token (not BOS/EOS)
            valid_indices = list(range(1, len(tokens) - 1))
            if not valid_indices:
                continue
            i = rng.choice(valid_indices)
            edit = EditOp(type=EditType.DEL, i=i)
            tokens = apply_edit(tokens, edit)
            
        elif op_type == EditType.SUB:
            # Substitute a random interior token
            valid_indices = list(range(1, len(tokens) - 1))
            if not valid_indices:
                continue
            i = rng.choice(valid_indices)
            new_tok = rng.choice(vocab_list)
            edit = EditOp(type=EditType.SUB, i=i, tok=new_tok)
            tokens = apply_edit(tokens, edit)
            
        elif op_type == EditType.INS:
            # Insert at a random gap (between tokens, not before BOS or after EOS)
            # Gaps are at positions 1 to len-1 (before tokens at those indices)
            valid_gaps = list(range(1, len(tokens)))  # Can insert before any token except BOS
            if not valid_gaps:
                continue
            g = rng.choice(valid_gaps)
            new_tok = rng.choice(vocab_list)
            edit = EditOp(type=EditType.INS, g=g, tok=new_tok)
            tokens = apply_edit(tokens, edit)
    
    return tokens


if __name__ == "__main__":
    from .tokenizer import tokenize, add_special
    
    # Test corruption
    print("Corruption test:")
    
    smiles = "CC(=O)O"
    tokens = tokenize(smiles)
    tokens_with_special = add_special(tokens)
    
    vocab = set(tokens + ["N", "S", "P", "F"])
    
    rng = random.Random(42)
    
    print(f"\nOriginal: {tokens_with_special}")
    
    for t in [0.2, 0.5, 0.8]:
        corrupted = corrupt(tokens_with_special, t, vocab, rng)
        print(f"t={t}: {corrupted}")
    
    print("\nCorruption test complete!")
