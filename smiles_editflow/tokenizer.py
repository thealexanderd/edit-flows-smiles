"""SMILES-aware tokenizer for molecular sequences.

Handles:
- Two-character atoms: Cl, Br
- Bracket expressions: [NH3+], [nH], [O-], [13CH3]
- Ring closures: single digits 0-9, multi-digit %10, %11, etc.
- Bonds: -, =, #, :, /, \\
- Branching: (, )
- Disconnected: .
- Aromatic: c, n, o, s, p
- Regular atoms: B, C, N, O, P, S, F, I
"""

import re
from typing import List, Dict, Tuple


# Special tokens
BOS = "<BOS>"
EOS = "<EOS>"
PAD = "<PAD>"
UNK = "<UNK>"

SPECIAL_TOKENS = [PAD, BOS, EOS, UNK]


def tokenize(smiles: str) -> List[str]:
    """
    Tokenize a SMILES string into a list of tokens.
    
    Args:
        smiles: SMILES string
        
    Returns:
        List of tokens
    """
    tokens = []
    i = 0
    n = len(smiles)
    
    while i < n:
        # Bracket expression: [ ... ]
        if smiles[i] == '[':
            j = i + 1
            while j < n and smiles[j] != ']':
                j += 1
            if j < n:  # Found closing bracket
                tokens.append(smiles[i:j+1])
                i = j + 1
            else:  # No closing bracket, treat as single char
                tokens.append(smiles[i])
                i += 1
                
        # Multi-digit ring closure: %NN
        elif smiles[i] == '%' and i + 2 < n and smiles[i+1].isdigit() and smiles[i+2].isdigit():
            tokens.append(smiles[i:i+3])
            i += 3
            
        # Two-character atoms: Cl, Br
        elif i + 1 < n and smiles[i:i+2] in ['Cl', 'Br']:
            tokens.append(smiles[i:i+2])
            i += 2
            
        # Single character tokens
        else:
            tokens.append(smiles[i])
            i += 1
    
    return tokens


def detokenize(tokens: List[str]) -> str:
    """
    Convert a list of tokens back to a SMILES string.
    
    Filters out special tokens (BOS, EOS, PAD) before joining.
    
    Args:
        tokens: List of tokens
        
    Returns:
        SMILES string
    """
    # Filter out special tokens
    filtered = [tok for tok in tokens if tok not in SPECIAL_TOKENS]
    return ''.join(filtered)


def add_special(tokens: List[str]) -> List[str]:
    """
    Add BOS and EOS tokens to a token sequence.
    
    Args:
        tokens: List of tokens
        
    Returns:
        [BOS] + tokens + [EOS]
    """
    return [BOS] + tokens + [EOS]


def build_vocab(smiles_list: List[str]) -> Tuple[Dict[str, int], Dict[int, str]]:
    """
    Build vocabulary from a list of SMILES strings.
    
    Args:
        smiles_list: List of SMILES strings
        
    Returns:
        (token_to_id, id_to_token) dictionaries
    """
    # Start with special tokens
    vocab_tokens = set(SPECIAL_TOKENS)
    
    # Collect all tokens from SMILES
    for smiles in smiles_list:
        tokens = tokenize(smiles)
        vocab_tokens.update(tokens)
    
    # Sort for deterministic ordering (special tokens first)
    sorted_tokens = SPECIAL_TOKENS + sorted([t for t in vocab_tokens if t not in SPECIAL_TOKENS])
    
    token_to_id = {tok: idx for idx, tok in enumerate(sorted_tokens)}
    id_to_token = {idx: tok for idx, tok in enumerate(sorted_tokens)}
    
    return token_to_id, id_to_token


def encode(tokens: List[str], token_to_id: Dict[str, int]) -> List[int]:
    """
    Encode tokens to IDs.
    
    Args:
        tokens: List of tokens
        token_to_id: Token to ID mapping
        
    Returns:
        List of token IDs
    """
    unk_id = token_to_id[UNK]
    return [token_to_id.get(tok, unk_id) for tok in tokens]


def decode(ids: List[int], id_to_token: Dict[int, str]) -> List[str]:
    """
    Decode IDs to tokens.
    
    Args:
        ids: List of token IDs
        id_to_token: ID to token mapping
        
    Returns:
        List of tokens
    """
    return [id_to_token[idx] for idx in ids]


def _self_check():
    """Self-check that tokenization + detokenization roundtrip works."""
    test_cases = [
        "CC(=O)O",  # Acetic acid
        "c1ccccc1",  # Benzene (aromatic)
        "C%10CCCCC%10",  # Ring closure with %10
        "C[NH3+]",  # Charged bracket
        "ClCBr",  # Two-char atoms
        "[nH]1cccc1",  # Bracket with aromatic
        "[O-]",  # Charged oxygen
        "[13CH3]",  # Isotope
        "CC(C)C",  # Branching
        "C#N",  # Triple bond
        "C/C=C/C",  # Stereochemistry
        "C.Cl",  # Disconnected
    ]
    
    print("Tokenizer self-check:")
    all_passed = True
    for smiles in test_cases:
        tokens = tokenize(smiles)
        reconstructed = detokenize(tokens)
        passed = reconstructed == smiles
        all_passed &= passed
        status = "✓" if passed else "✗"
        print(f"  {status} {smiles:20s} -> {tokens} -> {reconstructed}")
    
    if all_passed:
        print("All tokenization tests passed!")
    else:
        print("Some tokenization tests failed!")
    
    return all_passed


if __name__ == "__main__":
    _self_check()
