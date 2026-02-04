"""Unit tests for tokenizer."""

import pytest
from smiles_editflow.tokenizer import (
    tokenize,
    detokenize,
    add_special,
    build_vocab,
    encode,
    decode,
    BOS,
    EOS,
    PAD,
    UNK,
)


class TestTokenizer:
    """Test SMILES tokenization."""
    
    def test_simple_smiles(self):
        """Test tokenization of simple SMILES."""
        smiles = "CC(=O)O"
        tokens = tokenize(smiles)
        assert tokens == ["C", "C", "(", "=", "O", ")", "O"]
        
    def test_aromatic_smiles(self):
        """Test tokenization of aromatic SMILES."""
        smiles = "c1ccccc1"
        tokens = tokenize(smiles)
        assert tokens == ["c", "1", "c", "c", "c", "c", "c", "1"]
        
    def test_two_char_atoms(self):
        """Test tokenization of two-character atoms."""
        smiles = "ClCBr"
        tokens = tokenize(smiles)
        assert tokens == ["Cl", "C", "Br"]
        
    def test_bracket_expressions(self):
        """Test tokenization of bracket expressions."""
        smiles = "C[NH3+]"
        tokens = tokenize(smiles)
        assert tokens == ["C", "[NH3+]"]
        
        smiles2 = "[nH]1cccc1"
        tokens2 = tokenize(smiles2)
        assert tokens2 == ["[nH]", "1", "c", "c", "c", "c", "1"]
        
        smiles3 = "[O-]"
        tokens3 = tokenize(smiles3)
        assert tokens3 == ["[O-]"]
        
        smiles4 = "[13CH3]"
        tokens4 = tokenize(smiles4)
        assert tokens4 == ["[13CH3]"]
        
    def test_multi_digit_ring_closure(self):
        """Test tokenization of multi-digit ring closures."""
        smiles = "C%10CCCCC%10"
        tokens = tokenize(smiles)
        assert tokens == ["C", "%10", "C", "C", "C", "C", "C", "%10"]
        
    def test_stereochemistry(self):
        """Test tokenization with stereochemistry markers."""
        smiles = "C/C=C/C"
        tokens = tokenize(smiles)
        assert tokens == ["C", "/", "C", "=", "C", "/", "C"]
        
    def test_disconnected(self):
        """Test tokenization of disconnected molecules."""
        smiles = "C.Cl"
        tokens = tokenize(smiles)
        assert tokens == ["C", ".", "Cl"]
        
    def test_triple_bond(self):
        """Test tokenization with triple bond."""
        smiles = "C#N"
        tokens = tokenize(smiles)
        assert tokens == ["C", "#", "N"]
        
    def test_roundtrip(self):
        """Test that tokenization and detokenization are inverses."""
        test_cases = [
            "CC(=O)O",
            "c1ccccc1",
            "C%10CCCCC%10",
            "C[NH3+]",
            "ClCBr",
            "[nH]1cccc1",
            "[O-]",
            "[13CH3]",
            "CC(C)C",
            "C#N",
            "C/C=C/C",
            "C.Cl",
        ]
        
        for smiles in test_cases:
            tokens = tokenize(smiles)
            reconstructed = detokenize(tokens)
            assert reconstructed == smiles, f"Failed for {smiles}: got {reconstructed}"
            
    def test_add_special(self):
        """Test adding special tokens."""
        tokens = ["C", "C", "O"]
        with_special = add_special(tokens)
        assert with_special == [BOS, "C", "C", "O", EOS]
        
    def test_build_vocab(self):
        """Test vocabulary building."""
        smiles_list = ["CCO", "CCN", "c1ccccc1"]
        token_to_id, id_to_token = build_vocab(smiles_list)
        
        # Check special tokens are included
        assert PAD in token_to_id
        assert BOS in token_to_id
        assert EOS in token_to_id
        assert UNK in token_to_id
        
        # Check some regular tokens
        assert "C" in token_to_id
        assert "O" in token_to_id
        assert "N" in token_to_id
        assert "c" in token_to_id
        
        # Check inverse mapping
        for tok, idx in token_to_id.items():
            assert id_to_token[idx] == tok
            
    def test_encode_decode(self):
        """Test encoding and decoding tokens."""
        smiles_list = ["CCO"]
        token_to_id, id_to_token = build_vocab(smiles_list)
        
        tokens = ["C", "C", "O"]
        ids = encode(tokens, token_to_id)
        decoded = decode(ids, id_to_token)
        
        assert decoded == tokens
        
    def test_encode_unknown(self):
        """Test encoding with unknown tokens."""
        token_to_id = {PAD: 0, BOS: 1, EOS: 2, UNK: 3, "C": 4}
        
        tokens = ["C", "X"]  # X is unknown
        ids = encode(tokens, token_to_id)
        
        assert ids[0] == 4  # C
        assert ids[1] == 3  # UNK


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
