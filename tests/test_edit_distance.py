"""Unit tests for edit distance computation."""

import pytest
from smiles_editflow.edit_distance import (
    levenshtein_script,
    apply_edit,
    apply_script,
    edit_distance,
    EditOp,
    EditType,
)
from smiles_editflow.tokenizer import BOS, EOS


class TestEditDistance:
    """Test edit distance and script computation."""
    
    def test_identical_sequences(self):
        """Test that identical sequences have zero edit distance."""
        src = [BOS, "C", "C", EOS]
        tgt = [BOS, "C", "C", EOS]
        
        script = levenshtein_script(src, tgt)
        assert len(script) == 0
        assert edit_distance(src, tgt) == 0
        
    def test_single_deletion(self):
        """Test single deletion operation."""
        src = [BOS, "C", "C", "O", EOS]
        tgt = [BOS, "C", "C", EOS]
        
        script = levenshtein_script(src, tgt)
        assert len(script) == 1
        assert script[0].type == EditType.DEL
        
        result = apply_script(src, script)
        assert result == tgt
        
    def test_single_insertion(self):
        """Test single insertion operation."""
        src = [BOS, "C", "C", EOS]
        tgt = [BOS, "C", "C", "O", EOS]
        
        script = levenshtein_script(src, tgt)
        assert len(script) == 1
        assert script[0].type == EditType.INS
        
        result = apply_script(src, script)
        assert result == tgt
        
    def test_single_substitution(self):
        """Test single substitution operation."""
        src = [BOS, "C", "C", "N", EOS]
        tgt = [BOS, "C", "C", "O", EOS]
        
        script = levenshtein_script(src, tgt)
        assert len(script) == 1
        assert script[0].type == EditType.SUB
        assert script[0].tok == "O"
        
        result = apply_script(src, script)
        assert result == tgt
        
    def test_multiple_edits(self):
        """Test sequence with multiple edits."""
        src = [BOS, "C", "C", "N", EOS]
        tgt = [BOS, "C", "O", "O", EOS]
        
        script = levenshtein_script(src, tgt)
        result = apply_script(src, script)
        assert result == tgt
        
    def test_all_different(self):
        """Test sequences that are completely different."""
        src = [BOS, "C", "C", "C", EOS]
        tgt = [BOS, "O", "O", "O", EOS]
        
        script = levenshtein_script(src, tgt)
        result = apply_script(src, script)
        assert result == tgt
        
    def test_empty_to_nonempty(self):
        """Test going from empty interior to non-empty."""
        src = [BOS, EOS]
        tgt = [BOS, "C", "C", "O", EOS]
        
        script = levenshtein_script(src, tgt)
        result = apply_script(src, script)
        assert result == tgt
        
    def test_nonempty_to_empty(self):
        """Test going from non-empty to empty interior."""
        src = [BOS, "C", "C", "O", EOS]
        tgt = [BOS, EOS]
        
        script = levenshtein_script(src, tgt)
        result = apply_script(src, script)
        assert result == tgt
        
    def test_complex_edits(self):
        """Test more complex edit sequences."""
        src = [BOS, "C", "C", "(", "=", "O", ")", "O", EOS]
        tgt = [BOS, "C", "C", "O", EOS]
        
        script = levenshtein_script(src, tgt)
        result = apply_script(src, script)
        assert result == tgt
        
    def test_apply_deletion(self):
        """Test applying deletion edit."""
        tokens = [BOS, "C", "C", "O", EOS]
        edit = EditOp(type=EditType.DEL, i=2)  # Delete second C
        
        result = apply_edit(tokens, edit)
        assert result == [BOS, "C", "O", EOS]
        
    def test_apply_substitution(self):
        """Test applying substitution edit."""
        tokens = [BOS, "C", "C", "N", EOS]
        edit = EditOp(type=EditType.SUB, i=3, tok="O")  # Substitute N with O
        
        result = apply_edit(tokens, edit)
        assert result == [BOS, "C", "C", "O", EOS]
        
    def test_apply_insertion(self):
        """Test applying insertion edit."""
        tokens = [BOS, "C", "C", EOS]
        edit = EditOp(type=EditType.INS, g=3, tok="O")  # Insert O before EOS
        
        result = apply_edit(tokens, edit)
        assert result == [BOS, "C", "C", "O", EOS]
        
    def test_deterministic_script(self):
        """Test that script generation is deterministic."""
        src = [BOS, "C", "C", "N", EOS]
        tgt = [BOS, "C", "O", "O", EOS]
        
        script1 = levenshtein_script(src, tgt)
        script2 = levenshtein_script(src, tgt)
        
        assert len(script1) == len(script2)
        for op1, op2 in zip(script1, script2):
            assert op1.type == op2.type
            assert op1.i == op2.i
            assert op1.g == op2.g
            assert op1.tok == op2.tok
            
    def test_edit_distance_equals_script_length(self):
        """Test that edit distance equals script length."""
        test_cases = [
            ([BOS, "C", "C", EOS], [BOS, "C", "C", EOS]),
            ([BOS, "C", "C", "O", EOS], [BOS, "C", "C", EOS]),
            ([BOS, "C", "C", EOS], [BOS, "C", "C", "O", EOS]),
            ([BOS, "C", "C", "N", EOS], [BOS, "C", "C", "O", EOS]),
        ]
        
        for src, tgt in test_cases:
            script = levenshtein_script(src, tgt)
            dist = edit_distance(src, tgt)
            assert len(script) == dist


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
