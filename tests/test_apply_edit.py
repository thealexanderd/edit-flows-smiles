"""Unit tests for apply_edit correctness."""

import pytest
from smiles_editflow.edit_distance import (
    apply_edit,
    EditOp,
    EditType,
)
from smiles_editflow.tokenizer import BOS, EOS


class TestApplyEdit:
    """Test correctness of applying individual edits."""
    
    def test_delete_preserves_bos_eos(self):
        """Test that deletion preserves BOS and EOS."""
        tokens = [BOS, "C", "C", "O", EOS]
        
        # Delete first C
        edit = EditOp(type=EditType.DEL, i=1)
        result = apply_edit(tokens, edit)
        assert result[0] == BOS
        assert result[-1] == EOS
        
    def test_substitute_preserves_bos_eos(self):
        """Test that substitution preserves BOS and EOS."""
        tokens = [BOS, "C", "C", "N", EOS]
        
        # Substitute N with O
        edit = EditOp(type=EditType.SUB, i=3, tok="O")
        result = apply_edit(tokens, edit)
        assert result[0] == BOS
        assert result[-1] == EOS
        
    def test_insert_preserves_bos_eos(self):
        """Test that insertion preserves BOS and EOS."""
        tokens = [BOS, "C", "C", EOS]
        
        # Insert O between C and EOS
        edit = EditOp(type=EditType.INS, g=3, tok="O")
        result = apply_edit(tokens, edit)
        assert result[0] == BOS
        assert result[-1] == EOS
        
    def test_delete_at_beginning(self):
        """Test deletion at the beginning of sequence."""
        tokens = [BOS, "C", "C", "O", EOS]
        
        # Delete first C (index 1)
        edit = EditOp(type=EditType.DEL, i=1)
        result = apply_edit(tokens, edit)
        assert result == [BOS, "C", "O", EOS]
        
    def test_delete_at_end(self):
        """Test deletion at the end of sequence."""
        tokens = [BOS, "C", "C", "O", EOS]
        
        # Delete O (index 3)
        edit = EditOp(type=EditType.DEL, i=3)
        result = apply_edit(tokens, edit)
        assert result == [BOS, "C", "C", EOS]
        
    def test_delete_middle(self):
        """Test deletion in the middle of sequence."""
        tokens = [BOS, "C", "C", "C", "O", EOS]
        
        # Delete middle C (index 2)
        edit = EditOp(type=EditType.DEL, i=2)
        result = apply_edit(tokens, edit)
        assert result == [BOS, "C", "C", "O", EOS]
        
    def test_substitute_at_beginning(self):
        """Test substitution at the beginning."""
        tokens = [BOS, "C", "C", "O", EOS]
        
        # Substitute first C with N
        edit = EditOp(type=EditType.SUB, i=1, tok="N")
        result = apply_edit(tokens, edit)
        assert result == [BOS, "N", "C", "O", EOS]
        
    def test_substitute_at_end(self):
        """Test substitution at the end."""
        tokens = [BOS, "C", "C", "O", EOS]
        
        # Substitute O with N
        edit = EditOp(type=EditType.SUB, i=3, tok="N")
        result = apply_edit(tokens, edit)
        assert result == [BOS, "C", "C", "N", EOS]
        
    def test_substitute_middle(self):
        """Test substitution in the middle."""
        tokens = [BOS, "C", "C", "O", EOS]
        
        # Substitute second C with N
        edit = EditOp(type=EditType.SUB, i=2, tok="N")
        result = apply_edit(tokens, edit)
        assert result == [BOS, "C", "N", "O", EOS]
        
    def test_insert_after_bos(self):
        """Test insertion right after BOS."""
        tokens = [BOS, "C", "O", EOS]
        
        # Insert N after BOS (at gap 1, before first C)
        edit = EditOp(type=EditType.INS, g=1, tok="N")
        result = apply_edit(tokens, edit)
        assert result == [BOS, "N", "C", "O", EOS]
        
    def test_insert_before_eos(self):
        """Test insertion right before EOS."""
        tokens = [BOS, "C", "C", EOS]
        
        # Insert O before EOS (at gap 3)
        edit = EditOp(type=EditType.INS, g=3, tok="O")
        result = apply_edit(tokens, edit)
        assert result == [BOS, "C", "C", "O", EOS]
        
    def test_insert_middle(self):
        """Test insertion in the middle."""
        tokens = [BOS, "C", "O", EOS]
        
        # Insert N between C and O (at gap 2)
        edit = EditOp(type=EditType.INS, g=2, tok="N")
        result = apply_edit(tokens, edit)
        assert result == [BOS, "C", "N", "O", EOS]
        
    def test_insert_empty_interior(self):
        """Test insertion into empty interior."""
        tokens = [BOS, EOS]
        
        # Insert C after BOS
        edit = EditOp(type=EditType.INS, g=1, tok="C")
        result = apply_edit(tokens, edit)
        assert result == [BOS, "C", EOS]
        
    def test_delete_to_empty_interior(self):
        """Test deletion that leaves empty interior."""
        tokens = [BOS, "C", EOS]
        
        # Delete the only interior token
        edit = EditOp(type=EditType.DEL, i=1)
        result = apply_edit(tokens, edit)
        assert result == [BOS, EOS]
        
    def test_sequence_of_deletes(self):
        """Test applying multiple deletions in sequence."""
        tokens = [BOS, "C", "C", "C", "O", EOS]
        
        # Delete first C
        edit1 = EditOp(type=EditType.DEL, i=1)
        tokens = apply_edit(tokens, edit1)
        assert tokens == [BOS, "C", "C", "O", EOS]
        
        # Delete first C again (now at index 1)
        edit2 = EditOp(type=EditType.DEL, i=1)
        tokens = apply_edit(tokens, edit2)
        assert tokens == [BOS, "C", "O", EOS]
        
    def test_sequence_of_inserts(self):
        """Test applying multiple insertions in sequence."""
        tokens = [BOS, EOS]
        
        # Insert C
        edit1 = EditOp(type=EditType.INS, g=1, tok="C")
        tokens = apply_edit(tokens, edit1)
        assert tokens == [BOS, "C", EOS]
        
        # Insert another C (need to adjust gap position)
        edit2 = EditOp(type=EditType.INS, g=2, tok="C")
        tokens = apply_edit(tokens, edit2)
        assert tokens == [BOS, "C", "C", EOS]
        
    def test_mixed_operations(self):
        """Test applying mixed operations."""
        tokens = [BOS, "C", "N", EOS]
        
        # Substitute N with C
        edit1 = EditOp(type=EditType.SUB, i=2, tok="C")
        tokens = apply_edit(tokens, edit1)
        assert tokens == [BOS, "C", "C", EOS]
        
        # Insert O
        edit2 = EditOp(type=EditType.INS, g=3, tok="O")
        tokens = apply_edit(tokens, edit2)
        assert tokens == [BOS, "C", "C", "O", EOS]
        
        # Delete first C
        edit3 = EditOp(type=EditType.DEL, i=1)
        tokens = apply_edit(tokens, edit3)
        assert tokens == [BOS, "C", "O", EOS]
        
    def test_no_mutation_of_input(self):
        """Test that apply_edit doesn't mutate the input."""
        original = [BOS, "C", "C", "O", EOS]
        tokens = original.copy()
        
        edit = EditOp(type=EditType.DEL, i=1)
        result = apply_edit(tokens, edit)
        
        # Original should be unchanged
        assert tokens == original
        # Result should be different
        assert result != original


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
