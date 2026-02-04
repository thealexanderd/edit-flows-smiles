"""Token-level Levenshtein edit distance and script extraction.

Implements dynamic programming for computing minimal edit scripts between
token sequences, with special handling for BOS/EOS tokens.
"""

from typing import List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum


class EditType(Enum):
    """Types of edit operations."""
    DEL = "DEL"
    SUB = "SUB"
    INS = "INS"


@dataclass
class EditOp:
    """
    An edit operation.
    
    Attributes:
        type: Type of edit (DEL, SUB, INS)
        i: Position index for DEL/SUB (in full sequence including BOS/EOS)
        g: Gap position for INS (before token at index g)
        tok: Token for SUB/INS
    """
    type: EditType
    i: Optional[int] = None
    g: Optional[int] = None
    tok: Optional[str] = None
    
    def __repr__(self):
        if self.type == EditType.DEL:
            return f"DEL({self.i})"
        elif self.type == EditType.SUB:
            return f"SUB({self.i}, {self.tok})"
        elif self.type == EditType.INS:
            return f"INS({self.g}, {self.tok})"
        return f"EditOp({self.type})"


def levenshtein_script(src_tokens: List[str], tgt_tokens: List[str]) -> List[EditOp]:
    """
    Compute a minimal Levenshtein edit script from src to tgt.
    
    Assumes src and tgt both include BOS/EOS tokens. These special tokens
    must match and cannot be edited. We run DP on the interior tokens only,
    then map indices back to the full sequence.
    
    Args:
        src_tokens: Source token sequence (with BOS/EOS)
        tgt_tokens: Target token sequence (with BOS/EOS)
        
    Returns:
        List of EditOp that transforms src into tgt
    """
    from .tokenizer import BOS, EOS
    
    # Verify BOS/EOS match
    if len(src_tokens) < 2 or len(tgt_tokens) < 2:
        raise ValueError("Sequences must have at least BOS and EOS")
    if src_tokens[0] != BOS or tgt_tokens[0] != BOS:
        raise ValueError("Sequences must start with BOS")
    if src_tokens[-1] != EOS or tgt_tokens[-1] != EOS:
        raise ValueError("Sequences must end with EOS")
    
    # Extract interior tokens (without BOS/EOS)
    src_interior = src_tokens[1:-1]
    tgt_interior = tgt_tokens[1:-1]
    
    n = len(src_interior)
    m = len(tgt_interior)
    
    # DP table: dp[i][j] = min edit distance from src_interior[:i] to tgt_interior[:j]
    dp = [[float('inf')] * (m + 1) for _ in range(n + 1)]
    
    # Backpointer table: stores (prev_i, prev_j, op_type)
    backptr = [[None] * (m + 1) for _ in range(n + 1)]
    
    # Base cases
    dp[0][0] = 0
    for i in range(1, n + 1):
        dp[i][0] = i
        backptr[i][0] = (i - 1, 0, EditType.DEL)
    for j in range(1, m + 1):
        dp[0][j] = j
        backptr[0][j] = (0, j - 1, EditType.INS)
    
    # Fill DP table
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            # Match or substitute
            if src_interior[i - 1] == tgt_interior[j - 1]:
                cost = dp[i - 1][j - 1]  # Match (no edit)
                if cost < dp[i][j]:
                    dp[i][j] = cost
                    backptr[i][j] = (i - 1, j - 1, None)  # Match
            else:
                cost = dp[i - 1][j - 1] + 1  # Substitute
                if cost < dp[i][j]:
                    dp[i][j] = cost
                    backptr[i][j] = (i - 1, j - 1, EditType.SUB)
            
            # Delete from src
            cost = dp[i - 1][j] + 1
            if cost < dp[i][j]:
                dp[i][j] = cost
                backptr[i][j] = (i - 1, j, EditType.DEL)
            
            # Insert into src
            cost = dp[i][j - 1] + 1
            if cost < dp[i][j]:
                dp[i][j] = cost
                backptr[i][j] = (i, j - 1, EditType.INS)
    
    # Backtrack to build edit script
    script = []
    i, j = n, m
    while i > 0 or j > 0:
        if backptr[i][j] is None:
            break
        prev_i, prev_j, op_type = backptr[i][j]
        
        if op_type == EditType.DEL:
            # Delete from position i (interior), which is i+1 in full sequence (after BOS)
            full_idx = i  # i-th interior token is at position i in full (1-indexed after BOS)
            script.append(EditOp(type=EditType.DEL, i=full_idx))
        elif op_type == EditType.SUB:
            # Substitute at position i (interior), which is i+1 in full sequence
            full_idx = i
            script.append(EditOp(type=EditType.SUB, i=full_idx, tok=tgt_interior[j - 1]))
        elif op_type == EditType.INS:
            # Insert before position i (interior) in src, which corresponds to gap at position i+1 in full
            # In the full sequence with BOS at 0, interior starts at 1
            # If we're at interior position i, the gap is after position i in full (before i+1)
            # Actually, gap g means "insert before token at index g"
            # Interior position i maps to full position i+1 (since BOS is at 0)
            # We want to insert to match tgt_interior[j-1]
            # The gap should be at position i+1 (in full sequence coordinates)
            gap_idx = i + 1
            script.append(EditOp(type=EditType.INS, g=gap_idx, tok=tgt_interior[j - 1]))
        
        i, j = prev_i, prev_j
    
    # Reverse to get forward script
    script.reverse()
    
    return script


def apply_edit(tokens: List[str], edit: EditOp) -> List[str]:
    """
    Apply a single edit operation to a token sequence.
    
    Args:
        tokens: Token sequence
        edit: Edit operation
        
    Returns:
        Modified token sequence
    """
    result = tokens.copy()
    
    if edit.type == EditType.DEL:
        # Delete token at position i
        if 0 <= edit.i < len(result):
            result.pop(edit.i)
    elif edit.type == EditType.SUB:
        # Substitute token at position i
        if 0 <= edit.i < len(result):
            result[edit.i] = edit.tok
    elif edit.type == EditType.INS:
        # Insert token at gap g (before token at index g)
        if 0 <= edit.g <= len(result):
            result.insert(edit.g, edit.tok)
    
    return result


def apply_script(tokens: List[str], script: List[EditOp]) -> List[str]:
    """
    Apply a sequence of edit operations to a token sequence.
    
    Args:
        tokens: Initial token sequence
        script: List of edit operations
        
    Returns:
        Final token sequence after all edits
    """
    result = tokens.copy()
    for edit in script:
        result = apply_edit(result, edit)
    return result


def edit_distance(src_tokens: List[str], tgt_tokens: List[str]) -> int:
    """
    Compute the edit distance between two token sequences.
    
    Args:
        src_tokens: Source token sequence
        tgt_tokens: Target token sequence
        
    Returns:
        Minimum number of edits
    """
    script = levenshtein_script(src_tokens, tgt_tokens)
    return len(script)


if __name__ == "__main__":
    from .tokenizer import BOS, EOS
    
    # Test edit distance
    print("Edit distance tests:")
    
    # Test 1: Same sequence
    src = [BOS, "C", "C", EOS]
    tgt = [BOS, "C", "C", EOS]
    script = levenshtein_script(src, tgt)
    print(f"\nTest 1: {src} -> {tgt}")
    print(f"  Script: {script}")
    print(f"  Distance: {len(script)}")
    assert len(script) == 0
    
    # Test 2: Single deletion
    src = [BOS, "C", "C", "O", EOS]
    tgt = [BOS, "C", "C", EOS]
    script = levenshtein_script(src, tgt)
    result = apply_script(src, script)
    print(f"\nTest 2: {src} -> {tgt}")
    print(f"  Script: {script}")
    print(f"  Applied: {result}")
    print(f"  Match: {result == tgt}")
    assert result == tgt
    
    # Test 3: Single insertion
    src = [BOS, "C", "C", EOS]
    tgt = [BOS, "C", "C", "O", EOS]
    script = levenshtein_script(src, tgt)
    result = apply_script(src, script)
    print(f"\nTest 3: {src} -> {tgt}")
    print(f"  Script: {script}")
    print(f"  Applied: {result}")
    print(f"  Match: {result == tgt}")
    assert result == tgt
    
    # Test 4: Single substitution
    src = [BOS, "C", "C", "N", EOS]
    tgt = [BOS, "C", "C", "O", EOS]
    script = levenshtein_script(src, tgt)
    result = apply_script(src, script)
    print(f"\nTest 4: {src} -> {tgt}")
    print(f"  Script: {script}")
    print(f"  Applied: {result}")
    print(f"  Match: {result == tgt}")
    assert result == tgt
    
    # Test 5: Multiple edits
    src = [BOS, "C", "C", "N", EOS]
    tgt = [BOS, "C", "O", "O", EOS]
    script = levenshtein_script(src, tgt)
    result = apply_script(src, script)
    print(f"\nTest 5: {src} -> {tgt}")
    print(f"  Script: {script}")
    print(f"  Applied: {result}")
    print(f"  Match: {result == tgt}")
    assert result == tgt
    
    print("\nAll edit distance tests passed!")
