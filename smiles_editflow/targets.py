"""Build supervision targets from edit scripts for training."""

from typing import List, Optional, Tuple
import torch

from .edit_distance import EditOp, EditType


def build_targets(
    x_t_tokens: List[str],
    script: List[EditOp],
    token_to_id: dict
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[Tuple[str, int, int]]]:
    """
    Build training targets from x_t and the edit script to x1.
    
    Takes the FIRST edit in the script as the label (teacher-forced single-step).
    
    Args:
        x_t_tokens: Current corrupted sequence [BOS, ..., EOS]
        script: Edit script from x_t to x1
        token_to_id: Token vocabulary mapping
        
    Returns:
        (y_del, y_sub, y_ins, tok_target) where:
        - y_del: float tensor [len(x_t)] with 1 at deletion site
        - y_sub: float tensor [len(x_t)] with 1 at substitution site
        - y_ins: float tensor [len(x_t)+1] with 1 at insertion gap
        - tok_target: None or ("SUB"|"INS", position, token_id)
    """
    seq_len = len(x_t_tokens)
    
    # Initialize all-zero targets
    y_del = torch.zeros(seq_len, dtype=torch.float32)
    y_sub = torch.zeros(seq_len, dtype=torch.float32)
    y_ins = torch.zeros(seq_len + 1, dtype=torch.float32)
    tok_target = None
    
    # If script is empty, return no-op targets
    if not script:
        return y_del, y_sub, y_ins, tok_target
    
    # Take first edit as the label
    edit = script[0]
    
    if edit.type == EditType.DEL:
        # Mark deletion at position i
        if 0 <= edit.i < seq_len:
            y_del[edit.i] = 1.0
            
    elif edit.type == EditType.SUB:
        # Mark substitution at position i
        if 0 <= edit.i < seq_len:
            y_sub[edit.i] = 1.0
            # Add token target
            tok_id = token_to_id.get(edit.tok, token_to_id.get("<UNK>"))
            tok_target = ("SUB", edit.i, tok_id)
            
    elif edit.type == EditType.INS:
        # Mark insertion at gap g
        if 0 <= edit.g <= seq_len:
            y_ins[edit.g] = 1.0
            # Add token target
            tok_id = token_to_id.get(edit.tok, token_to_id.get("<UNK>"))
            tok_target = ("INS", edit.g, tok_id)
    
    return y_del, y_sub, y_ins, tok_target


def collate_batch(
    batch_data: List[Tuple[List[str], torch.Tensor, torch.Tensor, torch.Tensor, Optional[Tuple]]],
    token_to_id: dict,
    pad_id: int
) -> dict:
    """
    Collate a batch of training examples with padding.
    
    Args:
        batch_data: List of (x_t_tokens, y_del, y_sub, y_ins, tok_target)
        token_to_id: Token vocabulary
        pad_id: ID for PAD token
        
    Returns:
        Dictionary with batched tensors
    """
    batch_size = len(batch_data)
    
    # Find max sequence length in batch
    max_len = max(len(x_t) for x_t, _, _, _, _ in batch_data)
    
    # Initialize batched tensors
    token_ids = torch.full((batch_size, max_len), pad_id, dtype=torch.long)
    attn_mask = torch.zeros((batch_size, max_len), dtype=torch.bool)
    y_del_batch = torch.zeros((batch_size, max_len), dtype=torch.float32)
    y_sub_batch = torch.zeros((batch_size, max_len), dtype=torch.float32)
    y_ins_batch = torch.zeros((batch_size, max_len + 1), dtype=torch.float32)
    
    # Token targets stored as lists (variable length)
    tok_targets = []
    
    for i, (x_t, y_del, y_sub, y_ins, tok_target) in enumerate(batch_data):
        seq_len = len(x_t)
        
        # Encode tokens
        ids = [token_to_id.get(tok, token_to_id.get("<UNK>")) for tok in x_t]
        token_ids[i, :seq_len] = torch.tensor(ids, dtype=torch.long)
        attn_mask[i, :seq_len] = True
        
        # Copy targets
        y_del_batch[i, :seq_len] = y_del
        y_sub_batch[i, :seq_len] = y_sub
        y_ins_batch[i, :seq_len + 1] = y_ins
        
        tok_targets.append(tok_target)
    
    return {
        "token_ids": token_ids,
        "attn_mask": attn_mask,
        "y_del": y_del_batch,
        "y_sub": y_sub_batch,
        "y_ins": y_ins_batch,
        "tok_targets": tok_targets,
    }


if __name__ == "__main__":
    from .tokenizer import BOS, EOS, build_vocab
    from .edit_distance import levenshtein_script
    
    # Test target building
    print("Target building test:")
    
    # Build a simple vocab
    token_to_id, id_to_token = build_vocab(["CC(=O)O"])
    
    # Test case 1: Single deletion
    x_t = [BOS, "C", "C", "O", EOS]
    x1 = [BOS, "C", "C", EOS]
    script = levenshtein_script(x_t, x1)
    
    print(f"\nTest 1: {x_t} -> {x1}")
    print(f"  Script: {script}")
    
    y_del, y_sub, y_ins, tok_target = build_targets(x_t, script, token_to_id)
    print(f"  y_del: {y_del}")
    print(f"  y_sub: {y_sub}")
    print(f"  y_ins: {y_ins}")
    print(f"  tok_target: {tok_target}")
    
    # Test case 2: Single insertion
    x_t = [BOS, "C", "C", EOS]
    x1 = [BOS, "C", "C", "O", EOS]
    script = levenshtein_script(x_t, x1)
    
    print(f"\nTest 2: {x_t} -> {x1}")
    print(f"  Script: {script}")
    
    y_del, y_sub, y_ins, tok_target = build_targets(x_t, script, token_to_id)
    print(f"  y_del: {y_del}")
    print(f"  y_sub: {y_sub}")
    print(f"  y_ins: {y_ins}")
    print(f"  tok_target: {tok_target}")
    
    # Test case 3: Single substitution
    x_t = [BOS, "C", "C", "N", EOS]
    x1 = [BOS, "C", "C", "O", EOS]
    script = levenshtein_script(x_t, x1)
    
    print(f"\nTest 3: {x_t} -> {x1}")
    print(f"  Script: {script}")
    
    y_del, y_sub, y_ins, tok_target = build_targets(x_t, script, token_to_id)
    print(f"  y_del: {y_del}")
    print(f"  y_sub: {y_sub}")
    print(f"  y_ins: {y_ins}")
    print(f"  tok_target: {tok_target}")
    
    print("\nTarget building test complete!")
