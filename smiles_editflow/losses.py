"""Loss functions for training the edit flow model."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple


def compute_losses(
    p_del: torch.Tensor,
    p_sub: torch.Tensor,
    p_ins: torch.Tensor,
    sub_tok_logits: torch.Tensor,
    ins_tok_logits: torch.Tensor,
    y_del: torch.Tensor,
    y_sub: torch.Tensor,
    y_ins: torch.Tensor,
    tok_targets: List[Optional[Tuple[str, int, int]]],
    attn_mask: torch.Tensor,
    bos_id: int,
    eos_id: int,
    pad_id: int,
    alpha: float = 1.0,
) -> Dict[str, torch.Tensor]:
    """
    Compute masked losses for edit operations.
    
    Args:
        p_del: [B, S] deletion probabilities
        p_sub: [B, S] substitution probabilities
        p_ins: [B, S+1] insertion probabilities
        sub_tok_logits: [B, S, V] substitution token logits
        ins_tok_logits: [B, S+1, V] insertion token logits
        y_del: [B, S] deletion targets
        y_sub: [B, S] substitution targets
        y_ins: [B, S+1] insertion targets
        tok_targets: List of (type, pos, tok_id) or None for each batch item
        attn_mask: [B, S] attention mask (True for real tokens)
        bos_id: BOS token ID
        eos_id: EOS token ID
        pad_id: PAD token ID
        alpha: Weight for token loss
        
    Returns:
        Dictionary of losses and metrics
    """
    B, S = p_del.shape
    device = p_del.device
    
    # Create masks for valid positions
    # For DEL/SUB: exclude BOS, EOS, and PAD
    token_ids = torch.zeros(B, S, dtype=torch.long, device=device)  # Dummy for now
    
    # Valid mask for DEL/SUB: not BOS, not EOS, not PAD
    valid_del_sub = attn_mask.clone()  # Start with attention mask
    
    # We need to identify BOS/EOS positions
    # Assume BOS is always at position 0 and EOS is at the last valid position
    for b in range(B):
        valid_del_sub[b, 0] = False  # Mask BOS
        # Find last valid position (EOS)
        valid_positions = attn_mask[b].nonzero(as_tuple=True)[0]
        if len(valid_positions) > 0:
            last_valid = valid_positions[-1]
            valid_del_sub[b, last_valid] = False  # Mask EOS
    
    # Valid mask for INS: gaps between valid tokens (1 to S-1)
    # Gap 0 (before BOS) and gap S (after EOS) should be masked
    valid_ins = torch.zeros(B, S + 1, dtype=torch.bool, device=device)
    for b in range(B):
        valid_positions = attn_mask[b].nonzero(as_tuple=True)[0]
        if len(valid_positions) > 0:
            first_valid = valid_positions[0].item()
            last_valid = valid_positions[-1].item()
            # Valid gaps are from first_valid+1 to last_valid (inclusive)
            # These are gaps AFTER BOS and BEFORE/AT EOS position
            valid_ins[b, first_valid + 1:last_valid + 1] = True
    
    # BCE loss for DEL
    loss_del = F.binary_cross_entropy(
        p_del[valid_del_sub],
        y_del[valid_del_sub],
        reduction='mean'
    )
    
    # BCE loss for SUB
    loss_sub = F.binary_cross_entropy(
        p_sub[valid_del_sub],
        y_sub[valid_del_sub],
        reduction='mean'
    )
    
    # BCE loss for INS
    loss_ins = F.binary_cross_entropy(
        p_ins[valid_ins],
        y_ins[valid_ins],
        reduction='mean'
    )
    
    # Token CE loss
    loss_tok = torch.tensor(0.0, device=device)
    tok_correct = 0
    tok_total = 0
    
    for b, tok_target in enumerate(tok_targets):
        if tok_target is not None:
            op_type, pos, tok_id = tok_target
            
            if op_type == "SUB":
                if pos < S:
                    logits = sub_tok_logits[b, pos]
                    loss_tok = loss_tok + F.cross_entropy(
                        logits.unsqueeze(0),
                        torch.tensor([tok_id], device=device),
                        reduction='sum'
                    )
                    pred_id = logits.argmax().item()
                    if pred_id == tok_id:
                        tok_correct += 1
                    tok_total += 1
                    
            elif op_type == "INS":
                if pos <= S:
                    logits = ins_tok_logits[b, pos]
                    loss_tok = loss_tok + F.cross_entropy(
                        logits.unsqueeze(0),
                        torch.tensor([tok_id], device=device),
                        reduction='sum'
                    )
                    pred_id = logits.argmax().item()
                    if pred_id == tok_id:
                        tok_correct += 1
                    tok_total += 1
    
    if tok_total > 0:
        loss_tok = loss_tok / tok_total
        tok_accuracy = tok_correct / tok_total
    else:
        tok_accuracy = 0.0
    
    # Total loss
    total_loss = loss_del + loss_sub + loss_ins + alpha * loss_tok
    
    # Compute accuracies for edit type
    # Predict edit type: argmax over [p_del, p_sub, p_ins] for each valid position
    edit_type_correct = 0
    edit_type_total = 0
    
    for b in range(B):
        # Check if any target is active
        if y_del[b].sum() > 0 or y_sub[b].sum() > 0 or y_ins[b].sum() > 0:
            # Find the active target
            del_active = y_del[b].argmax()
            sub_active = y_sub[b].argmax()
            ins_active = y_ins[b].argmax()
            
            # Determine ground truth edit type
            if y_del[b, del_active] > 0.5:
                gt_type = 0  # DEL
                # Check if model predicts DEL at that position
                if p_del[b, del_active] > p_sub[b, del_active]:
                    edit_type_correct += 1
            elif y_sub[b, sub_active] > 0.5:
                gt_type = 1  # SUB
                if p_sub[b, sub_active] > p_del[b, sub_active]:
                    edit_type_correct += 1
            elif y_ins[b, ins_active] > 0.5:
                gt_type = 2  # INS
                # For INS, just check if p_ins is highest at that gap
                if ins_active < len(p_ins[b]) and p_ins[b, ins_active] > 0.5:
                    edit_type_correct += 1
            
            edit_type_total += 1
    
    edit_type_accuracy = edit_type_correct / edit_type_total if edit_type_total > 0 else 0.0
    
    return {
        "loss": total_loss,
        "loss_del": loss_del,
        "loss_sub": loss_sub,
        "loss_ins": loss_ins,
        "loss_tok": loss_tok,
        "edit_type_acc": edit_type_accuracy,
        "tok_acc": tok_accuracy,
    }


if __name__ == "__main__":
    # Test loss computation
    print("Loss computation test:")
    
    B, S, V = 2, 5, 10
    
    # Create dummy model outputs
    p_del = torch.rand(B, S)
    p_sub = torch.rand(B, S)
    p_ins = torch.rand(B, S + 1)
    sub_tok_logits = torch.randn(B, S, V)
    ins_tok_logits = torch.randn(B, S + 1, V)
    
    # Create dummy targets
    y_del = torch.zeros(B, S)
    y_sub = torch.zeros(B, S)
    y_ins = torch.zeros(B, S + 1)
    
    # Sample 1: deletion at position 2
    y_del[0, 2] = 1.0
    
    # Sample 2: substitution at position 1
    y_sub[1, 1] = 1.0
    
    tok_targets = [
        None,
        ("SUB", 1, 5),
    ]
    
    attn_mask = torch.ones(B, S, dtype=torch.bool)
    
    losses = compute_losses(
        p_del, p_sub, p_ins,
        sub_tok_logits, ins_tok_logits,
        y_del, y_sub, y_ins,
        tok_targets,
        attn_mask,
        bos_id=0, eos_id=1, pad_id=2,
        alpha=1.0,
    )
    
    print(f"  Total loss: {losses['loss'].item():.4f}")
    print(f"  DEL loss: {losses['loss_del'].item():.4f}")
    print(f"  SUB loss: {losses['loss_sub'].item():.4f}")
    print(f"  INS loss: {losses['loss_ins'].item():.4f}")
    print(f"  TOK loss: {losses['loss_tok'].item():.4f}")
    print(f"  Edit type acc: {losses['edit_type_acc']:.4f}")
    print(f"  Token acc: {losses['tok_acc']:.4f}")
    
    print("\nLoss computation test passed!")
