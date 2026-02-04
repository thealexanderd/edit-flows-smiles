"""Loss functions for training the edit flow model."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple


def _build_valid_masks(attn_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build valid masks for DEL/SUB positions and INS gaps."""
    B, S = attn_mask.shape
    device = attn_mask.device

    valid_del_sub = attn_mask.clone()
    valid_ins = torch.zeros(B, S + 1, dtype=torch.bool, device=device)

    for b in range(B):
        valid_positions = attn_mask[b].nonzero(as_tuple=True)[0]
        if len(valid_positions) > 0:
            valid_del_sub[b, 0] = False
            last_valid = valid_positions[-1]
            valid_del_sub[b, last_valid] = False

            first_valid = valid_positions[0].item()
            valid_ins[b, first_valid + 1:last_valid + 1] = True

    return valid_del_sub, valid_ins


def compute_losses_teacher_forced(
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
    """Compute masked BCE/CE losses for teacher-forced supervision."""
    B, S = p_del.shape
    device = p_del.device

    valid_del_sub, valid_ins = _build_valid_masks(attn_mask)

    loss_del = F.binary_cross_entropy(
        p_del[valid_del_sub],
        y_del[valid_del_sub],
        reduction="mean",
    )
    loss_sub = F.binary_cross_entropy(
        p_sub[valid_del_sub],
        y_sub[valid_del_sub],
        reduction="mean",
    )
    loss_ins = F.binary_cross_entropy(
        p_ins[valid_ins],
        y_ins[valid_ins],
        reduction="mean",
    )

    loss_tok = torch.tensor(0.0, device=device)
    tok_correct = 0
    tok_total = 0

    for b, tok_target in enumerate(tok_targets):
        if tok_target is not None:
            op_type, pos, tok_id = tok_target
            if op_type == "SUB" and pos < S:
                logits = sub_tok_logits[b, pos]
                loss_tok = loss_tok + F.cross_entropy(
                    logits.unsqueeze(0),
                    torch.tensor([tok_id], device=device),
                    reduction="sum",
                )
                pred_id = logits.argmax().item()
                tok_correct += int(pred_id == tok_id)
                tok_total += 1
            elif op_type == "INS" and pos <= S:
                logits = ins_tok_logits[b, pos]
                loss_tok = loss_tok + F.cross_entropy(
                    logits.unsqueeze(0),
                    torch.tensor([tok_id], device=device),
                    reduction="sum",
                )
                pred_id = logits.argmax().item()
                tok_correct += int(pred_id == tok_id)
                tok_total += 1

    if tok_total > 0:
        loss_tok = loss_tok / tok_total
        tok_accuracy = tok_correct / tok_total
    else:
        tok_accuracy = 0.0

    total_loss = loss_del + loss_sub + loss_ins + alpha * loss_tok

    edit_type_accuracy = 0.0
    edit_type_total = 0
    for b in range(B):
        if y_del[b].sum() > 0 or y_sub[b].sum() > 0 or y_ins[b].sum() > 0:
            del_active = y_del[b].argmax()
            sub_active = y_sub[b].argmax()
            ins_active = y_ins[b].argmax()
            if y_del[b, del_active] > 0.5:
                if p_del[b, del_active] > p_sub[b, del_active]:
                    edit_type_accuracy += 1
            elif y_sub[b, sub_active] > 0.5:
                if p_sub[b, sub_active] > p_del[b, sub_active]:
                    edit_type_accuracy += 1
            elif y_ins[b, ins_active] > 0.5:
                if ins_active < len(p_ins[b]) and p_ins[b, ins_active] > 0.5:
                    edit_type_accuracy += 1
            edit_type_total += 1

    edit_type_accuracy = edit_type_accuracy / edit_type_total if edit_type_total > 0 else 0.0

    return {
        "loss": total_loss,
        "loss_del": loss_del,
        "loss_sub": loss_sub,
        "loss_ins": loss_ins,
        "loss_tok": loss_tok,
        "edit_type_acc": edit_type_accuracy,
        "tok_acc": tok_accuracy,
    }


def compute_losses_editflows(
    del_rates: torch.Tensor,
    sub_rates: torch.Tensor,
    ins_rates: torch.Tensor,
    sub_tok_logits: torch.Tensor,
    ins_tok_logits: torch.Tensor,
    del_targets: List[List[int]],
    sub_targets: List[List[Tuple[int, int]]],
    ins_targets: List[List[Tuple[int, int]]],
    attn_mask: torch.Tensor,
    bos_id: int,
    eos_id: int,
    pad_id: int,
    beta: float = 1e-3,
    t_weight: Optional[torch.Tensor] = None,
    eps: float = 1e-6,
) -> Dict[str, torch.Tensor]:
    """Compute Edit Flows rate-based loss with CTMC-inspired objective."""
    B, S = del_rates.shape
    device = del_rates.device

    valid_del_sub, valid_ins = _build_valid_masks(attn_mask)

    loss_del_total = torch.tensor(0.0, device=device)
    loss_sub_total = torch.tensor(0.0, device=device)
    loss_ins_total = torch.tensor(0.0, device=device)
    loss_tok_total = torch.tensor(0.0, device=device)
    tok_correct = 0
    tok_total = 0

    for b in range(B):
        w = t_weight[b] if t_weight is not None else 1.0

        del_rate_b = del_rates[b]
        sub_rate_b = sub_rates[b]
        ins_rate_b = ins_rates[b]

        del_targets_b = del_targets[b]
        if del_targets_b:
            del_idx = torch.tensor(del_targets_b, device=device, dtype=torch.long)
            lpos = -torch.log(del_rate_b[del_idx] + eps).sum()
        else:
            lpos = torch.tensor(0.0, device=device)
        lneg = beta * del_rate_b[valid_del_sub[b]].sum()
        loss_del_total = loss_del_total + w * (lpos + lneg)

        sub_targets_b = sub_targets[b]
        if sub_targets_b:
            sub_idx = torch.tensor([p for p, _ in sub_targets_b], device=device, dtype=torch.long)
            lpos = -torch.log(sub_rate_b[sub_idx] + eps).sum()
        else:
            lpos = torch.tensor(0.0, device=device)
        lneg = beta * sub_rate_b[valid_del_sub[b]].sum()
        loss_sub_total = loss_sub_total + w * (lpos + lneg)

        ins_targets_b = ins_targets[b]
        if ins_targets_b:
            ins_idx = torch.tensor([g for g, _ in ins_targets_b], device=device, dtype=torch.long)
            lpos = -torch.log(ins_rate_b[ins_idx] + eps).sum()
        else:
            lpos = torch.tensor(0.0, device=device)
        lneg = beta * ins_rate_b[valid_ins[b]].sum()
        loss_ins_total = loss_ins_total + w * (lpos + lneg)

        for pos, tok_id in sub_targets_b:
            if pos < S:
                logits = sub_tok_logits[b, pos]
                loss_tok_total = loss_tok_total + w * F.cross_entropy(
                    logits.unsqueeze(0),
                    torch.tensor([tok_id], device=device),
                    reduction="sum",
                )
                pred_id = logits.argmax().item()
                tok_correct += int(pred_id == tok_id)
                tok_total += 1

        for gap, tok_id in ins_targets_b:
            if gap <= S:
                logits = ins_tok_logits[b, gap]
                loss_tok_total = loss_tok_total + w * F.cross_entropy(
                    logits.unsqueeze(0),
                    torch.tensor([tok_id], device=device),
                    reduction="sum",
                )
                pred_id = logits.argmax().item()
                tok_correct += int(pred_id == tok_id)
                tok_total += 1

    total_loss = (loss_del_total + loss_sub_total + loss_ins_total + loss_tok_total) / B
    loss_del = loss_del_total / B
    loss_sub = loss_sub_total / B
    loss_ins = loss_ins_total / B
    loss_tok = loss_tok_total / B
    tok_accuracy = tok_correct / tok_total if tok_total > 0 else 0.0

    return {
        "loss": total_loss,
        "loss_del": loss_del,
        "loss_sub": loss_sub,
        "loss_ins": loss_ins,
        "loss_tok": loss_tok,
        "edit_type_acc": 0.0,
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
    
    losses = compute_losses_teacher_forced(
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
