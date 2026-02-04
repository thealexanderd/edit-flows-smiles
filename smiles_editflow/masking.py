"""Utilities for masking special tokens in logits."""

from typing import Sequence
import torch


def mask_token_logits(
    logits: torch.Tensor,
    forbidden_ids: Sequence[int],
    mask_value: float = -1e9,
) -> torch.Tensor:
    """
    Mask forbidden token IDs in logits by setting them to a large negative value.

    Args:
        logits: Tensor with token dimension on the last axis.
        forbidden_ids: Token IDs to mask.
        mask_value: Value to assign to masked logits.

    Returns:
        Masked logits tensor (same shape as input).
    """
    if not forbidden_ids:
        return logits

    masked = logits.clone()
    idx = torch.tensor(list(forbidden_ids), device=logits.device, dtype=torch.long)
    masked.index_fill_(-1, idx, mask_value)
    return masked
