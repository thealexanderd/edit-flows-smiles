"""Tests for token logit masking."""

import torch

from smiles_editflow.masking import mask_token_logits
from smiles_editflow.tokenizer import build_vocab, BOS, EOS, PAD, UNK


def test_mask_token_logits_blocks_specials():
    token_to_id, _ = build_vocab(["CCO"])
    forbidden = [token_to_id[BOS], token_to_id[EOS], token_to_id[PAD], token_to_id[UNK]]

    logits = torch.zeros(4, len(token_to_id))
    for fid in forbidden:
        logits[:, fid] = 10.0

    masked = mask_token_logits(logits, forbidden)
    probs = torch.softmax(masked, dim=-1)

    for fid in forbidden:
        assert torch.all(probs[:, fid] < 1e-6)

    torch.manual_seed(0)
    samples = torch.multinomial(probs, num_samples=20, replacement=True)
    for fid in forbidden:
        assert not torch.any(samples == fid)
