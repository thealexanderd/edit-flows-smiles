"""Smoke test for Edit Flows training step."""

import torch

from smiles_editflow.model import EditFlowModel
from smiles_editflow.tokenizer import build_vocab
from smiles_editflow.train_step import train_step


def test_editflows_training_step_runs():
    smiles_list = [
        "CC(=O)O",
        "c1ccccc1",
        "CCO",
    ]

    token_to_id, id_to_token = build_vocab(smiles_list)
    vocab_set = set(token_to_id.keys())

    model = EditFlowModel(
        vocab_size=len(token_to_id),
        d_model=64,
        nhead=2,
        num_layers=1,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    result = train_step(
        model,
        optimizer,
        smiles_list,
        token_to_id,
        id_to_token,
        vocab_set,
        device="cpu",
        aligned_length=32,
        x0_mode="uniform",
        x0_max_len=6,
        kappa_power=3,
        beta=1e-3,
    )

    assert "loss" in result
    assert result["loss"] == result["loss"]
