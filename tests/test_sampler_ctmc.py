"""Smoke test for CTMC sampler."""

from smiles_editflow.model import EditFlowModel
from smiles_editflow.tokenizer import build_vocab
from smiles_editflow.sampler import sample_molecule


def test_ctmc_sampler_runs():
    smiles_list = ["CC(=O)O", "c1ccccc1", "CCO"]
    token_to_id, id_to_token = build_vocab(smiles_list)

    model = EditFlowModel(
        vocab_size=len(token_to_id),
        d_model=64,
        nhead=2,
        num_layers=1,
    )

    smiles, valid, intermediates = sample_molecule(
        model,
        token_to_id,
        id_to_token,
        device="cpu",
        max_steps=5,
        step_size=0.05,
        verbose=False,
    )

    assert isinstance(smiles, str)
    assert isinstance(valid, bool)
    assert isinstance(intermediates, list)
