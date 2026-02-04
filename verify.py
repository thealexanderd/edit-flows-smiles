#!/usr/bin/env python3
"""Quick verification script to test the implementation."""

import sys
sys.path.insert(0, '/home/adimit/editflows')

print("Testing SMILES Edit Flow Implementation")
print("=" * 60)

# Test 1: Tokenizer
print("\n1. Testing tokenizer...")
from smiles_editflow.tokenizer import tokenize, detokenize, add_special, build_vocab

test_smiles = ["CC(=O)O", "c1ccccc1", "C%10CCCCC%10", "ClCBr"]
for smi in test_smiles:
    tokens = tokenize(smi)
    reconstructed = detokenize(tokens)
    assert reconstructed == smi, f"Tokenizer failed for {smi}"
print("   ✓ Tokenizer working correctly")

# Test 2: Edit distance
print("\n2. Testing edit distance...")
from smiles_editflow.edit_distance import levenshtein_script, apply_script
from smiles_editflow.tokenizer import BOS, EOS

src = [BOS, "C", "C", "O", EOS]
tgt = [BOS, "C", "C", EOS]
script = levenshtein_script(src, tgt)
result = apply_script(src, script)
assert result == tgt, "Edit script application failed"
print("   ✓ Edit distance working correctly")

# Test 3: Alignment + targets
print("\n3. Testing alignment and targets...")
from smiles_editflow.alignment import make_alignment_fixed_N, sample_z_t, strip_epsilon, extract_targets
from smiles_editflow.edit_distance import align_with_epsilon

x0 = ["C", "C"]
x1 = ["C", "O", "C"]
a0, a1 = align_with_epsilon(x0, x1)
z0, z1 = make_alignment_fixed_N(a0, a1, len(a0))
z_t = sample_z_t(z0, z1, t=0.0, kappa=lambda u: u ** 3)
assert strip_epsilon(z_t) == x0
del_pos, sub_pairs, ins_pairs = extract_targets(z_t, z1)
assert len(del_pos) + len(sub_pairs) + len(ins_pairs) == len(levenshtein_script(add_special(x0), add_special(x1)))
print("   ✓ Alignment and target extraction working correctly")

# Test 4: Model forward pass
print("\n4. Testing model...")
import torch
from smiles_editflow.model import EditFlowModel

vocab_size = 50
model = EditFlowModel(vocab_size=vocab_size, d_model=64, nhead=2, num_layers=1)
token_ids = torch.randint(0, vocab_size, (2, 10))
attn_mask = torch.ones(2, 10, dtype=torch.bool)
t = torch.rand(2)

p_del, p_sub, p_ins, sub_tok_logits, ins_tok_logits = model(token_ids, attn_mask, t)
assert p_del.shape == (2, 10), "Model output shape incorrect"
print("   ✓ Model working correctly")

# Test 5: Training step
print("\n5. Testing training step...")
from smiles_editflow.train_step import train_step
import torch.optim as optim

smiles_list = ["CCO", "CC(=O)O", "c1ccccc1"]
token_to_id, id_to_token = build_vocab(smiles_list)
vocab_set = set(token_to_id.keys())

model = EditFlowModel(vocab_size=len(token_to_id), d_model=32, nhead=2, num_layers=1)
optimizer = optim.Adam(model.parameters(), lr=0.001)

result = train_step(model, optimizer, smiles_list, token_to_id, id_to_token, vocab_set)
assert "loss" in result, "Training step failed"
assert result["loss"] > 0, "Loss is not positive"
print("   ✓ Training step working correctly")

print("\n" + "=" * 60)
print("All tests passed! ✓")
print("=" * 60)
print("\nThe implementation is ready to use (Edit Flows mode).")
print("\nTo run training:")
print("  python3 smiles_editflow/train.py --tiny")
