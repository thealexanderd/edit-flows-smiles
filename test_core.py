#!/usr/bin/env python3
"""Minimal verification script - tests core components without PyTorch/RDKit."""

import sys
sys.path.insert(0, '/home/adimit/editflows')

print("SMILES Edit Flow - Core Component Tests")
print("=" * 60)

# Test 1: Tokenizer
print("\n✓ Test 1: Tokenizer")
from smiles_editflow.tokenizer import tokenize, detokenize, add_special, build_vocab, BOS, EOS

test_cases = [
    "CC(=O)O",      # Acetic acid
    "c1ccccc1",     # Benzene (aromatic)
    "C%10CCCCC%10", # Multi-digit ring closure
    "C[NH3+]",      # Charged bracket
    "ClCBr",        # Two-char atoms
]

print("  Testing tokenization roundtrip:")
for smi in test_cases:
    tokens = tokenize(smi)
    reconstructed = detokenize(tokens)
    assert reconstructed == smi, f"Failed for {smi}"
    print(f"    ✓ {smi}")

print("\n  Testing special tokens:")
tokens = tokenize("CCO")
with_special = add_special(tokens)
assert with_special[0] == BOS and with_special[-1] == EOS
print(f"    ✓ BOS/EOS added: {with_special}")

print("\n  Testing vocabulary building:")
vocab_smiles = ["CCO", "CCN", "c1ccccc1"]
token_to_id, id_to_token = build_vocab(vocab_smiles)
assert BOS in token_to_id and EOS in token_to_id
print(f"    ✓ Vocab size: {len(token_to_id)}")

# Test 2: Edit distance
print("\n✓ Test 2: Edit Distance")
from smiles_editflow.edit_distance import (
    levenshtein_script, apply_script, apply_edit, 
    EditOp, EditType, edit_distance
)

test_pairs = [
    ([BOS, "C", "C", EOS], [BOS, "C", "C", EOS], 0),  # Identical
    ([BOS, "C", "C", "O", EOS], [BOS, "C", "C", EOS], 1),  # Delete
    ([BOS, "C", "C", EOS], [BOS, "C", "C", "O", EOS], 1),  # Insert
    ([BOS, "C", "C", "N", EOS], [BOS, "C", "C", "O", EOS], 1),  # Substitute
]

print("  Testing edit scripts:")
for src, tgt, expected_dist in test_pairs:
    script = levenshtein_script(src, tgt)
    dist = edit_distance(src, tgt)
    result = apply_script(src, script)
    
    assert dist == expected_dist, f"Distance mismatch: {dist} != {expected_dist}"
    assert result == tgt, f"Script application failed: {result} != {tgt}"
    print(f"    ✓ {src[1:-1]} -> {tgt[1:-1]}: distance={dist}")

print("\n  Testing individual edit operations:")
# Delete
tokens = [BOS, "C", "C", "O", EOS]
edit = EditOp(type=EditType.DEL, i=2)
result = apply_edit(tokens, edit)
assert result == [BOS, "C", "O", EOS]
print(f"    ✓ DELETE: {tokens} -> {result}")

# Substitute
tokens = [BOS, "C", "C", "N", EOS]
edit = EditOp(type=EditType.SUB, i=3, tok="O")
result = apply_edit(tokens, edit)
assert result == [BOS, "C", "C", "O", EOS]
print(f"    ✓ SUBSTITUTE: {tokens} -> {result}")

# Insert
tokens = [BOS, "C", "C", EOS]
edit = EditOp(type=EditType.INS, g=3, tok="O")
result = apply_edit(tokens, edit)
assert result == [BOS, "C", "C", "O", EOS]
print(f"    ✓ INSERT: {tokens} -> {result}")

# Test 3: Corruption
print("\n✓ Test 3: Corruption")
from smiles_editflow.corruption import corrupt
import random

tokens = add_special(tokenize("CC(=O)O"))
vocab = {"C", "N", "O", "S", "P", "=", "(", ")"}

print("  Testing corruption at different levels:")
for t in [0.2, 0.5, 0.8]:
    rng = random.Random(42)
    x_t = corrupt(tokens, t, vocab, rng)
    assert x_t[0] == BOS and x_t[-1] == EOS, "BOS/EOS not preserved"
    print(f"    ✓ t={t}: {tokens} -> {x_t}")

# Test 4: Target building
print("\n✓ Test 4: Target Building")
from smiles_editflow.targets import build_targets

print("  Testing target construction:")
token_to_id, _ = build_vocab(["CC(=O)O"])

# Test deletion target
x_t = [BOS, "C", "C", "O", EOS]
x1 = [BOS, "C", "C", EOS]
script = levenshtein_script(x_t, x1)
y_del, y_sub, y_ins, tok_target = build_targets(x_t, script, token_to_id)

assert y_del.sum() == 1, "Should have exactly one deletion target"
print(f"    ✓ Deletion target built")

# Test insertion target
x_t = [BOS, "C", "C", EOS]
x1 = [BOS, "C", "C", "O", EOS]
script = levenshtein_script(x_t, x1)
y_del, y_sub, y_ins, tok_target = build_targets(x_t, script, token_to_id)

assert y_ins.sum() == 1, "Should have exactly one insertion target"
assert tok_target is not None, "Token target should exist"
print(f"    ✓ Insertion target built")

# Summary
print("\n" + "=" * 60)
print("ALL CORE TESTS PASSED! ✓")
print("=" * 60)
print("\nCore modules verified:")
print("  ✓ Tokenizer: SMILES tokenization with special handling")
print("  ✓ Edit Distance: Levenshtein script computation")
print("  ✓ Corruption: Random sequence corruption")
print("  ✓ Targets: Training target construction")
print("\nTo run full training (requires torch and rdkit):")
print("  pip install torch rdkit")
print("  python3 smiles_editflow/train.py --tiny")
