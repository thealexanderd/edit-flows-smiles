# SMILES Edit Flow

A sequence-editing pretraining pipeline for SMILES strings, inspired by the concept of "editing as a generative process". This implementation trains a transformer model to predict edit operations (insert, delete, substitute) that progressively refine corrupted molecular sequences toward valid target structures.

## Overview

Unlike traditional next-token language models, this model learns a **policy over edit operations**. Given a partially corrupted SMILES sequence `x_t` and a scalar corruption level `t ∈ (0,1)`, the model predicts the next edit operation that moves `x_t` closer to a valid target molecule.

### Key Features

- **SMILES-aware tokenization**: Handles bracket expressions, multi-digit ring closures, stereochemistry, and two-character atoms
- **Edit-based generation**: Models molecular generation as iterative sequence editing
- **Levenshtein-based supervision**: Uses minimal edit scripts for training targets
- **Transformer architecture**: Policy network with specialized heads for delete/substitute/insert operations
- **No third-party dependencies**: Rule-based tokenizer implemented from scratch

## Project Structure

```
smiles_editflow/
    __init__.py          # Package initialization
    tokenizer.py         # SMILES tokenization and vocabulary
    chemistry.py         # RDKit utilities for validation
    corruption.py        # Sequence corruption for x_t generation
    edit_distance.py     # Levenshtein distance and edit scripts
    targets.py           # Training target construction
    model.py             # Transformer encoder policy network
    losses.py            # Masked BCE and CE loss functions
    train_step.py        # Training step logic
    sampler.py           # Iterative editing sampler
    train.py             # Main training script

tests/
    test_tokenizer.py    # Tokenizer unit tests
    test_edit_distance.py # Edit distance tests
    test_apply_edit.py   # Edit application tests

README.md
```

## Installation

### Requirements

- Python 3.10+
- PyTorch
- RDKit
- pytest (for testing)

### Setup

```bash
# Install dependencies
pip install torch rdkit pytest

# Verify installation by running tests
python -m pytest tests/ -v
```

## Usage

### Quick Start - Tiny Mode

Test the pipeline on a small dataset to verify everything works:

```bash
# Run in tiny mode (50 molecules, 300 steps)
python smiles_editflow/train.py --tiny
```

This will:
- Use a built-in dataset of 50 molecules
- Train for 300 steps with frequent sampling
- Show loss decreasing and generate sample molecules

### Training on Custom Data

Prepare a text file with one SMILES string per line:

```bash
# Example: data/smiles.txt
CC(=O)O
c1ccccc1
CCO
CC(C)O
# ... more SMILES
```

Run training:

```bash
python smiles_editflow/train.py \
    --data data/smiles.txt \
    --max-data 1000 \
    --batch-size 8 \
    --steps 500 \
    --d-model 128 \
    --nhead 4 \
    --num-layers 3 \
    --lr 0.0003 \
    --sample-every 100 \
    --device cpu
```

### Training Parameters

- `--data`: Path to SMILES data file (default: `data/smiles.txt`)
- `--max-data`: Maximum number of SMILES to use (default: 1000)
- `--batch-size`: Training batch size (default: 8)
- `--steps`: Number of training steps (default: 500)
- `--d-model`: Model dimension (default: 128)
- `--nhead`: Number of attention heads (default: 4)
- `--num-layers`: Number of transformer layers (default: 3)
- `--lr`: Learning rate (default: 0.0003)
- `--sample-every`: Sample molecules every N steps (default: 100)
- `--device`: Device to use (`cpu` or `cuda`)
- `--tiny`: Use tiny mode for quick testing
- `--save-dir`: Directory to save model checkpoints (default: `checkpoints`)

### Expected Output

During training, you'll see:

```
Step    0 | Loss: 2.1234 | DEL: 0.693 | SUB: 0.693 | INS: 0.693 | TOK: 1.234 | EditAcc: 0.125 | TokAcc: 0.000
Step   10 | Loss: 1.8765 | DEL: 0.612 | SUB: 0.645 | INS: 0.621 | TOK: 0.998 | EditAcc: 0.250 | TokAcc: 0.125
...

============================================================
Sampling molecules at step 100:
------------------------------------------------------------
  1. [✓] CCO
  2. [✗] C(C)C)O
  3. [✓] c1ccccc1
  4. [✓] CC(=O)O
  5. [✗] CC(C
  
Valid: 3/5
============================================================
```

The model learns to:
1. Predict correct edit types (delete/substitute/insert)
2. Choose correct positions for edits
3. Select appropriate tokens for substitutions/insertions
4. Generate valid SMILES through iterative editing

## Testing

Run all unit tests:

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test file
python -m pytest tests/test_tokenizer.py -v

# Run with coverage
python -m pytest tests/ -v --cov=smiles_editflow
```

### Test Coverage

- **Tokenizer tests**: Bracket expressions, multi-digit ring closures, two-char atoms, stereochemistry
- **Edit distance tests**: Script generation, application correctness, determinism
- **Apply edit tests**: Operation correctness, BOS/EOS preservation, boundary cases

## Implementation Details

### Tokenization

The tokenizer handles complex SMILES features:

```python
from smiles_editflow.tokenizer import tokenize, detokenize

smiles = "C[NH3+]%10CCC%10"
tokens = tokenize(smiles)
# ['C', '[NH3+]', '%10', 'C', 'C', 'C', '%10']

reconstructed = detokenize(tokens)
# 'C[NH3+]%10CCC%10'
```

### Corruption Process

Given a target sequence `x1`, create corrupted `x_t`:

```python
from smiles_editflow.corruption import corrupt
from smiles_editflow.tokenizer import add_special

tokens = add_special(tokenize("CCO"))  # [BOS, C, C, O, EOS]
vocab = {"C", "N", "O", "S", "P"}

x_t = corrupt(tokens, t=0.5, vocab=vocab, rng=random.Random(42))
# Example: [BOS, C, N, EOS] (deleted O, substituted C->N)
```

### Edit Operations

Three edit types with specific index semantics:

- **DEL(i)**: Delete token at position `i` in current sequence
- **SUB(i, tok)**: Substitute token at position `i` with `tok`
- **INS(g, tok)**: Insert `tok` at gap `g` (before token at index `g`)

BOS and EOS tokens are never modified.

### Model Architecture

```
Input: token_ids [B, S], attn_mask [B, S], t [B]
  ↓
Token Embedding + Positional Encoding + Time Embedding
  ↓
Transformer Encoder (N layers)
  ↓
  ├─→ Delete head: p_del [B, S]
  ├─→ Substitute head: p_sub [B, S], sub_tok_logits [B, S, V]
  └─→ Insert head: p_ins [B, S+1], ins_tok_logits [B, S+1, V]
```

### Training Objective

For each training example:

1. Canonicalize and randomize SMILES (augmentation)
2. Tokenize to get `x1`
3. Sample corruption level `t ~ Uniform(0.05, 0.95)`
4. Corrupt to get `x_t`
5. Compute minimal edit script from `x_t` to `x1`
6. Take **first edit** as supervision (teacher forcing)
7. Train with masked BCE + CE losses

### Sampling Process

Generate molecules via iterative editing:

```python
from smiles_editflow.sampler import sample_molecule

smiles, is_valid, intermediates = sample_molecule(
    model,
    token_to_id,
    id_to_token,
    max_steps=50,
    sampling_strategy="sample",  # or "argmax"
    temperature=1.0,
    verbose=True
)
```

Starting from `[BOS, EOS]`, the model iteratively applies edits until convergence or max steps.

## Limitations and Future Work

Current implementation:

- ✅ Handles complex SMILES tokenization
- ✅ Levenshtein-based supervision
- ✅ Masked loss computation
- ✅ Iterative sampling with validity checking
- ⚠️ Small models (for demonstration)
- ⚠️ Simple time scheduling

Potential improvements:

- Larger models and datasets
- Better time scheduling strategies
- Multi-step training (predict multiple edits)
- Curriculum learning (easier molecules first)
- Property-conditional generation
- Reinforcement learning fine-tuning

## Citation

This implementation is inspired by the "Edit Flows" concept but is an independent implementation built from scratch for molecular sequence editing.

## License

MIT License - See LICENSE file for details

## Troubleshooting

### Import Errors

If you see `ModuleNotFoundError`, ensure you're running from the project root:

```bash
# From editflows/ directory
python smiles_editflow/train.py --tiny
```

### RDKit Installation Issues

If RDKit installation fails:

```bash
# Try conda
conda install -c conda-forge rdkit

# Or use rdkit-pypi
pip install rdkit-pypi
```

### Out of Memory

Reduce model size or batch size:

```bash
python smiles_editflow/train.py \
    --d-model 64 \
    --num-layers 2 \
    --batch-size 4
```

## Contact

For questions or issues, please open a GitHub issue.
