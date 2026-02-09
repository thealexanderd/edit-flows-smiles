# SMILES Edit Flow

A sequence-editing pretraining pipeline for SMILES strings, inspired by the concept of "editing as a generative process". This implementation trains a transformer model to predict edit operations (insert, delete, substitute) that progressively refine intermediate sequences toward valid target structures.

## Overview

Unlike traditional next-token language models, this model learns a **policy over edit operations**. Given an intermediate sequence `x_t` sampled from an epsilon-aligned mixture of `x0` and `x1` and a scalar time `t ∈ (0,1)`, the model predicts edit **rates** and token distributions for insertions, deletions, and substitutions.

### Key Features

- **SMILES-aware tokenization**: Handles bracket expressions, multi-digit ring closures, stereochemistry, and two-character atoms
- **Edit-based generation**: Models molecular generation as iterative sequence editing
- **Epsilon alignment supervision**: Builds multi-target edits from aligned `(x0, x1)` pairs
- **Rate-based training**: Learns CTMC-style edit rates with position and token heads
- **Transformer architecture**: Policy network with specialized heads for delete/substitute/insert operations
- **No third-party dependencies**: Rule-based tokenizer implemented from scratch

## Project Structure

```
smiles_editflow/
    __init__.py          # Package initialization
    tokenizer.py         # SMILES tokenization and vocabulary
    chemistry.py         # RDKit utilities for validation
    corruption.py        # Legacy corruption utilities (teacher-forced)
    edit_distance.py     # Levenshtein distance and edit scripts
    targets.py           # Training target construction
    model.py             # Transformer encoder policy network
    masking.py           # Token logit masking utilities
    losses.py            # Rate-based losses (plus legacy teacher-forced)
    train_step.py        # Training step logic
    sampler.py           # Iterative editing sampler
    train.py             # Main training script
    legacy/teacher_forced.py # Deprecated teacher-forced pipeline

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
- `--aligned-length`: Aligned length `N` for epsilon alignment (default: 160)
- `--x0-mode`: Source init mode (`uniform_halfhalf`, `uniform`, or `empty`)
- `--x0-max-len`: Max length for sampled `x0` in `uniform_halfhalf`/`uniform`
- `--beta`: Rate regularizer for Edit Flows
- `--kappa-power`: Power for `kappa(t) = t^power`

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

### Alignment and z_t Sampling

Given a source sequence `x0` and target `x1`, build epsilon-aligned pairs and sample `z_t`:

```python
from smiles_editflow.edit_distance import align_with_epsilon
from smiles_editflow.alignment import make_alignment_fixed_N, sample_z_t, strip_epsilon
from smiles_editflow.train_step import kappa
from smiles_editflow.tokenizer import tokenize

x0 = ["C"]  # example source
x1 = tokenize("CCO")

a0, a1 = align_with_epsilon(x0, x1)
z0, z1 = make_alignment_fixed_N(a0, a1, N=8)
z_t = sample_z_t(z0, z1, t=0.5, kappa=lambda u: kappa(u, 3))
x_t = strip_epsilon(z_t)
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
  ├─→ Delete head: λ_del [B, S]
  ├─→ Substitute head: λ_sub [B, S], sub_tok_logits [B, S, V]
  └─→ Insert head: λ_ins [B, S+1], ins_tok_logits [B, S+1, V]
```

### Training Objective

For each training example:

1. Canonicalize and randomize SMILES (augmentation)
2. Tokenize to get `x1`
3. Sample `x0`:
   - `empty`: `x0 = []`
   - `uniform`: random tokens sampled uniformly from vocab
   - `uniform_halfhalf` (paper variant): tokens sampled from empirical token marginal `p_emp`
4. Compute epsilon alignment `(a0, a1)` and pad to fixed length `N`
   - In `uniform_halfhalf`, alignment is forced to paper-style half/half:
     first `floor(L0/2)` `x0` tokens delete, remaining `ceil(L0/2)` substitute,
     and leftover `x1` tokens insert.
5. Sample `t` in `(0,1)` and draw `z_t` via `kappa(t) = t^3`
6. Strip epsilons to get `x_t`
7. Extract multiple edit targets from `(z_t, z1)`
8. Train with rate-based loss: `-log λ` on targets + `beta * sum λ` + token CE, weighted by `w(t) = dκ/dt`

### Sampling Process

Generate molecules via iterative editing:

```python
from smiles_editflow.sampler import sample_molecule

smiles, is_valid, intermediates = sample_molecule(
    model,
    token_to_id,
    id_to_token,
    max_steps=50,
    temperature=1.0,
    verbose=True
)
```

Starting from `[BOS, EOS]`, the model iteratively samples a single edit proportional to the predicted rates until convergence or max steps.

## Limitations and Future Work

Current implementation:

- ✅ Handles complex SMILES tokenization
- ✅ Epsilon alignment supervision
- ✅ Rate-based loss computation
- ✅ Iterative sampling with validity checking
- ⚠️ Small models (for demonstration)
- ⚠️ Simple time scheduling

Potential improvements:

- Larger models and datasets
- Richer time scheduling strategies
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
