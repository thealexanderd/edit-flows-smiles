"""Main training script for SMILES edit flow model."""

import os
import argparse
import torch
import torch.optim as optim
import math
from collections import Counter
from pathlib import Path

from smiles_editflow.tokenizer import build_vocab, tokenize, SPECIAL_TOKENS
from smiles_editflow.model import EditFlowModel
from smiles_editflow.train_step import train_step
from smiles_editflow.sampler import sample_molecule
from smiles_editflow.chemistry import filter_smiles, canonicalize_smiles


MODEL_PRESETS = {
    # Paper-scale targets. This implementation uses a PyTorch Transformer encoder,
    # not a Llama architecture, so this is an approximate size match.
    "paper_280m": {
        "d_model": 1024,
        "nhead": 16,
        "num_layers": 12,
        "dim_feedforward": 6963,
        "max_len": 1024,
        "aligned_length": 1024,
        "batch_size": 4096,
        "steps": 500000,
        "vocab_size": 32000,
        "lr": 3e-4,
        "warmup_steps": 2000,
        "lr_schedule": "cosine",
    },
    "paper_1_3b": {
        "d_model": 2048,
        "nhead": 32,
        "num_layers": 16,
        "dim_feedforward": 12288,
        "max_len": 1024,
        "aligned_length": 1024,
        "batch_size": 4096,
        "steps": 500000,
        "vocab_size": 32000,
        "lr": 3e-4,
        "warmup_steps": 2000,
        "lr_schedule": "cosine",
    },
}


def load_smiles_dataset(file_path: str, max_samples: int = None) -> list:
    """
    Load SMILES from a text file (one per line).
    
    Args:
        file_path: Path to SMILES file
        max_samples: Maximum number of samples to load
        
    Returns:
        List of SMILES strings
    """
    smiles_list = []
    
    if not os.path.exists(file_path):
        print(f"Warning: {file_path} not found, using built-in tiny dataset")
        # Tiny built-in dataset for testing
        return [
            "CC(=O)O",  # Acetic acid
            "c1ccccc1",  # Benzene
            "CCO",  # Ethanol
            "CC(C)O",  # Isopropanol
            "c1ccc(O)cc1",  # Phenol
            "CC(=O)C",  # Acetone
            "c1cccnc1",  # Pyridine
            "CC(N)C(=O)O",  # Alanine
            "c1cc(O)ccc1O",  # Hydroquinone
            "CCCC",  # Butane
        ] * 5  # Replicate for more training data
    
    with open(file_path, 'r') as f:
        for line in f:
            smiles = line.strip()
            if smiles and not smiles.startswith('#'):
                smiles_list.append(smiles)
                if max_samples and len(smiles_list) >= max_samples:
                    break
    
    return smiles_list


def main():
    parser = argparse.ArgumentParser(description="Train SMILES Edit Flow model")
    parser.add_argument(
        "--model-config",
        type=str,
        default="paper_280m",
        choices=["paper_280m", "paper_1_3b", "custom"],
        help="Model/training preset. 'custom' uses explicit CLI values.",
    )
    parser.add_argument("--data", type=str, default="data/smiles.txt", help="Path to SMILES data file")
    parser.add_argument("--max-data", type=int, default=1000, help="Maximum number of SMILES to use")
    parser.add_argument("--vocab-size", type=int, default=500, help="Build vocab from first N molecules")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--steps", type=int, default=500, help="Number of training steps")
    parser.add_argument("--d-model", type=int, default=128, help="Model dimension")
    parser.add_argument("--nhead", type=int, default=4, help="Number of attention heads")
    parser.add_argument("--num-layers", type=int, default=3, help="Number of transformer layers")
    parser.add_argument("--dim-feedforward", type=int, default=1024, help="Transformer feedforward dimension")
    parser.add_argument("--dropout", type=float, default=0.1, help="Transformer dropout")
    parser.add_argument("--max-len", type=int, default=512, help="Maximum sequence length for positional encoding")
    parser.add_argument("--lr", type=float, default=0.0003, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.0, help="Weight decay for AdamW")
    parser.add_argument("--warmup-steps", type=int, default=2000, help="Warmup steps for LR schedule")
    parser.add_argument("--lr-schedule", type=str, default="cosine", choices=["cosine", "constant"], help="Learning rate schedule")
    parser.add_argument("--sample-every", type=int, default=100, help="Sample molecules every N steps")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu or cuda)")
    parser.add_argument("--tiny", action="store_true", help="Tiny mode: overfit on 50 molecules")
    parser.add_argument("--save-dir", type=str, default="checkpoints", help="Directory to save model")
    parser.add_argument("--aligned-length", type=int, default=160, help="Aligned length N for edit flows")
    parser.add_argument(
        "--x0-mode",
        type=str,
        default="uniform_halfhalf",
        choices=["uniform_halfhalf", "uniform", "empty"],
        help="x0 initialization mode",
    )
    parser.add_argument("--x0-max-len", type=int, default=32, help="Max length for sampled x0 in uniform modes")
    parser.add_argument("--beta", type=float, default=1.0, help="Rate regularizer for edit flows")
    parser.add_argument("--kappa-power", type=int, default=3, help="Power for kappa(t)=t^power")
    
    args = parser.parse_args()

    if args.model_config != "custom":
        preset = MODEL_PRESETS[args.model_config]
        for key, value in preset.items():
            setattr(args, key, value)

    print("=" * 60)
    print("SMILES Edit Flow Training")
    print("=" * 60)
    if args.model_config != "custom":
        print(f"Preset: {args.model_config} (paper-scale approximation)")
        print("Note: architecture is Transformer encoder, not Llama/FlexAttention.")
    
    # Load data
    print(f"\nLoading SMILES from {args.data}...")
    all_smiles = load_smiles_dataset(args.data, args.max_data)
    print(f"Loaded {len(all_smiles)} SMILES")
    
    # Filter valid SMILES
    print("Filtering valid SMILES...")
    valid_smiles = []
    for smi in all_smiles:
        canonical = canonicalize_smiles(smi)
        if canonical and filter_smiles(canonical, max_len=80):
            valid_smiles.append(canonical)
    
    print(f"Kept {len(valid_smiles)} valid SMILES")
    
    if len(valid_smiles) == 0:
        print("Error: No valid SMILES found!")
        return
    
    # Tiny mode: use only 50 molecules
    if args.tiny:
        print("\n*** TINY MODE: Using only 50 molecules for overfitting test ***")
        valid_smiles = valid_smiles[:50]
        args.steps = 300
        args.sample_every = 50

    # Build empirical token marginal p_emp from training corpus.
    token_counter = Counter()
    for smi in valid_smiles:
        for tok in tokenize(smi):
            if tok not in SPECIAL_TOKENS:
                token_counter[tok] += 1

    if not token_counter:
        print("Error: Empirical token distribution is empty. Check training data/tokenizer.")
        return

    emp_tokens = sorted(token_counter.keys())
    total_count = sum(token_counter[t] for t in emp_tokens)
    emp_weights = [token_counter[t] / total_count for t in emp_tokens]
    print(f"Empirical token support: {len(emp_tokens)}")
    
    # Build vocabulary
    print(f"\nBuilding vocabulary from {min(args.vocab_size, len(valid_smiles))} molecules...")
    vocab_smiles = valid_smiles[:args.vocab_size]
    token_to_id, id_to_token = build_vocab(vocab_smiles)
    vocab_set = set(token_to_id.keys())
    
    print(f"Vocabulary size: {len(token_to_id)}")
    print(f"Sample tokens: {list(token_to_id.keys())[:20]}")
    
    # Create model
    print(f"\nCreating model...")
    model = EditFlowModel(
        vocab_size=len(token_to_id),
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        max_len=args.max_len,
    )
    model.to(args.device)
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")
    
    # Create optimizer (paper uses AdamW with betas 0.9, 0.95)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.95),
        weight_decay=args.weight_decay,
    )

    # Learning rate schedule with warmup + cosine decay (paper default)
    if args.lr_schedule == "cosine":
        warmup = max(0, args.warmup_steps)
        total = max(1, args.steps)

        def lr_lambda(step: int) -> float:
            if warmup > 0 and step < warmup:
                return float(step + 1) / float(warmup)
            if total <= warmup:
                return 1.0
            progress = (step - warmup) / float(total - warmup)
            progress = min(max(progress, 0.0), 1.0)
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    else:
        scheduler = None
    
    # Training loop
    print(f"\nStarting training for {args.steps} steps...")
    print("-" * 60)
    
    import random
    
    for step in range(args.steps):
        # Sample batch
        batch_smiles = random.sample(valid_smiles, min(args.batch_size, len(valid_smiles)))
        
        # Training step
        result = train_step(
            model,
            optimizer,
            batch_smiles,
            token_to_id,
            id_to_token,
            vocab_set,
            device=args.device,
            aligned_length=args.aligned_length,
            x0_mode=args.x0_mode,
            x0_max_len=args.x0_max_len,
            kappa_power=args.kappa_power,
            beta=args.beta,
            emp_tokens=emp_tokens,
            emp_weights=emp_weights,
        )
        
        # Log
        if step % 10 == 0:
            print(f"Step {step:4d} | Loss: {result['loss']:.4f} | "
                  f"DEL: {result['loss_del']:.3f} | SUB: {result['loss_sub']:.3f} | "
                  f"INS: {result['loss_ins']:.3f} | TOK: {result['loss_tok']:.3f} | "
                  f"EditAcc: {result.get('edit_type_acc', 0.0):.3f} | TokAcc: {result.get('tok_acc', 0.0):.3f}")

        if scheduler is not None:
            scheduler.step()
        
        # Sample molecules
        if (step + 1) % args.sample_every == 0:
            print("\n" + "=" * 60)
            print(f"Sampling molecules at step {step + 1}:")
            print("-" * 60)
            
            valid_count = 0
            for i in range(5):
                smiles, is_valid, intermediates = sample_molecule(
                    model,
                    token_to_id,
                    id_to_token,
                    device=args.device,
                    max_steps=30,
                    temperature=1.0,
                    verbose=False,
                )
                
                status = "✓" if is_valid else "✗"
                print(f"  {i+1}. [{status}] {smiles}")
                if is_valid:
                    valid_count += 1
            
            print(f"\nValid: {valid_count}/5")
            print("=" * 60 + "\n")
    
    # Save model
    print(f"\nTraining complete!")
    
    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)
        model_path = os.path.join(args.save_dir, "editflow_model.pt")
        torch.save({
            "model_state_dict": model.state_dict(),
            "token_to_id": token_to_id,
            "id_to_token": id_to_token,
            "args": vars(args),
        }, model_path)
        print(f"Model saved to {model_path}")
    
    # Final sampling
    print("\n" + "=" * 60)
    print("Final sampling (10 molecules):")
    print("-" * 60)
    
    valid_count = 0
    valid_molecules = []
    
    for i in range(10):
        smiles, is_valid, intermediates = sample_molecule(
            model,
            token_to_id,
            id_to_token,
            device=args.device,
            max_steps=40,
            temperature=0.8,
            verbose=False,
        )
        
        status = "✓" if is_valid else "✗"
        print(f"  {i+1:2d}. [{status}] {smiles}")
        if is_valid:
            valid_count += 1
            valid_molecules.append(smiles)
    
    print(f"\nValid: {valid_count}/10")
    
    if valid_molecules:
        print("\nValid molecules generated:")
        for i, smi in enumerate(valid_molecules[:5], 1):
            print(f"  {i}. {smi}")
    
    print("=" * 60)


if __name__ == "__main__":
    main()
