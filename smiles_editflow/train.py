"""Main training script for SMILES edit flow model."""

import os
import argparse
import torch
import torch.optim as optim
from pathlib import Path

from smiles_editflow.tokenizer import build_vocab
from smiles_editflow.model import EditFlowModel
from smiles_editflow.train_step import train_step
from smiles_editflow.sampler import sample_molecule
from smiles_editflow.chemistry import filter_smiles, canonicalize_smiles


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
    parser.add_argument("--data", type=str, default="data/smiles.txt", help="Path to SMILES data file")
    parser.add_argument("--max-data", type=int, default=1000, help="Maximum number of SMILES to use")
    parser.add_argument("--vocab-size", type=int, default=500, help="Build vocab from first N molecules")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--steps", type=int, default=500, help="Number of training steps")
    parser.add_argument("--d-model", type=int, default=128, help="Model dimension")
    parser.add_argument("--nhead", type=int, default=4, help="Number of attention heads")
    parser.add_argument("--num-layers", type=int, default=3, help="Number of transformer layers")
    parser.add_argument("--lr", type=float, default=0.0003, help="Learning rate")
    parser.add_argument("--sample-every", type=int, default=100, help="Sample molecules every N steps")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu or cuda)")
    parser.add_argument("--tiny", action="store_true", help="Tiny mode: overfit on 50 molecules")
    parser.add_argument("--save-dir", type=str, default="checkpoints", help="Directory to save model")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("SMILES Edit Flow Training")
    print("=" * 60)
    
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
    )
    model.to(args.device)
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")
    
    # Create optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
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
            alpha=1.0,
        )
        
        # Log
        if step % 10 == 0:
            print(f"Step {step:4d} | Loss: {result['loss']:.4f} | "
                  f"DEL: {result['loss_del']:.3f} | SUB: {result['loss_sub']:.3f} | "
                  f"INS: {result['loss_ins']:.3f} | TOK: {result['loss_tok']:.3f} | "
                  f"EditAcc: {result['edit_type_acc']:.3f} | TokAcc: {result['tok_acc']:.3f}")
        
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
                    sampling_strategy="sample" if step > 100 else "argmax",
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
            sampling_strategy="sample",
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
