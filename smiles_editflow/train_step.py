"""Training step logic for the edit flow model."""

import random
from typing import List, Dict
import torch

from .chemistry import randomized_smiles, canonicalize_smiles, is_valid_smiles
from .tokenizer import tokenize, add_special, encode, BOS, EOS, PAD
from .corruption import corrupt
from .edit_distance import levenshtein_script, apply_script
from .targets import build_targets
from .losses import compute_losses


def prepare_training_batch(
    batch_smiles: List[str],
    token_to_id: dict,
    id_to_token: dict,
    vocab_set: set,
    t_min: float = 0.05,
    t_max: float = 0.95,
    seed: int = None,
) -> Dict:
    """
    Prepare a training batch from SMILES strings.
    
    For each SMILES:
    1. Canonicalize and randomize (augmentation)
    2. Tokenize and add BOS/EOS
    3. Sample t ~ Uniform(t_min, t_max)
    4. Corrupt to get x_t
    5. Compute edit script from x_t to x1
    6. Build targets from first edit
    
    Args:
        batch_smiles: List of SMILES strings
        token_to_id: Token to ID mapping
        id_to_token: ID to token mapping
        vocab_set: Set of all tokens for corruption
        t_min: Minimum corruption level
        t_max: Maximum corruption level
        seed: Random seed
        
    Returns:
        Dictionary with batched data ready for training
    """
    if seed is not None:
        rng = random.Random(seed)
    else:
        rng = random.Random()
    
    batch_data = []
    
    for smiles in batch_smiles:
        # Canonicalize and randomize
        canonical = canonicalize_smiles(smiles)
        if canonical is None:
            continue
        
        randomized = randomized_smiles(canonical)
        if randomized is None:
            randomized = canonical
        
        # Tokenize
        tokens = tokenize(randomized)
        tokens_with_special = add_special(tokens)
        
        # Sample corruption level
        t = rng.uniform(t_min, t_max)
        
        # Corrupt
        x_t_tokens = corrupt(tokens_with_special, t, vocab_set, rng)
        
        # Compute edit script
        script = levenshtein_script(x_t_tokens, tokens_with_special)
        
        # Build targets
        y_del, y_sub, y_ins, tok_target = build_targets(x_t_tokens, script, token_to_id)
        
        # Sanity check: apply script should yield original
        if script:
            reconstructed = apply_script(x_t_tokens, script)
            if reconstructed != tokens_with_special:
                print(f"Warning: script application mismatch!")
                print(f"  x_t: {x_t_tokens}")
                print(f"  x1: {tokens_with_special}")
                print(f"  script: {script}")
                print(f"  reconstructed: {reconstructed}")
        
        batch_data.append((x_t_tokens, y_del, y_sub, y_ins, tok_target, t))
    
    if not batch_data:
        return None
    
    # Collate batch
    batch_size = len(batch_data)
    max_len = max(len(x_t) for x_t, _, _, _, _, _ in batch_data)
    
    pad_id = token_to_id[PAD]
    
    token_ids = torch.full((batch_size, max_len), pad_id, dtype=torch.long)
    attn_mask = torch.zeros((batch_size, max_len), dtype=torch.bool)
    y_del_batch = torch.zeros((batch_size, max_len), dtype=torch.float32)
    y_sub_batch = torch.zeros((batch_size, max_len), dtype=torch.float32)
    y_ins_batch = torch.zeros((batch_size, max_len + 1), dtype=torch.float32)
    t_batch = torch.zeros(batch_size, dtype=torch.float32)
    
    tok_targets = []
    
    for i, (x_t, y_del, y_sub, y_ins, tok_target, t) in enumerate(batch_data):
        seq_len = len(x_t)
        
        # Encode tokens
        ids = encode(x_t, token_to_id)
        token_ids[i, :seq_len] = torch.tensor(ids, dtype=torch.long)
        attn_mask[i, :seq_len] = True
        
        # Copy targets
        y_del_batch[i, :seq_len] = y_del
        y_sub_batch[i, :seq_len] = y_sub
        y_ins_batch[i, :seq_len + 1] = y_ins
        
        t_batch[i] = t
        
        tok_targets.append(tok_target)
    
    return {
        "token_ids": token_ids,
        "attn_mask": attn_mask,
        "y_del": y_del_batch,
        "y_sub": y_sub_batch,
        "y_ins": y_ins_batch,
        "tok_targets": tok_targets,
        "t": t_batch,
    }


def train_step(
    model,
    optimizer,
    batch_smiles: List[str],
    token_to_id: dict,
    id_to_token: dict,
    vocab_set: set,
    device: str = "cpu",
    alpha: float = 1.0,
) -> Dict:
    """
    Perform a single training step.
    
    Args:
        model: EditFlowModel
        optimizer: PyTorch optimizer
        batch_smiles: List of SMILES strings
        token_to_id: Token to ID mapping
        id_to_token: ID to token mapping
        vocab_set: Set of tokens for corruption
        device: Device to run on
        alpha: Weight for token loss
        
    Returns:
        Dictionary of losses and metrics
    """
    model.train()
    
    # Prepare batch
    batch = prepare_training_batch(
        batch_smiles,
        token_to_id,
        id_to_token,
        vocab_set,
    )
    
    if batch is None:
        return {"loss": 0.0, "error": "empty_batch"}
    
    # Move to device
    token_ids = batch["token_ids"].to(device)
    attn_mask = batch["attn_mask"].to(device)
    y_del = batch["y_del"].to(device)
    y_sub = batch["y_sub"].to(device)
    y_ins = batch["y_ins"].to(device)
    t = batch["t"].to(device)
    tok_targets = batch["tok_targets"]
    
    # Forward pass
    p_del, p_sub, p_ins, sub_tok_logits, ins_tok_logits = model(
        token_ids, attn_mask, t
    )
    
    # Compute losses
    losses = compute_losses(
        p_del, p_sub, p_ins,
        sub_tok_logits, ins_tok_logits,
        y_del, y_sub, y_ins,
        tok_targets,
        attn_mask,
        bos_id=token_to_id[BOS],
        eos_id=token_to_id[EOS],
        pad_id=token_to_id[PAD],
        alpha=alpha,
    )
    
    # Backward pass
    optimizer.zero_grad()
    losses["loss"].backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    
    # Convert to CPU for returning
    result = {k: v.item() if isinstance(v, torch.Tensor) else v for k, v in losses.items()}
    
    return result


if __name__ == "__main__":
    from .model import EditFlowModel
    from .tokenizer import build_vocab
    
    # Test training step
    print("Training step test:")
    
    # Sample SMILES
    smiles_list = [
        "CC(=O)O",
        "c1ccccc1",
        "CCO",
    ]
    
    # Build vocab
    token_to_id, id_to_token = build_vocab(smiles_list)
    vocab_set = set(token_to_id.keys())
    
    # Create model
    model = EditFlowModel(
        vocab_size=len(token_to_id),
        d_model=64,
        nhead=2,
        num_layers=1,
    )
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Run training step
    result = train_step(
        model,
        optimizer,
        smiles_list,
        token_to_id,
        id_to_token,
        vocab_set,
        device="cpu",
    )
    
    print(f"  Loss: {result['loss']:.4f}")
    print(f"  DEL loss: {result['loss_del']:.4f}")
    print(f"  SUB loss: {result['loss_sub']:.4f}")
    print(f"  INS loss: {result['loss_ins']:.4f}")
    print(f"  TOK loss: {result['loss_tok']:.4f}")
    
    print("\nTraining step test passed!")
