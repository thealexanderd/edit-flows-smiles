"""Training step logic for the edit flow model."""

import random
from typing import List, Dict, Optional, Tuple
import torch
import torch.nn.functional as F

from .chemistry import randomized_smiles, canonicalize_smiles
from .tokenizer import tokenize, add_special, encode, BOS, EOS, PAD, UNK
from .edit_distance import align_with_epsilon
from .alignment import make_alignment_fixed_N, sample_z_t, strip_epsilon, extract_targets
from .losses import compute_losses_editflows
from .masking import mask_token_logits


def kappa(t: float, power: int = 3) -> float:
    """Default schedule κ(t) = t^power."""
    return t ** power


def kappa_derivative(t: torch.Tensor, power: int = 3) -> torch.Tensor:
    """Derivative of κ(t) = t^power with respect to t."""
    return power * (t ** (power - 1))


def sample_t_open_interval(
    rng: random.Random,
    t_min: float,
    t_max: float,
    eps: float = 1e-6,
) -> float:
    """
    Sample t strictly within (t_min, t_max). If t_min == t_max, return t_min.
    """
    if t_min == t_max:
        t = t_min
    else:
        lower = t_min + eps
        upper = t_max - eps
        if lower >= upper:
            lower = t_min
            upper = t_max
        if lower == upper:
            t = lower
        else:
            t = rng.uniform(lower, upper)

    if t <= eps:
        return eps
    if t >= 1.0 - eps:
        return 1.0 - eps
    return t


def prepare_training_batch(
    batch_smiles: List[str],
    token_to_id: dict,
    id_to_token: dict,
    vocab_set: set,
    t_min: float = 0.0,
    t_max: float = 1.0,
    seed: int = None,
    aligned_length: int = 160,
    x0_mode: str = "uniform",
    x0_max_len: int = 32,
    kappa_power: int = 3,
) -> Optional[Dict]:
    """
    Prepare a training batch from SMILES strings.

    Uses epsilon-alignment + rate-based supervision (Edit Flows).
    Legacy teacher-forced batching is in smiles_editflow/legacy/teacher_forced.py.
    """
    if seed is not None:
        rng = random.Random(seed)
    else:
        rng = random.Random()

    vocab_tokens = [
        tok for tok in vocab_set
        if tok not in {BOS, EOS, PAD, UNK}
    ]

    batch_data = []

    for smiles in batch_smiles:
        canonical = canonicalize_smiles(smiles)
        if canonical is None:
            continue

        randomized = randomized_smiles(canonical)
        if randomized is None:
            randomized = canonical

        tokens = tokenize(randomized)

        t = sample_t_open_interval(rng, t_min, t_max)

        x1_tokens = tokens
        if x0_mode == "empty":
            x0_tokens = []
        else:
            max_len = max(0, x0_max_len)
            L0 = rng.randint(0, max_len)
            if vocab_tokens:
                x0_tokens = [rng.choice(vocab_tokens) for _ in range(L0)]
            else:
                x0_tokens = []

        a0, a1 = align_with_epsilon(x0_tokens, x1_tokens)
        if len(a0) > aligned_length:
            continue
        z0, z1 = make_alignment_fixed_N(a0, a1, aligned_length, rng)
        z_t = sample_z_t(z0, z1, t, lambda u: kappa(u, kappa_power), rng)

        x_t_interior = strip_epsilon(z_t)
        x_t_tokens = add_special(x_t_interior)

        del_pos, sub_pairs, ins_pairs = extract_targets(z_t, z1)

        del_pos_full = [i + 1 for i in del_pos]
        sub_targets_full: List[Tuple[int, int]] = []
        for i, tok in sub_pairs:
            tok_id = token_to_id.get(tok, token_to_id.get(UNK))
            sub_targets_full.append((i + 1, tok_id))
        ins_targets_full: List[Tuple[int, int]] = []
        for g, tok in ins_pairs:
            tok_id = token_to_id.get(tok, token_to_id.get(UNK))
            ins_targets_full.append((g + 1, tok_id))

        batch_data.append((x_t_tokens, del_pos_full, sub_targets_full, ins_targets_full, t))

    if not batch_data:
        return None

    batch_size = len(batch_data)
    max_len = max(len(x_t) for x_t, *_ in batch_data)
    pad_id = token_to_id[PAD]

    token_ids = torch.full((batch_size, max_len), pad_id, dtype=torch.long)
    attn_mask = torch.zeros((batch_size, max_len), dtype=torch.bool)
    t_batch = torch.zeros(batch_size, dtype=torch.float32)

    del_targets = []
    sub_targets = []
    ins_targets = []

    for i, (x_t, del_pos_full, sub_targets_full, ins_targets_full, t) in enumerate(batch_data):
        seq_len = len(x_t)
        ids = encode(x_t, token_to_id)
        token_ids[i, :seq_len] = torch.tensor(ids, dtype=torch.long)
        attn_mask[i, :seq_len] = True
        t_batch[i] = t
        del_targets.append(del_pos_full)
        sub_targets.append(sub_targets_full)
        ins_targets.append(ins_targets_full)

    return {
        "token_ids": token_ids,
        "attn_mask": attn_mask,
        "del_targets": del_targets,
        "sub_targets": sub_targets,
        "ins_targets": ins_targets,
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
    aligned_length: int = 160,
    x0_mode: str = "uniform",
    x0_max_len: int = 32,
    kappa_power: int = 3,
    beta: float = 1.0,
) -> Dict:
    """
    Perform a single training step.

    Edit Flows only (legacy teacher-forced path lives in
    smiles_editflow/legacy/teacher_forced.py).
    
    Args:
        model: EditFlowModel
        optimizer: PyTorch optimizer
        batch_smiles: List of SMILES strings
        token_to_id: Token to ID mapping
        id_to_token: ID to token mapping
        vocab_set: Set of tokens for x0 sampling
        device: Device to run on
        beta: Rate regularizer for Edit Flows
        
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
        aligned_length=aligned_length,
        x0_mode=x0_mode,
        x0_max_len=x0_max_len,
        kappa_power=kappa_power,
    )
    
    if batch is None:
        return {"loss": 0.0, "error": "empty_batch"}
    
    # Move to device
    token_ids = batch["token_ids"].to(device)
    attn_mask = batch["attn_mask"].to(device)
    t = batch["t"].to(device)
    
    # Forward pass
    del_logits, sub_logits, ins_logits, sub_tok_logits, ins_tok_logits = model(
        token_ids, attn_mask, t
    )

    forbidden_ids = [token_to_id[BOS], token_to_id[EOS], token_to_id[PAD], token_to_id[UNK]]
    sub_tok_logits = mask_token_logits(sub_tok_logits, forbidden_ids)
    ins_tok_logits = mask_token_logits(ins_tok_logits, forbidden_ids)

    del_rates = F.softplus(del_logits)
    sub_rates = F.softplus(sub_logits)
    ins_rates = F.softplus(ins_logits)

    del_targets = batch["del_targets"]
    sub_targets = batch["sub_targets"]
    ins_targets = batch["ins_targets"]

    kappa_val = t ** kappa_power
    pos_weight = kappa_derivative(t, power=kappa_power) / (1.0 - kappa_val + 1e-8)

    losses = compute_losses_editflows(
        del_rates, sub_rates, ins_rates,
        sub_tok_logits, ins_tok_logits,
        del_targets, sub_targets, ins_targets,
        attn_mask,
        bos_id=token_to_id[BOS],
        eos_id=token_to_id[EOS],
        pad_id=token_to_id[PAD],
        rate_weight=beta,
        pos_weight=pos_weight,
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
