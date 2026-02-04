"""Training step logic for the edit flow model."""

import random
from typing import List, Dict, Optional, Tuple
import torch
import torch.nn.functional as F

from .chemistry import randomized_smiles, canonicalize_smiles
from .tokenizer import tokenize, add_special, encode, BOS, EOS, PAD, UNK
from .corruption import corrupt
from .targets import build_targets
from .edit_distance import levenshtein_script, apply_script, align_with_epsilon
from .alignment import make_alignment_fixed_N, sample_z_t, strip_epsilon, extract_targets
from .losses import compute_losses_teacher_forced, compute_losses_editflows


def kappa(t: float, power: int = 3) -> float:
    """Default schedule κ(t) = t^power."""
    return t ** power


def kappa_derivative(t: torch.Tensor, power: int = 3) -> torch.Tensor:
    """Derivative of κ(t) = t^power with respect to t."""
    return power * (t ** (power - 1))


def prepare_training_batch(
    batch_smiles: List[str],
    token_to_id: dict,
    id_to_token: dict,
    vocab_set: set,
    t_min: float = 0.0,
    t_max: float = 1.0,
    seed: int = None,
    mode: str = "editflows",
    aligned_length: int = 160,
    x0_mode: str = "uniform",
    x0_max_len: int = 32,
    kappa_power: int = 3,
) -> Optional[Dict]:
    """
    Prepare a training batch from SMILES strings.

    Supports:
        - teacher_forced: corruption + first-edit supervision
        - editflows: epsilon-alignment + rate-based supervision
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

        t = rng.uniform(t_min, t_max)

        if mode == "teacher_forced":
            tokens_with_special = add_special(tokens)
            x_t_tokens = corrupt(tokens_with_special, t, vocab_set, rng)
            script = levenshtein_script(x_t_tokens, tokens_with_special)
            y_del, y_sub, y_ins, tok_target = build_targets(x_t_tokens, script, token_to_id)

            if script:
                reconstructed = apply_script(x_t_tokens, script)
                if reconstructed != tokens_with_special:
                    print("Warning: script application mismatch!")
                    print(f"  x_t: {x_t_tokens}")
                    print(f"  x1: {tokens_with_special}")
                    print(f"  script: {script}")
                    print(f"  reconstructed: {reconstructed}")

            batch_data.append((x_t_tokens, y_del, y_sub, y_ins, tok_target, t))
        else:
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

    if mode == "teacher_forced":
        y_del_batch = torch.zeros((batch_size, max_len), dtype=torch.float32)
        y_sub_batch = torch.zeros((batch_size, max_len), dtype=torch.float32)
        y_ins_batch = torch.zeros((batch_size, max_len + 1), dtype=torch.float32)
        tok_targets = []

        for i, (x_t, y_del, y_sub, y_ins, tok_target, t) in enumerate(batch_data):
            seq_len = len(x_t)
            ids = encode(x_t, token_to_id)
            token_ids[i, :seq_len] = torch.tensor(ids, dtype=torch.long)
            attn_mask[i, :seq_len] = True
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
    alpha: float = 1.0,
    mode: str = "editflows",
    aligned_length: int = 160,
    x0_mode: str = "uniform",
    x0_max_len: int = 32,
    kappa_power: int = 3,
    beta: float = 1e-3,
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
        mode=mode,
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
    tok_targets = batch.get("tok_targets")
    
    # Forward pass
    del_logits, sub_logits, ins_logits, sub_tok_logits, ins_tok_logits = model(
        token_ids, attn_mask, t
    )
    
    if mode == "teacher_forced":
        y_del = batch["y_del"].to(device)
        y_sub = batch["y_sub"].to(device)
        y_ins = batch["y_ins"].to(device)

        p_del = torch.sigmoid(del_logits)
        p_sub = torch.sigmoid(sub_logits)
        p_ins = torch.sigmoid(ins_logits)

        losses = compute_losses_teacher_forced(
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
    else:
        del_rates = F.softplus(del_logits)
        sub_rates = F.softplus(sub_logits)
        ins_rates = F.softplus(ins_logits)

        del_targets = batch["del_targets"]
        sub_targets = batch["sub_targets"]
        ins_targets = batch["ins_targets"]

        t_weight = kappa_derivative(t, power=kappa_power)

        losses = compute_losses_editflows(
            del_rates, sub_rates, ins_rates,
            sub_tok_logits, ins_tok_logits,
            del_targets, sub_targets, ins_targets,
            attn_mask,
            bos_id=token_to_id[BOS],
            eos_id=token_to_id[EOS],
            pad_id=token_to_id[PAD],
            beta=beta,
            t_weight=t_weight,
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
        mode="editflows",
    )
    
    print(f"  Loss: {result['loss']:.4f}")
    print(f"  DEL loss: {result['loss_del']:.4f}")
    print(f"  SUB loss: {result['loss_sub']:.4f}")
    print(f"  INS loss: {result['loss_ins']:.4f}")
    print(f"  TOK loss: {result['loss_tok']:.4f}")
    
    print("\nTraining step test passed!")
