"""Legacy teacher-forced training utilities (deprecated)."""

from typing import List, Dict, Optional, Tuple
import random
import torch
import torch.nn.functional as F

from ..chemistry import randomized_smiles, canonicalize_smiles
from ..tokenizer import tokenize, add_special, encode, BOS, EOS, PAD, UNK
from ..corruption import corrupt
from ..targets import build_targets
from ..edit_distance import levenshtein_script, apply_script
from ..losses import compute_losses_teacher_forced


def prepare_teacher_forced_batch(
    batch_smiles: List[str],
    token_to_id: dict,
    id_to_token: dict,
    vocab_set: set,
    t_min: float = 0.0,
    t_max: float = 1.0,
    seed: int = None,
) -> Optional[Dict]:
    """
    Prepare a teacher-forced batch (legacy).

    This path is deprecated and should not be used for Edit Flows training.
    """
    if seed is not None:
        rng = random.Random(seed)
    else:
        rng = random.Random()

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

    if not batch_data:
        return None

    batch_size = len(batch_data)
    max_len = max(len(x_t) for x_t, *_ in batch_data)
    pad_id = token_to_id[PAD]

    token_ids = torch.full((batch_size, max_len), pad_id, dtype=torch.long)
    attn_mask = torch.zeros((batch_size, max_len), dtype=torch.bool)
    t_batch = torch.zeros(batch_size, dtype=torch.float32)

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


def train_step_teacher_forced(
    model,
    optimizer,
    batch_smiles: List[str],
    token_to_id: dict,
    id_to_token: dict,
    vocab_set: set,
    device: str = "cpu",
    alpha: float = 1.0,
) -> Dict:
    """Legacy teacher-forced training step (deprecated)."""
    model.train()

    batch = prepare_teacher_forced_batch(
        batch_smiles,
        token_to_id,
        id_to_token,
        vocab_set,
    )
    if batch is None:
        return {"loss": 0.0, "error": "empty_batch"}

    token_ids = batch["token_ids"].to(device)
    attn_mask = batch["attn_mask"].to(device)
    t = batch["t"].to(device)
    tok_targets = batch["tok_targets"]

    del_logits, sub_logits, ins_logits, sub_tok_logits, ins_tok_logits = model(
        token_ids, attn_mask, t
    )

    p_del = torch.sigmoid(del_logits)
    p_sub = torch.sigmoid(sub_logits)
    p_ins = torch.sigmoid(ins_logits)

    losses = compute_losses_teacher_forced(
        p_del, p_sub, p_ins,
        sub_tok_logits, ins_tok_logits,
        batch["y_del"].to(device),
        batch["y_sub"].to(device),
        batch["y_ins"].to(device),
        tok_targets,
        attn_mask,
        bos_id=token_to_id[BOS],
        eos_id=token_to_id[EOS],
        pad_id=token_to_id[PAD],
        alpha=alpha,
    )

    optimizer.zero_grad()
    losses["loss"].backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

    return {k: v.item() if isinstance(v, torch.Tensor) else v for k, v in losses.items()}
