"""CTMC-style editing sampler for generating SMILES."""

import torch
from typing import List

from .tokenizer import BOS, EOS, PAD, UNK, encode, detokenize
from .chemistry import is_valid_smiles
from .masking import mask_token_logits


def sample_molecule(
    model,
    token_to_id: dict,
    id_to_token: dict,
    device: str = "cpu",
    max_steps: int = 400,
    step_size: float = 0.05,
    t_schedule: str = "ctmc",
    t_value: float = 0.5,
    temperature: float = 1.0,
    stop_threshold: float = 1e-3,
    stop_patience: int = 5,
    max_retries: int = 3,
    max_seq_len: int | None = None,
    verbose: bool = False,
) -> tuple[str, bool, List[str]]:
    """
    Generate a SMILES string by iterative editing.
    
    Args:
        model: Trained EditFlowModel
        token_to_id: Token to ID mapping
        id_to_token: ID to token mapping
        device: Device to run on
        max_steps: Maximum number of editing steps
        step_size: Step size h for CTMC simulation
        t_schedule: Time schedule ("ctmc", "constant", or "linear_decay")
        t_value: Time value for constant schedule
        temperature: Temperature for sampling
        stop_threshold: Minimum total rate to continue editing
        max_retries: Max retries for invalid edits
        max_seq_len: Maximum token sequence length including BOS/EOS.
            If None, inferred from model positional encoding length.
        verbose: Print intermediate steps
        
    Returns:
        (smiles_string, is_valid, intermediate_smiles_list)
    """
    model.eval()

    if max_seq_len is None:
        max_seq_len = int(model.pos_encoder.pe.size(0))
    if max_seq_len < 2:
        raise ValueError("max_seq_len must be >= 2 to fit BOS/EOS")
    
    # Start with empty sequence
    tokens = [BOS, EOS]
    
    intermediate_smiles = []
    t = 0.0
    
    with torch.no_grad():
        low_rate_steps = 0
        for step in range(max_steps):
            # Determine t for this step
            if t_schedule == "ctmc":
                t = min(1.0, t + step_size)
            elif t_schedule == "linear_decay":
                t = 0.9 - (0.8 * step / max_steps)  # Decay from 0.9 to 0.1
            elif t_schedule == "constant":
                t = t_value
            else:
                raise ValueError(f"Unknown t_schedule: {t_schedule}")
            
            # Encode tokens
            ids = encode(tokens, token_to_id)
            token_ids = torch.tensor([ids], dtype=torch.long, device=device)
            attn_mask = torch.ones_like(token_ids, dtype=torch.bool)
            t_tensor = torch.tensor([t], dtype=torch.float32, device=device)
            
            # Forward pass
            del_logits, sub_logits, ins_logits, sub_tok_logits, ins_tok_logits = model(
                token_ids, attn_mask, t_tensor
            )
            
            # Extract predictions for single sample
            del_rates = torch.nn.functional.softplus(del_logits[0]).cpu()  # [S]
            sub_rates = torch.nn.functional.softplus(sub_logits[0]).cpu()  # [S]
            ins_rates = torch.nn.functional.softplus(ins_logits[0]).cpu()  # [S+1]
            sub_tok_logits = sub_tok_logits[0].cpu()  # [S, V]
            ins_tok_logits = ins_tok_logits[0].cpu()  # [S+1, V]

            forbidden_ids = [token_to_id[BOS], token_to_id[EOS], token_to_id[PAD], token_to_id[UNK]]
            sub_tok_logits = mask_token_logits(sub_tok_logits, forbidden_ids)
            ins_tok_logits = mask_token_logits(ins_tok_logits, forbidden_ids)
            
            S = len(tokens)
            
            # Sample independent edit events with probability h * lambda
            del_set = set()
            sub_map = {}
            ins_map = {}

            for i in range(1, S - 1):
                rate = del_rates[i].item()
                p = min(1.0, step_size * rate)
                if torch.rand(1).item() < p:
                    del_set.add(i)

            for i in range(1, S - 1):
                rate = sub_rates[i].item()
                p = min(1.0, step_size * rate)
                if torch.rand(1).item() < p:
                    tok_probs = torch.softmax(sub_tok_logits[i] / temperature, dim=0)
                    tok_id = torch.multinomial(tok_probs, 1).item()
                    sub_map[i] = id_to_token[tok_id]

            if S < max_seq_len:
                for g in range(1, S):
                    rate = ins_rates[g].item()
                    p = min(1.0, step_size * rate)
                    if torch.rand(1).item() < p:
                        tok_probs = torch.softmax(ins_tok_logits[g] / temperature, dim=0)
                        tok_id = torch.multinomial(tok_probs, 1).item()
                        ins_map[g] = id_to_token[tok_id]

            if t_schedule != "ctmc":
                total_rate = del_rates[1:S-1].sum() + sub_rates[1:S-1].sum() + ins_rates[1:S].sum()
                if total_rate.item() < stop_threshold:
                    low_rate_steps += 1
                else:
                    low_rate_steps = 0
                if low_rate_steps >= stop_patience:
                    if verbose:
                        print(f"Step {step}: stopping due to low total rate")
                    break

            if not del_set and not sub_map and not ins_map:
                if t_schedule == "ctmc" and t >= 1.0:
                    break
                continue

            # Apply all edits simultaneously
            new_tokens = []
            for idx in range(S):
                if idx in ins_map:
                    new_tokens.append(ins_map[idx])
                if idx in del_set:
                    continue
                if idx in sub_map:
                    new_tokens.append(sub_map[idx])
                else:
                    new_tokens.append(tokens[idx])

            # Guardrail for positional encoding limits: cap to model-supported length.
            if len(new_tokens) > max_seq_len:
                interior = new_tokens[1:-1][: max_seq_len - 2]
                new_tokens = [BOS] + interior + [EOS]
            
            # Validate (detokenize and check)
            smiles_candidate = detokenize(new_tokens)
            is_valid = is_valid_smiles(smiles_candidate)
            
            if verbose:
                print(
                    f"Step {step}: DEL={len(del_set)} SUB={len(sub_map)} "
                    f"INS={len(ins_map)} len={len(new_tokens)} -> {smiles_candidate} (valid={is_valid})"
                )
            
            if not is_valid and verbose:
                print(f"  -> Invalid SMILES, continuing")
            
            tokens = new_tokens
            intermediate_smiles.append(smiles_candidate)
            
            # Optional: continue editing even if valid to refine
        
    # Final SMILES
    final_smiles = detokenize(tokens)
    final_valid = is_valid_smiles(final_smiles)
    
    return final_smiles, final_valid, intermediate_smiles


if __name__ == "__main__":
    from .model import EditFlowModel
    from .tokenizer import build_vocab
    
    # Test sampler
    print("Sampler test:")
    
    # Build vocab
    smiles_list = ["CC(=O)O", "c1ccccc1", "CCO"]
    token_to_id, id_to_token = build_vocab(smiles_list)
    
    # Create model (untrained)
    model = EditFlowModel(
        vocab_size=len(token_to_id),
        d_model=64,
        nhead=2,
        num_layers=1,
    )
    
    # Sample
    smiles, valid, intermediates = sample_molecule(
        model,
        token_to_id,
        id_to_token,
        device="cpu",
        max_steps=20,
        verbose=True,
    )
    
    print(f"\nFinal SMILES: {smiles}")
    print(f"Valid: {valid}")
    print(f"Steps taken: {len(intermediates)}")
