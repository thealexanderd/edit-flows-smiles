"""CTMC-style editing sampler for generating SMILES."""

import math
import torch
from typing import List, Optional, Tuple

from .tokenizer import BOS, EOS, PAD, encode, detokenize
from .edit_distance import EditOp, EditType, apply_edit
from .chemistry import is_valid_smiles


def sample_molecule(
    model,
    token_to_id: dict,
    id_to_token: dict,
    device: str = "cpu",
    max_steps: int = 400,
    step_size: float = 0.05,
    t_schedule: str = "constant",
    t_value: float = 0.5,
    temperature: float = 1.0,
    stop_threshold: float = 1e-3,
    stop_patience: int = 5,
    max_retries: int = 3,
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
        t_schedule: Time schedule ("constant" or "linear_decay")
        t_value: Time value for constant schedule
        sampling_strategy: "argmax" or "sample"
        temperature: Temperature for sampling
        threshold: Minimum score to continue editing
        max_retries: Max retries for invalid edits
        verbose: Print intermediate steps
        
    Returns:
        (smiles_string, is_valid, intermediate_smiles_list)
    """
    model.eval()
    
    # Start with empty sequence
    tokens = [BOS, EOS]
    
    intermediate_smiles = []
    
    with torch.no_grad():
        low_rate_steps = 0
        for step in range(max_steps):
            # Determine t for this step
            if t_schedule == "linear_decay":
                t = 0.9 - (0.8 * step / max_steps)  # Decay from 0.9 to 0.1
            else:
                t = t_value
            
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
            
            S = len(tokens)
            
            candidates: List[Tuple[str, int, float, Optional[torch.Tensor]]] = []
            for i in range(1, S - 1):
                rate = del_rates[i].item()
                weight = 1.0 - math.exp(-step_size * rate)
                candidates.append(("DEL", i, weight, None))
            for i in range(1, S - 1):
                rate = sub_rates[i].item()
                weight = 1.0 - math.exp(-step_size * rate)
                candidates.append(("SUB", i, weight, sub_tok_logits[i]))
            for g in range(1, S):
                rate = ins_rates[g].item()
                weight = 1.0 - math.exp(-step_size * rate)
                candidates.append(("INS", g, weight, ins_tok_logits[g]))

            total_rate = sum(c[2] for c in candidates)
            if total_rate < stop_threshold:
                low_rate_steps += 1
            else:
                low_rate_steps = 0
            if low_rate_steps >= stop_patience:
                if verbose:
                    print(f"Step {step}: stopping due to low total rate")
                break

            if total_rate <= 0:
                break

            weights = torch.tensor([c[2] for c in candidates], dtype=torch.float32)
            probs = weights / weights.sum()
            idx = torch.multinomial(probs, 1).item()
            op_type, pos, score, tok_logits = candidates[idx]

            if op_type in {"SUB", "INS"}:
                tok_probs = torch.softmax(tok_logits / temperature, dim=0)
                tok_id = torch.multinomial(tok_probs, 1).item()
                tok = id_to_token[tok_id]
            else:
                tok = None
            
            # Create edit operation
            if op_type == "DEL":
                edit = EditOp(type=EditType.DEL, i=pos)
            elif op_type == "SUB":
                edit = EditOp(type=EditType.SUB, i=pos, tok=tok)
            elif op_type == "INS":
                edit = EditOp(type=EditType.INS, g=pos, tok=tok)
            
            # Apply edit
            new_tokens = apply_edit(tokens, edit)
            
            # Validate (detokenize and check)
            smiles_candidate = detokenize(new_tokens)
            is_valid = is_valid_smiles(smiles_candidate)
            
            if verbose:
                print(f"Step {step}: {op_type} at {pos} (rate={score:.3f}) -> {smiles_candidate} (valid={is_valid})")
            
            if not is_valid:
                # Try next candidates by descending rate
                retry_count = 0
                found_valid = False
                alt_candidates = sorted(candidates, key=lambda x: x[2], reverse=True)
                for alt_idx in range(1, min(len(alt_candidates), max_retries + 1)):
                    alt_op_type, alt_pos, alt_score, alt_tok_logits = alt_candidates[alt_idx]
                    
                    # Sample token if needed
                    if alt_op_type in ["SUB", "INS"]:
                        tok_probs = torch.softmax(alt_tok_logits / temperature, dim=0)
                        alt_tok_id = torch.multinomial(tok_probs, 1).item()
                        alt_tok = id_to_token[alt_tok_id]
                    else:
                        alt_tok = None
                    
                    # Create and apply edit
                    if alt_op_type == "DEL":
                        alt_edit = EditOp(type=EditType.DEL, i=alt_pos)
                    elif alt_op_type == "SUB":
                        alt_edit = EditOp(type=EditType.SUB, i=alt_pos, tok=alt_tok)
                    else:
                        alt_edit = EditOp(type=EditType.INS, g=alt_pos, tok=alt_tok)
                    
                    alt_tokens = apply_edit(tokens, alt_edit)
                    alt_smiles = detokenize(alt_tokens)
                    
                    if is_valid_smiles(alt_smiles):
                        new_tokens = alt_tokens
                        smiles_candidate = alt_smiles
                        is_valid = True
                        found_valid = True
                        if verbose:
                            print(f"  -> Retry successful: {alt_smiles}")
                        break
                
                if not found_valid:
                    if verbose:
                        print(f"  -> All retries failed, keeping invalid edit")
                    # Accept invalid edit and continue (or could break here)
            
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
