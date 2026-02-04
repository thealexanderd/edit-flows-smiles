"""Iterative editing sampler for generating SMILES."""

import torch
import numpy as np
from typing import List, Optional

from .tokenizer import BOS, EOS, PAD, encode, decode, detokenize
from .edit_distance import EditOp, EditType, apply_edit
from .chemistry import is_valid_smiles


def sample_molecule(
    model,
    token_to_id: dict,
    id_to_token: dict,
    device: str = "cpu",
    max_steps: int = 50,
    t_schedule: str = "constant",
    t_value: float = 0.5,
    sampling_strategy: str = "argmax",
    temperature: float = 1.0,
    threshold: float = 0.1,
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
            p_del, p_sub, p_ins, sub_tok_logits, ins_tok_logits = model(
                token_ids, attn_mask, t_tensor
            )
            
            # Extract predictions for single sample
            p_del = p_del[0].cpu().numpy()  # [S]
            p_sub = p_sub[0].cpu().numpy()  # [S]
            p_ins = p_ins[0].cpu().numpy()  # [S+1]
            sub_tok_logits = sub_tok_logits[0].cpu()  # [S, V]
            ins_tok_logits = ins_tok_logits[0].cpu()  # [S+1, V]
            
            S = len(tokens)
            
            # Build candidate edits with scores
            candidates = []
            
            # Deletion candidates (exclude BOS at 0 and EOS at S-1)
            for i in range(1, S - 1):
                candidates.append(("DEL", i, p_del[i], None))
            
            # Substitution candidates (exclude BOS and EOS)
            for i in range(1, S - 1):
                candidates.append(("SUB", i, p_sub[i], sub_tok_logits[i]))
            
            # Insertion candidates (gaps between tokens, not before BOS or after EOS)
            for g in range(1, S):
                candidates.append(("INS", g, p_ins[g], ins_tok_logits[g]))
            
            if not candidates:
                break
            
            # Filter by threshold
            candidates = [(op, pos, score, tok_logits) for op, pos, score, tok_logits in candidates if score > threshold]
            
            if not candidates:
                if verbose:
                    print(f"Step {step}: No candidates above threshold")
                break
            
            # Sort by score
            candidates.sort(key=lambda x: x[2], reverse=True)
            
            # Select edit
            if sampling_strategy == "argmax":
                chosen = candidates[0]
            else:
                # Sample proportional to scores
                scores = np.array([c[2] for c in candidates])
                scores = scores / temperature
                probs = np.exp(scores) / np.exp(scores).sum()
                idx = np.random.choice(len(candidates), p=probs)
                chosen = candidates[idx]
            
            op_type, pos, score, tok_logits = chosen
            
            # Sample token if needed
            if op_type == "SUB" or op_type == "INS":
                if sampling_strategy == "argmax":
                    tok_id = tok_logits.argmax().item()
                else:
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
                print(f"Step {step}: {op_type} at {pos} (score={score:.3f}) -> {smiles_candidate} (valid={is_valid})")
            
            if not is_valid:
                # Try next candidate
                retry_count = 0
                found_valid = False
                for alt_idx in range(1, min(len(candidates), max_retries + 1)):
                    alt_op_type, alt_pos, alt_score, alt_tok_logits = candidates[alt_idx]
                    
                    # Sample token if needed
                    if alt_op_type in ["SUB", "INS"]:
                        alt_tok_id = alt_tok_logits.argmax().item()
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
            
            # Early stopping if valid and reasonable length
            if is_valid and len(tokens) > 5:
                # Could add more stopping criteria here
                pass
        
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
