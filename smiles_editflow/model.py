"""Edit policy network with selectable Transformer or Llama backbone."""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class TimeMLP(nn.Module):
    """MLP for encoding scalar time t into d_model dimensions."""
    
    def __init__(self, d_model: int):
        super().__init__()
        self.fc1 = nn.Linear(1, d_model)
        self.fc2 = nn.Linear(d_model, d_model)
    
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: [B] or [B, 1] time values
            
        Returns:
            [B, d_model] time embeddings
        """
        if t.dim() == 1:
            t = t.unsqueeze(-1)  # [B, 1]
        h = F.relu(self.fc1(t))
        h = self.fc2(h)
        return h


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""
    
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, S, d_model]
            
        Returns:
            [B, S, d_model] with positional encoding added
        """
        seq_len = x.size(1)
        return x + self.pe[:seq_len].unsqueeze(0)


class EditFlowModel(nn.Module):
    """
    Transformer encoder policy network for predicting edit operations.
    
    Outputs:
        - del_rate_logits: [B, S] rate logits for deletion at each position
        - sub_rate_logits: [B, S] rate logits for substitution at each position
        - ins_rate_logits: [B, S+1] rate logits for insertion at each gap
        - sub_tok_logits: [B, S, V] token logits for substitution
        - ins_tok_logits: [B, S+1, V] token logits for insertion
    """
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        max_len: int = 512,
        backbone: str = "transformer",
        num_kv_heads: int = None,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.backbone = backbone

        if backbone == "transformer":
            self.d_model = d_model
            self.token_embedding = nn.Embedding(vocab_size, d_model)
            self.pos_encoder = PositionalEncoding(d_model, max_len)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                batch_first=True,
            )
            self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            self.llama = None
        elif backbone == "llama":
            try:
                from transformers import LlamaConfig, LlamaModel
            except ImportError as exc:
                raise ImportError(
                    "Llama backbone requires 'transformers'. Install with: pip install transformers"
                ) from exc

            kv_heads = nhead if num_kv_heads is None else num_kv_heads
            if kv_heads <= 0 or nhead % kv_heads != 0:
                raise ValueError("num_kv_heads must be > 0 and divide nhead")

            llama_cfg = LlamaConfig(
                vocab_size=vocab_size,
                hidden_size=d_model,
                intermediate_size=dim_feedforward,
                num_hidden_layers=num_layers,
                num_attention_heads=nhead,
                num_key_value_heads=kv_heads,
                max_position_embeddings=max_len,
                attention_dropout=dropout,
                hidden_act="silu",
                rms_norm_eps=1e-5,
                tie_word_embeddings=False,
            )
            self.llama = LlamaModel(llama_cfg)
            self.d_model = llama_cfg.hidden_size
            self.token_embedding = None
            self.pos_encoder = None
            self.transformer_encoder = None
        else:
            raise ValueError(f"Unknown backbone: {backbone}")

        self.time_mlp = TimeMLP(self.d_model)
        
        # Per-position heads for DEL and SUB
        self.del_head = nn.Linear(self.d_model, 1)
        self.sub_head = nn.Linear(self.d_model, 1)
        self.sub_tok_head = nn.Linear(self.d_model, vocab_size)
        
        # Gap representation and INS heads
        self.gap_mlp = nn.Sequential(
            nn.Linear(self.d_model * 2, self.d_model),
            nn.ReLU(),
            nn.Linear(self.d_model, self.d_model),
        )
        self.ins_head = nn.Linear(self.d_model, 1)
        self.ins_tok_head = nn.Linear(self.d_model, vocab_size)
        
        self._init_custom_weights()

    def _init_custom_weights(self):
        """Initialize edit heads/time MLP without overriding backbone defaults."""
        modules = [
            self.time_mlp,
            self.del_head,
            self.sub_head,
            self.sub_tok_head,
            self.gap_mlp,
            self.ins_head,
            self.ins_tok_head,
        ]
        for module in modules:
            for p in module.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)
    
    def forward(
        self,
        token_ids: torch.Tensor,
        attn_mask: torch.Tensor,
        t: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            token_ids: [B, S] token IDs
            attn_mask: [B, S] attention mask (True for real tokens)
            t: [B] time values
            
        Returns:
            (del_rate_logits, sub_rate_logits, ins_rate_logits, sub_tok_logits, ins_tok_logits)
        """
        B, S = token_ids.shape
        
        if self.backbone == "transformer":
            x = self.token_embedding(token_ids)  # [B, S, d_model]
            x = self.pos_encoder(x)
            time_emb = self.time_mlp(t)  # [B, d_model]
            x = x + time_emb.unsqueeze(1)  # [B, S, d_model]
            padding_mask = ~attn_mask  # [B, S], True = ignore
            H = self.transformer_encoder(x, src_key_padding_mask=padding_mask)  # [B, S, d_model]
        else:
            # Llama backbone uses RoPE positions internally; inject time through embeddings.
            x = self.llama.get_input_embeddings()(token_ids)  # [B, S, d_model]
            time_emb = self.time_mlp(t)
            x = x + time_emb.unsqueeze(1)
            llama_mask = attn_mask.to(dtype=torch.long, device=token_ids.device)
            H = self.llama(
                inputs_embeds=x,
                attention_mask=llama_mask,
                use_cache=False,
                return_dict=True,
            ).last_hidden_state
        
        # Per-position deletion and substitution
        del_logits = self.del_head(H).squeeze(-1)  # [B, S]
        sub_logits = self.sub_head(H).squeeze(-1)  # [B, S]
        sub_tok_logits = self.sub_tok_head(H)  # [B, S, V]
        
        # Gap representation for insertion
        # Gaps: 0 (before pos 0), 1 (before pos 1), ..., S (after pos S-1)
        # We'll create S+1 gap representations
        
        # For interior gaps (1 to S-1), use concat of adjacent token hiddens
        # For boundary gaps, use special handling
        
        gap_hiddens = []
        
        # Gap 0: before first token (use zeros or first hidden)
        gap_0 = torch.zeros(B, self.d_model, device=H.device)
        gap_hiddens.append(gap_0)
        
        # Interior gaps: 1 to S-1
        for g in range(1, S):
            h_left = H[:, g - 1]  # Token before gap
            h_right = H[:, g]  # Token after gap
            gap_h = self.gap_mlp(torch.cat([h_left, h_right], dim=-1))  # [B, d_model]
            gap_hiddens.append(gap_h)
        
        # Gap S: after last token
        gap_S = torch.zeros(B, self.d_model, device=H.device)
        gap_hiddens.append(gap_S)
        
        # Stack gap hiddens: [B, S+1, d_model]
        gap_H = torch.stack(gap_hiddens, dim=1)
        
        # Insertion predictions
        ins_logits = self.ins_head(gap_H).squeeze(-1)  # [B, S+1]
        ins_tok_logits = self.ins_tok_head(gap_H)  # [B, S+1, V]
        
        return del_logits, sub_logits, ins_logits, sub_tok_logits, ins_tok_logits


if __name__ == "__main__":
    # Test model forward pass
    print("Model test:")
    
    vocab_size = 100
    B, S = 4, 10
    
    model = EditFlowModel(vocab_size=vocab_size, d_model=128, nhead=4, num_layers=2)
    
    token_ids = torch.randint(0, vocab_size, (B, S))
    attn_mask = torch.ones(B, S, dtype=torch.bool)
    attn_mask[:, -2:] = False  # Mask last 2 positions
    t = torch.rand(B)
    
    del_logits, sub_logits, ins_logits, sub_tok_logits, ins_tok_logits = model(token_ids, attn_mask, t)
    
    print(f"  del_logits shape: {del_logits.shape}")
    print(f"  sub_logits shape: {sub_logits.shape}")
    print(f"  ins_logits shape: {ins_logits.shape}")
    print(f"  sub_tok_logits shape: {sub_tok_logits.shape}")
    print(f"  ins_tok_logits shape: {ins_tok_logits.shape}")
    
    assert del_logits.shape == (B, S)
    assert sub_logits.shape == (B, S)
    assert ins_logits.shape == (B, S + 1)
    assert sub_tok_logits.shape == (B, S, vocab_size)
    assert ins_tok_logits.shape == (B, S + 1, vocab_size)
    
    print("\nModel test passed!")
