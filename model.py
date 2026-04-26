"""
PHENO TYPE — model.py
Transformer Encoder → 256-dim L2-normalised behavioural fingerprint.

Architecture (per spec Section 2 & 3):
  Embedding (100 → 128)  →  SinusoidalPE  →
  4× TransformerEncoderLayer (8 heads, d_ff=512)  →
  AttentionPooling  →  Linear(128 → 256)  →  L2 normalise
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────
# 1. Sinusoidal Positional Encoding
# ─────────────────────────────────────────────────────────────
class SinusoidalPositionalEncoding(nn.Module):
    """
    Fixed sinusoidal encoding — no learnable parameters.
    Generalises well; sufficient for 1200-position sequences.
    """

    def __init__(self, d_model: int, max_len: int = 1200, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe  = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, seq, d_model)"""
        return self.dropout(x + self.pe[:, : x.size(1)])


# ─────────────────────────────────────────────────────────────
# 2. Attention Pooling
# ─────────────────────────────────────────────────────────────
class AttentionPooling(nn.Module):
    """
    Learns which positions in the sequence matter most.
    Attention weights can be returned for interpretability / SHAP use.
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.query = nn.Linear(d_model, 1, bias=False)

    def forward(
        self,
        x: torch.Tensor,                        # (batch, seq, d_model)
        key_padding_mask: torch.Tensor = None,   # (batch, seq) bool — True = pad
        return_weights: bool = False,
    ):
        scores = self.query(x).squeeze(-1)       # (batch, seq)
        if key_padding_mask is not None:
            scores = scores.masked_fill(key_padding_mask, float('-inf'))
        weights = torch.softmax(scores, dim=1)
        # Guard: all-PAD rows produce all -inf → softmax NaN.
        # Replace those rows with uniform weights so output degrades gracefully.
        nan_rows = weights.isnan().any(dim=1)
        if nan_rows.any():
            weights[nan_rows] = 1.0 / weights.size(1)
        weights = weights.unsqueeze(-1)          # (batch, seq, 1)
        pooled  = (weights * x).sum(dim=1)      # (batch, d_model)

        if return_weights:
            return pooled, weights.squeeze(-1)
        return pooled


# ─────────────────────────────────────────────────────────────
# 3. Full Behaviour Encoder
# ─────────────────────────────────────────────────────────────
class BehaviourEncoder(nn.Module):
    """
    Converts a (batch, 1200) int64 token sequence into a
    (batch, 256) L2-normalised fingerprint vector.

    Hyperparameters from spec Section 2.5:
        vocab_size      = 100   (0..99, PAD=0)
        d_model         = 128
        nhead           = 8
        num_layers      = 4
        d_ff            = 512
        fingerprint_dim = 256
        dropout         = 0.1
    """

    def __init__(
        self,
        vocab_size:      int = 100,
        d_model:         int = 128,
        nhead:           int = 8,
        num_layers:      int = 4,
        d_ff:            int = 512,
        fingerprint_dim: int = 256,
        dropout:         float = 0.1,
    ):
        super().__init__()

        # Token embedding — padding_idx=0 so PAD tokens get zero gradient
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)

        self.pos_enc = SinusoidalPositionalEncoding(d_model, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model         = d_model,
            nhead           = nhead,
            dim_feedforward = d_ff,
            dropout         = dropout,
            activation      = 'gelu',
            batch_first     = True,
            norm_first      = True,   # Pre-LN: more stable training
        )
        # enable_nested_tensor=False: required when norm_first=True to silence
        # the PyTorch warning about nested tensors being unsupported with Pre-LN.
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers, enable_nested_tensor=False
        )

        self.pool = AttentionPooling(d_model)
        self.proj = nn.Linear(d_model, fingerprint_dim)

    def forward(
        self,
        tokens: torch.Tensor,           # (batch, 1200) int64
        return_attn_weights: bool = False,
    ):
        pad_mask = (tokens == 0)                     # True where PAD
        x = self.embedding(tokens)                   # (batch, 1200, 128)
        x = self.pos_enc(x)

        # PyTorch's TransformerEncoder takes src_key_padding_mask
        x = self.transformer(x, src_key_padding_mask=pad_mask)

        if return_attn_weights:
            pooled, weights = self.pool(x, pad_mask, return_weights=True)
        else:
            pooled = self.pool(x, pad_mask)

        fp = self.proj(pooled)                       # (batch, 256)
        fp = F.normalize(fp, dim=1)                  # L2 norm → unit hypersphere

        if return_attn_weights:
            return fp, weights
        return fp

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ─────────────────────────────────────────────────────────────
# Quick sanity check
# ─────────────────────────────────────────────────────────────
if __name__ == '__main__':
    model = BehaviourEncoder()
    print(f'Parameters: {model.count_parameters():,}')

    dummy = torch.randint(0, 100, (4, 1200))
    fp    = model(dummy)
    print(f'Output shape : {fp.shape}')                          # (4, 256)
    print(f'L2 norms     : {fp.norm(dim=1)}')                    # all ≈ 1.0
