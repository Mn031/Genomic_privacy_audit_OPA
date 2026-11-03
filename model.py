"""
model.py - Minimal Causal Transformer Architecture (SimpleDNALM) for Genomic LM (OPA).
Based on "Quantifying Memorization and Privacy Risks in Genomic Language Models" (arXiv:2603.08913v3).

Architecture Specifications:
  - Token & Learned Positional Embeddings (seq_len <= 256)
  - 4 Transformer Decoder Blocks
  - Hidden Dimension d_model: 512
  - Multi-Head Attention: 8 heads (head_dim = 64)
  - Feedforward Dimension d_ff: 2048
  - Activation: GELU
  - Dropout: 0.05
  - Causal Mask: Lower-triangular masking for autoregressive next-token prediction
  - LM Head: Linear projection (d_model -> vocab_size=8)
  - Parameter Footprint: ~12.6M - 12.88M parameters
  - Flower FL / OPA Interface: get_weights() and set_weights() ndarray converters
"""

import math
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    """Multi-Head Causal Self-Attention block."""

    def __init__(self, d_model: int = 512, n_heads: int = 8, dropout: float = 0.05, max_seq_len: int = 256) -> None:
        super().__init__()
        assert d_model % n_heads == 0, f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        # Key, Query, Value projections combined into single linear layer for computational efficiency
        self.c_attn = nn.Linear(d_model, 3 * d_model)
        # Output projection
        self.c_proj = nn.Linear(d_model, d_model)

        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        # Causal mask: lower-triangular matrix ensuring tokens only attend to past and present tokens
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(max_seq_len, max_seq_len)).view(1, 1, max_seq_len, max_seq_len),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()  # Batch size, Sequence length, Embedding dimension (d_model)

        # Calculate query, key, values for all heads in batch and move head forward to the batch dim
        qkv = self.c_attn(x)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)  # (B, nh, T, hs)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)  # (B, nh, T, hs)

        # Scaled dot-product attention with causal mask
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)

        y = att @ v  # (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # Re-assemble all head outputs side by side

        # Output projection
        y = self.resid_dropout(self.c_proj(y))
        return y


class TransformerBlock(nn.Module):
    """Standard Transformer Decoder Block: Pre-LN Attention + Pre-LN Feedforward."""

    def __init__(self, d_model: int = 512, n_heads: int = 8, d_ff: int = 2048, dropout: float = 0.05, max_seq_len: int = 256) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model=d_model, n_heads=n_heads, dropout=dropout, max_seq_len=max_seq_len)
        self.ln_2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-LayerNorm residual connections
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class SimpleDNALM(nn.Module):
    """
    SimpleDNALM: Minimal Causal Transformer Baseline for Genomic Sequences.
    Conforms to the experimental specifications in arXiv:2603.08913v3:
      - 4 transformer layers
      - d_model = 512
      - 8 attention heads
      - d_ff = 2048
      - GELU activation, 0.05 dropout
      - Total parameters: ~12.6M - 12.88M
    """

    def __init__(
        self,
        vocab_size: int = 8,
        d_model: int = 512,
        num_layers: int = 4,
        n_heads: int = 8,
        d_ff: int = 2048,
        max_seq_len: int = 256,
        dropout: float = 0.05,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_layers = num_layers
        self.max_seq_len = max_seq_len

        # Token & Positional Embeddings
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)
        self.emb_dropout = nn.Dropout(dropout)

        # 4 Causal Transformer Layers
        self.blocks = nn.ModuleList([
            TransformerBlock(
                d_model=d_model,
                n_heads=n_heads,
                d_ff=d_ff,
                dropout=dropout,
                max_seq_len=max_seq_len,
            )
            for _ in range(num_layers)
        ])

        # Final LayerNorm & Language Modeling Projection Head
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # Weight tying: share token embedding weights with output projection head
        self.token_embedding.weight = self.lm_head.weight

        # Parameter initialization
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass for next-token prediction.
        
        Args:
            input_ids: Tensor of shape (batch_size, seq_len)
            targets: Optional tensor of shape (batch_size, seq_len)
            
        Returns:
            Tuple of (logits, loss). If targets is None, loss is None.
        """
        device = input_ids.device
        B, T = input_ids.size()
        assert T <= self.max_seq_len, f"Sequence length {T} exceeds max_seq_len {self.max_seq_len}"

        positions = torch.arange(0, T, dtype=torch.long, device=device).unsqueeze(0)  # (1, T)

        # Sum token and positional embeddings
        x = self.token_embedding(input_ids) + self.position_embedding(positions)
        x = self.emb_dropout(x)

        # Pass through Transformer Blocks
        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)
        logits = self.lm_head(x)  # (B, T, vocab_size)

        loss = None
        if targets is not None:
            # Cross-entropy loss ignoring padding (index 0)
            loss = F.cross_entropy(logits.view(-1, self.vocab_size), targets.view(-1), ignore_index=0)

        return logits, loss

    def get_weights(self) -> List[np.ndarray]:
        """Extracts model parameters as a list of NumPy arrays for Flower FL / OPA aggregation."""
        return [param.detach().cpu().numpy() for param in self.parameters()]

    def set_weights(self, weights: List[np.ndarray]) -> None:
        """Sets model parameters from a list of NumPy arrays."""
        for param, new_w in zip(self.parameters(), weights):
            param.data.copy_(torch.from_numpy(new_w).to(param.device))

