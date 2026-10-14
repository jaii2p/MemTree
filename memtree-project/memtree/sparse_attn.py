"""Linear sparse causal attention: sliding window + global sink tokens.

Cost per query is O(w + g), so total cost is O(T * (w + g)) -- linear in T.
Mirrors memtree.ref_numpy.sparse_attention_* exactly.
"""
import math

import numpy as np
import torch
import torch.nn as nn

from .ref_numpy import sparse_indices


def sparse_attention(q, k, v, window: int, n_global: int):
    """q, k, v: (B, H, T, d). Returns (B, H, T, d)."""
    B, H, T, d = q.shape
    idx = sparse_indices(T, window, n_global)  # (T, W) numpy, -1 padded
    idx_t = torch.as_tensor(idx, device=q.device)
    valid = idx_t >= 0  # (T, W)
    safe = idx_t.clamp(min=0)
    kg = k[:, :, safe]  # (B, H, T, W, d)
    vg = v[:, :, safe]
    scores = torch.einsum("bhtd,bhtwd->bhtw", q, kg) / math.sqrt(d)
    scores = scores.masked_fill(~valid, float("-inf"))  # every row keeps its own token
    p = torch.softmax(scores, dim=-1)
    return torch.einsum("bhtw,bhtwd->bhtd", p, vg)


class SparseCausalSelfAttention(nn.Module):
    """Drop-in self-attention block with linear sparse context."""

    def __init__(self, d_model: int, n_heads: int, window: int, n_global: int):
        super().__init__()
        assert d_model % n_heads == 0
        self.h, self.dh = n_heads, d_model // n_heads
        self.window, self.n_global = window, n_global
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x):
        B, T, D = x.shape
        q, k, v = self.qkv(x).view(B, T, 3, self.h, self.dh).permute(2, 0, 3, 1, 4)
        y = sparse_attention(q, k, v, self.window, self.n_global)
        return self.out(y.transpose(1, 2).reshape(B, T, D))
