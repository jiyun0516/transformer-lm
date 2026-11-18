"""A4 Milestone 1: attention (STARTER).

softmax, scaled dot-product attention, the causal mask, RoPE and causal
multi-head self-attention. Submit this file for Milestone 1.

Rules: torch tensors, Modules and autograd are allowed, but NOT
torch.nn.MultiheadAttention, torch.nn.functional.scaled_dot_product_attention,
torch.nn.functional.softmax, torch.nn.LayerNorm/RMSNorm, torch.optim, or
torch.nn.functional.cross_entropy.
"""
import math

import torch
import torch.nn as nn


def softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Numerically stable softmax (subtract the max before exponentiating)."""
    shifted = x - x.amax(dim=dim, keepdim=True)
    exp_shifted = shifted.exp()
    return exp_shifted / exp_shifted.sum(dim=dim, keepdim=True)


def scaled_dot_product_attention(Q, K, V, mask=None):
    """Attention(Q, K, V) = softmax(QK^T / sqrt(d_k) + mask) V.

    Q: (..., n_q, d_k), K: (..., n_k, d_k), V: (..., n_k, d_v)
    mask: optional bool tensor broadcastable to (..., n_q, n_k); positions
    where mask is False are excluded (set to -inf before the softmax).
    """
    scores = (Q @ K.transpose(-2, -1)) / math.sqrt(Q.shape[-1])

    if mask is None:
        return softmax(scores, dim=-1) @ V

    scores = scores.masked_fill(~mask, float("-inf"))

    # Give queries with no visible keys a zero output and finite gradients.
    has_visible_key = mask.any(dim=-1, keepdim=True)
    scores = scores.masked_fill(~has_visible_key, 0.0)
    weights = softmax(scores, dim=-1).masked_fill(~mask, 0.0)
    return weights @ V


def causal_mask(n: int, device=None) -> torch.Tensor:
    """(n, n) bool tensor: True where key position <= query position."""
    positions = torch.arange(n, device=device)
    return positions[:, None] >= positions[None, :]


class RotaryEmbedding(nn.Module):
    """RoPE: rotate (even, odd) channel pairs by position-dependent angles.

    theta_i = base^(-2i/d); at position p, pair i rotates by p * theta_i.
    """

    def __init__(self, dim: int, max_len: int = 4096, base: float = 10000.0):
        super().__init__()
        if dim <= 0 or dim % 2 != 0:
            raise ValueError("RoPE requires a positive, even channel dimension")

        self.dim = dim
        self.max_len = max_len
        self.base = base

    def forward(self, x, positions=None):
        """x: (..., seq, dim). Rotate pairs (x[2i], x[2i+1])."""
        if positions is None:
            positions = torch.arange(x.shape[-2], device=x.device)

        # Compute trigonometric values in at least float32.
        angle_dtype = torch.float64 if x.dtype == torch.float64 else torch.float32
        channels = torch.arange(0, self.dim, 2, device=x.device, dtype=angle_dtype)
        frequencies = self.base ** (-channels / self.dim)
        angles = positions.to(device=x.device, dtype=angle_dtype)[..., None] * frequencies
        cos = angles.cos().to(dtype=x.dtype)
        sin = angles.sin().to(dtype=x.dtype)

        even = x[..., 0::2]
        odd = x[..., 1::2]
        rotated_even = even * cos - odd * sin
        rotated_odd = even * sin + odd * cos
        return torch.stack((rotated_even, rotated_odd), dim=-1).flatten(-2)


class MultiHeadAttention(nn.Module):
    """Causal multi-head self-attention with bias-free projections."""

    def __init__(self, d_model: int, n_heads: int, max_len: int = 4096):
        super().__init__()
        if d_model <= 0 or n_heads <= 0 or d_model % n_heads != 0:
            raise ValueError("d_model must be positive and divisible by n_heads")

        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)
        self.rope = RotaryEmbedding(self.d_head, max_len=max_len)

    def forward(self, x):
        head_shape = (*x.shape[:-1], self.n_heads, self.d_head)
        q = self.q_proj(x).reshape(head_shape).transpose(-3, -2)
        k = self.k_proj(x).reshape(head_shape).transpose(-3, -2)
        v = self.v_proj(x).reshape(head_shape).transpose(-3, -2)

        q = self.rope(q)
        k = self.rope(k)
        mask = causal_mask(x.shape[-2], device=x.device)
        attended = scaled_dot_product_attention(q, k, v, mask)

        joined = attended.transpose(-3, -2).reshape(x.shape)
        return self.o_proj(joined)
        


