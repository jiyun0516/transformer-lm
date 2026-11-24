"""A4 Milestone 2: the transformer language model (STARTER).

Modern (Llama-style) decoder-only LM: RMSNorm (pre-norm), RoPE, SwiGLU FFN,
no bias terms, weight-untied LM head, and cross-entropy from scratch. The
attention parts come from your Milestone 1 file, attention.py. AdamW, the
cosine learning-rate schedule with warmup and gradient clipping are given
at the end of the file (read them; you do not write them).

Rules: torch tensors, Modules and autograd are allowed, but NOT
torch.nn.MultiheadAttention, torch.nn.functional.scaled_dot_product_attention,
torch.nn.functional.softmax, torch.nn.LayerNorm/RMSNorm, torch.optim, or
torch.nn.functional.cross_entropy.
"""
import math

import torch
import torch.nn as nn

from attention import (softmax, scaled_dot_product_attention, causal_mask,  # noqa: F401
                       RotaryEmbedding, MultiHeadAttention)


# --------------------------- milestone 2 -----------------------------------

class RMSNorm(nn.Module):
    """x * g / RMS(x), RMS(x) = sqrt(mean(x^2) + eps). Computed in fp32."""

    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        x_float = x.float()
        rms = torch.sqrt(x_float.square().mean(dim=-1, keepdim=True) + self.eps)
        output = (x_float / rms) * self.weight.float()
        return output.to(dtype=x.dtype)


class SwiGLU(nn.Module):
    """FFN(x) = W2( silu(W1 x) * W3 x ), silu(z) = z * sigmoid(z)."""

    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)
        self.w3 = nn.Linear(d_model, d_ff, bias=False)

    def forward(self, x):
        gate = self.w1(x)
        activated = gate * torch.sigmoid(gate)
        return self.w2(activated * self.w3(x))


class TransformerBlock(nn.Module):
    """Pre-norm block: x += attn(norm(x)); x += ffn(norm(x))."""

    def __init__(self, d_model, n_heads, d_ff, max_len=4096):
        super().__init__()
        self.attn_norm = RMSNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads, max_len=max_len)
        self.ffn_norm = RMSNorm(d_model)
        self.ffn = SwiGLU(d_model, d_ff)

    def forward(self, x):
        x = x + self.attn(self.attn_norm(x))
        x = x + self.ffn(self.ffn_norm(x))
        return x


class TransformerLM(nn.Module):
    def __init__(self, vocab_size, d_model=128, n_layers=4, n_heads=4,
                 d_ff=None, max_len=512):
        super().__init__()
        if d_ff is None:
            d_ff = 4 * d_model
        if max_len <= 0:
            raise ValueError("max_len must be positive")

        self.max_len = max_len
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, max_len=max_len)
            for _ in range(n_layers)
        ])
        self.norm = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, token_ids):
        """token_ids: (B, T) int64 -> logits (B, T, vocab_size)."""
        x = self.embedding(token_ids)
        for block in self.blocks:
            x = block(x)
        return self.lm_head(self.norm(x))
 
    @torch.no_grad()
    def generate(self, prefix_ids, max_new_tokens, temperature=1.0, seed=None):
        """Autoregressive sampling. prefix_ids: (T,) or (1, T)."""
        if prefix_ids.ndim not in (1, 2):
            raise ValueError("prefix_ids must have shape (T,) or (1, T)")
        if prefix_ids.ndim == 2 and prefix_ids.shape[0] != 1:
            raise ValueError("generate supports one sequence at a time")
        if prefix_ids.shape[-1] == 0:
            raise ValueError("prefix_ids must contain at least one token")
        if max_new_tokens < 0 or temperature < 0:
            raise ValueError("max_new_tokens and temperature must be nonnegative")

        was_1d = prefix_ids.ndim == 1
        tokens = prefix_ids.unsqueeze(0) if was_1d else prefix_ids
        tokens = tokens.clone()

        generator = None
        if seed is not None:
            generator = torch.Generator(device="cpu")
            generator.manual_seed(seed)

        for _ in range(max_new_tokens):
            context = tokens[:, -self.max_len:]
            logits = self(context)[:, -1, :].float()

            if temperature == 0:
                next_token = logits.argmax(dim=-1, keepdim=True)
            else:
                probabilities = softmax(logits / temperature, dim=-1)
                next_token = torch.multinomial(
                    probabilities.cpu(),
                    num_samples=1,
                    generator=generator,
                ).to(device=tokens.device)

            tokens = torch.cat((tokens, next_token), dim=-1)

        return tokens.squeeze(0) if was_1d else tokens


def cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Mean cross-entropy from raw logits, computed stably via log-sum-exp.

    logits: (N, V) or (B, T, V); targets: matching int64 shape.
    """
    if logits.dtype in (torch.float16, torch.bfloat16):
        logits = logits.float()

    shifted = logits - logits.amax(dim=-1, keepdim=True)
    log_normalizer = shifted.exp().sum(dim=-1).log()
    target_scores = shifted.gather(
        dim=-1,
        index=targets.unsqueeze(-1),
    ).squeeze(-1)

    return (log_normalizer - target_scores).mean()


# --------------------------- given -----------------------------------------
# AdamW, the cosine schedule and gradient clipping are given (taught Thu 19 Nov).

class AdamW:
    """AdamW (decoupled weight decay), implemented from scratch."""

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0.01):
        self.params = [p for p in params if p.requires_grad]
        self.lr, self.betas, self.eps, self.wd = lr, betas, eps, weight_decay
        self.t = 0
        self.m = [torch.zeros_like(p) for p in self.params]
        self.v = [torch.zeros_like(p) for p in self.params]

    def zero_grad(self):
        for p in self.params:
            p.grad = None

    @torch.no_grad()
    def step(self):
        self.t += 1
        b1, b2 = self.betas
        for p, m, v in zip(self.params, self.m, self.v):
            if p.grad is None:
                continue
            g = p.grad
            p.mul_(1 - self.lr * self.wd)  # decoupled decay (applied first)
            m.mul_(b1).add_(g, alpha=1 - b1)
            v.mul_(b2).addcmul_(g, g, value=1 - b2)
            m_hat = m / (1 - b1 ** self.t)
            v_hat = v / (1 - b2 ** self.t)
            p.add_(m_hat / (v_hat.sqrt() + self.eps), alpha=-self.lr)


def cosine_lr(step, max_lr, min_lr, warmup_steps, total_steps):
    """Linear warmup 0->max_lr, then cosine decay max_lr->min_lr, then min_lr."""
    if step < warmup_steps:
        return max_lr * step / max(warmup_steps, 1)
    if step >= total_steps:
        return min_lr
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * progress))


def clip_grad_norm(params, max_norm, eps=1e-6):
    """Scale all gradients so their global L2 norm is at most max_norm.
    Returns the pre-clip norm as a float."""
    grads = [p.grad for p in params if p.grad is not None]
    total = torch.sqrt(sum((g ** 2).sum() for g in grads))
    if total > max_norm:
        scale = max_norm / (total + eps)
        for g in grads:
            g.mul_(scale)
    return float(total)
