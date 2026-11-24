"""A4 Milestone 2 public tests: transformer.py (it imports your attention.py)."""
import math
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..",
                                os.environ.get("SUBMISSION_DIR", "starter")))
import transformer as tf  # noqa: E402

torch.manual_seed(0)


def test_rmsnorm_matches_reference():
    x = torch.randn(2, 5, 16)
    layer = tf.RMSNorm(16)
    with torch.no_grad():
        layer.weight.copy_(torch.randn(16).abs() + 0.5)
    ref = x / torch.sqrt(x.pow(2).mean(-1, keepdim=True) + 1e-5) * layer.weight
    torch.testing.assert_close(layer(x), ref, atol=1e-5, rtol=1e-5)


def test_lm_has_one_attention_per_block():
    """The LM stacks n_layers blocks, each with its own attention."""
    model = tf.TransformerLM(vocab_size=50, d_model=32, n_layers=2, n_heads=4)
    logits = model(torch.randint(0, 50, (2, 7)))
    assert logits.shape == (2, 7, 50)
    attn = [m for m in model.modules() if isinstance(m, tf.MultiHeadAttention)]
    assert len(attn) == 2


def test_swiglu_matches_reference_formula():
    ffn = tf.SwiGLU(8, 16)
    x = torch.randn(3, 8)
    ref = ffn.w2(F.silu(ffn.w1(x)) * ffn.w3(x))
    torch.testing.assert_close(ffn(x), ref, atol=1e-6, rtol=1e-6)


def test_block_is_residual():
    blk = tf.TransformerBlock(32, 4, 64)
    # zero out output projections -> block must be the identity
    with torch.no_grad():
        blk.attn.o_proj.weight.zero_()
        blk.ffn.w2.weight.zero_()
    x = torch.randn(2, 6, 32)
    torch.testing.assert_close(blk(x), x)


def test_lm_shapes_and_causality():
    model = tf.TransformerLM(vocab_size=50, d_model=32, n_layers=2, n_heads=4)
    ids = torch.randint(0, 50, (2, 9))
    logits = model(ids)
    assert logits.shape == (2, 9, 50)
    ids2 = ids.clone()
    ids2[:, 6:] = torch.randint(0, 50, (2, 3))
    torch.testing.assert_close(model(ids)[:, :6], model(ids2)[:, :6],
                               atol=1e-5, rtol=1e-5)


def test_cross_entropy_matches_torch():
    logits = torch.randn(4, 7, 50) * 30
    targets = torch.randint(0, 50, (4, 7))
    mine = tf.cross_entropy(logits, targets)
    ref = F.cross_entropy(logits.reshape(-1, 50), targets.reshape(-1))
    torch.testing.assert_close(mine, ref, atol=1e-5, rtol=1e-5)
    assert torch.isfinite(mine)


def test_tiny_model_overfits_repeating_sequence():
    """End-to-end: the full stack must drive loss near zero on a trivial task."""
    torch.manual_seed(3)
    vocab = 10
    seq = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9] * 3])
    x, y = seq[:, :-1], seq[:, 1:]
    model = tf.TransformerLM(vocab, d_model=32, n_layers=2, n_heads=4,
                             max_len=64)
    opt = tf.AdamW(model.parameters(), lr=3e-3, weight_decay=0.0)
    first = None
    for step in range(300):
        logits = model(x)
        loss = tf.cross_entropy(logits, y)
        if first is None:
            first = float(loss.detach())
        opt.zero_grad()
        loss.backward()
        tf.clip_grad_norm(model.parameters(), 1.0)
        opt.step()
    final = float(loss.detach())
    assert first > 1.5 and final < 0.1, f"{first=} {final=}"


def test_generate_shapes_and_determinism():
    torch.manual_seed(4)
    model = tf.TransformerLM(vocab_size=20, d_model=32, n_layers=1, n_heads=4)
    prefix = torch.tensor([1, 2, 3])
    out1 = model.generate(prefix, max_new_tokens=5, seed=0)
    out2 = model.generate(prefix, max_new_tokens=5, seed=0)
    assert out1.shape == (8,)
    assert torch.equal(out1[:3], prefix)
    assert torch.equal(out1, out2)


def test_reflect_causality():
    """Public reflection of a private hard test: causality."""
    torch.manual_seed(21)
    blk = tf.TransformerBlock(16, 2, 32)
    x = torch.randn(1, 6, 16)
    x2 = x.clone()
    x2[0, 4] += 1.0
    torch.testing.assert_close(blk(x)[0, :4], blk(x2)[0, :4], atol=1e-5, rtol=1e-5)


def test_reflect_numerical_stability():
    """Public reflection of a private hard test: numerical_stability (cross-entropy, RMSNorm)."""
    torch.manual_seed(23)
    logits = torch.randn(3, 9) * 100
    targets = torch.randint(0, 9, (3,))
    ce = tf.cross_entropy(logits, targets)
    assert torch.isfinite(ce)
    torch.testing.assert_close(ce, F.cross_entropy(logits, targets), atol=1e-5, rtol=1e-5)
    norm = tf.RMSNorm(8)
    h = torch.randn(2, 8)
    torch.testing.assert_close(norm(h * 10), norm(h), atol=1e-5, rtol=1e-5)


def test_reflect_generate_sampling():
    """Public reflection of a private hard test: generate_sampling."""
    torch.manual_seed(25)
    model = tf.TransformerLM(vocab_size=12, d_model=16, n_layers=1, n_heads=2)
    prefix = torch.tensor([5, 11])
    out = model.generate(prefix, max_new_tokens=6, temperature=0.5, seed=1)
    assert out.shape == (8,) and int(out.min()) >= 0 and int(out.max()) < 12
    assert torch.equal(out, model.generate(prefix, max_new_tokens=6, temperature=0.5, seed=1))
