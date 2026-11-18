"""A4 Milestone 1 public tests: attention.py."""
import math
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..",
                                os.environ.get("SUBMISSION_DIR", "starter")))
import attention as tf  # noqa: E402

torch.manual_seed(0)


def test_softmax_matches_torch_and_is_stable():
    x = torch.randn(4, 7) * 50  # large values: naive exp overflows
    mine = tf.softmax(x, dim=-1)
    torch.testing.assert_close(mine, F.softmax(x, dim=-1))
    assert torch.isfinite(mine).all()
    torch.testing.assert_close(mine.sum(-1), torch.ones(4))


def test_attention_matches_torch_sdpa():
    Q, K, V = (torch.randn(2, 3, 5, 8) for _ in range(3))
    mine = tf.scaled_dot_product_attention(Q, K, V)
    ref = F.scaled_dot_product_attention(Q, K, V)
    torch.testing.assert_close(mine, ref, atol=1e-5, rtol=1e-5)
    # with causal mask
    mask = tf.causal_mask(5)
    mine_c = tf.scaled_dot_product_attention(Q, K, V, mask=mask)
    ref_c = F.scaled_dot_product_attention(Q, K, V, is_causal=True)
    torch.testing.assert_close(mine_c, ref_c, atol=1e-5, rtol=1e-5)


def test_causal_mask_shape_and_content():
    m = tf.causal_mask(4)
    assert m.dtype == torch.bool and m.shape == (4, 4)
    assert m[0, 0] and m[3, 0] and not m[0, 1]


def test_rope_preserves_norm_and_relative_positions():
    d = 16
    rope = tf.RotaryEmbedding(d, max_len=64)
    x = torch.randn(1, 10, d)
    out = rope(x)
    # rotation preserves the norm of each vector
    torch.testing.assert_close(out.norm(dim=-1), x.norm(dim=-1),
                               atol=1e-5, rtol=1e-5)
    # relative property: <rope(q,i), rope(k,j)> depends only on (i - j)
    q = torch.randn(d)
    k = torch.randn(d)
    def dot_at(i, j):
        qi = rope(q.view(1, 1, d), positions=torch.tensor([i]))[0, 0]
        kj = rope(k.view(1, 1, d), positions=torch.tensor([j]))[0, 0]
        return float(qi @ kj)
    assert abs(dot_at(3, 1) - dot_at(10, 8)) < 1e-4
    assert abs(dot_at(0, 0) - float(q @ k)) < 1e-5  # position 0 = identity


def test_mha_output_is_causal():
    torch.manual_seed(1)
    mha = tf.MultiHeadAttention(d_model=32, n_heads=4)
    x = torch.randn(1, 8, 32)
    base = mha(x)
    x2 = x.clone()
    x2[0, 5:] = torch.randn(3, 32)  # perturb only future positions
    out2 = mha(x2)
    torch.testing.assert_close(base[0, :5], out2[0, :5], atol=1e-5, rtol=1e-5)
    assert not torch.allclose(base[0, 5:], out2[0, 5:])


def test_mha_projections_are_d_model_wide():
    """Each of q_proj, k_proj, v_proj maps d_model to d_model (all heads at once)."""
    mha = tf.MultiHeadAttention(d_model=32, n_heads=4)
    assert mha.k_proj.weight.shape == (32, 32)
    x = torch.randn(1, 6, 32)
    assert mha(x).shape == (1, 6, 32)


def test_reflect_mha_naive_loop():
    """Public reflection of a private hard test: mha_naive_loop."""
    torch.manual_seed(20)
    T, d_model, H = 4, 16, 2
    dh = d_model // H
    mha = tf.MultiHeadAttention(d_model=d_model, n_heads=H)
    x = torch.randn(1, T, d_model)
    with torch.no_grad():
        q = mha.q_proj(x).view(T, H, dh)
        k = mha.k_proj(x).view(T, H, dh)
        v = mha.v_proj(x).view(T, H, dh)
        o = torch.zeros(T, H, dh)
        for h in range(H):
            qh, kh = mha.rope(q[:, h]), mha.rope(k[:, h])
            for t in range(T):
                w = F.softmax(qh[t] @ kh[: t + 1].T / math.sqrt(dh), dim=-1)
                o[t, h] = w @ v[: t + 1, h]
        ref = mha.o_proj(o.reshape(1, T, d_model))
    torch.testing.assert_close(mha(x), ref, atol=1e-5, rtol=1e-5)


def test_reflect_rope_relative():
    """Public reflection of a private hard test: rope_relative."""
    torch.manual_seed(22)
    d = 8
    rope = tf.RotaryEmbedding(d, max_len=32)
    q, k = torch.randn(d), torch.randn(d)

    def dot_at(i, j):
        qi = rope(q.view(1, 1, d), positions=torch.tensor([i]))[0, 0]
        kj = rope(k.view(1, 1, d), positions=torch.tensor([j]))[0, 0]
        return float(qi @ kj)

    assert abs(dot_at(2, 0) - dot_at(5, 3)) < 1e-4


def test_reflect_softmax_stability():
    """Public reflection of a private hard test: softmax_stability."""
    torch.manual_seed(23)
    x = torch.randn(2, 6) * 1e3
    assert torch.isfinite(tf.softmax(x, dim=-1)).all()
