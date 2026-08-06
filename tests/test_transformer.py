import numpy as np
import pytest

from src.transformer import (
    FeedForward,
    MultiHeadAttention,
    TransformerBlock,
    causal_mask,
    gelu,
    layer_norm,
    scaled_dot_product_attention,
)


def test_layer_norm_zero_mean_unit_var():
    rng = np.random.default_rng(0)
    x = rng.normal(loc=5.0, scale=3.0, size=(4, 6, 16))
    gamma = np.ones(16)
    beta = np.zeros(16)
    y = layer_norm(x, gamma, beta)
    assert np.allclose(y.mean(axis=-1), 0.0, atol=1e-6)
    assert np.allclose(y.var(axis=-1), 1.0, atol=1e-4)


def test_layer_norm_affine_shift_and_scale():
    x = np.array([[1.0, 2.0, 3.0, 4.0]])
    gamma = np.full(4, 2.0)
    beta = np.full(4, -1.0)
    y = layer_norm(x, gamma, beta)
    # after standardizing, affine is 2 * x_hat - 1
    x_hat = (x - x.mean()) / np.sqrt(x.var() + 1e-5)
    assert np.allclose(y, 2.0 * x_hat - 1.0)


def test_gelu_known_values():
    assert gelu(np.array([0.0]))[0] == pytest.approx(0.0)
    # tanh approx of Φ: positive inputs stay positive, negatives shrink toward 0
    assert gelu(np.array([1.0]))[0] > 0.8
    assert gelu(np.array([-1.0]))[0] < 0.0
    assert gelu(np.array([-1.0]))[0] > -0.2


def test_attention_shapes_and_row_stochastic():
    rng = np.random.default_rng(1)
    q = rng.normal(size=(2, 5, 8))
    out, w = scaled_dot_product_attention(q, q, q)
    assert out.shape == (2, 5, 8)
    assert w.shape == (2, 5, 5)
    assert np.allclose(w.sum(axis=-1), 1.0)


def test_causal_mask_blocks_future():
    rng = np.random.default_rng(2)
    t, d = 6, 4
    q = rng.normal(size=(t, d))
    _, w = scaled_dot_product_attention(q, q, q, mask=causal_mask(t))
    assert np.allclose(np.triu(w, k=1), 0.0)
    assert np.allclose(w.sum(axis=-1), 1.0)


def test_empty_sequence_attention():
    q = np.zeros((0, 4))
    out, w = scaled_dot_product_attention(q, q, q)
    assert out.shape == (0, 4)
    assert w.shape == (0, 0)


def test_single_token_attention_is_identity_weight():
    q = np.array([[1.0, 0.0]])
    v = np.array([[3.0, 4.0]])
    out, w = scaled_dot_product_attention(q, q, v)
    assert w[0, 0] == pytest.approx(1.0)
    assert np.allclose(out, v)


def test_multihead_shapes():
    mha = MultiHeadAttention(d_model=32, n_heads=4, seed=0)
    x = np.random.default_rng(3).normal(size=(2, 7, 32))
    out, w = mha.forward(x)
    assert out.shape == (2, 7, 32)
    assert w.shape == (2, 4, 7, 7)
    assert np.allclose(w.sum(axis=-1), 1.0)
    assert len(mha.parameters()) == 4


def test_multihead_requires_divisible_d_model():
    with pytest.raises(ValueError, match="divisible"):
        MultiHeadAttention(d_model=30, n_heads=4)


def test_multihead_rejects_bad_feature_dim():
    mha = MultiHeadAttention(d_model=16, n_heads=2, seed=0)
    with pytest.raises(ValueError, match="d_model"):
        mha.forward(np.zeros((3, 15)))


def test_multihead_causal():
    mha = MultiHeadAttention(d_model=16, n_heads=2, seed=1)
    x = np.random.default_rng(4).normal(size=(5, 16))
    _, w = mha.forward(x, mask=causal_mask(5))
    assert np.allclose(np.triu(w, k=1), 0.0)


def test_ffn_expands_and_contracts():
    ffn = FeedForward(d_model=16, d_ff=64, seed=0)
    x = np.random.default_rng(5).normal(size=(3, 8, 16))
    y = ffn.forward(x)
    assert y.shape == x.shape
    assert ffn.W1.shape == (16, 64)
    assert ffn.W2.shape == (64, 16)
    assert len(ffn.parameters()) == 4


def test_block_preserves_sequence_shape():
    block = TransformerBlock(d_model=32, n_heads=4, d_ff=64, seed=0)
    x = np.random.default_rng(6).normal(size=(2, 9, 32))
    result = block.forward(x)
    assert result.values.shape == (2, 9, 32)
    assert result.attn_weights.shape == (2, 4, 9, 9)
    assert np.all(np.isfinite(result.values))


def test_block_residual_keeps_input_when_sublayers_near_zero():
    # zero the branch weights so the residual stream is the only path
    block = TransformerBlock(d_model=8, n_heads=2, d_ff=16, seed=0)
    for p in block.attn.parameters():
        p.fill(0.0)
    for p in block.ffn.parameters():
        p.fill(0.0)
    x = np.random.default_rng(7).normal(size=(4, 8))
    y = block.forward(x).values
    assert np.allclose(y, x)


def test_block_causal_mask():
    block = TransformerBlock(d_model=16, n_heads=2, seed=2)
    x = np.random.default_rng(8).normal(size=(6, 16))
    result = block.forward(x, mask=causal_mask(6))
    assert np.allclose(np.triu(result.attn_weights, k=1), 0.0)


def test_block_empty_sequence():
    block = TransformerBlock(d_model=8, n_heads=2, seed=0)
    x = np.zeros((0, 8))
    result = block.forward(x)
    assert result.values.shape == (0, 8)
    assert result.attn_weights.shape == (2, 0, 0)


def test_block_single_token():
    block = TransformerBlock(d_model=8, n_heads=2, seed=3)
    x = np.random.default_rng(9).normal(size=(1, 8))
    result = block.forward(x)
    assert result.values.shape == (1, 8)
    assert result.attn_weights.shape == (2, 1, 1)
    assert result.attn_weights[0, 0, 0] == pytest.approx(1.0)


def test_block_rejects_bad_shapes_and_ctor():
    block = TransformerBlock(d_model=16, n_heads=2, seed=0)
    assert len(block.parameters()) == 12  # 4 attn + 4 ffn + 4 LN
    with pytest.raises(ValueError, match="d_model"):
        block.forward(np.zeros((3, 10)))
    with pytest.raises(ValueError):
        TransformerBlock(d_model=0, n_heads=1)
    with pytest.raises(ValueError):
        TransformerBlock(d_model=8, n_heads=3)
    with pytest.raises(ValueError):
        TransformerBlock(d_model=8, n_heads=2, eps=0.0)


def test_stacked_blocks_stay_finite():
    rng = np.random.default_rng(10)
    x = rng.normal(size=(2, 5, 32))
    for i in range(4):
        x = TransformerBlock(d_model=32, n_heads=4, seed=100 + i).forward(x).values
    assert np.all(np.isfinite(x))
    assert x.shape == (2, 5, 32)
