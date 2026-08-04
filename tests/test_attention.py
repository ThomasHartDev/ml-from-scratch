import numpy as np
import pytest

from src.attention import (
    SelfAttentionHead,
    causal_mask,
    scaled_dot_product_attention,
    softmax,
)


def test_softmax_rows_sum_to_one():
    x = np.array([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0], [-5.0, 10.0, -5.0]])
    s = softmax(x, axis=-1)
    assert s.shape == x.shape
    assert np.allclose(s.sum(axis=-1), 1.0)
    assert np.all(s >= 0.0)


def test_softmax_is_shift_invariant():
    x = np.array([[1.0, 3.0, -2.0]])
    assert np.allclose(softmax(x), softmax(x + 100.0))


def test_softmax_large_logits_stay_finite():
    x = np.array([[1e3, 1e3 + 50.0, -1e3]])
    s = softmax(x)
    assert np.all(np.isfinite(s))
    assert s[0, 1] == pytest.approx(1.0, abs=1e-12)
    assert s[0, 0] == pytest.approx(0.0, abs=1e-12)


def test_attention_shapes_unbatched():
    t_q, t_k, d_k, d_v = 4, 5, 8, 6
    q = np.zeros((t_q, d_k))
    k = np.zeros((t_k, d_k))
    v = np.zeros((t_k, d_v))
    out, w = scaled_dot_product_attention(q, k, v)
    assert out.shape == (t_q, d_v)
    assert w.shape == (t_q, t_k)
    assert np.allclose(w.sum(axis=-1), 1.0)


def test_attention_shapes_batched():
    b, t, d = 3, 7, 16
    q = np.random.default_rng(0).normal(size=(b, t, d))
    out, w = scaled_dot_product_attention(q, q, q)
    assert out.shape == (b, t, d)
    assert w.shape == (b, t, t)
    assert np.allclose(w.sum(axis=-1), 1.0)


def test_uniform_keys_give_uniform_weights():
    # identical keys => equal similarities => uniform average of values
    t, d = 4, 8
    q = np.ones((t, d))
    k = np.ones((t, d))
    v = np.arange(t * d, dtype=np.float64).reshape(t, d)
    out, w = scaled_dot_product_attention(q, k, v)
    assert np.allclose(w, 1.0 / t)
    assert np.allclose(out, v.mean(axis=0, keepdims=True))


def test_one_hot_when_query_matches_one_key():
    # orthogonal keys, query aligned with key 1 => mass on that key only
    k = np.eye(3, dtype=np.float64)
    q = np.array([[0.0, 40.0, 0.0]])
    v = np.array([[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
    out, w = scaled_dot_product_attention(q, k, v, scale=False)
    assert w[0, 1] == pytest.approx(1.0, abs=1e-12)
    assert np.allclose(out, v[1:2], atol=1e-10)


def test_scale_prevents_saturation_for_large_d_k():
    # without 1/√d_k, unit-variance scores have std ~√d_k and softmax collapses
    rng = np.random.default_rng(1)
    t, d_k = 8, 256
    q = rng.normal(size=(t, d_k))
    k = rng.normal(size=(t, d_k))
    v = rng.normal(size=(t, d_k))
    _, w_scaled = scaled_dot_product_attention(q, k, v, scale=True)
    _, w_raw = scaled_dot_product_attention(q, k, v, scale=False)
    # max weight closer to uniform under scaling; raw is more peaky
    assert w_scaled.max(axis=-1).mean() < w_raw.max(axis=-1).mean()

    # scaled entropy higher on average (less near-one-hot)
    def _entropy(w: np.ndarray) -> np.ndarray:
        p = np.clip(w, 1e-12, 1.0)
        return -np.sum(p * np.log(p), axis=-1)

    assert _entropy(w_scaled).mean() > _entropy(w_raw).mean()


def test_causal_mask_blocks_future():
    t, d = 5, 4
    rng = np.random.default_rng(2)
    q = rng.normal(size=(t, d))
    k = rng.normal(size=(t, d))
    v = rng.normal(size=(t, d))
    mask = causal_mask(t)
    _, w = scaled_dot_product_attention(q, k, v, mask=mask)
    assert np.allclose(w, np.tril(w))
    assert np.allclose(np.triu(w, k=1), 0.0)
    assert np.allclose(w.sum(axis=-1), 1.0)


def test_causal_mask_first_row_is_self_only():
    mask = causal_mask(3)
    q = np.ones((3, 2))
    k = np.ones((3, 2))
    v = np.eye(3, 2)
    _, w = scaled_dot_product_attention(q, k, v, mask=mask)
    assert w[0, 0] == pytest.approx(1.0)
    assert w[0, 1] == pytest.approx(0.0)
    assert w[0, 2] == pytest.approx(0.0)


def test_empty_sequence():
    q = np.zeros((0, 4))
    k = np.zeros((0, 4))
    v = np.zeros((0, 3))
    out, w = scaled_dot_product_attention(q, k, v)
    assert out.shape == (0, 3)
    assert w.shape == (0, 0)


def test_single_token():
    q = np.array([[1.0, 2.0]])
    k = np.array([[1.0, 2.0]])
    v = np.array([[3.0, 4.0, 5.0]])
    out, w = scaled_dot_product_attention(q, k, v)
    assert w.shape == (1, 1)
    assert w[0, 0] == pytest.approx(1.0)
    assert np.allclose(out, v)


def test_mismatched_d_k_raises():
    with pytest.raises(ValueError, match="d_k"):
        scaled_dot_product_attention(
            np.zeros((2, 3)), np.zeros((2, 4)), np.zeros((2, 5))
        )


def test_mismatched_t_k_raises():
    with pytest.raises(ValueError, match="T_k"):
        scaled_dot_product_attention(
            np.zeros((2, 4)), np.zeros((3, 4)), np.zeros((5, 4))
        )


def test_rank_one_input_raises():
    with pytest.raises(ValueError, match="2 dims"):
        scaled_dot_product_attention(np.zeros(3), np.zeros(3), np.zeros(3))


def test_self_attention_head_shapes_and_row_stochastic():
    head = SelfAttentionHead(d_model=16, d_k=8, d_v=8, seed=0)
    x = np.random.default_rng(3).normal(size=(2, 5, 16))
    result = head.forward(x)
    assert result.values.shape == (2, 5, 16)
    assert result.weights.shape == (2, 5, 5)
    assert np.allclose(result.weights.sum(axis=-1), 1.0)
    assert np.all(np.isfinite(result.values))


def test_self_attention_without_output_proj():
    head = SelfAttentionHead(d_model=12, d_k=6, d_v=4, use_output_proj=False, seed=1)
    x = np.random.default_rng(4).normal(size=(3, 12))
    result = head.forward(x)
    assert result.values.shape == (3, 4)
    assert len(head.parameters()) == 3


def test_self_attention_parameters_count_with_output_proj():
    head = SelfAttentionHead(d_model=8, seed=0)
    assert len(head.parameters()) == 4
    assert head.W_q.shape == (8, 8)
    assert head.W_o is not None
    assert head.W_o.shape == (8, 8)


def test_self_attention_rejects_bad_feature_dim():
    head = SelfAttentionHead(d_model=8, seed=0)
    with pytest.raises(ValueError, match="d_model"):
        head.forward(np.zeros((4, 7)))


def test_self_attention_invalid_dims():
    with pytest.raises(ValueError):
        SelfAttentionHead(d_model=0)
    with pytest.raises(ValueError):
        SelfAttentionHead(d_model=4, d_k=0)


def test_self_attention_causal_is_autoregressive():
    head = SelfAttentionHead(d_model=8, d_k=8, use_output_proj=False, seed=5)
    x = np.random.default_rng(6).normal(size=(6, 8))
    mask = causal_mask(6)
    result = head.forward(x, mask=mask)
    assert np.allclose(np.triu(result.weights, k=1), 0.0)


def test_attention_is_permutation_equivariant_on_values_path():
    """Reordering keys/values reorders the weight columns the same way."""
    rng = np.random.default_rng(7)
    t, d = 5, 4
    q = rng.normal(size=(t, d))
    k = rng.normal(size=(t, d))
    v = rng.normal(size=(t, d))
    perm = np.array([2, 0, 4, 1, 3])
    _, w = scaled_dot_product_attention(q, k, v)
    _, w_perm = scaled_dot_product_attention(q, k[perm], v[perm])
    assert np.allclose(w[:, perm], w_perm)


def test_copy_task_identity_weights_retrieve_values():
    """With W=I and matching Q/K, a query peaking on position j copies v_j."""
    head = SelfAttentionHead(d_model=4, d_k=4, d_v=4, use_output_proj=False, seed=0)
    head.W_q = np.eye(4)
    head.W_k = np.eye(4)
    head.W_v = np.eye(4)
    # distinct one-hot-ish keys along the diagonal of content
    x = np.eye(4) * 10.0
    result = head.forward(x, scale=False)
    # each position matches itself strongly
    assert np.allclose(np.argmax(result.weights, axis=-1), np.arange(4))
    assert np.allclose(result.values, x, atol=1e-4)
