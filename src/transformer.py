"""Minimal Pre-LN self-attention block: multi-head attention, FFN, residual, LayerNorm.

Forward-only: no backward/grad path. Pre-LN layout (stable for stacking):

    y = x + MultiHeadAttn(LayerNorm(x))
    z = y + FFN(LayerNorm(y))

Multi-head attention runs h scaled-dot-product heads on d_k = d_model/h slices,
concatenates, and projects with W_o. Optional causal mask supports decoder-style
use; omit the mask for bidirectional encoder-style attention. The FFN is a
shared two-layer MLP per token. Residuals keep a residual stream; LayerNorm
holds its scale.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]


def softmax(x: Array, axis: int = -1) -> Array:
    """Numerically stable softmax. All-(-inf) rows return zeros (no mass)."""
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return np.zeros_like(x, dtype=np.float64)
    max_x = np.max(x, axis=axis, keepdims=True)
    # All-masked rows: max is -inf; emit zeros without nan/RuntimeWarning.
    valid = np.isfinite(max_x)
    with np.errstate(invalid="ignore"):
        shifted = np.where(valid, x - max_x, 0.0)
    exp = np.where(valid, np.exp(shifted), 0.0)
    denom = np.sum(exp, axis=axis, keepdims=True)
    out = np.zeros_like(exp)
    np.divide(exp, denom, out=out, where=denom > 0)
    return out


def causal_mask(seq_len: int) -> Array:
    if seq_len < 0:
        raise ValueError("seq_len must be non-negative")
    return np.tril(np.ones((seq_len, seq_len), dtype=bool))


def layer_norm(
    x: Array,
    gamma: Array,
    beta: Array,
    *,
    eps: float = 1e-5,
) -> Array:
    """Per-token normalize last axis, then affine (γ, β shaped (d_model,))."""
    x = np.asarray(x, dtype=np.float64)
    if x.shape[-1] != gamma.shape[0] or gamma.shape != beta.shape:
        raise ValueError("gamma/beta must match last dim of x")
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    return gamma * (x - mean) / np.sqrt(var + eps) + beta


def gelu(x: Array) -> Array:
    """GELU via the tanh approximation (Hendrycks & Gimpel)."""
    x = np.asarray(x, dtype=np.float64)
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x**3)))


def scaled_dot_product_attention(
    q: Array,
    k: Array,
    v: Array,
    *,
    mask: Array | None = None,
) -> tuple[Array, Array]:
    """Attention(Q,K,V) = softmax(Q Kᵀ / √d_k) V. Returns (out, weights)."""
    q = np.asarray(q, dtype=np.float64)
    k = np.asarray(k, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    if q.ndim < 2 or k.ndim < 2 or v.ndim < 2:
        raise ValueError("q, k, v need at least (seq, feature)")
    if q.shape[-1] != k.shape[-1]:
        raise ValueError(f"d_k mismatch: {q.shape[-1]} vs {k.shape[-1]}")
    if k.shape[-2] != v.shape[-2]:
        raise ValueError(f"T_k mismatch: {k.shape[-2]} vs {v.shape[-2]}")
    d_k = q.shape[-1]
    if d_k == 0:
        raise ValueError("d_k must be positive")
    scores = np.matmul(q, np.swapaxes(k, -1, -2)) / np.sqrt(float(d_k))
    if mask is not None:
        mask_arr = np.asarray(mask)
        if mask_arr.dtype != np.bool_:
            mask_arr = mask_arr.astype(bool)
        scores = np.where(mask_arr, scores, np.float64(-np.inf))
    weights = np.nan_to_num(softmax(scores, axis=-1), nan=0.0, posinf=0.0, neginf=0.0)
    return np.matmul(weights, v), weights


def _xavier(rng: np.random.Generator, fan_in: int, fan_out: int) -> Array:
    std = np.sqrt(2.0 / (fan_in + fan_out))
    return rng.normal(0.0, std, size=(fan_in, fan_out)).astype(np.float64)


@dataclass
class BlockOutput:
    values: Array
    attn_weights: Array


class MultiHeadAttention:
    """h parallel heads on d_k = d_model/h, then concat and W_o."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        *,
        rng: np.random.Generator | None = None,
        seed: int | None = None,
    ) -> None:
        if d_model <= 0 or n_heads <= 0:
            raise ValueError("d_model and n_heads must be positive")
        if d_model % n_heads != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
            )
        if rng is None:
            rng = np.random.default_rng(seed)
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.W_q = _xavier(rng, d_model, d_model)
        self.W_k = _xavier(rng, d_model, d_model)
        self.W_v = _xavier(rng, d_model, d_model)
        self.W_o = _xavier(rng, d_model, d_model)

    def parameters(self) -> list[Array]:
        return [self.W_q, self.W_k, self.W_v, self.W_o]

    def _split_heads(self, x: Array) -> Array:
        *lead, t, _ = x.shape
        return np.moveaxis(x.reshape(*lead, t, self.n_heads, self.d_k), -2, -3)

    def _merge_heads(self, x: Array) -> Array:
        x = np.moveaxis(x, -3, -2)
        *lead, t, _, _ = x.shape
        return x.reshape(*lead, t, self.d_model)

    def forward(
        self, x: Array, *, mask: Array | None = None
    ) -> tuple[Array, Array]:
        x = np.asarray(x, dtype=np.float64)
        if x.ndim < 2:
            raise ValueError("x must have shape (..., T, d_model)")
        if x.shape[-1] != self.d_model:
            raise ValueError(
                f"last dim of x is {x.shape[-1]}, expected d_model={self.d_model}"
            )
        q = self._split_heads(x @ self.W_q)
        k = self._split_heads(x @ self.W_k)
        v = self._split_heads(x @ self.W_v)
        out, weights = scaled_dot_product_attention(q, k, v, mask=mask)
        return self._merge_heads(out) @ self.W_o, weights


class FeedForward:
    """Position-wise MLP: d_model → d_ff → d_model with GELU between."""

    def __init__(
        self,
        d_model: int,
        d_ff: int | None = None,
        *,
        rng: np.random.Generator | None = None,
        seed: int | None = None,
    ) -> None:
        if d_model <= 0:
            raise ValueError("d_model must be positive")
        d_ff = 4 * d_model if d_ff is None else d_ff
        if d_ff <= 0:
            raise ValueError("d_ff must be positive")
        if rng is None:
            rng = np.random.default_rng(seed)
        self.d_model = d_model
        self.d_ff = d_ff
        self.W1 = _xavier(rng, d_model, d_ff)
        self.b1 = np.zeros(d_ff, dtype=np.float64)
        self.W2 = _xavier(rng, d_ff, d_model)
        self.b2 = np.zeros(d_model, dtype=np.float64)

    def parameters(self) -> list[Array]:
        return [self.W1, self.b1, self.W2, self.b2]

    def forward(self, x: Array) -> Array:
        x = np.asarray(x, dtype=np.float64)
        if x.shape[-1] != self.d_model:
            raise ValueError(
                f"last dim of x is {x.shape[-1]}, expected d_model={self.d_model}"
            )
        return gelu(x @ self.W1 + self.b1) @ self.W2 + self.b2


class TransformerBlock:
    """Pre-LN self-attention block: residual MHA then residual FFN.

    Forward-only: `parameters()` returns weight tensors for inspection, not a
    trainable optimizer interface (no backward/grad path). Attention weights
    have shape (..., n_heads, T, T). Optional causal mask for decoder-style use.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int | None = None,
        *,
        eps: float = 1e-5,
        rng: np.random.Generator | None = None,
        seed: int | None = None,
    ) -> None:
        if d_model <= 0:
            raise ValueError("d_model must be positive")
        if eps <= 0.0:
            raise ValueError("eps must be positive")
        if rng is None:
            rng = np.random.default_rng(seed)
        self.d_model = d_model
        self.eps = float(eps)
        self.attn = MultiHeadAttention(d_model, n_heads, rng=rng)
        self.ffn = FeedForward(d_model, d_ff, rng=rng)
        self.ln1_g = np.ones(d_model, dtype=np.float64)
        self.ln1_b = np.zeros(d_model, dtype=np.float64)
        self.ln2_g = np.ones(d_model, dtype=np.float64)
        self.ln2_b = np.zeros(d_model, dtype=np.float64)

    def parameters(self) -> list[Array]:
        return [
            *self.attn.parameters(),
            *self.ffn.parameters(),
            self.ln1_g,
            self.ln1_b,
            self.ln2_g,
            self.ln2_b,
        ]

    def forward(self, x: Array, *, mask: Array | None = None) -> BlockOutput:
        x = np.asarray(x, dtype=np.float64)
        if x.ndim < 2:
            raise ValueError("x must have shape (..., T, d_model)")
        if x.shape[-1] != self.d_model:
            raise ValueError(
                f"last dim of x is {x.shape[-1]}, expected d_model={self.d_model}"
            )
        h, weights = self.attn.forward(
            layer_norm(x, self.ln1_g, self.ln1_b, eps=self.eps),
            mask=mask,
        )
        x = x + h
        x = x + self.ffn.forward(layer_norm(x, self.ln2_g, self.ln2_b, eps=self.eps))
        return BlockOutput(values=x, attn_weights=weights)
