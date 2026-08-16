"""Scaled dot-product self-attention, single head.

The attention mechanism (Vaswani et al., 2017) lets each position build a
weighted average of value vectors, with weights from the similarity of query
and key vectors. For one head the core is:

    Attention(Q, K, V) = softmax(Q Kᵀ / √d_k) V

The 1/√d_k scale keeps the logits from growing like √d_k when entries are
unit-variance, which would otherwise push softmax into a near-one-hot regime
and kill gradients. Self-attention means Q, K, and V are linear projections of
the same sequence X, so every token can look at every other token (or a
restricted subset under a mask).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]


def softmax(x: Array, axis: int = -1) -> Array:
    """Numerically stable softmax along `axis`."""
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return np.zeros_like(x, dtype=np.float64)
    shifted = x - np.max(x, axis=axis, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=axis, keepdims=True)


def scaled_dot_product_attention(
    q: Array,
    k: Array,
    v: Array,
    *,
    mask: Array | None = None,
    scale: bool = True,
) -> tuple[Array, Array]:
    """Compute Attention(Q,K,V) and the attention weight matrix.

    Shapes (trailing dims; leading batch dims are broadcast):
      q: (..., T_q, d_k)
      k: (..., T_k, d_k)
      v: (..., T_k, d_v)
      mask: broadcastable to (..., T_q, T_k); True/1 keeps the position,
            False/0 sets its logit to -inf before softmax (causal / padding).

    Returns:
      output: (..., T_q, d_v)
      weights: (..., T_q, T_k) rows sum to 1 over the key axis
    """
    q = np.asarray(q, dtype=np.float64)
    k = np.asarray(k, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)

    if q.ndim < 2 or k.ndim < 2 or v.ndim < 2:
        raise ValueError("q, k, v must have at least 2 dims (seq, feature)")
    if q.shape[-1] != k.shape[-1]:
        raise ValueError(
            f"q and k must share d_k, got {q.shape[-1]} vs {k.shape[-1]}"
        )
    if k.shape[-2] != v.shape[-2]:
        raise ValueError(
            f"k and v must share T_k, got {k.shape[-2]} vs {v.shape[-2]}"
        )

    d_k = q.shape[-1]
    # (..., T_q, d_k) @ (..., d_k, T_k) -> (..., T_q, T_k)
    scores = np.matmul(q, np.swapaxes(k, -1, -2))
    if scale:
        if d_k == 0:
            raise ValueError("d_k must be positive when scale=True")
        scores = scores / np.sqrt(float(d_k))

    if mask is not None:
        mask_arr = np.asarray(mask)
        if mask_arr.dtype != np.bool_:
            mask_arr = mask_arr.astype(bool)
        # False positions become -inf so softmax weight is exactly 0
        scores = np.where(mask_arr, scores, np.float64(-np.inf))

    weights = softmax(scores, axis=-1)
    # all-masked rows yield 0/0 -> nan; treat as zero mass
    weights = np.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)
    output = np.matmul(weights, v)
    return output, weights


def causal_mask(seq_len: int) -> Array:
    """Lower-triangular boolean mask: position i may attend to j <= i."""
    if seq_len < 0:
        raise ValueError("seq_len must be non-negative")
    return np.tril(np.ones((seq_len, seq_len), dtype=bool))


@dataclass
class AttentionOutput:
    """Forward result: projected values plus the weight matrix for inspection."""

    values: Array
    weights: Array


class SelfAttentionHead:
    """Single-head self-attention: Q, K, V from the same sequence via W_q/k/v.

    X is (..., T, d_model). Optional W_o maps the head output back to d_model.
    Parameters are plain numpy arrays so training can plug into the same
    optimizer loop used by the MLP, or stay frozen for pure attention demos.
    """

    def __init__(
        self,
        d_model: int,
        d_k: int | None = None,
        d_v: int | None = None,
        *,
        use_output_proj: bool = True,
        rng: np.random.Generator | None = None,
        seed: int | None = None,
    ) -> None:
        if d_model <= 0:
            raise ValueError("d_model must be positive")
        d_k = d_model if d_k is None else d_k
        d_v = d_k if d_v is None else d_v
        if d_k <= 0 or d_v <= 0:
            raise ValueError("d_k and d_v must be positive")

        if rng is None:
            rng = np.random.default_rng(seed)

        # Xavier-style scale for the linear maps into the head
        def _init(fan_in: int, fan_out: int) -> Array:
            std = np.sqrt(2.0 / (fan_in + fan_out))
            return rng.normal(0.0, std, size=(fan_in, fan_out)).astype(np.float64)

        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.use_output_proj = use_output_proj

        self.W_q = _init(d_model, d_k)
        self.W_k = _init(d_model, d_k)
        self.W_v = _init(d_model, d_v)
        self.W_o: Array | None
        if use_output_proj:
            self.W_o = _init(d_v, d_model)
        else:
            self.W_o = None

    def parameters(self) -> list[Array]:
        params = [self.W_q, self.W_k, self.W_v]
        if self.W_o is not None:
            params.append(self.W_o)
        return params

    def forward(
        self,
        x: Array,
        *,
        mask: Array | None = None,
        scale: bool = True,
    ) -> AttentionOutput:
        x = np.asarray(x, dtype=np.float64)
        if x.ndim < 2:
            raise ValueError("x must have shape (..., T, d_model)")
        if x.shape[-1] != self.d_model:
            raise ValueError(
                f"last dim of x is {x.shape[-1]}, expected d_model={self.d_model}"
            )

        q = np.matmul(x, self.W_q)
        k = np.matmul(x, self.W_k)
        v = np.matmul(x, self.W_v)
        out, weights = scaled_dot_product_attention(q, k, v, mask=mask, scale=scale)
        if self.W_o is not None:
            out = np.matmul(out, self.W_o)
        return AttentionOutput(values=out, weights=weights)
