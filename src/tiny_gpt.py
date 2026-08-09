"""Char-level decoder-only transformer language model (a tiny GPT).

Next-token prediction over characters: each position predicts the next
character from a causal window of earlier ones. One residual block with
single-head scaled dot-product self-attention and a ReLU FFN, plus token and
learned position embeddings and a linear LM head. Gradients are written out by
hand so the chain through attention is visible end to end.

    logits_t = f(x_{≤t});  L = mean CE(softmax(logits_t), x_{t+1})

Causal masking zeros the upper triangle of the score matrix so position t
cannot read t+1..T, which is what makes generation autoregressive.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from src.optimizers import Adam, Optimizer

Array = NDArray[np.float64]
IntArray = NDArray[np.int64]


def _softmax_last(x: Array) -> Array:
    x = x - x.max(axis=-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=-1, keepdims=True)


def _xavier(rng: np.random.Generator, n_in: int, n_out: int) -> Array:
    return rng.normal(0.0, np.sqrt(2.0 / (n_in + n_out)), size=(n_in, n_out))


class CharTokenizer:
    """Sorted unique characters of a corpus; encode/decode as int ids."""

    def __init__(self, text: str) -> None:
        if not text:
            raise ValueError("tokenizer needs non-empty text")
        self.chars = sorted(set(text))
        self.stoi = {c: i for i, c in enumerate(self.chars)}
        self.itos = {i: c for i, c in enumerate(self.chars)}

    @property
    def vocab_size(self) -> int:
        return len(self.chars)

    def encode(self, text: str) -> IntArray:
        try:
            return np.array([self.stoi[c] for c in text], dtype=np.int64)
        except KeyError as e:
            raise ValueError(f"unknown character {e.args[0]!r}") from e

    def decode(self, ids: IntArray | list[int]) -> str:
        return "".join(self.itos[int(i)] for i in np.asarray(ids).reshape(-1))


@dataclass
class TinyGPT:
    """Trainable one-block char LM. Parameters are plain numpy arrays."""

    vocab_size: int
    d_model: int
    block_size: int
    tok_emb: Array
    pos_emb: Array
    W_q: Array
    W_k: Array
    W_v: Array
    W_o: Array
    W1: Array
    b1: Array
    W2: Array
    b2: Array
    W_lm: Array
    b_lm: Array

    @classmethod
    def create(
        cls,
        vocab_size: int,
        d_model: int = 32,
        block_size: int = 16,
        seed: int = 0,
    ) -> TinyGPT:
        if vocab_size < 1 or d_model < 1 or block_size < 1:
            raise ValueError("vocab_size, d_model, block_size must be positive")
        rng = np.random.default_rng(seed)
        d = d_model
        return cls(
            vocab_size=vocab_size,
            d_model=d,
            block_size=block_size,
            tok_emb=rng.normal(0.0, 0.02, size=(vocab_size, d)),
            pos_emb=rng.normal(0.0, 0.02, size=(block_size, d)),
            W_q=_xavier(rng, d, d),
            W_k=_xavier(rng, d, d),
            W_v=_xavier(rng, d, d),
            W_o=_xavier(rng, d, d),
            W1=_xavier(rng, d, 4 * d),
            b1=np.zeros(4 * d),
            W2=_xavier(rng, 4 * d, d),
            b2=np.zeros(d),
            W_lm=_xavier(rng, d, vocab_size),
            b_lm=np.zeros(vocab_size),
        )

    def parameters(self) -> list[Array]:
        return [
            self.tok_emb,
            self.pos_emb,
            self.W_q,
            self.W_k,
            self.W_v,
            self.W_o,
            self.W1,
            self.b1,
            self.W2,
            self.b2,
            self.W_lm,
            self.b_lm,
        ]

    def _as_batch(self, idx: IntArray, *, max_t: int) -> IntArray:
        arr = np.asarray(idx, dtype=np.int64)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if arr.ndim != 2 or arr.shape[1] == 0:
            raise ValueError("idx must be non-empty (T,) or (B, T)")
        if arr.shape[1] > max_t:
            raise ValueError(f"length {arr.shape[1]} exceeds limit {max_t}")
        if arr.min() < 0 or arr.max() >= self.vocab_size:
            raise ValueError("token id out of vocabulary range")
        return arr

    def logits(self, idx: IntArray) -> Array:
        """(B, T, V) next-token logits for each position (causal)."""
        return self._forward(self._as_batch(idx, max_t=self.block_size))[0]

    def _forward(self, idx: IntArray) -> tuple[Array, dict[str, Array]]:
        b, t = idx.shape
        d = self.d_model
        x = self.tok_emb[idx] + self.pos_emb[:t]
        q, k, v = x @ self.W_q, x @ self.W_k, x @ self.W_v
        scale = 1.0 / np.sqrt(float(d))
        scores = (q @ np.swapaxes(k, -1, -2)) * scale
        scores = np.where(np.tril(np.ones((t, t), dtype=bool)), scores, -np.inf)
        attn = np.nan_to_num(_softmax_last(scores), nan=0.0)
        att_out = attn @ v
        y = x + att_out @ self.W_o
        z1 = y @ self.W1 + self.b1
        a1 = np.maximum(z1, 0.0)
        z = y + a1 @ self.W2 + self.b2
        logits = z @ self.W_lm + self.b_lm
        cache = {
            "idx": idx,
            "x": x,
            "q": q,
            "k": k,
            "v": v,
            "attn": attn,
            "att_out": att_out,
            "y": y,
            "z1": z1,
            "a1": a1,
            "z": z,
        }
        return logits, cache

    def loss_and_grads(self, idx: IntArray) -> tuple[float, list[Array]]:
        """Next-token CE on idx[:, :-1] → idx[:, 1:]."""
        arr = self._as_batch(idx, max_t=self.block_size + 1)
        if arr.shape[1] < 2:
            raise ValueError("need at least 2 tokens for next-token loss")
        inp, tgt = arr[:, :-1], arr[:, 1:]
        logits, cache = self._forward(inp)
        b, t, _ = logits.shape
        log_p = logits - np.logaddexp.reduce(logits, axis=-1, keepdims=True)
        rows = np.arange(b)[:, None]
        cols = np.arange(t)[None, :]
        loss = float((-log_p[rows, cols, tgt]).mean())
        dlogits = _softmax_last(logits)
        dlogits[rows, cols, tgt] -= 1.0
        dlogits /= b * t
        return loss, self._backward(dlogits, cache)

    def _backward(self, dlogits: Array, cache: dict[str, Array]) -> list[Array]:
        idx, x = cache["idx"], cache["x"]
        q, k, v = cache["q"], cache["k"], cache["v"]
        attn, att_out = cache["attn"], cache["att_out"]
        y, z1, a1, z = cache["y"], cache["z1"], cache["a1"], cache["z"]
        b, t, d = x.shape
        scale = 1.0 / np.sqrt(float(d))
        flat = b * t

        dW_lm = z.reshape(flat, d).T @ dlogits.reshape(flat, -1)
        db_lm = dlogits.sum(axis=(0, 1))
        dz = dlogits @ self.W_lm.T

        dy = dz.copy()
        da1 = dz @ self.W2.T
        dW2 = a1.reshape(flat, -1).T @ dz.reshape(flat, d)
        db2 = dz.sum(axis=(0, 1))
        dz1 = da1 * (z1 > 0.0)
        dW1 = y.reshape(flat, d).T @ dz1.reshape(flat, -1)
        db1 = dz1.sum(axis=(0, 1))
        dy += dz1 @ self.W1.T

        dx = dy.copy()
        datt_out = dy @ self.W_o.T
        dW_o = att_out.reshape(flat, d).T @ dy.reshape(flat, d)
        dattn = datt_out @ np.swapaxes(v, -1, -2)
        dv = np.swapaxes(attn, -1, -2) @ datt_out
        sum_d = (dattn * attn).sum(axis=-1, keepdims=True)
        dscores = attn * (dattn - sum_d)
        dscores = np.where(np.tril(np.ones((t, t), dtype=bool)), dscores, 0.0)
        dq = (dscores @ k) * scale
        dk = (np.swapaxes(dscores, -1, -2) @ q) * scale
        dW_q = x.reshape(flat, d).T @ dq.reshape(flat, d)
        dW_k = x.reshape(flat, d).T @ dk.reshape(flat, d)
        dW_v = x.reshape(flat, d).T @ dv.reshape(flat, d)
        dx += dq @ self.W_q.T + dk @ self.W_k.T + dv @ self.W_v.T

        d_tok = np.zeros_like(self.tok_emb)
        np.add.at(d_tok, idx.reshape(-1), dx.reshape(flat, d))
        d_pos = np.zeros_like(self.pos_emb)
        d_pos[:t] = dx.sum(axis=0)
        return [
            d_tok,
            d_pos,
            dW_q,
            dW_k,
            dW_v,
            dW_o,
            dW1,
            db1,
            dW2,
            db2,
            dW_lm,
            db_lm,
        ]

    def generate(
        self,
        idx: IntArray,
        max_new_tokens: int,
        *,
        temperature: float = 1.0,
        seed: int | None = None,
    ) -> IntArray:
        """Autoregressive sample; returns prompt plus new tokens."""
        if max_new_tokens < 0 or temperature <= 0.0:
            raise ValueError("max_new_tokens >= 0 and temperature > 0 required")
        rng = np.random.default_rng(seed)
        out = self._as_batch(idx, max_t=self.block_size)
        for _ in range(max_new_tokens):
            ctx = out[:, -self.block_size :]
            probs = _softmax_last(self.logits(ctx)[:, -1, :] / temperature)
            nxt = np.array(
                [rng.choice(self.vocab_size, p=probs[i]) for i in range(out.shape[0])],
                dtype=np.int64,
            )
            out = np.concatenate([out, nxt[:, None]], axis=1)
        return out


def fit(
    text: str,
    *,
    d_model: int = 32,
    block_size: int = 16,
    steps: int = 400,
    batch_size: int = 16,
    lr: float = 0.05,
    seed: int = 0,
    optimizer: Optimizer | None = None,
) -> tuple[TinyGPT, CharTokenizer, list[float]]:
    """Train on `text` by next-token CE; returns model, tokenizer, loss history."""
    if len(text) < block_size + 1:
        raise ValueError("text must be longer than block_size")
    tok = CharTokenizer(text)
    data = tok.encode(text)
    model = TinyGPT.create(
        tok.vocab_size, d_model=d_model, block_size=block_size, seed=seed
    )
    opt: Optimizer = Adam(lr=lr) if optimizer is None else optimizer
    rng = np.random.default_rng(seed + 1)
    history: list[float] = []
    params = model.parameters()
    max_start = data.size - block_size - 1
    for _ in range(steps):
        starts = rng.integers(0, max_start + 1, size=batch_size)
        batch = np.stack([data[s : s + block_size + 1] for s in starts])
        loss, grads = model.loss_and_grads(batch)
        opt.step(params, grads)
        history.append(loss)
    return model, tok, history
