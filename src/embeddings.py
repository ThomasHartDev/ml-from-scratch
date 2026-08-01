"""Skip-gram word embeddings trained from scratch.

The distributional hypothesis says a word's meaning is the company it keeps.
Skip-gram turns that into a supervised task: given a center word, predict each
nearby context word inside a sliding window. After training, each vocabulary
word is a dense vector (a row of the input embedding matrix). Cosine nearest
neighbors recover words that shared similar contexts.

The model keeps two matrices, matching the original word2vec form: W_in (V x D)
embeds the center word and W_out (V x D) scores every context candidate. The
probability is a full softmax over the vocabulary, p(o|c) ∝ exp(W_out[o] · W_in[c]).
That is exact and fine for the small corpora here; production systems replace the
softmax with negative sampling so each step is O(k) instead of O(V).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]

_TOKEN = re.compile(r"[a-z]+")


def tokenize(text: str) -> list[str]:
    """Lowercase alphabetic tokens; punctuation and digits are dropped."""
    return _TOKEN.findall(text.lower())


def build_vocab(
    tokens: list[str], min_count: int = 1
) -> tuple[dict[str, int], list[str]]:
    if min_count < 1:
        raise ValueError("min_count must be at least 1")
    counts: dict[str, int] = {}
    for t in tokens:
        counts[t] = counts.get(t, 0) + 1
    words = sorted(w for w, c in counts.items() if c >= min_count)
    if not words:
        raise ValueError("vocabulary is empty after min_count filter")
    word_to_idx = {w: i for i, w in enumerate(words)}
    return word_to_idx, words


def skipgram_pairs(
    token_ids: list[int], window: int
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """Emit (center, context) id pairs for every token inside the window.

    The window is symmetric and clipped at sentence ends when callers pass one
    sentence at a time; a flat corpus just uses sequence edges.
    """
    if window < 1:
        raise ValueError("window must be at least 1")
    centers: list[int] = []
    contexts: list[int] = []
    n = len(token_ids)
    for i, c in enumerate(token_ids):
        lo = max(0, i - window)
        hi = min(n, i + window + 1)
        for j in range(lo, hi):
            if j == i:
                continue
            centers.append(c)
            contexts.append(token_ids[j])
    if not centers:
        return (
            np.zeros(0, dtype=np.int64),
            np.zeros(0, dtype=np.int64),
        )
    return (
        np.asarray(centers, dtype=np.int64),
        np.asarray(contexts, dtype=np.int64),
    )


def _softmax(z: Array) -> Array:
    z = z - z.max(axis=-1, keepdims=True)
    ez = np.exp(z)
    return ez / ez.sum(axis=-1, keepdims=True)


def cosine_similarity(a: Array, b: Array) -> float:
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {a.shape} vs {b.shape}")
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


@dataclass
class SkipGramModel:
    """Trained skip-gram: vocab maps plus center and context embedding tables."""

    word_to_idx: dict[str, int]
    idx_to_word: list[str]
    W_in: Array
    W_out: Array

    @property
    def dim(self) -> int:
        return int(self.W_in.shape[1])

    @property
    def vocab_size(self) -> int:
        return int(self.W_in.shape[0])

    def embed(self, word: str) -> Array:
        if word not in self.word_to_idx:
            raise KeyError(f"unknown word: {word!r}")
        return self.W_in[self.word_to_idx[word]].copy()

    def nearest(self, word: str, k: int = 5) -> list[tuple[str, float]]:
        """Top-k cosine neighbors of `word` among the rest of the vocabulary."""
        if k < 1:
            raise ValueError("k must be at least 1")
        if word not in self.word_to_idx:
            raise KeyError(f"unknown word: {word!r}")
        query = self.W_in[self.word_to_idx[word]]
        qn = float(np.linalg.norm(query))
        if qn == 0.0:
            return []
        norms = np.linalg.norm(self.W_in, axis=1)
        # zero vectors contribute nothing useful; skip them and the query itself
        scores = (self.W_in @ query) / np.maximum(norms * qn, 1e-12)
        scores[self.word_to_idx[word]] = -np.inf
        scores[norms == 0.0] = -np.inf
        k_eff = min(k, self.vocab_size - 1)
        if k_eff < 1:
            return []
        # partial sort is enough; vocab stays small in this module
        top = np.argpartition(-scores, k_eff - 1)[:k_eff]
        top = top[np.argsort(-scores[top])]
        return [(self.idx_to_word[int(i)], float(scores[i])) for i in top]


def make_toy_corpus() -> str:
    """A tiny multi-sentence corpus with a few clear co-occurrence clusters.

    City names share the "capital of" pattern; animals share sit/chase patterns;
    royalty and people share ruled/walked patterns. Skip-gram should pull those
    groups closer than unrelated words after enough epochs.
    """
    sentences = [
        "paris is the capital of france",
        "berlin is the capital of germany",
        "london is the capital of england",
        "rome is the capital of italy",
        "madrid is the capital of spain",
        "the cat sat on the mat",
        "the dog sat on the rug",
        "the cat chased the mouse",
        "the dog chased the cat",
        "cats and dogs live in the house",
        "the king ruled the kingdom with the queen",
        "the queen ruled the kingdom with the king",
        "a man walked with a woman in the park",
        "a woman walked with a man in the park",
        "the boy played with the girl in the yard",
        "the girl played with the boy in the yard",
    ]
    # repeat so rare content words see enough center/context pairs
    return " . ".join(sentences * 8)


def fit(
    text: str,
    dim: int = 32,
    window: int = 2,
    lr: float = 0.05,
    epochs: int = 80,
    batch_size: int = 64,
    min_count: int = 1,
    seed: int = 0,
) -> tuple[SkipGramModel, list[float]]:
    """Train skip-gram on `text` with minibatch SGD; return model and loss curve.

    Loss is mean negative log-likelihood of the context word under the full
    softmax. Parameters update from the closed-form softmax gradient
    `p - onehot(context)`.
    """
    if dim < 1:
        raise ValueError("dim must be at least 1")
    if lr <= 0.0:
        raise ValueError("lr must be positive")
    if epochs < 1:
        raise ValueError("epochs must be at least 1")
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")

    tokens = tokenize(text)
    if len(tokens) < 2:
        raise ValueError("need at least two tokens to form a skip-gram pair")

    word_to_idx, idx_to_word = build_vocab(tokens, min_count=min_count)
    ids = [word_to_idx[t] for t in tokens if t in word_to_idx]
    centers, contexts = skipgram_pairs(ids, window)
    if centers.size == 0:
        raise ValueError("no skip-gram pairs produced; try a larger window or corpus")

    v = len(idx_to_word)
    rng = np.random.default_rng(seed)
    # small Gaussian so early softmax is not saturated
    scale = 0.1
    W_in = rng.standard_normal((v, dim)) * scale
    W_out = rng.standard_normal((v, dim)) * scale

    n = centers.shape[0]
    history: list[float] = []
    for _ in range(epochs):
        order = rng.permutation(n)
        total_loss = 0.0
        seen = 0
        for start in range(0, n, batch_size):
            idx = order[start : start + batch_size]
            c = centers[idx]
            o = contexts[idx]
            m = c.shape[0]

            h = W_in[c]  # (m, D)
            scores = h @ W_out.T  # (m, V)
            p = _softmax(scores)
            # NLL of the true context id
            rows = np.arange(m)
            total_loss += float(-np.sum(np.log(np.clip(p[rows, o], 1e-12, 1.0))))
            seen += m

            # dL/dscores = p - onehot(o); average over the batch
            ds = p
            ds[rows, o] -= 1.0
            ds /= m

            dW_out = ds.T @ h  # (V, D)
            dh = ds @ W_out  # (m, D)

            W_out -= lr * dW_out
            # scatter-add dh into the center rows that appeared in the batch
            np.add.at(W_in, c, -lr * dh)

        history.append(total_loss / max(seen, 1))

    model = SkipGramModel(
        word_to_idx=word_to_idx,
        idx_to_word=idx_to_word,
        W_in=W_in,
        W_out=W_out,
    )
    return model, history
