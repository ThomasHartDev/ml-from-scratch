"""Tests for the char-level tiny GPT: shapes, causal mask, grads, learning."""

from __future__ import annotations

import numpy as np
import pytest

from src.tiny_gpt import CharTokenizer, TinyGPT, fit


def test_tokenizer_roundtrip():
    tok = CharTokenizer("abca")
    assert tok.vocab_size == 3
    assert tok.decode(tok.encode("cab")) == "cab"
    with pytest.raises(ValueError, match="non-empty"):
        CharTokenizer("")
    with pytest.raises(ValueError, match="unknown"):
        tok.encode("z")


def test_logits_shape_and_causal_mask():
    model = TinyGPT.create(vocab_size=5, d_model=8, block_size=6, seed=0)
    logits = model.logits(np.array([[0, 1, 2, 3]], dtype=np.int64))
    assert logits.shape == (1, 4, 5)
    a = model.logits(np.array([[0, 1, 2]], dtype=np.int64))
    b = model.logits(np.array([[0, 1, 4]], dtype=np.int64))
    np.testing.assert_allclose(a[0, :2], b[0, :2], atol=1e-12)
    assert not np.allclose(a[0, 2], b[0, 2])


def test_rejects_empty_oob_and_overlong():
    model = TinyGPT.create(vocab_size=4, d_model=4, block_size=4, seed=0)
    with pytest.raises(ValueError, match="non-empty"):
        model.logits(np.zeros((1, 0), dtype=np.int64))
    with pytest.raises(ValueError, match="exceeds"):
        model.logits(np.zeros((1, 5), dtype=np.int64))
    with pytest.raises(ValueError, match="vocabulary"):
        model.logits(np.array([[0, 9]], dtype=np.int64))
    with pytest.raises(ValueError, match="at least 2"):
        model.loss_and_grads(np.array([[1]], dtype=np.int64))


def test_generate_extends_prompt():
    model = TinyGPT.create(vocab_size=4, d_model=8, block_size=8, seed=1)
    prompt = np.array([[0, 1]], dtype=np.int64)
    out = model.generate(prompt, max_new_tokens=3, seed=0)
    assert out.shape == (1, 5)
    np.testing.assert_array_equal(out[0, :2], [0, 1])
    with pytest.raises(ValueError, match="temperature"):
        model.generate(prompt, 1, temperature=0.0)


def test_grads_match_finite_differences():
    model = TinyGPT.create(vocab_size=6, d_model=4, block_size=5, seed=2)
    idx = np.array([[0, 1, 2, 3, 1], [1, 2, 0, 1, 2]], dtype=np.int64)
    loss, grads = model.loss_and_grads(idx)
    assert np.isfinite(loss)
    eps = 1e-5
    for pi in (10, 2):  # W_lm, W_q
        p = model.parameters()[pi].reshape(-1)
        num = np.zeros(min(6, p.size))
        for j in range(num.size):
            old = p[j]
            p[j] = old + eps
            lp, _ = model.loss_and_grads(idx)
            p[j] = old - eps
            lm, _ = model.loss_and_grads(idx)
            p[j] = old
            num[j] = (lp - lm) / (2 * eps)
        np.testing.assert_allclose(
            grads[pi].reshape(-1)[: num.size], num, rtol=2e-2, atol=5e-3
        )


def test_learns_toy_alternating_sequence():
    text = "AB" * 80
    model, tok, history = fit(
        text, d_model=16, block_size=8, steps=250, batch_size=16, lr=0.08, seed=0
    )
    assert history[-1] < history[0]
    assert history[-1] < 0.35
    a_id, b_id = tok.stoi["A"], tok.stoi["B"]
    la = model.logits(np.array([[a_id, b_id, a_id]], dtype=np.int64))[0, -1]
    lb = model.logits(np.array([[a_id, b_id, a_id, b_id]], dtype=np.int64))[0, -1]
    assert la[b_id] > la[a_id]
    assert lb[a_id] > lb[b_id]
    decoded = tok.decode(model.generate(np.array([[a_id]]), 6, seed=3)[0])
    assert decoded.count("AB") + decoded.count("BA") >= 2


def test_fit_rejects_short_text():
    with pytest.raises(ValueError, match="longer"):
        fit("AB", block_size=8, steps=1)
