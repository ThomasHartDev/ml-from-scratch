import numpy as np
import pytest

from src.embeddings import (
    SkipGramModel,
    build_vocab,
    cosine_similarity,
    fit,
    make_toy_corpus,
    skipgram_pairs,
    tokenize,
)


def test_tokenize_lowercases_and_drops_punctuation():
    assert tokenize("Hello, WORLD! 123 cats.") == ["hello", "world", "cats"]
    assert tokenize("") == []
    assert tokenize("... ---") == []


def test_build_vocab_sorts_and_respects_min_count():
    tokens = ["b", "a", "b", "c", "a", "a"]
    w2i, i2w = build_vocab(tokens, min_count=2)
    assert i2w == ["a", "b"]
    assert w2i["a"] == 0 and w2i["b"] == 1
    assert "c" not in w2i


def test_build_vocab_rejects_empty_after_filter():
    with pytest.raises(ValueError, match="empty"):
        build_vocab(["once"], min_count=2)
    with pytest.raises(ValueError, match="min_count"):
        build_vocab(["a"], min_count=0)


def test_skipgram_pairs_window_and_edges():
    # tokens: 0 1 2 3, window=1 → edges have one context, middle have two
    centers, contexts = skipgram_pairs([0, 1, 2, 3], window=1)
    pairs = sorted(zip(centers.tolist(), contexts.tolist(), strict=True))
    assert pairs == sorted(
        [
            (0, 1),
            (1, 0),
            (1, 2),
            (2, 1),
            (2, 3),
            (3, 2),
        ]
    )


def test_skipgram_pairs_window_two_includes_skip():
    centers, contexts = skipgram_pairs([10, 20, 30], window=2)
    pairs = set(zip(centers.tolist(), contexts.tolist(), strict=True))
    assert (10, 30) in pairs and (30, 10) in pairs
    assert (10, 10) not in pairs


def test_skipgram_pairs_empty_and_single_token():
    c, o = skipgram_pairs([], window=2)
    assert c.size == 0 and o.size == 0
    c, o = skipgram_pairs([7], window=2)
    assert c.size == 0 and o.size == 0


def test_skipgram_pairs_rejects_bad_window():
    with pytest.raises(ValueError, match="window"):
        skipgram_pairs([1, 2], window=0)


def test_cosine_similarity_basics():
    assert cosine_similarity([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)
    assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)
    assert cosine_similarity([1.0, 0.0], [-1.0, 0.0]) == pytest.approx(-1.0)
    assert cosine_similarity([0.0, 0.0], [1.0, 2.0]) == 0.0
    with pytest.raises(ValueError, match="shape"):
        cosine_similarity([1.0], [1.0, 2.0])


def test_fit_rejects_too_short_and_bad_hparams():
    with pytest.raises(ValueError, match="two tokens"):
        fit("hello")
    with pytest.raises(ValueError, match="dim"):
        fit("a b c", dim=0)
    with pytest.raises(ValueError, match="lr"):
        fit("a b c", lr=0.0)
    with pytest.raises(ValueError, match="epochs"):
        fit("a b c", epochs=0)


def test_loss_decreases_on_toy_corpus():
    model, history = fit(
        make_toy_corpus(),
        dim=24,
        window=2,
        lr=0.08,
        epochs=40,
        batch_size=64,
        seed=0,
    )
    assert len(history) == 40
    assert history[-1] < history[0]
    assert model.vocab_size == len(model.idx_to_word)
    assert model.dim == 24
    assert model.W_in.shape == (model.vocab_size, 24)
    assert model.W_out.shape == (model.vocab_size, 24)


def test_nearest_recovers_city_cluster():
    # capitals share the same skip-gram contexts ("is", "the", "capital", "of")
    model, _ = fit(
        make_toy_corpus(),
        dim=32,
        window=2,
        lr=0.1,
        epochs=100,
        batch_size=32,
        seed=1,
    )
    neighbors = [w for w, _ in model.nearest("paris", k=5)]
    cities = {"berlin", "london", "rome", "madrid"}
    assert len(cities.intersection(neighbors)) >= 1, neighbors


def test_nearest_recovers_animal_cluster():
    model, _ = fit(
        make_toy_corpus(),
        dim=32,
        window=2,
        lr=0.1,
        epochs=100,
        batch_size=32,
        seed=2,
    )
    neighbors = [w for w, _ in model.nearest("cat", k=6)]
    animals = {"dog", "cats", "dogs", "mouse"}
    assert len(animals.intersection(neighbors)) >= 1, neighbors


def test_embed_and_nearest_unknown_word():
    model, _ = fit("the cat sat on the mat", dim=8, epochs=5, seed=0)
    vec = model.embed("cat")
    assert vec.shape == (8,)
    with pytest.raises(KeyError, match="unknown"):
        model.embed("zebra")
    with pytest.raises(KeyError, match="unknown"):
        model.nearest("zebra")
    with pytest.raises(ValueError, match="k"):
        model.nearest("cat", k=0)


def test_nearest_k_capped_by_vocab():
    model, _ = fit("alpha beta gamma", dim=4, window=1, epochs=20, seed=0)
    # vocab size 3 → at most 2 neighbors
    nn = model.nearest("alpha", k=10)
    assert len(nn) == 2
    assert all(isinstance(s, float) for _, s in nn)
    # self never appears
    assert all(w != "alpha" for w, _ in nn)


def test_softmax_gradient_matches_finite_differences():
    """One-pair NLL gradient vs central differences on W_in[center] and W_out."""
    rng = np.random.default_rng(0)
    v, d = 5, 3
    W_in = rng.standard_normal((v, d)) * 0.2
    W_out = rng.standard_normal((v, d)) * 0.2
    center, context = 1, 3

    def loss() -> float:
        scores = W_out @ W_in[center]
        scores = scores - scores.max()
        p = np.exp(scores)
        p = p / p.sum()
        return float(-np.log(p[context] + 1e-12))

    h = W_in[center]
    scores = W_out @ h
    p = np.exp(scores - scores.max())
    p = p / p.sum()
    ds = p.copy()
    ds[context] -= 1.0
    dW_out = np.outer(ds, h)
    dh = W_out.T @ ds

    eps = 1e-5
    for i in range(d):
        orig = W_in[center, i]
        W_in[center, i] = orig + eps
        hi = loss()
        W_in[center, i] = orig - eps
        lo = loss()
        W_in[center, i] = orig
        assert (hi - lo) / (2 * eps) == pytest.approx(dh[i], abs=1e-4)

    for i in range(v):
        for j in range(d):
            orig = W_out[i, j]
            W_out[i, j] = orig + eps
            hi = loss()
            W_out[i, j] = orig - eps
            lo = loss()
            W_out[i, j] = orig
            assert (hi - lo) / (2 * eps) == pytest.approx(dW_out[i, j], abs=1e-4)


def test_model_vocab_maps_are_consistent():
    model = SkipGramModel(
        word_to_idx={"a": 0, "b": 1},
        idx_to_word=["a", "b"],
        W_in=np.eye(2),
        W_out=np.eye(2),
    )
    assert model.nearest("a", k=1)[0][0] == "b"
    assert cosine_similarity(model.embed("a"), [1.0, 0.0]) == pytest.approx(1.0)
