import numpy as np
import pytest

from src.mlp import (
    ACTIVATIONS,
    MLP,
    _backprop,
    _forward_logits,
    _softmax,
    accuracy,
    cross_entropy,
    fit,
    make_spiral,
    make_xor,
)


def test_softmax_rows_sum_to_one_and_survive_huge_logits():
    z = np.array([[1000.0, -1000.0, 0.0], [0.0, 0.0, 0.0]])
    p = _softmax(z)
    assert np.allclose(p.sum(axis=1), 1.0)
    assert np.all(np.isfinite(p))
    assert p[0, 0] == pytest.approx(1.0)
    assert np.allclose(p[1], 1.0 / 3.0)


@pytest.mark.parametrize("activation", ["tanh", "relu"])
def test_backprop_matches_finite_differences(activation):
    rng = np.random.default_rng(3)
    x = rng.standard_normal((6, 3))
    y = np.eye(4)[rng.integers(0, 4, size=6)]
    weights = [rng.standard_normal((3, 5)), rng.standard_normal((5, 4))]
    biases = [rng.standard_normal(5), rng.standard_normal(4)]
    act = ACTIVATIONS[activation]

    def loss() -> float:
        logits = _forward_logits(weights, biases, act, x)
        return cross_entropy(_softmax(logits), y)

    grad_w, grad_b = _backprop(weights, biases, act, x, y)

    eps = 1e-6
    for layer in range(len(weights)):
        for idx in np.ndindex(weights[layer].shape):
            orig = weights[layer][idx]
            weights[layer][idx] = orig + eps
            hi = loss()
            weights[layer][idx] = orig - eps
            lo = loss()
            weights[layer][idx] = orig
            numeric = (hi - lo) / (2.0 * eps)
            assert numeric == pytest.approx(grad_w[layer][idx], abs=1e-5)
        for j in range(biases[layer].shape[0]):
            orig = biases[layer][j]
            biases[layer][j] = orig + eps
            hi = loss()
            biases[layer][j] = orig - eps
            lo = loss()
            biases[layer][j] = orig
            numeric = (hi - lo) / (2.0 * eps)
            assert numeric == pytest.approx(grad_b[layer][j], abs=1e-5)


def test_learns_xor_that_a_linear_model_cannot():
    X, y = make_xor(n=400, seed=0)
    model, history = fit(X, y, hidden=(16,), activation="tanh", epochs=300, seed=0)
    assert accuracy(model, X, y) > 0.97
    assert history[-1] < history[0]


def test_learns_three_class_spiral():
    X, y = make_spiral(n=600, classes=3, seed=1)
    model, history = fit(
        X, y, hidden=(32, 32), activation="relu", lr=0.3, epochs=400, seed=1
    )
    assert accuracy(model, X, y) > 0.9
    assert len(history) == 400


def test_loss_curve_trends_down():
    X, y = make_xor(n=200, seed=2)
    _, history = fit(X, y, hidden=(8,), epochs=150, seed=2)
    # compare averaged windows so a single noisy epoch can't fail it
    assert np.mean(history[-20:]) < np.mean(history[:20])


def test_predict_recovers_original_label_values():
    X, y = make_xor(n=200, seed=0)
    y_labeled = np.where(y == 0, 7, 42)
    model, _ = fit(X, y_labeled, hidden=(16,), epochs=200, seed=0)
    preds = model.predict(X)
    assert set(np.unique(preds)).issubset({7, 42})
    assert accuracy(model, X, y_labeled) > 0.95


def test_predict_proba_is_a_distribution():
    X, y = make_spiral(n=300, classes=3, seed=0)
    model, _ = fit(X, y, hidden=(16,), epochs=50, seed=0)
    p = model.predict_proba(X)
    assert p.shape == (X.shape[0], 3)
    assert np.allclose(p.sum(axis=1), 1.0)
    assert np.all(p >= 0.0)


def test_constant_feature_does_not_break_standardization():
    X, y = make_xor(n=160, seed=0)
    X = np.hstack([X, np.full((X.shape[0], 1), 5.0)])  # zero-variance column
    model, history = fit(X, y, hidden=(16,), epochs=200, seed=0)
    assert np.all(np.isfinite(history))
    assert accuracy(model, X, y) > 0.95


def test_accepts_1d_input_as_single_feature():
    rng = np.random.default_rng(0)
    x = rng.standard_normal(120)
    y = (x > 0).astype(np.int64)
    model, _ = fit(x, y, hidden=(8,), epochs=200, seed=0)
    assert accuracy(model, x, y) > 0.95


def test_deterministic_under_seed():
    X, y = make_xor(n=200, seed=0)
    m1, h1 = fit(X, y, hidden=(16,), epochs=50, seed=123)
    m2, h2 = fit(X, y, hidden=(16,), epochs=50, seed=123)
    assert h1 == h2
    pairs = zip(m1.weights, m2.weights, strict=True)
    assert all(np.array_equal(a, b) for a, b in pairs)


def test_empty_input_raises():
    with pytest.raises(ValueError, match="at least one sample"):
        fit(np.empty((0, 2)), np.array([]), epochs=1)


def test_mismatched_lengths_raise():
    with pytest.raises(ValueError, match="rows but"):
        fit(np.zeros((4, 2)), np.zeros(3), epochs=1)


def test_single_class_raises():
    with pytest.raises(ValueError, match="two classes"):
        fit(np.zeros((5, 2)), np.zeros(5), epochs=1)


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"activation": "sigmoid"}, "unknown activation"),
        ({"lr": 0.0}, "lr must be positive"),
        ({"epochs": 0}, "epochs must be positive"),
        ({"batch_size": 0}, "batch_size must be positive"),
        ({"hidden": (16, 0)}, "hidden layer sizes"),
    ],
)
def test_bad_hyperparameters_raise(kwargs, match):
    X, y = make_xor(n=40, seed=0)
    with pytest.raises(ValueError, match=match):
        fit(X, y, **kwargs)


def test_model_is_frozen():
    model = MLP(
        weights=[np.zeros((2, 2))],
        biases=[np.zeros(2)],
        activation="tanh",
        classes=np.array([0, 1]),
        mu=np.zeros(2),
        sigma=np.ones(2),
    )
    with pytest.raises(AttributeError):
        model.activation = "relu"  # type: ignore[misc]
