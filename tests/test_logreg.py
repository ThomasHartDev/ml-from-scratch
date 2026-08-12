import dataclasses

import numpy as np
import pytest

from src.logreg import (
    LogisticModel,
    accuracy,
    bce_loss,
    decision_boundary,
    fit_sgd,
)

def make_blobs(n=400, w=(3.0, -2.0), b=0.5, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-3.0, 3.0, size=(n, len(w)))
    z = X @ np.array(w) + b
    p = 1.0 / (1.0 + np.exp(-z))
    y = (rng.uniform(size=n) < p).astype(np.int64)
    return X, y

def separable_data(n=200, seed=1):
    rng = np.random.default_rng(seed)
    pos = rng.uniform(1.0, 3.0, size=(n // 2, 2))
    neg = rng.uniform(-3.0, -1.0, size=(n // 2, 2))
    X = np.vstack([pos, neg])
    y = np.concatenate([np.ones(n // 2), np.zeros(n // 2)]).astype(np.int64)
    return X, y

def test_sigmoid_is_stable_on_extreme_logits():
    model = LogisticModel(weights=np.array([1.0]), bias=0.0)
    huge = np.array([-1000.0, 1000.0])
    p = model.predict_proba(huge)
    assert p[0] == pytest.approx(0.0, abs=1e-12)
    assert p[1] == pytest.approx(1.0, abs=1e-12)
    assert np.all(np.isfinite(p))

def test_bce_matches_naive_formula_in_safe_range():
    X, y = make_blobs(n=50, seed=3)
    model = LogisticModel(weights=np.array([0.4, -0.6]), bias=0.1)
    p = model.predict_proba(X)
    naive = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
    assert bce_loss(model, X, y) == pytest.approx(naive, abs=1e-9)

def test_bce_finite_on_saturated_logits():
    model = LogisticModel(weights=np.array([50.0]), bias=0.0)
    X = np.array([[1.0]])
    y = np.array([0])
    loss = bce_loss(model, X, y)
    assert np.isfinite(loss)
    assert loss == pytest.approx(50.0, abs=1e-6)  # softplus(50) - 0 ≈ 50

def test_recovers_true_weights_direction():
    X, y = make_blobs(n=4000, w=(3.0, -2.0), b=0.5, seed=7)
    model, history = fit_sgd(X, y, lr=0.2, epochs=400, batch_size=64)
    # logistic weights are only identified up to the noise, so check direction
    est = np.append(model.weights, model.bias)
    true = np.array([3.0, -2.0, 0.5])
    cos = est @ true / (np.linalg.norm(est) * np.linalg.norm(true))
    assert cos > 0.98
    assert history[-1] < history[0]

def test_perfectly_separable_data_is_classified_perfectly():
    X, y = separable_data(n=300, seed=2)
    model, _ = fit_sgd(X, y, lr=0.3, epochs=300, batch_size=32)
    assert accuracy(model, X, y) == pytest.approx(1.0)

def test_loss_decreases_monotone_enough():
    X, y = make_blobs(n=1000, seed=5)
    _, history = fit_sgd(X, y, lr=0.2, epochs=200, batch_size=64)
    assert len(history) == 200
    assert history[-1] < 0.9 * history[0]

def test_l2_shrinks_weights():
    X, y = separable_data(n=300, seed=4)
    plain, _ = fit_sgd(X, y, lr=0.2, epochs=200, l2=0.0)
    ridged, _ = fit_sgd(X, y, lr=0.2, epochs=200, l2=2.0)
    assert np.linalg.norm(ridged.weights) < np.linalg.norm(plain.weights)

def test_predict_uses_threshold():
    model = LogisticModel(weights=np.array([1.0]), bias=0.0)
    X = np.array([[0.2]])  # sigmoid(0.2) ≈ 0.55
    assert model.predict(X)[0] == 1
    assert model.predict(X, threshold=0.6)[0] == 0

def test_decision_boundary_line():
    model = LogisticModel(weights=np.array([2.0, -4.0]), bias=1.0)
    x1 = np.array([-1.0, 0.0, 1.0])
    x2 = decision_boundary(model, x1)
    pts = np.column_stack([x1, x2])
    np.testing.assert_allclose(model.decision_function(pts), 0.0, atol=1e-9)
    np.testing.assert_allclose(model.predict_proba(pts), 0.5, atol=1e-9)

def test_decision_boundary_rejects_non_2d_models():
    with pytest.raises(ValueError):
        decision_boundary(LogisticModel(weights=np.array([1.0]), bias=0.0), [0.0])

def test_decision_boundary_rejects_horizontal():
    with pytest.raises(ValueError):
        decision_boundary(LogisticModel(weights=np.array([1.0, 0.0]), bias=0.0), [0.0])

def test_predict_proba_shape_and_range():
    X, y = make_blobs(n=30, seed=6)
    model, _ = fit_sgd(X, y, lr=0.1, epochs=50)
    p = model.predict_proba(X)
    assert p.shape == (30,)
    assert np.all((p >= 0.0) & (p <= 1.0))

def test_constant_feature_does_not_break_standardization():
    X, y = separable_data(n=100, seed=8)
    X = np.hstack([X, np.ones((X.shape[0], 1))])  # zero-variance column
    model, _ = fit_sgd(X, y, lr=0.2, epochs=200)
    assert np.all(np.isfinite(model.weights))
    assert accuracy(model, X, y) > 0.95

def test_single_feature_1d_input():
    rng = np.random.default_rng(9)
    x = rng.uniform(-3.0, 3.0, size=200)
    y = (x > 0).astype(np.int64)
    model, _ = fit_sgd(x, y, lr=0.3, epochs=300)
    assert model.weights.shape == (1,)
    assert model.predict(np.array([2.0]))[0] == 1
    assert model.predict(np.array([-2.0]))[0] == 0

def test_empty_input_raises():
    with pytest.raises(ValueError):
        fit_sgd(np.zeros((0, 2)), np.zeros(0))

def test_mismatched_lengths_raise():
    with pytest.raises(ValueError):
        fit_sgd(np.zeros((5, 2)), np.zeros(4))

def test_non_binary_labels_raise():
    X = np.zeros((4, 2))
    with pytest.raises(ValueError):
        fit_sgd(X, np.array([0, 1, 2, 1]))
    model = LogisticModel(weights=np.zeros(2), bias=0.0)
    with pytest.raises(ValueError):
        bce_loss(model, X, np.array([0, 1, 0, 5]))

def test_invalid_hyperparams_raise():
    X, y = make_blobs(n=20)
    for kwargs in ({"lr": 0.0}, {"epochs": 0}, {"batch_size": 0}, {"l2": -1.0}):
        with pytest.raises(ValueError):
            fit_sgd(X, y, **kwargs)

def test_logistic_model_is_frozen():
    m = LogisticModel(weights=np.array([1.0]), bias=0.0)
    with pytest.raises(dataclasses.FrozenInstanceError):
        m.bias = 2.0
