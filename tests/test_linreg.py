import dataclasses

import numpy as np
import pytest

from src.linreg import LinearModel, fit_normal_equation, fit_sgd, mse


def make_data(n=200, w=(2.0, -3.0, 0.5), b=1.5, noise=0.0, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-2.0, 2.0, size=(n, len(w)))
    y = X @ np.array(w) + b + noise * rng.standard_normal(n)
    return X, y


def test_normal_equation_recovers_exact_weights_noiseless():
    X, y = make_data(noise=0.0)
    model = fit_normal_equation(X, y)
    np.testing.assert_allclose(model.weights, [2.0, -3.0, 0.5], atol=1e-9)
    assert model.bias == pytest.approx(1.5, abs=1e-9)
    assert mse(model, X, y) == pytest.approx(0.0, abs=1e-18)


def test_sgd_recovers_weights_noiseless():
    X, y = make_data(noise=0.0)
    model, history = fit_sgd(X, y, lr=0.1, epochs=300, batch_size=32)
    np.testing.assert_allclose(model.weights, [2.0, -3.0, 0.5], atol=1e-2)
    assert model.bias == pytest.approx(1.5, abs=1e-2)
    assert history[-1] < history[0]


def test_normal_equation_and_sgd_agree():
    # convex loss + noiseless data: both methods land on the same minimum
    X, y = make_data(noise=0.05, seed=7)
    exact = fit_normal_equation(X, y)
    approx, _ = fit_sgd(X, y, lr=0.1, epochs=500, batch_size=64)
    np.testing.assert_allclose(approx.weights, exact.weights, atol=2e-2)
    assert approx.bias == pytest.approx(exact.bias, abs=2e-2)


def test_single_feature_and_1d_input():
    rng = np.random.default_rng(1)
    x = rng.uniform(-1.0, 1.0, size=50)
    y = 4.0 * x - 2.0
    model = fit_normal_equation(x, y)
    assert model.weights.shape == (1,)
    assert model.weights[0] == pytest.approx(4.0, abs=1e-9)
    assert model.bias == pytest.approx(-2.0, abs=1e-9)
    assert model.predict(np.array([0.5]))[0] == pytest.approx(0.0, abs=1e-9)


def test_sgd_history_is_monotone_enough():
    X, y = make_data(noise=0.1, seed=3)
    _, history = fit_sgd(X, y, lr=0.05, epochs=200, batch_size=32)
    assert len(history) == 200
    # not strictly monotone with minibatches, but the trend must fall hard
    assert history[-1] < 0.1 * history[0]


def test_ridge_shrinks_weights_but_not_intercept():
    X, y = make_data(noise=0.0, seed=2)
    plain = fit_normal_equation(X, y, l2=0.0)
    ridged = fit_normal_equation(X, y, l2=50.0)
    assert np.linalg.norm(ridged.weights) < np.linalg.norm(plain.weights)
    # intercept stays near its true value; only slopes are penalized
    assert ridged.bias == pytest.approx(1.5, abs=0.2)


def test_rank_deficient_falls_back_to_pseudoinverse():
    # duplicate column makes X^T X singular; solve would blow up
    rng = np.random.default_rng(4)
    base = rng.uniform(-1.0, 1.0, size=(30, 1))
    X = np.hstack([base, base])  # perfectly collinear
    y = (3.0 * base[:, 0] + 1.0)
    model = fit_normal_equation(X, y)
    # split across the two identical columns, but predictions still fit
    np.testing.assert_allclose(model.predict(X), y, atol=1e-6)


def test_predict_shapes():
    X, y = make_data(n=10, seed=5)
    model = fit_normal_equation(X, y)
    preds = model.predict(X)
    assert preds.shape == (10,)


def test_empty_input_raises():
    with pytest.raises(ValueError):
        fit_normal_equation(np.zeros((0, 3)), np.zeros(0))
    with pytest.raises(ValueError):
        fit_sgd(np.zeros((0, 3)), np.zeros(0))


def test_mismatched_lengths_raise():
    with pytest.raises(ValueError):
        fit_normal_equation(np.zeros((5, 2)), np.zeros(4))


def test_invalid_hyperparams_raise():
    X, y = make_data(n=20)
    for kwargs in ({"lr": 0.0}, {"epochs": 0}, {"batch_size": 0}, {"l2": -1.0}):
        with pytest.raises(ValueError):
            fit_sgd(X, y, **kwargs)
    with pytest.raises(ValueError):
        fit_normal_equation(X, y, l2=-1.0)


def test_batch_larger_than_dataset_runs_full_batch():
    X, y = make_data(n=20, noise=0.0, seed=6)
    model, history = fit_sgd(X, y, lr=0.1, epochs=400, batch_size=256)
    np.testing.assert_allclose(model.weights, [2.0, -3.0, 0.5], atol=1e-2)
    assert len(history) == 400


def test_linear_model_is_frozen():
    m = LinearModel(weights=np.array([1.0]), bias=0.0)
    with pytest.raises(dataclasses.FrozenInstanceError):
        m.bias = 2.0
