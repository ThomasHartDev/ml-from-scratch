import numpy as np
import pytest

from src.mlp import accuracy, fit, make_xor
from src.optimizers import SGD, Adam, Momentum, compare_optimizers

def test_sgd_step_matches_closed_form():
    p = np.array([1.0, -2.0, 3.0])
    g = np.array([0.5, 1.0, -1.0])
    orig = p.copy()
    SGD(lr=0.2).step([p], [g])
    assert np.allclose(p, orig - 0.2 * g)

def test_momentum_accumulates_velocity():
    p = np.array([0.0])
    opt = Momentum(lr=1.0, beta=0.5)
    opt.step([p], [np.array([2.0])])  # v = 2, p = -2
    assert p[0] == pytest.approx(-2.0)
    opt.step([p], [np.array([2.0])])  # v = 0.5*2 + 2 = 3, p = -5
    assert p[0] == pytest.approx(-5.0)

def test_adam_first_step_bias_corrected():
    p = np.array([0.0])
    g = np.array([1.0])
    opt = Adam(lr=0.1, beta1=0.9, beta2=0.999, eps=1e-8)
    opt.step([p], [g])
    expected = -0.1 * 1.0 / (np.sqrt(1.0) + 1e-8)
    assert p[0] == pytest.approx(expected)
    assert opt.t == 1

def test_adam_second_moment_shrinks_noisy_steps():
    p = np.zeros(1)
    opt = Adam(lr=0.1)
    for _ in range(20):
        opt.step([p], [np.array([0.1])])
    pos_after_steady = p[0]
    p2 = np.zeros(1)
    opt2 = Adam(lr=0.1)
    for _ in range(19):
        opt2.step([p2], [np.array([0.1])])
    before_spike = p2[0]
    opt2.step([p2], [np.array([10.0])])
    spike_delta = abs(p2[0] - before_spike)
    steady_step = abs(pos_after_steady) / 20
    assert spike_delta < 100 * steady_step

def test_invalid_hyperparams():
    with pytest.raises(ValueError):
        SGD(lr=0.0)
    with pytest.raises(ValueError):
        Momentum(lr=0.1, beta=1.0)
    with pytest.raises(ValueError):
        Momentum(lr=0.1, beta=-0.1)
    with pytest.raises(ValueError):
        Adam(lr=0.01, beta1=1.0)
    with pytest.raises(ValueError):
        Adam(lr=0.01, beta2=-0.1)
    with pytest.raises(ValueError):
        Adam(lr=0.01, eps=0.0)

def test_shape_and_length_mismatch_raise():
    p = np.zeros(3)
    with pytest.raises(ValueError):
        SGD(lr=0.1).step([p], [np.zeros(2)])
    with pytest.raises(ValueError):
        SGD(lr=0.1).step([p], [np.zeros(3), np.zeros(3)])

def test_empty_params_is_noop():
    for opt in (SGD(lr=0.1), Momentum(lr=0.1), Adam(lr=0.01)):
        opt.step([], [])

def test_mlp_fit_with_each_optimizer_learns_xor():
    X, y = make_xor(n=400, seed=0)
    configs = [
        SGD(lr=0.1),
        Momentum(lr=0.05, beta=0.9),
        Adam(lr=0.01),
    ]
    for opt in configs:
        model, history = fit(
            X, y, hidden=(16,), activation="tanh", epochs=200, seed=0, optimizer=opt
        )
        assert history[-1] < history[0]
        assert accuracy(model, X, y) > 0.9

def test_compare_optimizers_all_descend_and_differ():
    X, y = make_xor(n=300, seed=1)
    histories = compare_optimizers(
        X, y, hidden=(16,), epochs=120, seed=1, batch_size=32
    )
    assert set(histories) == {"sgd", "momentum", "adam"}
    for name, h in histories.items():
        assert len(h) == 120
        assert h[-1] < h[0], f"{name} did not reduce loss"
    assert histories["sgd"] != histories["adam"]
    assert histories["sgd"] != histories["momentum"]

def test_adam_outpaces_sgd_early_on_xor():
    X, y = make_xor(n=400, seed=2)
    histories = compare_optimizers(
        X,
        y,
        factories={
            "sgd": lambda: SGD(lr=0.05),
            "adam": lambda: Adam(lr=0.02),
        },
        hidden=(16,),
        epochs=40,
        seed=2,
    )
    mid = 20
    assert histories["adam"][mid] < histories["sgd"][mid]

def test_default_fit_still_uses_sgd():
    X, y = make_xor(n=200, seed=3)
    m1, h1 = fit(X, y, hidden=(8,), epochs=50, lr=0.1, seed=3)
    m2, h2 = fit(
        X, y, hidden=(8,), epochs=50, seed=3, optimizer=SGD(lr=0.1)
    )
    assert h1 == h2
    assert np.allclose(m1.weights[0], m2.weights[0])
