import numpy as np
import pytest

from src.autograd import SGD, Value, minimize, value_sum

def test_sgd_step_updates_data():
    p = Value(5.0)
    p.grad = 2.0
    opt = SGD([p], lr=0.1)
    opt.step()
    assert p.data == pytest.approx(5.0 - 0.1 * 2.0)

def test_zero_grad_resets():
    p = Value(1.0)
    p.grad = 9.0
    opt = SGD([p], lr=0.1)
    opt.zero_grad()
    assert p.grad == 0.0

def test_minimize_quadratic_finds_minimum():
    x = Value(-4.0)
    history = minimize(lambda: (x - 3.0) ** 2, [x], steps=200, lr=0.1)
    assert x.data == pytest.approx(3.0, abs=1e-3)
    assert history[-1] < history[0]
    assert history[-1] == pytest.approx(0.0, abs=1e-4)

def test_momentum_reaches_minimum():
    x = Value(-4.0)
    history = minimize(
        lambda: (x - 3.0) ** 2, [x], steps=300, lr=0.02, momentum=0.9
    )
    assert x.data == pytest.approx(3.0, abs=1e-2)
    assert history[-1] < 1e-3
    y = Value(-4.0)
    plain = minimize(lambda: (y - 3.0) ** 2, [y], steps=300, lr=0.02)
    assert history[-1] < plain[-1]

def test_linear_regression_recovers_weights():
    rng = np.random.default_rng(0)
    xs = rng.uniform(-2, 2, size=40)
    ys = 2.0 * xs + -1.0
    w = Value(0.0)
    b = Value(0.0)

    def loss():
        errs = [(w * float(x) + b - float(y)) ** 2 for x, y in zip(xs, ys, strict=True)]
        return value_sum(errs) * (1.0 / len(xs))

    minimize(loss, [w, b], steps=400, lr=0.05)
    assert w.data == pytest.approx(2.0, abs=1e-2)
    assert b.data == pytest.approx(-1.0, abs=1e-2)

def test_invalid_hyperparams():
    with pytest.raises(ValueError):
        SGD([Value(1.0)], lr=0.0)
    with pytest.raises(ValueError):
        SGD([Value(1.0)], lr=0.1, momentum=1.0)
    with pytest.raises(ValueError):
        SGD([Value(1.0)], lr=0.1, momentum=-0.1)

def test_empty_params_is_noop():
    opt = SGD([], lr=0.1)
    opt.zero_grad()
    opt.step()  # must not raise
