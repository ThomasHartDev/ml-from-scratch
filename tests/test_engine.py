import math

import numpy as np
import pytest

from src.autograd import Value, value_sum

def numeric_grad(f, x, eps=1e-6):
    return (f(x + eps) - f(x - eps)) / (2 * eps)

def test_add_and_mul_grads():
    a = Value(2.0)
    b = Value(-3.0)
    c = a * b + b
    c.backward()
    assert c.data == pytest.approx(-9.0)
    assert a.grad == pytest.approx(-3.0)
    assert b.grad == pytest.approx(a.data + 1.0)

def test_shared_node_accumulates():
    a = Value(3.0)
    b = a + a
    b.backward()
    assert b.data == pytest.approx(6.0)
    assert a.grad == pytest.approx(2.0)

def test_pow_grad_matches_numeric():
    def f(x):
        return x**3

    x0 = 1.7
    x = Value(x0)
    y = x**3
    y.backward()
    assert x.grad == pytest.approx(numeric_grad(f, x0), rel=1e-4)

def test_division():
    a = Value(6.0)
    b = Value(3.0)
    c = a / b
    c.backward()
    assert c.data == pytest.approx(2.0)
    assert a.grad == pytest.approx(1.0 / 3.0)
    assert b.grad == pytest.approx(-6.0 / 9.0)

def test_rsub_and_rtruediv():
    x = Value(4.0)
    y = 10.0 - x
    z = 8.0 / x
    y.backward()
    assert y.data == pytest.approx(6.0)
    assert x.grad == pytest.approx(-1.0)
    x.grad = 0.0
    z.backward()
    assert z.data == pytest.approx(2.0)
    assert x.grad == pytest.approx(-8.0 / 16.0)

@pytest.mark.parametrize("fn_name", ["relu", "tanh", "exp", "sigmoid"])
def test_activation_grads_match_numeric(fn_name):
    def forward(x):
        v = Value(x)
        return getattr(v, fn_name)()

    for x0 in (-1.3, 0.4, 2.1):
        v = Value(x0)
        out = getattr(v, fn_name)()
        out.backward()
        expected = numeric_grad(lambda x: forward(x).data, x0)
        assert v.grad == pytest.approx(expected, rel=1e-4, abs=1e-6)

def test_relu_at_zero_is_dead():
    x = Value(0.0)
    y = x.relu()
    y.backward()
    assert y.data == 0.0
    assert x.grad == 0.0

def test_log_grad_and_domain():
    x = Value(2.5)
    y = x.log()
    y.backward()
    assert y.data == pytest.approx(math.log(2.5))
    assert x.grad == pytest.approx(1.0 / 2.5)
    with pytest.raises(ValueError):
        Value(0.0).log()
    with pytest.raises(ValueError):
        Value(-1.0).log()

def test_pow_rejects_value_exponent():
    with pytest.raises(TypeError):
        Value(2.0) ** Value(3.0)

def test_softmax_cross_entropy_matches_numpy():
    logits = [Value(1.0), Value(2.0), Value(0.5)]
    target = 1
    exps = [z.exp() for z in logits]
    denom = value_sum(exps)
    probs = [e / denom for e in exps]
    loss = -probs[target].log()
    loss.backward()

    raw = np.array([1.0, 2.0, 0.5])
    sm = np.exp(raw) / np.exp(raw).sum()
    expected = sm.copy()
    expected[target] -= 1.0
    got = np.array([z.grad for z in logits])
    np.testing.assert_allclose(got, expected, rtol=1e-6, atol=1e-8)
    assert loss.data == pytest.approx(-math.log(sm[target]))

def test_deep_chain_gradient():
    x = Value(1.1)
    out = x
    for _ in range(9):
        out = out * x
    out.backward()
    assert out.data == pytest.approx(1.1**10, rel=1e-9)
    assert x.grad == pytest.approx(10 * 1.1**9, rel=1e-6)

def test_backward_twice_needs_manual_zero():
    a = Value(3.0)
    b = Value(4.0)

    def build():
        return a * b

    out = build()
    out.backward()
    first = a.grad
    out2 = build()
    out2.backward()
    assert a.grad == pytest.approx(2 * first)
