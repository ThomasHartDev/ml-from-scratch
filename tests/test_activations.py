import numpy as np
import pytest

from src.activations import (
    ACTIVATIONS,
    forward_variance_profile,
    get_activation,
    he_normal,
    he_uniform,
    init_weights,
    recommended_init,
    xavier_normal,
    xavier_uniform,
)


def _numeric_deriv(fn, z0: float, eps: float = 1e-6) -> float:
    return (fn(z0 + eps) - fn(z0 - eps)) / (2.0 * eps)


@pytest.mark.parametrize("name", sorted(ACTIVATIONS))
def test_forward_backward_shapes_and_finite(name):
    act = get_activation(name)
    z = np.linspace(-3.0, 3.0, 21).reshape(3, 7)
    a = act.forward(z)
    g = act.backward(z, np.ones_like(z))
    assert a.shape == z.shape
    assert g.shape == z.shape
    assert np.all(np.isfinite(a))
    assert np.all(np.isfinite(g))


@pytest.mark.parametrize(
    "name,points",
    [
        ("tanh", (-1.5, -0.3, 0.0, 0.7, 2.0)),
        ("sigmoid", (-2.0, -0.5, 0.0, 0.5, 2.0)),
        ("relu", (-1.0, 0.5, 1.5)),
        ("leaky_relu", (-1.0, 0.5, 1.5)),
        ("linear", (-2.0, 0.0, 3.0)),
    ],
)
def test_derivative_matches_finite_differences(name, points):
    act = get_activation(name)
    for z0 in points:
        z = np.array([[z0]], dtype=np.float64)
        analytic = float(act.backward(z, np.ones_like(z))[0, 0])

        def _f(t: float) -> float:
            return float(act.forward(np.array([[t]]))[0, 0])

        numeric = _numeric_deriv(_f, z0)
        assert analytic == pytest.approx(numeric, abs=1e-5, rel=1e-4)


def test_relu_dead_at_and_below_zero():
    act = get_activation("relu")
    z = np.array([[-2.0, 0.0, 3.0]])
    assert np.allclose(act.forward(z), [[0.0, 0.0, 3.0]])
    assert np.allclose(act.backward(z, np.ones_like(z)), [[0.0, 0.0, 1.0]])


def test_leaky_relu_leaks_on_negative():
    act = get_activation("leaky_relu")
    z = np.array([[-2.0, 4.0]])
    a = act.forward(z)
    assert a[0, 0] == pytest.approx(-0.02)
    assert a[0, 1] == pytest.approx(4.0)
    g = act.backward(z, np.ones_like(z))
    assert g[0, 0] == pytest.approx(0.01)
    assert g[0, 1] == pytest.approx(1.0)


def test_sigmoid_stays_in_unit_interval_on_huge_logits():
    act = get_activation("sigmoid")
    z = np.array([[1e3, -1e3, 0.0]])
    s = act.forward(z)
    assert s[0, 0] == pytest.approx(1.0)
    assert s[0, 1] == pytest.approx(0.0)
    assert s[0, 2] == pytest.approx(0.5)
    assert np.all(np.isfinite(s))


def test_tanh_range_and_oddness():
    act = get_activation("tanh")
    z = np.array([[-2.0, 0.0, 2.0]])
    t = act.forward(z)
    assert t[0, 1] == pytest.approx(0.0)
    assert t[0, 0] == pytest.approx(-t[0, 2])
    assert np.all(np.abs(t) <= 1.0 + 1e-12)


def test_unknown_activation_lists_choices():
    with pytest.raises(ValueError, match="unknown activation"):
        get_activation("swish")


@pytest.mark.parametrize(
    "fn,expected_var",
    [
        (xavier_normal, lambda fi, fo: 2.0 / (fi + fo)),
        (he_normal, lambda fi, fo: 2.0 / fi),
    ],
)
def test_normal_init_sample_variance_matches_formula(fn, expected_var):
    fan_in, fan_out = 64, 128
    rng = np.random.default_rng(0)
    # draw many matrices so sample var converges to the theoretical scale
    samples = np.concatenate(
        [fn(fan_in, fan_out, rng).ravel() for _ in range(40)]
    )
    assert float(np.var(samples)) == pytest.approx(
        expected_var(fan_in, fan_out), rel=0.08
    )


def test_xavier_uniform_stays_inside_bound():
    fan_in, fan_out = 50, 30
    bound = np.sqrt(6.0 / (fan_in + fan_out))
    w = xavier_uniform(fan_in, fan_out, np.random.default_rng(1))
    assert w.shape == (fan_in, fan_out)
    assert np.all(w >= -bound - 1e-12)
    assert np.all(w <= bound + 1e-12)


def test_he_uniform_stays_inside_bound():
    fan_in, fan_out = 40, 20
    bound = np.sqrt(6.0 / fan_in)
    w = he_uniform(fan_in, fan_out, np.random.default_rng(2))
    assert w.shape == (fan_in, fan_out)
    assert np.all(np.abs(w) <= bound + 1e-12)


def test_gain_scales_std():
    rng = np.random.default_rng(3)
    base = xavier_normal(80, 80, rng, gain=1.0)
    rng = np.random.default_rng(3)
    scaled = xavier_normal(80, 80, rng, gain=2.0)
    assert float(np.std(scaled)) == pytest.approx(2.0 * float(np.std(base)), rel=0.05)


def test_init_weights_dispatch_and_recommended():
    assert recommended_init("relu") == "he"
    assert recommended_init("leaky_relu") == "he"
    assert recommended_init("tanh") == "xavier"
    assert recommended_init("sigmoid") == "xavier"
    assert recommended_init("linear") == "xavier"

    w = init_weights(10, 5, "he", np.random.default_rng(0))
    assert w.shape == (10, 5)
    w2 = init_weights(10, 5, "xavier", np.random.default_rng(0), distribution="uniform")
    assert w2.shape == (10, 5)
    w3 = init_weights(10, 5, "naive", np.random.default_rng(0))
    assert w3.shape == (10, 5)


def test_bad_fans_and_schemes_raise():
    with pytest.raises(ValueError, match="fan_in"):
        xavier_normal(0, 4)
    with pytest.raises(ValueError, match="fan_in"):
        he_normal(3, 0)
    with pytest.raises(ValueError, match="gain"):
        he_normal(4, 4, gain=0.0)
    with pytest.raises(ValueError, match="unknown init"):
        init_weights(4, 4, "orthogonal")  # type: ignore[arg-type]


def test_forward_profile_rejects_bad_args():
    with pytest.raises(ValueError, match="n_layers"):
        forward_variance_profile(0, 16, "relu", "he")
    with pytest.raises(ValueError, match="width"):
        forward_variance_profile(4, 0, "relu", "he")
    with pytest.raises(ValueError, match="n_samples"):
        forward_variance_profile(4, 16, "relu", "he", n_samples=0)


def test_naive_init_explodes_variance_with_depth():
    """Unit-scale weights without the 1/sqrt(fan) factor blow up signal var."""
    vars_naive = forward_variance_profile(
        n_layers=6, width=64, activation="linear", scheme="naive", seed=0
    )
    # each layer multiplies var by ~width under N(0,1) weights
    assert vars_naive[-1] > 1e6
    assert vars_naive[-1] > vars_naive[0] * 100


def test_xavier_keeps_tanh_variance_stable():
    vars_x = forward_variance_profile(
        n_layers=8, width=64, activation="tanh", scheme="xavier", seed=1
    )
    # tanh is contractive, but should not collapse to machine zero
    assert all(0.01 < v < 2.0 for v in vars_x)
    assert max(vars_x) / min(vars_x) < 20.0


def test_he_keeps_relu_variance_stable_while_xavier_fades():
    vars_he = forward_variance_profile(
        n_layers=10, width=64, activation="relu", scheme="he", seed=2
    )
    vars_x = forward_variance_profile(
        n_layers=10, width=64, activation="relu", scheme="xavier", seed=2
    )
    # He restores the half of the signal ReLU drops; variance stays O(1)
    assert all(0.05 < v < 5.0 for v in vars_he)
    # Xavier under-scales ReLU stacks, so later layers quietly die
    assert vars_x[-1] < vars_he[-1] * 0.5
    assert vars_x[-1] < 0.5


def test_profile_length_matches_depth():
    profile = forward_variance_profile(
        n_layers=5, width=32, activation="relu", scheme="he", seed=0
    )
    assert len(profile) == 5
