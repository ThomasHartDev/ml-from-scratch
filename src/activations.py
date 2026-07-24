"""Activation functions and variance-preserving weight initialization.

A deep stack of affine layers multiplies variances: if each weight matrix has
entries of order one, the pre-activation variance grows or shrinks exponentially
with depth, and gradients follow. Activations change the story further (ReLU
zeros half its inputs; tanh saturates), so the right scale depends on both fan
sizes and the nonlinearity. Glorot/Xavier balances fan-in and fan-out for
symmetric activations; He/Kaiming uses only fan-in and a factor of two for ReLU.
This module implements the common activations with their local derivatives and
those two init schemes, plus a forward variance profile that makes the blow-up
under naive N(0,1) weights measurable.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]

InitScheme = Literal["xavier", "he", "naive"]
ActivationName = Literal["linear", "tanh", "sigmoid", "relu", "leaky_relu"]


@dataclass(frozen=True)
class Activation:
    """Named nonlinearity with a local backward for the chain rule.

    `backward(z, grad_out)` multiplies the upstream gradient by act'(z), the
    form a hand-written backprop step needs. `z` is the pre-activation.
    """

    name: str
    forward: Callable[[Array], Array]
    backward: Callable[[Array, Array], Array]


def _linear_forward(z: Array) -> Array:
    return np.asarray(z, dtype=np.float64)


def _linear_backward(z: Array, grad_out: Array) -> Array:
    return np.asarray(grad_out, dtype=np.float64) * np.ones_like(z, dtype=np.float64)


def _tanh_forward(z: Array) -> Array:
    return np.tanh(z)


def _tanh_backward(z: Array, grad_out: Array) -> Array:
    t = np.tanh(z)
    return grad_out * (1.0 - t * t)


def _sigmoid_forward(z: Array) -> Array:
    # sign branch keeps exp argument non-positive so it never overflows
    out = np.empty_like(z, dtype=np.float64)
    pos = z >= 0.0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


def _sigmoid_backward(z: Array, grad_out: Array) -> Array:
    s = _sigmoid_forward(z)
    return grad_out * s * (1.0 - s)


def _relu_forward(z: Array) -> Array:
    return np.maximum(z, 0.0)


def _relu_backward(z: Array, grad_out: Array) -> Array:
    # subgradient at 0 is taken as 0 (dead unit stays dead)
    return grad_out * (z > 0.0)


def _leaky_relu_forward(z: Array, alpha: float = 0.01) -> Array:
    return np.where(z > 0.0, z, alpha * z)


def _leaky_relu_backward(z: Array, grad_out: Array, alpha: float = 0.01) -> Array:
    return grad_out * np.where(z > 0.0, 1.0, alpha)


ACTIVATIONS: dict[str, Activation] = {
    "linear": Activation("linear", _linear_forward, _linear_backward),
    "tanh": Activation("tanh", _tanh_forward, _tanh_backward),
    "sigmoid": Activation("sigmoid", _sigmoid_forward, _sigmoid_backward),
    "relu": Activation("relu", _relu_forward, _relu_backward),
    "leaky_relu": Activation("leaky_relu", _leaky_relu_forward, _leaky_relu_backward),
}


def get_activation(name: str) -> Activation:
    if name not in ACTIVATIONS:
        known = ", ".join(sorted(ACTIVATIONS))
        raise ValueError(f"unknown activation {name!r}; choose one of: {known}")
    return ACTIVATIONS[name]


def recommended_init(activation: str) -> InitScheme:
    """Pick Xavier for symmetric/saturating acts, He for ReLU-family."""
    act = get_activation(activation)
    if act.name in ("relu", "leaky_relu"):
        return "he"
    return "xavier"


def _validate_fans(fan_in: int, fan_out: int) -> None:
    if fan_in <= 0 or fan_out <= 0:
        raise ValueError(
            f"fan_in and fan_out must be positive, got {fan_in}, {fan_out}"
        )


def xavier_normal(
    fan_in: int,
    fan_out: int,
    rng: np.random.Generator | None = None,
    gain: float = 1.0,
) -> Array:
    """Glorot normal: N(0, gain^2 * 2 / (fan_in + fan_out)).

    Balances variance of the forward pass (fan_in) and the backward pass
    (fan_out). Default gain is 1; tanh is usually left at 1, linear too.
    """
    _validate_fans(fan_in, fan_out)
    if gain <= 0.0:
        raise ValueError("gain must be positive")
    rng = rng if rng is not None else np.random.default_rng()
    std = gain * np.sqrt(2.0 / (fan_in + fan_out))
    return rng.standard_normal((fan_in, fan_out)) * std


def xavier_uniform(
    fan_in: int,
    fan_out: int,
    rng: np.random.Generator | None = None,
    gain: float = 1.0,
) -> Array:
    """Glorot uniform: U(-a, a) with a = gain * sqrt(6 / (fan_in + fan_out))."""
    _validate_fans(fan_in, fan_out)
    if gain <= 0.0:
        raise ValueError("gain must be positive")
    rng = rng if rng is not None else np.random.default_rng()
    bound = gain * np.sqrt(6.0 / (fan_in + fan_out))
    return rng.uniform(-bound, bound, size=(fan_in, fan_out))


def he_normal(
    fan_in: int,
    fan_out: int,
    rng: np.random.Generator | None = None,
    gain: float = 1.0,
) -> Array:
    """Kaiming normal: N(0, gain^2 * 2 / fan_in).

    ReLU zeros roughly half the units, so the 2 restores unit variance after
    the rectification. Only fan_in appears because the analysis is for the
    forward signal; gain multiplies the std (leave at 1 for plain ReLU).
    """
    _validate_fans(fan_in, fan_out)
    if gain <= 0.0:
        raise ValueError("gain must be positive")
    rng = rng if rng is not None else np.random.default_rng()
    std = gain * np.sqrt(2.0 / fan_in)
    return rng.standard_normal((fan_in, fan_out)) * std


def he_uniform(
    fan_in: int,
    fan_out: int,
    rng: np.random.Generator | None = None,
    gain: float = 1.0,
) -> Array:
    """Kaiming uniform: U(-a, a) with a = gain * sqrt(6 / fan_in)."""
    _validate_fans(fan_in, fan_out)
    if gain <= 0.0:
        raise ValueError("gain must be positive")
    rng = rng if rng is not None else np.random.default_rng()
    bound = gain * np.sqrt(6.0 / fan_in)
    return rng.uniform(-bound, bound, size=(fan_in, fan_out))


def init_weights(
    fan_in: int,
    fan_out: int,
    scheme: InitScheme | str,
    rng: np.random.Generator | None = None,
    *,
    distribution: Literal["normal", "uniform"] = "normal",
    gain: float = 1.0,
) -> Array:
    """Dispatch to Xavier, He, or naive N(0,1) (the last is the control case)."""
    rng = rng if rng is not None else np.random.default_rng()
    if scheme == "naive":
        _validate_fans(fan_in, fan_out)
        if distribution == "normal":
            return rng.standard_normal((fan_in, fan_out))
        return rng.uniform(-1.0, 1.0, size=(fan_in, fan_out))
    if scheme == "xavier":
        if distribution == "normal":
            return xavier_normal(fan_in, fan_out, rng, gain=gain)
        return xavier_uniform(fan_in, fan_out, rng, gain=gain)
    if scheme == "he":
        if distribution == "normal":
            return he_normal(fan_in, fan_out, rng, gain=gain)
        return he_uniform(fan_in, fan_out, rng, gain=gain)
    raise ValueError(f"unknown init scheme {scheme!r}; use xavier, he, or naive")


def forward_variance_profile(
    n_layers: int,
    width: int,
    activation: str,
    scheme: InitScheme | str,
    *,
    n_samples: int = 2000,
    seed: int = 0,
    distribution: Literal["normal", "uniform"] = "normal",
) -> list[float]:
    """Mean activation variance after each of `n_layers` affine+act layers.

    Starts from unit-variance isotropic inputs and applies W_l with the given
    init, then the activation. A good scheme keeps the returned variances near
    order 1; naive N(0,1) weights explode (or vanish after saturating acts).
    """
    if n_layers <= 0:
        raise ValueError("n_layers must be positive")
    if width <= 0:
        raise ValueError("width must be positive")
    if n_samples <= 0:
        raise ValueError("n_samples must be positive")

    act = get_activation(activation)
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n_samples, width))
    variances: list[float] = []
    for _ in range(n_layers):
        w = init_weights(width, width, scheme, rng, distribution=distribution)
        z = x @ w
        x = act.forward(z)
        variances.append(float(np.var(x)))
    return variances
