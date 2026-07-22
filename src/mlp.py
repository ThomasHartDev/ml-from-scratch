"""Multilayer perceptron with backpropagation derived by hand.

A single linear model draws one hyperplane, so it cannot separate classes that
are not linearly separable (XOR is the classic counterexample). Stacking affine
layers with a nonlinearity between them lets the network bend the decision
surface, and the weights are learned by backpropagation: one forward pass caches
every pre-activation and activation, then the chain rule is walked backward layer
by layer.

The whole method is one recursion on the "delta" (the gradient of the loss with
respect to a layer's pre-activation z). With softmax + cross-entropy at the head,
the output delta collapses to `p - y_onehot`. From there each earlier delta is
`(delta_next @ W_next.T) * act'(z)`, and the parameter gradients fall out as
`dW = a_prev.T @ delta / m` and `db = mean(delta)`. Everything is vectorized over
a minibatch with numpy, but no autodiff: the gradients are written out by hand so
the matrix calculus stays visible.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]


@dataclass(frozen=True)
class _Activation:
    forward: object  # Callable[[Array], Array]
    backward: object  # Callable[[Array, Array], Array]: (z, grad_out) -> grad_in


def _tanh_forward(z: Array) -> Array:
    return np.tanh(z)


def _tanh_backward(z: Array, grad_out: Array) -> Array:
    t = np.tanh(z)
    return grad_out * (1.0 - t * t)


def _relu_forward(z: Array) -> Array:
    return np.maximum(z, 0.0)


def _relu_backward(z: Array, grad_out: Array) -> Array:
    return grad_out * (z > 0.0)


ACTIVATIONS: dict[str, _Activation] = {
    "tanh": _Activation(_tanh_forward, _tanh_backward),
    "relu": _Activation(_relu_forward, _relu_backward),
}


def _softmax(z: Array) -> Array:
    # subtract the row max first so exp never overflows on large logits
    z = z - z.max(axis=1, keepdims=True)
    ez = np.exp(z)
    return ez / ez.sum(axis=1, keepdims=True)


@dataclass(frozen=True)
class MLP:
    """A trained network: weights, biases, activation name, and label order.

    `weights[i]` maps layer i to layer i+1, so the last entry produces logits
    over the classes. `classes` records the original label values in the order
    the softmax columns correspond to, so `predict` can map argmax back to them.
    """

    weights: list[Array]
    biases: list[Array]
    activation: str
    classes: Array
    mu: Array
    sigma: Array

    def logits(self, X: Array) -> Array:
        act = ACTIVATIONS[self.activation]
        a = (_as_2d(X) - self.mu) / self.sigma
        for i, (w, b) in enumerate(zip(self.weights, self.biases, strict=True)):
            z = a @ w + b
            a = z if i == len(self.weights) - 1 else act.forward(z)
        return a

    def predict_proba(self, X: Array) -> Array:
        return _softmax(self.logits(X))

    def predict(self, X: Array) -> Array:
        idx = np.argmax(self.logits(X), axis=1)
        return self.classes[idx]


def _as_2d(X: Array) -> Array:
    arr = np.asarray(X, dtype=np.float64)
    return arr.reshape(-1, 1) if arr.ndim == 1 else arr


def _check_xy(X: Array, y: Array) -> tuple[Array, Array]:
    Xm = _as_2d(X)
    yv = np.asarray(y).reshape(-1)
    if Xm.shape[0] == 0:
        raise ValueError("need at least one sample")
    if Xm.shape[0] != yv.shape[0]:
        raise ValueError(f"X has {Xm.shape[0]} rows but y has {yv.shape[0]} entries")
    return Xm, yv


def _init_params(
    sizes: list[int], activation: str, rng: np.random.Generator
) -> tuple[list[Array], list[Array]]:
    """He init for relu, Xavier for tanh, so signal variance holds across depth.

    Both scale the initial weights by the fan-in; picking the wrong one for the
    nonlinearity is a real way to make a deep net fail to train, so it is chosen
    from the activation rather than left as a constant.
    """
    weights: list[Array] = []
    biases: list[Array] = []
    for fan_in, fan_out in zip(sizes[:-1], sizes[1:], strict=True):
        scale = np.sqrt((2.0 if activation == "relu" else 1.0) / fan_in)
        weights.append(rng.standard_normal((fan_in, fan_out)) * scale)
        biases.append(np.zeros(fan_out))
    return weights, biases


def cross_entropy(proba: Array, y_onehot: Array) -> float:
    # clip keeps log finite when a class probability rounds to exactly 0
    p = np.clip(proba, 1e-12, 1.0)
    return float(-np.mean(np.sum(y_onehot * np.log(p), axis=1)))


def accuracy(model: MLP, X: Array, y: Array) -> float:
    Xm, yv = _check_xy(X, y)
    return float(np.mean(model.predict(Xm) == yv))


def fit(
    X: Array,
    y: Array,
    hidden: tuple[int, ...] = (16, 16),
    activation: str = "tanh",
    lr: float = 0.1,
    epochs: int = 200,
    batch_size: int = 32,
    seed: int = 0,
) -> tuple[MLP, list[float]]:
    """Train an MLP classifier by minibatch SGD, returning it and the loss curve.

    Inputs are standardized to zero mean and unit variance so one learning rate
    works across features, and the standardizer is stored on the model so it
    consumes raw X at predict time. Labels may be any hashable values; they are
    mapped to softmax columns and remembered on the model. Returns the trained
    model and the per-epoch training cross-entropy.
    """
    if activation not in ACTIVATIONS:
        raise ValueError(f"unknown activation {activation!r}")
    if lr <= 0.0:
        raise ValueError("lr must be positive")
    if epochs <= 0:
        raise ValueError("epochs must be positive")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if any(h <= 0 for h in hidden):
        raise ValueError("hidden layer sizes must be positive")

    Xm, yv = _check_xy(X, y)
    n, d = Xm.shape

    classes, y_idx = np.unique(yv, return_inverse=True)
    k = classes.shape[0]
    if k < 2:
        raise ValueError("need at least two classes")
    y_onehot = np.eye(k)[y_idx]

    mu = Xm.mean(axis=0)
    sigma = Xm.std(axis=0)
    sigma = np.where(sigma == 0.0, 1.0, sigma)  # constant features carry no info
    Xs = (Xm - mu) / sigma

    act = ACTIVATIONS[activation]
    rng = np.random.default_rng(seed)
    sizes = [d, *hidden, k]
    weights, biases = _init_params(sizes, activation, rng)

    history: list[float] = []
    for _ in range(epochs):
        order = rng.permutation(n)
        for start in range(0, n, batch_size):
            idx = order[start : start + batch_size]
            xb, yb = Xs[idx], y_onehot[idx]
            _sgd_step(weights, biases, act, xb, yb, lr)
        proba = _softmax(_forward_logits(weights, biases, act, Xs))
        history.append(cross_entropy(proba, y_onehot))

    model = MLP(
        weights=weights,
        biases=biases,
        activation=activation,
        classes=classes,
        mu=mu,
        sigma=sigma,
    )
    return model, history


def _forward_logits(
    weights: list[Array], biases: list[Array], act: _Activation, x: Array
) -> Array:
    a = x
    last = len(weights) - 1
    for i, (w, b) in enumerate(zip(weights, biases, strict=True)):
        z = a @ w + b
        a = z if i == last else act.forward(z)
    return a


def _backprop(
    weights: list[Array],
    biases: list[Array],
    act: _Activation,
    xb: Array,
    yb: Array,
) -> tuple[list[Array], list[Array]]:
    """Forward with caches, then the delta recursion, returning per-layer grads.

    `zs[i]` is the pre-activation of layer i and `activations[i]` its input, so
    the backward pass reuses them without recomputation. The output delta is the
    softmax-cross-entropy shortcut (p - y) / m, and each earlier delta multiplies
    by the next weight and the local activation derivative.
    """
    m = xb.shape[0]
    activations: list[Array] = [xb]
    zs: list[Array] = []
    a = xb
    last = len(weights) - 1
    for i, (w, b) in enumerate(zip(weights, biases, strict=True)):
        z = a @ w + b
        zs.append(z)
        a = z if i == last else act.forward(z)
        activations.append(a)

    grad_w = [np.zeros_like(w) for w in weights]
    grad_b = [np.zeros_like(b) for b in biases]
    delta = (_softmax(zs[-1]) - yb) / m
    for i in reversed(range(len(weights))):
        grad_w[i] = activations[i].T @ delta
        grad_b[i] = delta.sum(axis=0)
        if i > 0:
            delta = act.backward(zs[i - 1], delta @ weights[i].T)
    return grad_w, grad_b


def _sgd_step(
    weights: list[Array],
    biases: list[Array],
    act: _Activation,
    xb: Array,
    yb: Array,
    lr: float,
) -> None:
    grad_w, grad_b = _backprop(weights, biases, act, xb, yb)
    for i in range(len(weights)):
        weights[i] -= lr * grad_w[i]
        biases[i] -= lr * grad_b[i]


def make_xor(n: int = 400, noise: float = 0.15, seed: int = 0) -> tuple[Array, Array]:
    """XOR: four gaussian blobs at the corners, labeled by parity of the corner.

    The canonical not-linearly-separable toy. A single hyperplane gets at most
    75% here; an MLP with a hidden layer reaches ~100%.
    """
    rng = np.random.default_rng(seed)
    centers = np.array([[-1.0, -1.0], [1.0, 1.0], [-1.0, 1.0], [1.0, -1.0]])
    labels = np.array([0, 0, 1, 1])
    per = n // 4
    X = np.vstack([c + noise * rng.standard_normal((per, 2)) for c in centers])
    y = np.repeat(labels, per)
    return X, y


def make_spiral(
    n: int = 300, classes: int = 3, noise: float = 0.2, seed: int = 0
) -> tuple[Array, Array]:
    """Interleaved spiral arms, one class per arm. Needs a curved boundary."""
    rng = np.random.default_rng(seed)
    per = n // classes
    X = np.zeros((per * classes, 2))
    y = np.zeros(per * classes, dtype=np.int64)
    for c in range(classes):
        r = np.linspace(0.0, 1.0, per)
        theta = np.linspace(c * 4.0, (c + 1) * 4.0, per)
        theta = theta + noise * rng.standard_normal(per)
        X[c * per : (c + 1) * per] = np.c_[r * np.sin(theta), r * np.cos(theta)]
        y[c * per : (c + 1) * per] = c
    return X, y
