"""Binary logistic regression trained with cross-entropy loss.

Logistic regression models P(y=1 | x) as sigmoid(w·x + b). Its loss is the
negative log-likelihood of the Bernoulli labels, which is convex, so gradient
descent finds the global optimum. The gradient has the same shape as linear
regression: the residual (predicted probability minus label) times the input.
The one thing you cannot do naively is the loss itself, because sigmoid and log
overflow on large logits, so both the loss and the sigmoid are written in their
numerically stable forms here. The decision boundary is the line where the
logit is zero, and for two features that is a straight line you can plot.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]


@dataclass(frozen=True)
class LogisticModel:
    weights: Array
    bias: float

    def decision_function(self, X: Array) -> Array:
        """Raw logits w·x + b, before the sigmoid."""
        return _as_2d(X) @ self.weights + self.bias

    def predict_proba(self, X: Array) -> Array:
        return _sigmoid(self.decision_function(X))

    def predict(self, X: Array, threshold: float = 0.5) -> Array:
        return (self.predict_proba(X) >= threshold).astype(np.int64)


def _sigmoid(z: Array) -> Array:
    # branch on sign so neither exp overflows: exp(-|z|) is always in (0, 1]
    out = np.empty_like(z, dtype=np.float64)
    pos = z >= 0.0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


def bce_loss(model: LogisticModel, X: Array, y: Array) -> float:
    """Mean binary cross-entropy in logits, via softplus(z) - y·z.

    softplus(z) = log(1 + e^z) computed as logaddexp(0, z) never overflows, and
    the identity BCE = softplus(z) - y·z avoids ever calling log on a sigmoid
    that has saturated to exactly 0 or 1.
    """
    Xm, yv = _check_xy(X, y)
    z = Xm @ model.weights + model.bias
    return float(np.mean(np.logaddexp(0.0, z) - yv * z))


def decision_boundary(model: LogisticModel, x1: Array) -> Array:
    """For a 2-feature model, the x2 values where P(y=1)=0.5 along given x1.

    The boundary is w0·x1 + w1·x2 + b = 0, so x2 = -(b + w0·x1) / w1. Only
    defined when the model has exactly two features and w1 is nonzero (a
    horizontal boundary has no functional form in x2).
    """
    if model.weights.shape != (2,):
        raise ValueError("decision_boundary is only defined for two features")
    w0, w1 = float(model.weights[0]), float(model.weights[1])
    if w1 == 0.0:
        raise ValueError("boundary is vertical in x2; w1 is zero")
    x1v = np.asarray(x1, dtype=np.float64)
    return -(model.bias + w0 * x1v) / w1


def accuracy(model: LogisticModel, X: Array, y: Array) -> float:
    Xm, yv = _check_xy(X, y)
    return float(np.mean(model.predict(Xm) == yv))


def _as_2d(X: Array) -> Array:
    arr = np.asarray(X, dtype=np.float64)
    return arr.reshape(-1, 1) if arr.ndim == 1 else arr


def _check_xy(X: Array, y: Array) -> tuple[Array, Array]:
    Xm = _as_2d(X)
    yv = np.asarray(y, dtype=np.float64).reshape(-1)
    if Xm.shape[0] == 0:
        raise ValueError("need at least one sample")
    if Xm.shape[0] != yv.shape[0]:
        raise ValueError(f"X has {Xm.shape[0]} rows but y has {yv.shape[0]} entries")
    if not np.all((yv == 0.0) | (yv == 1.0)):
        raise ValueError("labels must be binary 0/1")
    return Xm, yv


def fit_sgd(
    X: Array,
    y: Array,
    lr: float = 0.1,
    epochs: int = 200,
    batch_size: int = 32,
    l2: float = 0.0,
    seed: int = 0,
) -> tuple[LogisticModel, list[float]]:
    """Fit logistic regression by minibatch gradient descent on standardized X.

    The gradient of the mean cross-entropy w.r.t. the logits is (p - y), the
    same residual form as least squares, so a batch update is (1/m) Xᵀ(p - y).
    Features are standardized to unit variance first for conditioning, then the
    weights are mapped back to raw feature space so the returned model consumes
    unscaled X. L2 shrinks the weights but not the bias. Returns the model and
    the per-epoch training cross-entropy.
    """
    if lr <= 0.0:
        raise ValueError("lr must be positive")
    if epochs <= 0:
        raise ValueError("epochs must be positive")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if l2 < 0.0:
        raise ValueError("l2 must be non-negative")

    Xm, yv = _check_xy(X, y)
    n, d = Xm.shape

    mu = Xm.mean(axis=0)
    sigma = Xm.std(axis=0)
    sigma = np.where(sigma == 0.0, 1.0, sigma)  # constant features carry no info
    Xs = (Xm - mu) / sigma

    rng = np.random.default_rng(seed)
    w = np.zeros(d)
    b = 0.0
    history: list[float] = []

    for _ in range(epochs):
        order = rng.permutation(n)
        for start in range(0, n, batch_size):
            idx = order[start : start + batch_size]
            xb, yb = Xs[idx], yv[idx]
            m = xb.shape[0]
            resid = _sigmoid(xb @ w + b) - yb
            grad_w = (xb.T @ resid) / m + l2 * w
            grad_b = float(resid.sum()) / m
            w -= lr * grad_w
            b -= lr * grad_b
        z_full = Xs @ w + b
        history.append(float(np.mean(np.logaddexp(0.0, z_full) - yv * z_full)))

    weights = w / sigma
    bias = b - float(np.sum(w * mu / sigma))
    return LogisticModel(weights=weights, bias=bias), history
