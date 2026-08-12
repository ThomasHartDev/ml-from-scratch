from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]

@dataclass(frozen=True)
class LinearModel:
    weights: Array
    bias: float

    def predict(self, X: Array) -> Array:
        return _as_2d(X) @ self.weights + self.bias

def mse(model: LinearModel, X: Array, y: Array) -> float:
    resid = model.predict(X) - np.asarray(y, dtype=np.float64)
    return float(np.mean(resid**2))

def _as_2d(X: Array) -> Array:
    arr = np.asarray(X, dtype=np.float64)
    return arr.reshape(-1, 1) if arr.ndim == 1 else arr

def _check_xy(X: Array, y: Array) -> tuple[Array, Array]:
    Xm = _as_2d(X)
    yv = np.asarray(y, dtype=np.float64).reshape(-1)
    if Xm.shape[0] == 0:
        raise ValueError("need at least one sample")
    if Xm.shape[0] != yv.shape[0]:
        raise ValueError(
            f"X has {Xm.shape[0]} rows but y has {yv.shape[0]} entries"
        )
    return Xm, yv

def fit_normal_equation(X: Array, y: Array, l2: float = 0.0) -> LinearModel:
    if l2 < 0.0:
        raise ValueError("l2 must be non-negative")
    Xm, yv = _check_xy(X, y)
    n, d = Xm.shape
    A = np.hstack([np.ones((n, 1)), Xm])

    gram = A.T @ A
    if l2 > 0.0:
        reg = np.eye(d + 1)
        reg[0, 0] = 0.0
        gram = gram + l2 * reg

    rhs = A.T @ yv
    try:
        theta = np.linalg.solve(gram, rhs)
    except np.linalg.LinAlgError:
        theta = np.linalg.lstsq(A, yv, rcond=None)[0]

    return LinearModel(weights=theta[1:], bias=float(theta[0]))

def fit_sgd(
    X: Array,
    y: Array,
    lr: float = 0.05,
    epochs: int = 100,
    batch_size: int = 32,
    l2: float = 0.0,
    seed: int = 0,
) -> tuple[LinearModel, list[float]]:
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
            resid = xb @ w + b - yb
            m = xb.shape[0]
            grad_w = (2.0 / m) * (xb.T @ resid) + 2.0 * l2 * w
            grad_b = (2.0 / m) * float(resid.sum())
            w -= lr * grad_w
            b -= lr * grad_b
        resid_full = Xs @ w + b - yv
        history.append(float(np.mean(resid_full**2)))

    weights = w / sigma
    bias = b - float(np.sum(w * mu / sigma))
    return LinearModel(weights=weights, bias=bias), history
