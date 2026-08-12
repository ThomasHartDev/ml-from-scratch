from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]

class Optimizer(Protocol):
    def step(self, params: Sequence[Array], grads: Sequence[Array]) -> None: ...

def _validate_pairs(params: Sequence[Array], grads: Sequence[Array]) -> None:
    if len(params) != len(grads):
        raise ValueError(
            f"params has {len(params)} tensors but grads has {len(grads)}"
        )
    for i, (p, g) in enumerate(zip(params, grads, strict=True)):
        if p.shape != g.shape:
            raise ValueError(
                f"param/grad shape mismatch at index {i}: {p.shape} vs {g.shape}"
            )

class SGD:

    def __init__(self, lr: float = 0.1) -> None:
        if lr <= 0.0:
            raise ValueError("lr must be positive")
        self.lr = float(lr)

    def step(self, params: Sequence[Array], grads: Sequence[Array]) -> None:
        _validate_pairs(params, grads)
        for p, g in zip(params, grads, strict=True):
            p -= self.lr * g

class Momentum:

    def __init__(self, lr: float = 0.05, beta: float = 0.9) -> None:
        if lr <= 0.0:
            raise ValueError("lr must be positive")
        if not 0.0 <= beta < 1.0:
            raise ValueError("beta must be in [0, 1)")
        self.lr = float(lr)
        self.beta = float(beta)
        self._v: list[Array] | None = None

    def step(self, params: Sequence[Array], grads: Sequence[Array]) -> None:
        _validate_pairs(params, grads)
        if self._v is None:
            self._v = [np.zeros_like(p) for p in params]
        elif len(self._v) != len(params):
            raise ValueError("param list length changed between steps")
        for i, (p, g) in enumerate(zip(params, grads, strict=True)):
            self._v[i] = self.beta * self._v[i] + g
            p -= self.lr * self._v[i]

class Adam:

    def __init__(
        self,
        lr: float = 0.01,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
    ) -> None:
        if lr <= 0.0:
            raise ValueError("lr must be positive")
        if not 0.0 <= beta1 < 1.0:
            raise ValueError("beta1 must be in [0, 1)")
        if not 0.0 <= beta2 < 1.0:
            raise ValueError("beta2 must be in [0, 1)")
        if eps <= 0.0:
            raise ValueError("eps must be positive")
        self.lr = float(lr)
        self.beta1 = float(beta1)
        self.beta2 = float(beta2)
        self.eps = float(eps)
        self._m: list[Array] | None = None
        self._v: list[Array] | None = None
        self.t = 0

    def step(self, params: Sequence[Array], grads: Sequence[Array]) -> None:
        _validate_pairs(params, grads)
        if self._m is None:
            self._m = [np.zeros_like(p) for p in params]
            self._v = [np.zeros_like(p) for p in params]
        assert self._v is not None
        if len(self._m) != len(params):
            raise ValueError("param list length changed between steps")

        self.t += 1
        # bias correction grows toward 1; without it early steps are tiny
        bc1 = 1.0 - self.beta1**self.t
        bc2 = 1.0 - self.beta2**self.t
        for i, (p, g) in enumerate(zip(params, grads, strict=True)):
            self._m[i] = self.beta1 * self._m[i] + (1.0 - self.beta1) * g
            self._v[i] = self.beta2 * self._v[i] + (1.0 - self.beta2) * (g * g)
            m_hat = self._m[i] / bc1
            v_hat = self._v[i] / bc2
            p -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

def compare_optimizers(
    X: Array,
    y: Array,
    *,
    factories: dict[str, Callable[[], Optimizer]] | None = None,
    hidden: tuple[int, ...] = (16,),
    activation: str = "tanh",
    epochs: int = 150,
    batch_size: int = 32,
    seed: int = 0,
) -> dict[str, list[float]]:
    # late import: optimizers is usable without pulling the full MLP module
    from src.mlp import fit

    if factories is None:
        factories = {
            "sgd": lambda: SGD(lr=0.1),
            "momentum": lambda: Momentum(lr=0.05, beta=0.9),
            "adam": lambda: Adam(lr=0.01),
        }
    histories: dict[str, list[float]] = {}
    for name, factory in factories.items():
        _, history = fit(
            X,
            y,
            hidden=hidden,
            activation=activation,
            epochs=epochs,
            batch_size=batch_size,
            seed=seed,
            optimizer=factory(),
        )
        histories[name] = history
    return histories
