"""Gradient descent over `Value` parameters.

`SGD` holds references to the leaf `Value`s it optimizes and steps each one
against its accumulated gradient. `momentum` adds a velocity term, the same
update PyTorch's SGD uses. `zero_grad` must run before each backward pass
because gradients accumulate rather than overwrite.
"""

from __future__ import annotations

from collections.abc import Sequence

from .engine import Value


class SGD:
    def __init__(
        self,
        params: Sequence[Value],
        lr: float = 0.01,
        momentum: float = 0.0,
    ) -> None:
        if lr <= 0.0:
            raise ValueError("lr must be positive")
        if not 0.0 <= momentum < 1.0:
            raise ValueError("momentum must be in [0, 1)")
        self.params = list(params)
        self.lr = lr
        self.momentum = momentum
        self._velocity = [0.0 for _ in self.params]

    def zero_grad(self) -> None:
        for p in self.params:
            p.grad = 0.0

    def step(self) -> None:
        for i, p in enumerate(self.params):
            if self.momentum:
                self._velocity[i] = self.momentum * self._velocity[i] + p.grad
                p.data -= self.lr * self._velocity[i]
            else:
                p.data -= self.lr * p.grad


def minimize(
    loss_fn,
    params: Sequence[Value],
    steps: int = 100,
    lr: float = 0.01,
    momentum: float = 0.0,
) -> list[float]:
    """Run `steps` of gradient descent, returning the loss at each step."""
    opt = SGD(params, lr=lr, momentum=momentum)
    history: list[float] = []
    for _ in range(steps):
        opt.zero_grad()
        loss = loss_fn()
        loss.backward()
        opt.step()
        history.append(loss.data)
    return history
