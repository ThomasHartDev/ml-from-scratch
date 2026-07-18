"""Scalar reverse-mode autograd. Placeholder until the engine lands."""

from __future__ import annotations


class Value:
    def __init__(self, data: float) -> None:
        self.data = float(data)
        self.grad = 0.0

    def __repr__(self) -> str:
        return f"Value(data={self.data:.6g}, grad={self.grad:.6g})"
