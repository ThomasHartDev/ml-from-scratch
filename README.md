# ml-from-scratch

Core machine-learning and deep-learning building blocks implemented from scratch in Python, numpy only. The point is to show the math behind modern AI, not to call a library.

## What this demonstrates

Every deep learning framework is, at its core, a graph that records operations and replays them backward to compute gradients. This repo builds that machinery by hand and then uses it to train models, so the chain rule, backpropagation, and gradient descent are all visible in a few hundred lines you can read end to end. Each concept lands as its own tested module with the intuition and the math written out, working up to a from-scratch attention block that learns a toy task.

## Concepts demonstrated

- Reverse-mode automatic differentiation (backpropagation) over a dynamically built computation graph
- The chain rule applied per-operation via local backward closures
- Reverse topological ordering of the graph so each node's gradient is complete before it is used
- Gradient accumulation for nodes reused multiple times in a graph (a real source of bugs)
- Gradient descent: vanilla SGD and SGD with momentum
- Numerical gradient checking (central finite differences) as a correctness oracle
- Softmax with numerically-checked cross-entropy loss, including the `softmax - onehot` gradient identity
- Linear regression trained purely through the autograd engine

## What's implemented

- **Gradient descent + autodiff-lite**: a scalar reverse-mode autograd engine (`Value`) with `+`, `*`, `**`, division, and `relu`/`tanh`/`exp`/`log`/`sigmoid`, plus an `SGD` optimizer (with momentum) and a `minimize` training loop. Gradients are checked against finite differences and against numpy for softmax cross-entropy.

## Usage

```python
from src.autograd import Value, minimize

# minimize f(x) = (x - 3)^2 with gradient descent
x = Value(-4.0)
minimize(lambda: (x - 3.0) ** 2, [x], steps=200, lr=0.1)
print(x.data)  # ~3.0
```

Differentiate an arbitrary scalar expression:

```python
from src.autograd import Value

a = Value(2.0)
b = Value(-3.0)
c = (a * b + b).tanh()
c.backward()
print(a.grad, b.grad)  # d c / d a, d c / d b
```

## Running the tests

```bash
pip install -r requirements-dev.txt
pytest
```

Lint with `ruff check .`. CI runs both on every push and pull request across Python 3.10 and 3.12.
