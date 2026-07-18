# ml-from-scratch

Core machine-learning and deep-learning building blocks implemented from scratch in Python, numpy only. The point is to show the math behind modern AI, not to call a library.

## What this demonstrates

Every deep learning framework is, at its core, a graph that records operations and replays them backward to compute gradients. This repo builds that machinery by hand and then uses it to train models, so the chain rule, backpropagation, and gradient descent stay visible in a few hundred lines you can read end to end. Each concept lands as its own tested module with the intuition and the math written out.

## Concepts demonstrated

- Reverse-mode automatic differentiation (backpropagation)
- Gradient descent and numerical gradient checking

## What's implemented

Nothing yet. First module lands next.

## Running the tests

```bash
pip install -r requirements-dev.txt
pytest
```

Lint with `ruff check .`. CI runs both on every push and pull request across Python 3.10 and 3.12.
