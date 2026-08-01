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
- Ordinary least squares in closed form via the normal equation
- Ridge regularization that shrinks slopes without penalizing the intercept
- Minibatch stochastic gradient descent with feature standardization for conditioning
- The equivalence of the closed-form and iterative solutions on a convex loss
- Pseudoinverse fallback for rank-deficient design matrices
- Binary logistic regression as a Bernoulli likelihood, trained by minimizing cross-entropy
- The `sigmoid(z) - y` gradient identity, the same residual form that least squares has
- Numerically stable sigmoid (sign-branched) and log loss via `softplus(z) - y·z` (`logaddexp`)
- The linear decision boundary as the level set where the logit is zero
- A multilayer perceptron: stacked affine layers with `tanh`/`relu` nonlinearities
- Backpropagation derived by hand as a delta recursion, vectorized over a minibatch (no autodiff)
- The `softmax - onehot` output-delta shortcut and `delta_prev = (delta @ Wᵀ) ⊙ act'(z)` for hidden layers
- Xavier vs He weight initialization chosen from the activation to keep signal variance stable across depth
- Learning a non-linearly-separable target (XOR, interleaved spirals) that a single linear model cannot fit
- Local activation derivatives for the chain rule (`act'(z)`): tanh, sigmoid, ReLU, leaky ReLU, identity
- Glorot/Xavier initialization: variance `2/(fan_in+fan_out)` balancing forward and backward signal
- He/Kaiming initialization: variance `2/fan_in` restoring the half of the signal ReLU drops
- Forward variance profiling across depth: naive `N(0,1)` weights explode; correct scales stay `O(1)`
- Vanishing activations under mismatched init (Xavier on a deep ReLU stack) measured, not just described
- First-order optimizers as pure parameter updates decoupled from the loss and backprop
- Vanilla SGD: θ ← θ − ηg
- Heavy-ball / Polyak momentum: velocity accumulation that carries steps through elongated valleys
- Adam (adaptive moment estimation): bias-corrected first and second moments, per-parameter step sizes
- Fair convergence comparison: same MLP init, same minibatches, only the update rule changes
- Distributional hypothesis: word meaning from co-occurrence context
- Skip-gram language modeling: predict context words given a center word inside a sliding window
- Two-matrix word2vec parameterization (center embeddings W_in, context embeddings W_out)
- Full-softmax training objective with the `softmax − onehot` gradient identity
- Dense word vectors and cosine nearest-neighbor retrieval as a semantic similarity demo

## What's implemented

- **Gradient descent + autodiff-lite**: a scalar reverse-mode autograd engine (`Value`) with `+`, `*`, `**`, division, and `relu`/`tanh`/`exp`/`log`/`sigmoid`, plus an `SGD` optimizer (with momentum) and a `minimize` training loop. Gradients are checked against finite differences and against numpy for softmax cross-entropy.
- **Linear regression, two ways**: `fit_normal_equation` solves ordinary least squares exactly by setting the gradient to zero and solving `(AᵀA + λR)θ = Aᵀy`, with optional ridge and a least-norm fallback when `AᵀA` is singular. `fit_sgd` fits the same model with minibatch gradient descent on standardized features, then maps the weights back to raw feature space. Both are checked to recover the true coefficients and to agree with each other, so the exact-vs-iterative tradeoff is measurable.
- **Logistic regression + cross-entropy, decision boundary demo**: `src/logreg.py` trains a binary classifier by minibatch SGD, using the fact that the gradient of the mean cross-entropy with respect to the logits is exactly `sigmoid(z) - y`, the same residual form linear regression has. The sigmoid branches on the sign of the logit so `exp` never overflows, and the loss is computed as `softplus(z) - y·z` through `numpy.logaddexp` so a confidently-wrong prediction gives a large finite loss instead of `inf`. `decision_boundary` returns the straight line where the model sits at 50 percent for a two-feature problem, the level set `w·x + b = 0`.
- **Multilayer perceptron with hand-derived backprop**: `src/mlp.py` stacks affine layers with `tanh` or `relu` and a softmax head, and trains a multiclass classifier by minibatch SGD. The backward pass is written out by hand as one recursion on the per-layer delta rather than delegated to an autodiff engine: the output delta is the `softmax - onehot` residual, each hidden delta is `(delta_next @ W_nextᵀ) ⊙ act'(z)`, and the parameter gradients are `dW = a_prevᵀ @ delta` and `db = Σ delta`. Weights use He init for `relu` and Xavier for `tanh` so the signal variance holds across depth. The gradients are verified against central finite differences to a tight tolerance, and the model learns XOR and a three-arm spiral, targets a single hyperplane provably cannot separate. Ships with `make_xor` and `make_spiral` toy generators.
- **Activation functions + weight initialization (Xavier/He) and why they matter**: `src/activations.py` is the dedicated treatment of the nonlinearity and the initial scale. Each activation (`linear`, `tanh`, `sigmoid`, `relu`, `leaky_relu`) exposes `forward` and a local `backward(z, grad_out)` that multiplies by `act'(z)`, so a hand-written backprop step can drop it in. Xavier/Glorot draws `N(0, 2/(fan_in+fan_out))` (or the matching uniform bound) to keep both forward and backward variance stable for symmetric activations; He/Kaiming draws `N(0, 2/fan_in)` so a ReLU stack does not quietly die after a few layers. `forward_variance_profile` stacks affine+activation layers from unit-variance noise and returns the per-layer activation variance: naive `N(0,1)` weights explode, He keeps a ReLU stack `O(1)`, and Xavier on the same ReLU stack fades, which is the usual silent failure mode when the scheme and the nonlinearity disagree.
- **SGD, Momentum, Adam from scratch, convergence compared on the same net**: `src/optimizers.py` implements the three standard first-order update rules as numpy-only classes that own their state (velocity for momentum, bias-corrected moments for Adam) and mutate a flat list of parameter arrays in place. The MLP training loop calls `optimizer.step(params, grads)` after the hand-written backprop pass, so swapping the rule never touches the gradient math. `compare_optimizers` retrains the same architecture on the same data with the same seed for each factory, which keeps init and minibatch order fixed and isolates the update rule; on XOR, all three cut loss, and Adam typically pulls ahead of plain SGD early because its per-parameter rates absorb the uneven scale of the gradient.
- **Word embeddings (skip-gram) trained on a small corpus, nearest-neighbor demo**: `src/embeddings.py` learns dense vectors by predicting each window neighbor of a center word (skip-gram). Two matrices store center and context embeddings; the score for context o given center c is the dot product `W_out[o] · W_in[c]`, turned into a distribution with a full softmax over the vocabulary. Minibatch SGD minimizes the mean negative log-likelihood, using the closed-form `p − onehot` gradient on the logits (verified against finite differences). After training, `nearest(word, k)` ranks the rest of the vocab by cosine similarity on the center rows. A built-in toy corpus repeats short sentences about capitals, animals, and people so co-occurrence clusters show up in the neighbor lists without any external data.

## Usage

```python
from src.autograd import Value, minimize

# minimize f(x) = (x - 3)^2 with gradient descent
x = Value(-4.0)
minimize(lambda: (x - 3.0) ** 2, [x], steps=200, lr=0.1)
print(x.data)  # ~3.0
```

Fit a line two ways and compare:

```python
import numpy as np
from src.linreg import fit_normal_equation, fit_sgd, mse

rng = np.random.default_rng(0)
X = rng.uniform(-2, 2, size=(200, 3))
y = X @ np.array([2.0, -3.0, 0.5]) + 1.5

exact = fit_normal_equation(X, y)              # closed form, no tuning
approx, history = fit_sgd(X, y, lr=0.1, epochs=300)  # iterative, streams over data

print(exact.weights, exact.bias)   # ~[2, -3, 0.5], 1.5
print(mse(approx, X, y))           # ~0, matches the closed form
```

Fit a binary classifier and read its decision boundary:

```python
import numpy as np
from src.logreg import fit_sgd, accuracy, decision_boundary

rng = np.random.default_rng(0)
X = rng.uniform(-3, 3, size=(400, 2))
y = (X @ np.array([3.0, -2.0]) + 0.5 > 0).astype(int)

model, history = fit_sgd(X, y, lr=0.2, epochs=300)
print(accuracy(model, X, y))          # ~1.0 on separable data
print(model.predict_proba(X[:3]))     # calibrated probabilities in (0, 1)

# the 50 percent line, ready to plot against the two features
xs = np.linspace(-3, 3, 50)
boundary = decision_boundary(model, xs)
```

Train a small network on a target no line can separate:

```python
from src.mlp import fit, accuracy, make_xor

X, y = make_xor(n=400, seed=0)
model, history = fit(X, y, hidden=(16,), activation="tanh", epochs=300)

print(accuracy(model, X, y))     # ~1.0; a linear model caps out at 0.75 here
print(history[0], history[-1])   # cross-entropy falls over training
print(model.predict_proba(X[:3]))  # per-class probabilities that sum to 1
```

Train the same net under SGD, momentum, and Adam and compare loss curves:

```python
from src.mlp import make_xor, fit, accuracy
from src.optimizers import SGD, Momentum, Adam, compare_optimizers

X, y = make_xor(n=400, seed=0)

# plug any optimizer into the existing MLP trainer
model, history = fit(X, y, hidden=(16,), epochs=200, optimizer=Adam(lr=0.01))
print(accuracy(model, X, y), history[0], history[-1])

# same init + minibatches; only the update rule changes
curves = compare_optimizers(X, y, epochs=150, seed=0)
for name, h in curves.items():
    print(name, h[0], "->", h[-1])
```

Train skip-gram embeddings and query nearest neighbors:

```python
from src.embeddings import fit, make_toy_corpus, cosine_similarity

model, history = fit(make_toy_corpus(), dim=32, window=2, epochs=100, seed=0)
print(history[0], "->", history[-1])          # NLL falls over training
print(model.nearest("paris", k=4))            # other capital cities rise
print(model.nearest("cat", k=4))              # dog / mouse / cats nearby
print(cosine_similarity(model.embed("king"), model.embed("queen")))
```

Compare init schemes by watching activation variance with depth:

```python
from src.activations import (
    get_activation,
    he_normal,
    recommended_init,
    forward_variance_profile,
)

act = get_activation("relu")
print(recommended_init("relu"))   # "he"
print(he_normal(64, 64).std())    # ~sqrt(2/64)

# unit-scale noise, ten ReLU layers: He stays O(1), Xavier fades, naive explodes
print(forward_variance_profile(10, 64, "relu", "he")[-1])
print(forward_variance_profile(10, 64, "relu", "xavier")[-1])
print(forward_variance_profile(6, 64, "linear", "naive")[-1])
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
