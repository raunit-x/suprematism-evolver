"""Activation functions for CPPN nodes.

Each function operates element-wise on numpy arrays for vectorized evaluation.
The choice of activation functions determines the visual vocabulary:
  - sin/cos  -> repetition, stripes, interference patterns
  - gaussian -> radial symmetry, soft blobs
  - sigmoid  -> smooth binary partitioning (Malevich figure/ground)
  - step     -> hard edges, flat regions (Suprematist geometry)
  - abs      -> bilateral symmetry
  - sawtooth -> repeating ramps, hatching feel (Basquiat texture)
"""

import numpy as np


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60, 60)))


def tanh(x):
    return np.tanh(x)


def sin(x):
    return np.sin(x)


def cos(x):
    return np.cos(x)


def gaussian(x):
    return np.exp(-x * x / 2.0)


def abs_fn(x):
    return np.abs(x)


def relu(x):
    return np.maximum(0.0, x)


def step(x):
    return np.where(x > 0.0, 1.0, 0.0)


def linear(x):
    return x


def square(x):
    return x * x


def sawtooth(x):
    return (x / np.pi) - np.floor(x / np.pi + 0.5)


def inv(x):
    """Inverse / negation -- flips sign."""
    return -x


def softplus(x):
    return np.log1p(np.exp(np.clip(x, -60, 60)))


# Registry: name -> function
ACTIVATIONS = {
    "sigmoid": sigmoid,
    "tanh": tanh,
    "sin": sin,
    "cos": cos,
    "gaussian": gaussian,
    "abs": abs_fn,
    "relu": relu,
    "step": step,
    "linear": linear,
    "square": square,
    "sawtooth": sawtooth,
    "inv": inv,
    "softplus": softplus,
}

# Subsets biased toward each artist's aesthetic
MALEVICH_ACTIVATIONS = ["sigmoid", "step", "gaussian", "tanh", "abs", "linear"]
BASQUIAT_ACTIVATIONS = ["sin", "sawtooth", "relu", "tanh", "gaussian", "softplus", "square"]
ALL_ACTIVATIONS = list(ACTIVATIONS.keys())


def get_activation(name: str):
    """Look up activation function by name."""
    if name not in ACTIVATIONS:
        raise ValueError(f"Unknown activation: {name!r}. Available: {list(ACTIVATIONS.keys())}")
    return ACTIVATIONS[name]
