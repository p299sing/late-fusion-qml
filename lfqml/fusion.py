"""
fusion.py -- classical readout heads.

* LogisticHead  -- a linear (logistic-regression) head. Used both for the trained
  circuit's readout (B0/B1) and as the LATE-FUSION head (B2) over raw subcircuit
  features. Convex -> trains reliably with L-BFGS.
* MLPHead       -- a small 1-hidden-layer head for INTERMEDIATE/nonlinear fusion
  (the 'attention/MLP' point of the fusion spectrum). Optional.

Heads expose fit(F, y) and predict_proba(F) where F is a (n_samples, n_feat) matrix.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def _bce(p, y, eps=1e-9):
    p = np.clip(p, eps, 1 - eps)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


class LogisticHead:
    """Linear logistic-regression readout with L2 regularisation."""

    def __init__(self, l2: float = 1e-3):
        self.l2 = l2
        self.w = None
        self.b = 0.0

    def fit(self, F: np.ndarray, y: np.ndarray, maxiter: int = 200):
        F = np.atleast_2d(F)
        d = F.shape[1]

        def loss(z):
            w, b = z[:d], z[d]
            p = _sigmoid(F @ w + b)
            return _bce(p, y) + self.l2 * float(w @ w)

        z0 = np.zeros(d + 1)
        res = minimize(loss, z0, method="L-BFGS-B", options={"maxiter": maxiter})
        self.w, self.b = res.x[:d], res.x[d]
        return self

    def predict_proba(self, F: np.ndarray) -> np.ndarray:
        return _sigmoid(np.atleast_2d(F) @ self.w + self.b)


class MLPHead:
    """One-hidden-layer tanh MLP head (nonlinear / intermediate fusion)."""

    def __init__(self, hidden: int = 8, l2: float = 1e-3, seed: int = 0):
        self.hidden = hidden
        self.l2 = l2
        self.seed = seed
        self.params = None
        self._shapes = None

    def _unpack(self, z, d):
        h = self.hidden
        i = 0
        W1 = z[i:i + d * h].reshape(d, h); i += d * h
        b1 = z[i:i + h]; i += h
        W2 = z[i:i + h]; i += h
        b2 = z[i]
        return W1, b1, W2, b2

    def fit(self, F, y, maxiter: int = 300):
        F = np.atleast_2d(F)
        d = F.shape[1]
        h = self.hidden
        n_params = d * h + h + h + 1
        rng = np.random.default_rng(self.seed)

        def forward(z):
            W1, b1, W2, b2 = self._unpack(z, d)
            a = np.tanh(F @ W1 + b1)
            return _sigmoid(a @ W2 + b2)

        def loss(z):
            p = forward(z)
            return _bce(p, y) + self.l2 * float(z @ z)

        z0 = 0.1 * rng.standard_normal(n_params)
        res = minimize(loss, z0, method="L-BFGS-B", options={"maxiter": maxiter})
        self.params, self._d = res.x, d
        return self

    def predict_proba(self, F):
        F = np.atleast_2d(F)
        W1, b1, W2, b2 = self._unpack(self.params, self._d)
        a = np.tanh(F @ W1 + b1)
        return _sigmoid(a @ W2 + b2)
