"""
train.py -- train the target QNN (baseline B0), jointly with a linear readout.

We optimise z = [theta_quantum | head_w | head_b] to minimise binary cross-entropy
on the UNCUT circuit's readout features.  Gradients are obtained numerically by
L-BFGS-B (parameter counts here are small, ~20-40).  A parameter-shift gradient is
a drop-in speed upgrade later; see CODE_FLOW.md ("Performance / scaling up").

The trained circuit is then handed to models.py, which derives every readout
baseline (B0, B1, B2, B6, B7, Q-dial) from this ONE circuit so that the *only*
variable across baselines is the readout method -- exactly what H1/H3 require.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.optimize import minimize

from dataclasses import replace
from .circuits import (QNNConfig, init_params, param_count, prepare,
                       default_observables, full_features, Observable)
from .fusion import _sigmoid, _bce
from . import cutting, qsim


def _make_feature_fn(cfg, X, paulis, cap=1024):
    """Return f(q) -> subcircuit-feature matrix, memoized on the quantum params q.

    In the fusion trainers, loss(z) with z=[q | head] recomputes quantum features on
    EVERY L-BFGS finite-difference probe -- including probes that only perturb the
    classical head, for which the features are identical. Memoizing on q.tobytes()
    makes those probes reuse the cached features (an EXACT optimization, no change to
    results), giving a large speedup for wide heads where |head| >> |q|.
    """
    cache: dict[bytes, np.ndarray] = {}

    def features(q):
        key = q.tobytes()
        F = cache.get(key)
        if F is None:
            F = np.array([cutting.subcircuit_raw_features(prepare(cfg, q, x), paulis=paulis)
                          for x in X])
            if len(cache) >= cap:
                cache.clear()
            cache[key] = F
        return F

    return features


@dataclass
class TrainedQNN:
    cfg: QNNConfig
    params: np.ndarray            # trained quantum parameters (theta)
    head_w: np.ndarray            # trained linear readout weights
    head_b: float
    observables: list             # readout observables (order matches head_w)
    history: dict


def b0_feature_matrix(cfg: QNNConfig, theta: np.ndarray, X: np.ndarray,
                      observables: list[Observable]) -> np.ndarray:
    """Uncut readout features for every sample: shape (n_samples, n_obs)."""
    F = np.empty((len(X), len(observables)))
    for i, x in enumerate(X):
        pc = prepare(cfg, theta, x)
        F[i] = full_features(pc, observables)
    return F


def train_qnn(cfg: QNNConfig, X: np.ndarray, y: np.ndarray, seed: int = 0,
              l2: float = 1e-3, maxiter: int = 150, beta: float = 0.0,
              verbose: bool = False) -> TrainedQNN:
    """Jointly train quantum params + linear head on the uncut (B0) features.

    beta > 0 adds the FUSION-FRIENDLY entanglement penalty (H2): it penalises the
    mean A:B cut-entanglement of the trained circuit, pushing the model to solve
    the task with as little cross-cut quantum correlation as possible.  Sweeping
    beta shifts the whole accuracy/cost Pareto frontier toward cheaper operating
    points (knee -> smaller Q).  beta = 0 recovers the unregularised model.
    """
    obs = default_observables(cfg)
    P = param_count(cfg)
    d = len(obs)
    theta0 = init_params(cfg, seed=seed)
    z0 = np.concatenate([theta0, np.zeros(d + 1)])

    evals = {"n": 0}
    _fcache: dict[bytes, np.ndarray] = {}

    def b0_features(theta):
        # cache on theta so head-only finite-difference probes reuse features (exact)
        key = theta.tobytes()
        F = _fcache.get(key)
        if F is None:
            F = b0_feature_matrix(cfg, theta, X, obs)
            if len(_fcache) >= 1024:
                _fcache.clear()
            _fcache[key] = F
        return F

    def unpack(z):
        return z[:P], z[P:P + d], z[P + d]

    def mean_entanglement(theta):
        # averaged over a subsample of inputs to keep the penalty cheap
        from .circuits import full_state
        from . import qsim
        idx = np.linspace(0, len(X) - 1, min(len(X), 24)).astype(int)
        vals = []
        for i in idx:
            pc = prepare(cfg, theta, X[i])
            psi = full_state(pc)
            vals.append(qsim.cut_entanglement_entropy(psi, cfg.A_qubits, cfg.n))
        return float(np.mean(vals))

    def loss(z):
        theta, w, b = unpack(z)
        F = b0_features(theta)
        p = _sigmoid(F @ w + b)
        evals["n"] += 1
        val = _bce(p, y) + l2 * float(w @ w) + 1e-4 * float(theta @ theta)
        if beta > 0:
            val += beta * mean_entanglement(theta)
        return val

    res = minimize(loss, z0, method="L-BFGS-B", options={"maxiter": maxiter})
    theta, w, b = unpack(res.x)
    if verbose:
        print(f"  train_qnn: final loss={res.fun:.4f}  loss_evals={evals['n']}  beta={beta}")
    return TrainedQNN(cfg=cfg, params=theta, head_w=w, head_b=b,
                      observables=obs, history={"loss": float(res.fun), "beta": beta})


# --------------------------------------------------------------------------- #
# INDEPENDENT / fusion-native training (no coupling; subcircuits + head jointly)
# --------------------------------------------------------------------------- #
@dataclass
class TrainedFusion:
    cfg: QNNConfig
    q_params: np.ndarray          # subcircuit params (NO coupling)
    mlp_params: np.ndarray        # fusion head (1 hidden tanh layer)
    hidden: int
    paulis: tuple

    def _features(self, X):
        return np.array([cutting.subcircuit_raw_features(
            prepare(self.cfg, self.q_params, x), paulis=self.paulis) for x in X])

    def predict_proba(self, X):
        F = self._features(X)
        return _mlp_forward(self.mlp_params, F, F.shape[1], self.hidden)


def _mlp_forward(mp, F, d, h):
    W1 = mp[:d * h].reshape(d, h)
    b1 = mp[d * h:d * h + h]
    W2 = mp[d * h + h:d * h + 2 * h]
    b2 = mp[d * h + 2 * h]
    return _sigmoid(np.tanh(F @ W1 + b1) @ W2 + b2)


def train_fusion_independent(cfg: QNNConfig, X, y, seed=0, hidden=8,
                             paulis=(qsim.Z,), maxiter=150, verbose=False) -> TrainedFusion:
    """Train subcircuits + a classical fusion head JOINTLY, with NO coupling gate.

    This is genuine late fusion "from scratch": the two subcircuits never interact
    quantumly (no cross-cut entanglement), but their parameters are optimised so
    that their local measurements are *fusion-friendly* features for the head.
    Contrast with the frozen-B0 fusion baseline (B2), where subcircuits were trained
    as part of a coupled circuit and then re-used. Tests whether independent training
    closes the residual reconstruction gap.
    """
    cfg = replace(cfg, trainable_coupling=False, coupling_init=0.0)  # no coupling
    P = param_count(cfg)
    # feature dimension = |paulis| * n_qubits
    d = len(paulis) * cfg.n
    n_mlp = d * hidden + hidden + hidden + 1
    rng = np.random.default_rng(seed)
    z0 = np.concatenate([init_params(cfg, seed=seed),
                         0.1 * rng.standard_normal(n_mlp)])

    features = _make_feature_fn(cfg, X, paulis)

    def loss(z):
        q, mp = z[:P], z[P:]
        F = features(q)
        p = _mlp_forward(mp, F, d, hidden)
        return _bce(p, y) + 1e-4 * float(q @ q) + 1e-4 * float(mp @ mp)

    res = minimize(loss, z0, method="L-BFGS-B", options={"maxiter": maxiter})
    if verbose:
        print(f"  train_fusion_independent: final loss={res.fun:.4f}")
    return TrainedFusion(cfg=cfg, q_params=res.x[:P], mlp_params=res.x[P:],
                         hidden=hidden, paulis=paulis)


@dataclass
class TrainedKawase:
    """Faithful Kawase (arXiv:2312.13650) baseline: partitioned-feature subcircuits
    combined by a PARAMETER-FREE sum of expectation values (only a global scale+bias),
    trained end-to-end. Contrast with our trained fusion head (TrainedFusion)."""
    cfg: QNNConfig
    q_params: np.ndarray
    scale: float
    bias: float

    def predict_proba(self, X):
        s = np.array([cutting.subcircuit_raw_features(
            prepare(self.cfg, self.q_params, x)).sum() for x in X])  # equal-weight sum
        return _sigmoid(self.scale * s + self.bias)


def train_kawase(cfg: QNNConfig, X, y, seed=0, maxiter=150) -> TrainedKawase:
    """Kawase-style distributed QNN: subcircuits + parameter-free expectation sum."""
    cfg = replace(cfg, trainable_coupling=False, coupling_init=0.0)
    P = param_count(cfg)
    rng = np.random.default_rng(seed)
    z0 = np.concatenate([init_params(cfg, seed=seed), [1.0, 0.0]])  # q, scale, bias
    _scache: dict[bytes, np.ndarray] = {}

    def sums(q):  # cache on q so scale/bias probes reuse the expectation sum (exact)
        key = q.tobytes()
        s = _scache.get(key)
        if s is None:
            s = np.array([cutting.subcircuit_raw_features(prepare(cfg, q, x)).sum() for x in X])
            if len(_scache) >= 1024:
                _scache.clear()
            _scache[key] = s
        return s

    def loss(z):
        q, scale, bias = z[:P], z[P], z[P + 1]
        p = _sigmoid(scale * sums(q) + bias)
        return _bce(p, y) + 1e-4 * float(q @ q)

    res = minimize(loss, z0, method="L-BFGS-B", options={"maxiter": maxiter})
    return TrainedKawase(cfg=cfg, q_params=res.x[:P], scale=res.x[P], bias=res.x[P + 1])


# --------------------------------------------------------------------------- #
# G12: multiclass late fusion (softmax head over subcircuit features).
# --------------------------------------------------------------------------- #
def _softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def _softmax_forward(mp, F, d, h, K):
    """One tanh hidden layer -> K-way softmax. mp packs [W1,b1,W2,b2]."""
    o = 0
    W1 = mp[o:o + d * h].reshape(d, h); o += d * h
    b1 = mp[o:o + h]; o += h
    W2 = mp[o:o + h * K].reshape(h, K); o += h * K
    b2 = mp[o:o + K]
    return _softmax(np.tanh(F @ W1 + b1) @ W2 + b2)


@dataclass
class TrainedFusionMC:
    """Multiclass (K-way softmax) independent late fusion for G12."""
    cfg: QNNConfig
    q_params: np.ndarray
    mlp_params: np.ndarray
    hidden: int
    n_classes: int
    paulis: tuple

    def predict_proba(self, X):
        F = np.array([cutting.subcircuit_raw_features(
            prepare(self.cfg, self.q_params, x), paulis=self.paulis) for x in X])
        return _softmax_forward(self.mlp_params, F, F.shape[1], self.hidden, self.n_classes)

    def predict(self, X):
        return self.predict_proba(X).argmax(axis=1)


def train_fusion_multiclass(cfg: QNNConfig, X, y, n_classes, seed=0, hidden=8,
                            paulis=(qsim.Z,), maxiter=150) -> TrainedFusionMC:
    """K-way late fusion from scratch: independent subcircuits + softmax fusion head.
    Demonstrates the method is not limited to binary labels (G12)."""
    cfg = replace(cfg, trainable_coupling=False, coupling_init=0.0)
    P = param_count(cfg)
    d = len(paulis) * cfg.n
    K = n_classes
    n_mlp = d * hidden + hidden + hidden * K + K
    rng = np.random.default_rng(seed)
    z0 = np.concatenate([init_params(cfg, seed=seed), 0.1 * rng.standard_normal(n_mlp)])
    Y = np.eye(K)[y]  # one-hot
    features = _make_feature_fn(cfg, X, paulis)

    def loss(z):
        q, mp = z[:P], z[P:]
        F = features(q)
        p = _softmax_forward(mp, F, d, hidden, K)
        ce = -np.mean(np.sum(Y * np.log(p + 1e-9), axis=1))
        return ce + 1e-4 * float(q @ q) + 1e-4 * float(mp @ mp)

    res = minimize(loss, z0, method="L-BFGS-B", options={"maxiter": maxiter})
    return TrainedFusionMC(cfg=cfg, q_params=res.x[:P], mlp_params=res.x[P:],
                           hidden=hidden, n_classes=K, paulis=paulis)
