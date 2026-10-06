"""
deep_circuits.py -- DEEP coupled QNN with L coupling layers (paper revision item E1).

The main library (circuits.py) has exactly ONE coupling layer, which is what makes the
gate-cut reconstruction in cutting.py exactly factorisable.  Reviewer R2 asked whether
"coupling depth" is only a structural observation on untrained circuits
(exp_multilayer.py) or actually predicts fusion-accuracy degradation on TRAINED models.
This module provides the trained version:

    DEEP COUPLED (reference; == exact reconstruction, cost 9^(kL) so never actually cut)
        encode(x)  ->  [ local_A, local_B, coupling(phi_l) ] x L  ->  local_A, local_B
        readout = default_observables (<Z_q> per qubit + <Z_a Z_b> per cross pair)
        + linear logistic head, trained jointly (mirrors train.train_qnn).

    UNCOUPLED FUSION (same circuit with every RZZ removed -> A (x) B product state)
        readout = <Z_q> per qubit (as cutting.subcircuit_raw_features with paulis=(Z,))
        + MLP head (hidden=8), trained end-to-end (mirrors train.train_fusion_independent).

Parameter layout  [layer_0_A | layer_0_B | layer_1_A | layer_1_B | ... | layer_L_A | layer_L_B | phi_0..phi_{L-1}]
with each local block = (RY, RZ) per qubit.  At L=1 this is EXACTLY circuits.QNNConfig(depth=1)'s
layout [pre_A | pre_B | post_A | post_B | coupling], and init_params draws the same numbers, so
L=1 reproduces the single-coupling-layer model (tests/test_deep_circuits.py checks the states).

Local blocks are the same hardware-efficient layer as circuits.local_var_layer (RY,RZ per qubit
+ CZ chain within the register, + closing CZ if the register has >2 qubits).  Coupling layer l
applies RZZ(phi_l) on every boundary pair of cfg.cross_pairs (ONE angle per layer shared across
the k pairs, exactly as the single-layer model shares one phi across its k cross gates).

Two evaluation paths are provided:
  * state_single(...)  -- one input, built from qsim.apply_1q / apply_2q (the reference).
  * states_batch(...)  -- all inputs at once, same maths vectorised over a leading sample axis.
The batch path is what the trainers use (L-BFGS-B with numerical gradients needs thousands of
forward passes); the test asserts both agree to machine precision.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import numpy as np
from scipy.optimize import minimize

from . import qsim
from .circuits import QNNConfig, default_observables, Observable
from .fusion import _sigmoid, _bce
from .train import _mlp_forward


# --------------------------------------------------------------------------- #
# Config + parameter bookkeeping
# --------------------------------------------------------------------------- #
@dataclass
class DeepConfig:
    n_A: int = 2
    n_B: int = 2
    L: int = 1                       # number of coupling layers (L+1 local layers)
    n_cuts: int = 1                  # k cross RZZ gates per coupling layer
    coupled: bool = True             # False -> every RZZ removed (product circuit)
    coupling_init: float = np.pi / 2

    @property
    def n(self) -> int:
        return self.n_A + self.n_B

    @property
    def A_qubits(self) -> list[int]:
        return list(range(self.n_A))

    @property
    def B_qubits(self) -> list[int]:
        return list(range(self.n_A, self.n))

    @property
    def cross_pairs(self) -> list[tuple[int, int]]:
        # identical to QNNConfig.cross_pairs
        return QNNConfig(n_A=self.n_A, n_B=self.n_B, n_cuts=self.n_cuts).cross_pairs

    @property
    def n_local_params(self) -> int:
        return (self.L + 1) * 2 * self.n

    @property
    def n_coupling_params(self) -> int:
        return self.L if self.coupled else 0

    @property
    def n_params(self) -> int:
        return self.n_local_params + self.n_coupling_params

    def as_qnn_config(self) -> QNNConfig:
        """The single-layer QNNConfig this reduces to when L == 1."""
        return QNNConfig(n_A=self.n_A, n_B=self.n_B, depth=1, n_cuts=self.n_cuts,
                         trainable_coupling=self.coupled, coupling_init=self.coupling_init)


def init_params(cfg: DeepConfig, seed: int = 0) -> np.ndarray:
    """0.1*N(0,1) local angles, couplings started at coupling_init (as circuits.init_params)."""
    rng = np.random.default_rng(seed)
    p = 0.1 * rng.standard_normal(cfg.n_params)
    if cfg.coupled:
        p[cfg.n_local_params:] = cfg.coupling_init
    return p


def coupling_angles(cfg: DeepConfig, params: np.ndarray) -> np.ndarray:
    if not cfg.coupled:
        return np.zeros(cfg.L)
    return np.asarray(params[cfg.n_local_params:cfg.n_local_params + cfg.L], dtype=float)


def _local_block(params: np.ndarray, layer: int, reg: str, cfg: DeepConfig) -> np.ndarray:
    """Slice of the (RY,RZ)-per-qubit angles of local layer `layer` on register reg."""
    off = layer * 2 * cfg.n
    if reg == "A":
        return params[off: off + 2 * cfg.n_A]
    return params[off + 2 * cfg.n_A: off + 2 * cfg.n]


# --------------------------------------------------------------------------- #
# Reference path: ONE input, built from qsim primitives
# --------------------------------------------------------------------------- #
def _local_layer_single(psi, block, qubits, n):
    idx = 0
    for q in qubits:
        psi = qsim.apply_1q(psi, qsim.ry(block[idx]), q, n); idx += 1
        psi = qsim.apply_1q(psi, qsim.rz(block[idx]), q, n); idx += 1
    for k in range(len(qubits) - 1):
        psi = qsim.apply_2q(psi, qsim.CZ, qubits[k], qubits[k + 1], n)
    if len(qubits) > 2:
        psi = qsim.apply_2q(psi, qsim.CZ, qubits[-1], qubits[0], n)
    return psi


def state_single(cfg: DeepConfig, params: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Full n-qubit state for one input, using qsim.apply_1q / apply_2q only."""
    n = cfg.n
    psi = qsim.zero_state(n)
    for q in range(n):                                     # angle encoding (RY)
        psi = qsim.apply_1q(psi, qsim.ry(float(x[q])), q, n)
    phis = coupling_angles(cfg, params)
    for l in range(cfg.L + 1):
        psi = _local_layer_single(psi, _local_block(params, l, "A", cfg), cfg.A_qubits, n)
        psi = _local_layer_single(psi, _local_block(params, l, "B", cfg), cfg.B_qubits, n)
        if l < cfg.L and cfg.coupled:
            for (a, b) in cfg.cross_pairs:
                psi = qsim.apply_2q(psi, qsim.rzz(phis[l]), a, b, n)
    return psi


# --------------------------------------------------------------------------- #
# Batched path: all inputs at once (leading sample axis), same maths
# --------------------------------------------------------------------------- #
def apply_1q_batch(t: np.ndarray, U: np.ndarray, q: int) -> np.ndarray:
    """t has shape (N, 2, ..., 2); apply U to qubit q (axis q+1) of every sample."""
    out = np.tensordot(U, t, axes=([1], [q + 1]))          # new axis 0 = q's out index
    return np.moveaxis(out, 0, q + 1)


def apply_2q_batch(t: np.ndarray, U: np.ndarray, qa: int, qb: int) -> np.ndarray:
    U4 = U.reshape(2, 2, 2, 2)
    out = np.tensordot(U4, t, axes=([2, 3], [qa + 1, qb + 1]))
    return np.moveaxis(out, [0, 1], [qa + 1, qb + 1])


def _encode_batch(X: np.ndarray, n: int) -> np.ndarray:
    """Product state (x)_q RY(x_q)|0> for every row of X -> (N, 2, ..., 2)."""
    N = X.shape[0]
    t = np.ones((N, 1), dtype=complex)
    for q in range(n):
        amp = np.stack([np.cos(X[:, q] / 2), np.sin(X[:, q] / 2)], axis=1).astype(complex)
        t = (t[:, :, None] * amp[:, None, :]).reshape(N, -1)
    return t.reshape([N] + [2] * n)


def _local_layer_batch(t, block, qubits):
    idx = 0
    for q in qubits:
        t = apply_1q_batch(t, qsim.ry(block[idx]), q); idx += 1
        t = apply_1q_batch(t, qsim.rz(block[idx]), q); idx += 1
    for k in range(len(qubits) - 1):
        t = apply_2q_batch(t, qsim.CZ, qubits[k], qubits[k + 1])
    if len(qubits) > 2:
        t = apply_2q_batch(t, qsim.CZ, qubits[-1], qubits[0])
    return t


def states_batch(cfg: DeepConfig, params: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Full states for every input: shape (N, 2**n)."""
    X = np.atleast_2d(X)
    t = _encode_batch(X, cfg.n)
    phis = coupling_angles(cfg, params)
    for l in range(cfg.L + 1):
        t = _local_layer_batch(t, _local_block(params, l, "A", cfg), cfg.A_qubits)
        t = _local_layer_batch(t, _local_block(params, l, "B", cfg), cfg.B_qubits)
        if l < cfg.L and cfg.coupled:
            for (a, b) in cfg.cross_pairs:
                t = apply_2q_batch(t, qsim.rzz(phis[l]), a, b)
    return t.reshape(X.shape[0], -1)


def expval_batch(psis: np.ndarray, paulis: dict[int, np.ndarray], n: int) -> np.ndarray:
    """<psi_i| (x) paulis |psi_i> for every row (real part)."""
    t = psis.reshape([psis.shape[0]] + [2] * n)
    for q, P in paulis.items():
        t = apply_1q_batch(t, P, q)
    phi = t.reshape(psis.shape[0], -1)
    return np.real(np.einsum("ni,ni->n", psis.conj(), phi))


def observables(cfg: DeepConfig) -> list[Observable]:
    """Same readout set the single-layer B0 model uses: <Z_q> all qubits + <Z_a Z_b> cross pairs."""
    return default_observables(cfg.as_qnn_config())


def feature_matrix(cfg: DeepConfig, params: np.ndarray, X: np.ndarray,
                   obs: list[Observable]) -> np.ndarray:
    """(N, n_obs) matrix of <O_i> on the full (coupled or uncoupled) state."""
    psis = states_batch(cfg, params, X)
    qcfg = cfg.as_qnn_config()
    return np.stack([expval_batch(psis, o.global_dict(qcfg), cfg.n) for o in obs], axis=1)


def local_z_features(cfg: DeepConfig, params: np.ndarray, X: np.ndarray) -> np.ndarray:
    """<Z_q> for every qubit, i.e. the register-local marginal features late fusion sees.

    For the UNCOUPLED circuit this equals cutting.subcircuit_raw_features(paulis=(Z,)).
    For the COUPLED circuit it is Tr(rho_A Z_q) / Tr(rho_B Z_q) of the reduced states.
    """
    psis = states_batch(cfg, params, X)
    return np.stack([expval_batch(psis, {q: qsim.Z}, cfg.n) for q in range(cfg.n)], axis=1)


def cut_entropies(cfg: DeepConfig, params: np.ndarray, X: np.ndarray) -> np.ndarray:
    """A:B entanglement entropy (bits) of the full state for every input."""
    psis = states_batch(cfg, params, X)
    return np.array([qsim.cut_entanglement_entropy(p, cfg.A_qubits, cfg.n) for p in psis])


def connected_correlations(cfg: DeepConfig, params: np.ndarray, X: np.ndarray) -> np.ndarray:
    """|<Z_a Z_b> - <Z_a><Z_b>| for every input and boundary pair -> (N, k).

    This is exactly the cross-cut information marginal (fusion) features cannot recover.
    """
    psis = states_batch(cfg, params, X)
    cols = []
    for (a, b) in cfg.cross_pairs:
        zz = expval_batch(psis, {a: qsim.Z, b: qsim.Z}, cfg.n)
        za = expval_batch(psis, {a: qsim.Z}, cfg.n)
        zb = expval_batch(psis, {b: qsim.Z}, cfg.n)
        cols.append(np.abs(zz - za * zb))
    return np.stack(cols, axis=1)


# --------------------------------------------------------------------------- #
# Trainers (mirror train.train_qnn and train.train_fusion_independent)
# --------------------------------------------------------------------------- #
@dataclass
class TrainedDeepQNN:
    cfg: DeepConfig
    params: np.ndarray
    head_w: np.ndarray
    head_b: float
    observables: list
    history: dict

    def predict_proba(self, X):
        F = feature_matrix(self.cfg, self.params, X, self.observables)
        return _sigmoid(F @ self.head_w + self.head_b)


def train_deep_coupled(cfg: DeepConfig, X, y, seed: int = 0, l2: float = 1e-3,
                       maxiter: int = 90) -> TrainedDeepQNN:
    """Jointly train the deep coupled circuit + linear head (same loss/optimiser as train_qnn:
    BCE + l2*|w|^2 + 1e-4*|theta|^2, L-BFGS-B with numerical gradients, feature cache on theta)."""
    obs = observables(cfg)
    P, d = cfg.n_params, len(obs)
    z0 = np.concatenate([init_params(cfg, seed=seed), np.zeros(d + 1)])
    cache: dict[bytes, np.ndarray] = {}
    evals = {"n": 0}

    def feats(theta):
        key = theta.tobytes()
        F = cache.get(key)
        if F is None:
            F = feature_matrix(cfg, theta, X, obs)
            if len(cache) >= 1024:
                cache.clear()
            cache[key] = F
        return F

    def loss(z):
        theta, w, b = z[:P], z[P:P + d], z[P + d]
        p = _sigmoid(feats(theta) @ w + b)
        evals["n"] += 1
        return _bce(p, y) + l2 * float(w @ w) + 1e-4 * float(theta @ theta)

    res = minimize(loss, z0, method="L-BFGS-B", options={"maxiter": maxiter})
    return TrainedDeepQNN(cfg=cfg, params=res.x[:P], head_w=res.x[P:P + d], head_b=res.x[P + d],
                          observables=obs,
                          history={"loss": float(res.fun), "nit": int(res.nit),
                                   "loss_evals": evals["n"], "status": int(res.status)})


@dataclass
class TrainedDeepFusion:
    cfg: DeepConfig                 # coupled=False
    q_params: np.ndarray
    mlp_params: np.ndarray
    hidden: int

    def predict_proba(self, X):
        F = local_z_features(self.cfg, self.q_params, X)
        return _mlp_forward(self.mlp_params, F, F.shape[1], self.hidden)


def train_deep_fusion(cfg: DeepConfig, X, y, seed: int = 0, hidden: int = 8,
                      maxiter: int = 120) -> TrainedDeepFusion:
    """Uncoupled late fusion trained end-to-end: L+1 local layers per register (no RZZ anywhere)
    + MLP head on <Z_q>, same loss/init/optimiser as train.train_fusion_independent
    (BCE + 1e-4|q|^2 + 1e-4|mlp|^2).  At L=1 it is parameter-for-parameter identical to
    train_fusion_independent(QNNConfig(depth=1), paulis=(Z,))."""
    cfg = replace(cfg, coupled=False)
    P = cfg.n_params
    d = cfg.n
    n_mlp = d * hidden + hidden + hidden + 1
    rng = np.random.default_rng(seed)
    z0 = np.concatenate([init_params(cfg, seed=seed), 0.1 * rng.standard_normal(n_mlp)])
    cache: dict[bytes, np.ndarray] = {}

    def feats(q):
        key = q.tobytes()
        F = cache.get(key)
        if F is None:
            F = local_z_features(cfg, q, X)
            if len(cache) >= 1024:
                cache.clear()
            cache[key] = F
        return F

    def loss(z):
        q, mp = z[:P], z[P:]
        p = _mlp_forward(mp, feats(q), d, hidden)
        return _bce(p, y) + 1e-4 * float(q @ q) + 1e-4 * float(mp @ mp)

    res = minimize(loss, z0, method="L-BFGS-B", options={"maxiter": maxiter})
    return TrainedDeepFusion(cfg=cfg, q_params=res.x[:P], mlp_params=res.x[P:], hidden=hidden)
