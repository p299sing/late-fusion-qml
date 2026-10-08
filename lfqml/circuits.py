"""
circuits.py -- the target QNN and its A|B partition (the "cut architecture").

Architecture (deliberately cut-friendly)
----------------------------------------
Two qubit-disjoint registers A and B.  Data flow, left to right:

    A-reg:  encode(x_A) -> local_var(theta_pre_A) --.                 .-- local_var(theta_post_A) -> readout_A
                                                     |  COUPLING LAYER |
    B-reg:  encode(x_B) -> local_var(theta_pre_B) --'  (k cross gates) '-- local_var(theta_post_B) -> readout_B

* Everything before the coupling layer is register-local, so the pre-coupling
  state is an exact product  |psi_A> ⊗ |psi_B>.
* The ONLY gates that entangle A with B live in the coupling layer (k cross
  gates on boundary qubits) -- these are the "cuts".
* Everything after the coupling layer is register-local, so post-coupling gates
  fold into local observables and the readout observables factorize as O_A ⊗ O_B.

These three properties are exactly what let cutting.py reconstruct the uncut
expectation values to machine precision (see tests/test_cutting.py).  If you add
cross-register gates OUTSIDE the coupling layer you break the clean single-layer
factorization -- see CODE_FLOW.md ("Scope & how to extend").

Ansatz families
---------------
QNNConfig.ansatz selects the register-local variational block (the coupling
layer is RZZ in both cases, so the cut machinery is unchanged):
  "A" (default)  RY,RZ per qubit + CZ ring,            x depth   -- the original block
  "B"            RX,RY per qubit + CNOT ladder (no wrap), x depth -- robustness check
Both use 2 angles per qubit per layer, so the parameter layout is identical.

Entangler
---------
The coupling gate is RZZ(phi).  phi is a knob on entangling power: phi=0 is the
identity (no cut-entanglement), phi=pi/2 is maximally entangling.  During
training phi can be fixed or made trainable; the *trained* circuit's realised
cut-entanglement is what H1/H3 correlate against.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np

from . import qsim


# --------------------------------------------------------------------------- #
# Parameter bookkeeping
# --------------------------------------------------------------------------- #
@dataclass
class QNNConfig:
    n_A: int = 2                 # qubits in register A
    n_B: int = 2                 # qubits in register B
    depth: int = 1               # variational layers on each side of the coupling
    n_cuts: int = 1              # number of cross gates in the coupling layer
    trainable_coupling: bool = True   # is the RZZ angle a trained parameter?
    coupling_init: float = np.pi / 2  # initial RZZ angle (max-entangling by default)
    ansatz: str = "A"            # local block family: "A" = RY,RZ + CZ ring (default);
                                 #                     "B" = RX,RY + CNOT ladder (no wrap)

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
        """Global (a, b) qubit pairs joined by the coupling layer.

        Pairs the last qubits of A with the first qubits of B (boundary qubits).
        """
        pairs = []
        for j in range(self.n_cuts):
            a = self.n_A - 1 - j          # walk inward from A's boundary
            b = self.n_A + j              # walk inward from B's boundary
            pairs.append((a, b))
        return pairs


# --------------------------------------------------------------------------- #
# Parameter vector layout helpers
# --------------------------------------------------------------------------- #
# We store all variational angles in a flat vector for easy optimisation.
# Layout: [pre_A | pre_B | post_A | post_B | (coupling if trainable)]
# Each register block has depth * n_reg * 2 angles (RY, RZ per qubit per layer).

def n_block_params(n_reg: int, depth: int) -> int:
    return depth * n_reg * 2


def param_count(cfg: QNNConfig) -> int:
    per_side_A = n_block_params(cfg.n_A, cfg.depth)
    per_side_B = n_block_params(cfg.n_B, cfg.depth)
    total = 2 * per_side_A + 2 * per_side_B    # pre/post for A and B
    if cfg.trainable_coupling:
        total += 1
    return total


def init_params(cfg: QNNConfig, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    p = 0.1 * rng.standard_normal(param_count(cfg))
    if cfg.trainable_coupling:
        p[-1] = cfg.coupling_init            # start coupling at chosen angle
    return p


def _slice(cfg: QNNConfig):
    """Return dict of (start, stop) slices for each parameter block."""
    a = n_block_params(cfg.n_A, cfg.depth)
    b = n_block_params(cfg.n_B, cfg.depth)
    i = 0
    s = {}
    s["pre_A"] = (i, i + a); i += a
    s["pre_B"] = (i, i + b); i += b
    s["post_A"] = (i, i + a); i += a
    s["post_B"] = (i, i + b); i += b
    if cfg.trainable_coupling:
        s["coupling"] = (i, i + 1); i += 1
    return s


def coupling_angle(cfg: QNNConfig, params: np.ndarray) -> float:
    if cfg.trainable_coupling:
        return float(params[_slice(cfg)["coupling"][0]])
    return cfg.coupling_init


# --------------------------------------------------------------------------- #
# Building blocks (act on a register-local substate)
# --------------------------------------------------------------------------- #
def encode(psi: np.ndarray, x: np.ndarray, n_reg: int) -> np.ndarray:
    """Angle-encode features x (length n_reg) as RY rotations, one per qubit."""
    for q in range(n_reg):
        psi = qsim.apply_1q(psi, qsim.ry(float(x[q])), q, n_reg)
    return psi


# Ansatz families for the register-local variational block.  Each entry is
# (first 1q rotation, second 1q rotation, 2q entangler, wrap-around ring?).
# Both families use 2 angles per qubit per layer, so the parameter layout
# (param_count / init_params / _slice) is identical for "A" and "B".
ANSATZ_FAMILIES = {
    "A": (qsim.ry, qsim.rz, qsim.CZ, True),    # RY,RZ per qubit + CZ ring (original)
    "B": (qsim.rx, qsim.ry, qsim.CX, False),   # RX,RY per qubit + CNOT ladder (no wrap)
}


def _ansatz_gates(ansatz):
    """Resolve an ansatz spec (string or QNNConfig) to its gate family."""
    if isinstance(ansatz, QNNConfig):
        ansatz = ansatz.ansatz
    try:
        return ANSATZ_FAMILIES[ansatz]
    except KeyError:
        raise ValueError(f"unknown ansatz {ansatz!r}; choose from {sorted(ANSATZ_FAMILIES)}")


def local_var_layer(psi: np.ndarray, block: np.ndarray, n_reg: int, depth: int,
                    ansatz="A") -> np.ndarray:
    """A hardware-efficient variational block, x depth.

    ansatz "A" (default): RY,RZ per qubit + CZ ring  (wrap-around CZ if n_reg > 2).
    ansatz "B":           RX,RY per qubit + CNOT ladder CX(q, q+1), q = 0..n_reg-2.
    `ansatz` may be the family string or a QNNConfig (its .ansatz is used).
    """
    rot1, rot2, ent, wrap = _ansatz_gates(ansatz)
    idx = 0
    for _ in range(depth):
        for q in range(n_reg):
            psi = qsim.apply_1q(psi, rot1(block[idx]), q, n_reg); idx += 1
            psi = qsim.apply_1q(psi, rot2(block[idx]), q, n_reg); idx += 1
        # entangling layer within the register (only if >1 qubit)
        for q in range(n_reg - 1):
            psi = qsim.apply_2q(psi, ent, q, q + 1, n_reg)
        if wrap and n_reg > 2:
            psi = qsim.apply_2q(psi, ent, n_reg - 1, 0, n_reg)
    return psi


# --------------------------------------------------------------------------- #
# The structured, cut-ready representation of one forward pass
# --------------------------------------------------------------------------- #
@dataclass
class PreparedCircuit:
    """Everything cutting.py / models.py needs for one input x.

    Holds the pre-coupling product substates and the callables that finish the
    A and B halves.  This is the single source of truth shared by the uncut
    path (B0), exact reconstruction (B1), the Q-dial, and late fusion (B2).
    """
    cfg: QNNConfig
    psiA_pre: np.ndarray          # pre-coupling substate on A (2**n_A,)
    psiB_pre: np.ndarray          # pre-coupling substate on B (2**n_B,)
    postA: np.ndarray             # post-coupling variational params for A
    postB: np.ndarray             # post-coupling variational params for B
    phi: float                    # coupling RZZ angle

    def finish_A(self, psiA: np.ndarray) -> np.ndarray:
        """Apply A's post-coupling local gates to an A-substate."""
        return local_var_layer(psiA, self.postA, self.cfg.n_A, self.cfg.depth, self.cfg.ansatz)

    def finish_B(self, psiB: np.ndarray) -> np.ndarray:
        return local_var_layer(psiB, self.postB, self.cfg.n_B, self.cfg.depth, self.cfg.ansatz)

    # local (register-relative) boundary indices touched by the coupling layer
    def local_cross_pairs(self):
        """Return [(a_local, b_local), ...] with indices *within* each register."""
        out = []
        for (a, b) in self.cfg.cross_pairs:
            out.append((a, b - self.cfg.n_A))
        return out


def prepare(cfg: QNNConfig, params: np.ndarray, x: np.ndarray) -> PreparedCircuit:
    """Run encoding + pre-coupling layers on each register; return cut-ready state.

    x is the full feature vector of length n; the first n_A entries feed A, the
    rest feed B (feature partitioning across subcircuits, cf. Kawase 2312.13650).
    """
    s = _slice(cfg)
    xA, xB = x[: cfg.n_A], x[cfg.n_A:]

    psiA = qsim.zero_state(cfg.n_A)
    psiA = encode(psiA, xA, cfg.n_A)
    psiA = local_var_layer(psiA, params[slice(*s["pre_A"])], cfg.n_A, cfg.depth, cfg.ansatz)

    psiB = qsim.zero_state(cfg.n_B)
    psiB = encode(psiB, xB, cfg.n_B)
    psiB = local_var_layer(psiB, params[slice(*s["pre_B"])], cfg.n_B, cfg.depth, cfg.ansatz)

    return PreparedCircuit(
        cfg=cfg,
        psiA_pre=psiA,
        psiB_pre=psiB,
        postA=params[slice(*s["post_A"])],
        postB=params[slice(*s["post_B"])],
        phi=coupling_angle(cfg, params),
    )


# --------------------------------------------------------------------------- #
# The uncut ("gold standard", B0) forward pass and its readout features
# --------------------------------------------------------------------------- #
def coupling_unitary_4x4(phi: float) -> np.ndarray:
    """The two-qubit entangler used at every cross pair."""
    return qsim.rzz(phi)


def full_state(pc: PreparedCircuit) -> np.ndarray:
    """Assemble the full uncut state |psi> on n qubits (B0 path)."""
    cfg = pc.cfg
    psi = np.kron(pc.psiA_pre, pc.psiB_pre)          # product before coupling
    for (a, b) in cfg.cross_pairs:                   # coupling layer
        psi = qsim.apply_2q(psi, coupling_unitary_4x4(pc.phi), a, b, cfg.n)
    # post-coupling local layers (applied on the full register, but local)
    psi = _apply_block_on_subset(psi, pc.postA, cfg.A_qubits, cfg)
    psi = _apply_block_on_subset(psi, pc.postB, cfg.B_qubits, cfg)
    return psi


def _apply_block_on_subset(psi, block, qubits, cfg: QNNConfig):
    """Apply a local_var_layer to a subset of qubits of the FULL state.

    Used only by the uncut path; the cut path applies post gates on substates.
    """
    n = cfg.n
    n_reg = len(qubits)
    rot1, rot2, ent, wrap = _ansatz_gates(cfg)
    idx = 0
    for _ in range(cfg.depth):
        for k, q in enumerate(qubits):
            psi = qsim.apply_1q(psi, rot1(block[idx]), q, n); idx += 1
            psi = qsim.apply_1q(psi, rot2(block[idx]), q, n); idx += 1
        for k in range(n_reg - 1):
            psi = qsim.apply_2q(psi, ent, qubits[k], qubits[k + 1], n)
        if wrap and n_reg > 2:
            psi = qsim.apply_2q(psi, ent, qubits[-1], qubits[0], n)
    return psi


# --------------------------------------------------------------------------- #
# Readout observables (all factorize as O_A ⊗ O_B, so all are reconstructable)
# --------------------------------------------------------------------------- #
@dataclass
class Observable:
    """A product observable, stored as its A-part and B-part (local Pauli dicts).

    A_part maps A-local qubit index -> 2x2 Pauli; B_part likewise (B-local).
    """
    A_part: dict = field(default_factory=dict)
    B_part: dict = field(default_factory=dict)
    label: str = ""

    def global_dict(self, cfg: QNNConfig) -> dict:
        d = dict(self.A_part)
        for q, P in self.B_part.items():
            d[q + cfg.n_A] = P
        return d


def default_observables(cfg: QNNConfig) -> list[Observable]:
    """Feature set: <Z> on every qubit + <Z_a Z_b> on every cross pair.

    These are the readout features fed to the classifier head.  Every one is a
    product observable, hence exactly reconstructable from the two subcircuits.
    """
    obs = []
    for q in range(cfg.n_A):
        obs.append(Observable(A_part={q: qsim.Z}, label=f"ZA{q}"))
    for q in range(cfg.n_B):
        obs.append(Observable(B_part={q: qsim.Z}, label=f"ZB{q}"))
    for (a, b) in cfg.cross_pairs:                    # cross correlators
        obs.append(Observable(A_part={a: qsim.Z}, B_part={b - cfg.n_A: qsim.Z},
                              label=f"ZZ{a}-{b}"))
    return obs


def full_features(pc: PreparedCircuit, observables: list[Observable]) -> np.ndarray:
    """Uncut (B0) feature vector: <O_i> of the full circuit for each observable."""
    psi = full_state(pc)
    return np.array([qsim.expval(psi, o.global_dict(pc.cfg), pc.cfg.n) for o in observables])
