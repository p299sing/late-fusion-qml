"""
qsim.py -- A tiny, transparent state-vector simulator for the late-fusion QML project.

Why a hand-rolled simulator (instead of Qiskit) for the scientific core?
  * The controlled H1/H3 experiments run on <= ~10 qubits, where a dense state
    vector is trivially fast.
  * We need three things Qiskit does not hand us cheaply and transparently:
      (1) EXACT gate-cut reconstruction we can verify to machine precision,
      (2) partial (truncated) reconstruction for the "Q-dial",
      (3) the exact cut-entanglement entropy of the *trained* circuit.
    All three fall out of having direct access to amplitudes and sub-operators.
  * It is short and fully commented, so you can modify the physics yourself.

Qiskit / qiskit-addon-cutting integration is a SEPARATE, later path used only for
the camera-ready benchmark credibility. See CODE_FLOW.md ("Integration points").

Conventions
-----------
* A pure state on ``n`` qubits is a complex numpy array of shape (2**n,).
* Qubit 0 is the MOST significant index (big-endian), i.e. basis index
  ``b = sum_q bit_q * 2**(n-1-q)``.  This matches Qiskit's *little-endian* only
  after a relabel; we stay internally consistent and never mix conventions.
* Gates are applied by reshaping the state to a rank-n tensor and contracting.
"""

from __future__ import annotations

import numpy as np

# --------------------------------------------------------------------------- #
# Single-qubit gate matrices (2x2). All are plain numpy arrays.
# --------------------------------------------------------------------------- #
I2 = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)

# The four single-qubit Paulis, indexed 0..3 = I,X,Y,Z (used everywhere for the
# Pauli-basis bookkeeping of cuts and observables).
PAULIS = [I2, X, Y, Z]
PAULI_LABELS = ["I", "X", "Y", "Z"]


def rx(theta: float) -> np.ndarray:
    """Rotation about X by angle theta."""
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.array([[c, -1j * s], [-1j * s, c]], dtype=complex)


def ry(theta: float) -> np.ndarray:
    """Rotation about Y by angle theta."""
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.array([[c, -s], [s, c]], dtype=complex)


def rz(theta: float) -> np.ndarray:
    """Rotation about Z by angle theta."""
    e = np.exp(-1j * theta / 2)
    return np.array([[e, 0], [0, np.conj(e)]], dtype=complex)


# Two-qubit gates (4x4), ordered basis |q_a q_b> with q_a the high bit.
CZ = np.diag([1, 1, 1, -1]).astype(complex)
CX = np.array(
    [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=complex
)


def rzz(theta: float) -> np.ndarray:
    """exp(-i theta/2 Z⊗Z) -- a tunable entangler; theta controls entangling power."""
    return np.diag(
        [np.exp(-1j * theta / 2), np.exp(1j * theta / 2),
         np.exp(1j * theta / 2), np.exp(-1j * theta / 2)]
    ).astype(complex)


# --------------------------------------------------------------------------- #
# State construction
# --------------------------------------------------------------------------- #
def zero_state(n: int) -> np.ndarray:
    """|00...0> on n qubits."""
    psi = np.zeros(2 ** n, dtype=complex)
    psi[0] = 1.0
    return psi


# --------------------------------------------------------------------------- #
# Gate application
# --------------------------------------------------------------------------- #
def apply_1q(psi: np.ndarray, U: np.ndarray, q: int, n: int) -> np.ndarray:
    """
    Apply single-qubit operator U (need NOT be unitary -- we also apply
    projectors/Pauli operators when evaluating cut terms) to qubit q.
    """
    t = psi.reshape([2] * n)
    # Move axis q to front, contract with U, move back.
    t = np.tensordot(U, t, axes=([1], [q]))          # new axis 0 is q's out index
    t = np.moveaxis(t, 0, q)
    return t.reshape(-1)


def apply_2q(psi: np.ndarray, U: np.ndarray, qa: int, qb: int, n: int) -> np.ndarray:
    """Apply 4x4 two-qubit operator U to (qa, qb) with qa the high bit."""
    t = psi.reshape([2] * n)
    U4 = U.reshape(2, 2, 2, 2)                        # [a_out, b_out, a_in, b_in]
    t = np.tensordot(U4, t, axes=([2, 3], [qa, qb]))  # axes 0,1 are a_out,b_out
    t = np.moveaxis(t, [0, 1], [qa, qb])
    return t.reshape(-1)


# --------------------------------------------------------------------------- #
# Observables / expectation values
# --------------------------------------------------------------------------- #
def apply_pauli_string(psi: np.ndarray, paulis: dict[int, np.ndarray], n: int) -> np.ndarray:
    """Apply a tensor product of single-qubit operators {qubit: 2x2 matrix}."""
    out = psi
    for q, P in paulis.items():
        out = apply_1q(out, P, q, n)
    return out


def expval(psi: np.ndarray, paulis: dict[int, np.ndarray], n: int) -> float:
    """<psi| (⊗ paulis) |psi>  -- real part (observables are Hermitian)."""
    phi = apply_pauli_string(psi, paulis, n)
    return float(np.real(np.vdot(psi, phi)))


def matrix_element(bra: np.ndarray, ket: np.ndarray,
                   paulis: dict[int, np.ndarray], n: int) -> complex:
    """<bra| (⊗ paulis) |ket>  -- general (complex) matrix element.

    Needed by the exact gate-cut reconstruction, which sums cross terms
    <A_nu| O_A | A_mu> between different Schmidt branches.
    """
    phi = apply_pauli_string(ket, paulis, n)
    return complex(np.vdot(bra, phi))


# --------------------------------------------------------------------------- #
# Reduced density matrices & entanglement (the "quantumness" readouts)
# --------------------------------------------------------------------------- #
def reduced_density_matrix(psi: np.ndarray, keep: list[int], n: int) -> np.ndarray:
    """Partial trace of |psi><psi| keeping the qubits in ``keep`` (ordered)."""
    keep = list(keep)
    trace_out = [q for q in range(n) if q not in keep]
    t = psi.reshape([2] * n)
    # Reorder axes so kept qubits come first, traced qubits last.
    perm = keep + trace_out
    t = np.transpose(t, perm)
    dk = 2 ** len(keep)
    dt = 2 ** len(trace_out)
    t = t.reshape(dk, dt)
    return t @ t.conj().T                              # rho on kept subsystem


def von_neumann_entropy(rho: np.ndarray, base: float = 2.0) -> float:
    """S(rho) = -Tr rho log rho, in the given base (default: bits)."""
    evals = np.linalg.eigvalsh(rho)
    evals = evals[evals > 1e-12]
    return float(-np.sum(evals * (np.log(evals) / np.log(base))))


def cut_entanglement_entropy(psi: np.ndarray, subsystem_A: list[int], n: int) -> float:
    """
    Entanglement entropy across the A:B bipartition of a pure state -- our
    canonical measure of 'cut-entanglement' (the quantumness that reconstruction
    restores and late fusion discards). For a pure state this equals S(rho_A).
    """
    rho_A = reduced_density_matrix(psi, subsystem_A, n)
    return von_neumann_entropy(rho_A)
