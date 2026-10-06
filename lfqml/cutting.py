"""
cutting.py -- exact gate-cut reconstruction, the Q-dial, and subcircuit features.

The physics in one paragraph
----------------------------
Each cross gate G is decomposed by its operator-Schmidt decomposition into a sum
of local factors,  G = sum_mu g_mu (L_mu ⊗ R_mu),  with L_mu acting on the
A-boundary qubit and R_mu on the B-boundary qubit.  Because the pre-coupling
state is a product |psi_A>⊗|psi_B> and the post-coupling gates + readout are
local (see circuits.py), the finished full state is an exact sum over branches

    |Psi> = sum_{mu-vector} c_mu  |A_mu> ⊗ |B_mu>,      c_mu = prod_j g_{mu_j}

where |A_mu> = finish_A( applied L's on psi_A ) and |B_mu> likewise.  Any product
observable O_A⊗O_B then reconstructs EXACTLY as

    <O> = sum_{mu,nu} conj(c_nu) c_mu <A_nu|O_A|A_mu> <B_nu|O_B|B_mu>.

* Full sum  -> exact expectation value  == uncut circuit (baseline B1 == B0).
* Truncated sum keeping a fraction Q of the coefficient 1-norm mass -> the Q-dial
  (partial reconstruction).  Q=1 exact; smaller Q is cheaper and less "quantum".
* Dropping the cross (mu != nu) structure and instead giving a classifier the raw
  per-subcircuit measurements -> late fusion (baseline B2).

Because reconstruction is exact by construction, tests/test_cutting.py verifies
<O>_reconstructed == <O>_uncut to ~1e-12, which validates the whole edifice.

Cost model
----------
For the paper's *cost* axis we report the standard analytic sampling overhead of
gate cutting (gamma factor), not the simulator's own term count.  See
`sampling_overhead` below and CODE_FLOW.md ("Cost accounting").
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
import numpy as np

from . import qsim
from .circuits import PreparedCircuit, Observable, coupling_unitary_4x4


# --------------------------------------------------------------------------- #
# Operator-Schmidt decomposition of a two-qubit gate
# --------------------------------------------------------------------------- #
def operator_schmidt(G4: np.ndarray, tol: float = 1e-12):
    """Decompose a 4x4 gate as G = sum_mu s_mu (L_mu ⊗ R_mu).

    Returns (coeffs, Ls, Rs) with L_mu, R_mu 2x2 complex matrices.
    Verified against the original gate in tests/test_cutting.py.
    """
    G = G4.reshape(2, 2, 2, 2)                    # [a_out, b_out, a_in, b_in]
    M = np.transpose(G, (0, 2, 1, 3)).reshape(4, 4)   # [(a_out,a_in),(b_out,b_in)]
    U, S, Vh = np.linalg.svd(M)
    coeffs, Ls, Rs = [], [], []
    for mu in range(len(S)):
        if S[mu] < tol:
            continue
        coeffs.append(S[mu])
        Ls.append(U[:, mu].reshape(2, 2))         # a_out x a_in
        Rs.append(Vh[mu, :].reshape(2, 2))        # b_out x b_in
    return np.array(coeffs, dtype=complex), Ls, Rs


# --------------------------------------------------------------------------- #
# Branch enumeration: build all |A_mu> and |B_mu> once per input
# --------------------------------------------------------------------------- #
@dataclass
class CutDecomposition:
    """Precomputed branch states + coefficients for one PreparedCircuit."""
    A_branches: list          # list of A-substates |A_mu>
    B_branches: list          # list of B-substates |B_mu>
    coeffs: np.ndarray        # c_mu = prod_j g_{mu_j}, aligned with the lists
    n_A: int
    n_B: int
    gamma_per_cut: float      # 1-norm of one cut's Schmidt coeffs (for cost model)
    n_cuts: int


def build_cut(pc: PreparedCircuit) -> CutDecomposition:
    """Enumerate the Schmidt branches of the coupling layer for input `pc`."""
    cfg = pc.cfg
    G = coupling_unitary_4x4(pc.phi)
    coeffs1, Ls, Rs = operator_schmidt(G)
    chi = len(coeffs1)
    gamma_per_cut = float(np.sum(np.abs(coeffs1)))
    local_pairs = pc.local_cross_pairs()          # [(a_local, b_local), ...]

    A_branches, B_branches, cvals = [], [], []
    # multi-index mu-vector over all cuts
    for muvec in product(range(chi), repeat=cfg.n_cuts):
        c = 1.0 + 0j
        psiA = pc.psiA_pre.copy()
        psiB = pc.psiB_pre.copy()
        for j, mu in enumerate(muvec):
            a_local, b_local = local_pairs[j]
            c *= coeffs1[mu]
            psiA = qsim.apply_1q(psiA, Ls[mu], a_local, cfg.n_A)
            psiB = qsim.apply_1q(psiB, Rs[mu], b_local, cfg.n_B)
        A_branches.append(pc.finish_A(psiA))
        B_branches.append(pc.finish_B(psiB))
        cvals.append(c)

    return CutDecomposition(
        A_branches=A_branches, B_branches=B_branches,
        coeffs=np.array(cvals, dtype=complex),
        n_A=cfg.n_A, n_B=cfg.n_B,
        gamma_per_cut=gamma_per_cut, n_cuts=cfg.n_cuts,
    )


# --------------------------------------------------------------------------- #
# Reconstruction of one observable (full or Q-truncated)
# --------------------------------------------------------------------------- #
def _term_weights(cd: CutDecomposition):
    """|c_nu * c_mu| for every (nu, mu) pair -> the 1-norm mass of each term."""
    c = cd.coeffs
    return np.abs(np.outer(np.conj(c), c))        # W[nu, mu]


def reconstruct_observable(cd: CutDecomposition, obs: Observable,
                           Q: float = 1.0) -> float:
    """Reconstruct <O_A ⊗ O_B>, keeping the top fraction Q of coefficient mass.

    Q = 1.0 -> exact (== uncut).  0 <= Q < 1 -> partial reconstruction (Q-dial).
    """
    nb = len(cd.coeffs)
    # A-side and B-side matrix elements <branch_nu| O |branch_mu>.
    MA = np.zeros((nb, nb), dtype=complex)
    MB = np.zeros((nb, nb), dtype=complex)
    for nu in range(nb):
        for mu in range(nb):
            MA[nu, mu] = qsim.matrix_element(cd.A_branches[nu], cd.A_branches[mu],
                                             obs.A_part, cd.n_A)
            MB[nu, mu] = qsim.matrix_element(cd.B_branches[nu], cd.B_branches[mu],
                                             obs.B_part, cd.n_B)
    c = cd.coeffs
    full_term = np.outer(np.conj(c), c) * MA * MB   # W[nu,mu] contribution matrix

    if Q >= 1.0:
        return float(np.real(np.sum(full_term)))

    # Q-dial: keep the largest-mass (nu,mu) terms until cumulative mass >= Q*total.
    mass = _term_weights(cd)
    total = mass.sum()
    order = np.argsort(mass.ravel())[::-1]          # descending
    keep = np.zeros(nb * nb, dtype=bool)
    acc = 0.0
    for idx in order:
        if acc >= Q * total:      # threshold met BEFORE adding: Q=0 keeps no terms
            break
        keep[idx] = True
        acc += mass.ravel()[idx]
    keep = keep.reshape(nb, nb)
    return float(np.real(np.sum(full_term[keep])))


def reconstruct_features(cd: CutDecomposition, observables: list[Observable],
                         Q: float = 1.0) -> np.ndarray:
    """Reconstruct the full readout feature vector at Q-dial level Q."""
    return np.array([reconstruct_observable(cd, o, Q) for o in observables])


# --------------------------------------------------------------------------- #
# Raw per-subcircuit features (the inputs available to LATE FUSION, B2)
# --------------------------------------------------------------------------- #
def subcircuit_raw_features(pc: PreparedCircuit,
                            paulis=(qsim.Z,)) -> np.ndarray:
    """Measurements from each subcircuit run INDEPENDENTLY (coupling removed).

    For each requested single-qubit Pauli, measure it on every qubit of the
    finished A-subcircuit and B-subcircuit.  These are the only quantities late
    fusion may use -- it never sees a genuine cross-cut correlator.
    """
    psiA = pc.finish_A(pc.psiA_pre.copy())
    psiB = pc.finish_B(pc.psiB_pre.copy())
    feats = []
    for P in paulis:
        for q in range(pc.cfg.n_A):
            feats.append(qsim.expval(psiA, {q: P}, pc.cfg.n_A))
        for q in range(pc.cfg.n_B):
            feats.append(qsim.expval(psiB, {q: P}, pc.cfg.n_B))
    return np.array(feats)


# --------------------------------------------------------------------------- #
# Cost model (analytic, matches the circuit-cutting literature)
# --------------------------------------------------------------------------- #
def sampling_overhead(cd: CutDecomposition, Q: float = 1.0) -> float:
    """Sampling overhead of the retained reconstruction terms, floored at 1.

    At Q=1 this is the full joint 1-norm mass sum_{nu,mu}|c_nu||c_mu| =
    gamma_S^(2k) (gamma_S = per-cut operator-Schmidt 1-norm; an exact-simulation
    convention DISTINCT from the physical QPD overhead, which for RZZ(phi) cuts
    is (1+2|sin phi|)^2 per cut = 9 at phi=pi/2 -- see qiskit_cut.py /
    exp_qiskit for the physical validation; neither bounds the other at general
    trained angles). The Q-dial keeps
    the largest-|c_nu c_mu| terms until their cumulative mass reaches Q of the
    total, so the retained overhead is the ACTUAL kept mass (~ Q * gamma_S^(2k),
    linear in Q), floored at 1 because fusion always pays its own O(1)
    measurements. NOTE: an earlier version returned (Q*gamma)^(2k), which
    understates intermediate-Q cost; results/*.json produced before this fix
    are migrated by scripts/migrate_qdial_cost.py.
    """
    g = cd.gamma_per_cut
    k = cd.n_cuts
    total = float(g ** (2 * k))                    # full joint 1-norm mass
    if Q >= 1.0:
        return max(1.0, total)
    mass = _term_weights(cd)
    tot = mass.sum()
    order = np.argsort(mass.ravel())[::-1]
    acc = 0.0
    for idx in order:
        if acc >= Q * tot:        # Q=0 keeps no terms -> overhead floors at 1
            break
        acc += mass.ravel()[idx]
    return max(1.0, float(acc))


def n_subexperiments(cd: CutDecomposition, Q: float = 1.0) -> int:
    """Number of (nu, mu) reconstruction terms actually evaluated at level Q."""
    nb = len(cd.coeffs)
    if Q >= 1.0:
        return nb * nb
    mass = _term_weights(cd)
    total = mass.sum()
    order = np.sort(mass.ravel())[::-1]
    acc, cnt = 0.0, 0
    for m in order:
        if acc >= Q * total:      # Q=0 evaluates no reconstruction terms
            break
        acc += m; cnt += 1
    return cnt
