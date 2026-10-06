"""Sanity tests for the state-vector simulator."""
import numpy as np
from lfqml import qsim


def test_single_qubit_rotation():
    # RY(pi) on |0> -> |1>; <Z> should go +1 -> -1.
    psi = qsim.zero_state(1)
    assert abs(qsim.expval(psi, {0: qsim.Z}, 1) - 1.0) < 1e-9
    psi = qsim.apply_1q(psi, qsim.ry(np.pi), 0, 1)
    assert abs(qsim.expval(psi, {0: qsim.Z}, 1) + 1.0) < 1e-9


def test_bell_state_entanglement():
    # H on q0 then CX(0,1) makes a Bell pair: entropy across {0}|{1} = 1 bit.
    n = 2
    psi = qsim.zero_state(n)
    psi = qsim.apply_1q(psi, qsim.H, 0, n)
    psi = qsim.apply_2q(psi, qsim.CX, 0, 1, n)
    S = qsim.cut_entanglement_entropy(psi, [0], n)
    assert abs(S - 1.0) < 1e-9
    # Product state has zero entanglement.
    prod = qsim.apply_1q(qsim.zero_state(n), qsim.ry(0.7), 0, n)
    prod = qsim.apply_1q(prod, qsim.ry(1.1), 1, n)
    assert qsim.cut_entanglement_entropy(prod, [0], n) < 1e-9


def test_two_qubit_gate_matches_dense():
    # Compare apply_2q against a dense kron for a random 3-qubit case.
    rng = np.random.default_rng(0)
    n = 3
    psi = rng.normal(size=2 ** n) + 1j * rng.normal(size=2 ** n)
    psi /= np.linalg.norm(psi)
    out = qsim.apply_2q(psi, qsim.CZ, 0, 2, n)
    # Dense reference: CZ on qubits (0,2), identity on 1. Build by permutation.
    dense = np.zeros_like(psi)
    for idx in range(2 ** n):
        bits = [(idx >> (n - 1 - q)) & 1 for q in range(n)]
        sign = -1.0 if (bits[0] == 1 and bits[2] == 1) else 1.0
        dense[idx] = sign * psi[idx]
    assert np.allclose(out, dense, atol=1e-12)


if __name__ == "__main__":
    test_single_qubit_rotation()
    test_bell_state_entanglement()
    test_two_qubit_gate_matches_dense()
    print("qsim tests passed.")
