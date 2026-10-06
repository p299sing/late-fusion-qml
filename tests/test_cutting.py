"""
The load-bearing correctness tests.

If exact reconstruction (Q=1) reproduces the uncut circuit's expectation values
to machine precision, the whole cut/Q-dial/fusion machinery is trustworthy.
"""
import numpy as np

from lfqml import qsim
from lfqml.circuits import (QNNConfig, init_params, prepare, default_observables,
                            full_features, coupling_unitary_4x4)
from lfqml import cutting


def test_operator_schmidt_reconstructs_gate():
    for phi in [0.0, 0.3, np.pi / 2, 1.9]:
        G = coupling_unitary_4x4(phi)
        coeffs, Ls, Rs = cutting.operator_schmidt(G)
        rebuilt = sum(c * np.kron(L, R) for c, L, R in zip(coeffs, Ls, Rs))
        assert np.allclose(rebuilt, G, atol=1e-12), phi
    # CZ too (Schmidt rank 2)
    coeffs, Ls, Rs = cutting.operator_schmidt(qsim.CZ)
    rebuilt = sum(c * np.kron(L, R) for c, L, R in zip(coeffs, Ls, Rs))
    assert np.allclose(rebuilt, qsim.CZ, atol=1e-12)


def test_exact_reconstruction_equals_uncut():
    """B1 (full reconstruction) == B0 (uncut) for many random circuits/inputs."""
    rng = np.random.default_rng(1)
    for (nA, nB, cuts, depth) in [(2, 2, 1, 1), (2, 2, 2, 1), (3, 2, 2, 2), (2, 3, 1, 2)]:
        cfg = QNNConfig(n_A=nA, n_B=nB, depth=depth, n_cuts=cuts)
        obs = default_observables(cfg)
        for trial in range(5):
            params = init_params(cfg, seed=trial) + 0.5 * rng.standard_normal(
                init_params(cfg).shape)
            x = rng.uniform(-np.pi, np.pi, size=cfg.n)
            pc = prepare(cfg, params, x)

            f_uncut = full_features(pc, obs)
            cd = cutting.build_cut(pc)
            f_recon = cutting.reconstruct_features(cd, obs, Q=1.0)

            err = np.max(np.abs(f_uncut - f_recon))
            assert err < 1e-10, f"cfg={cfg} trial={trial} err={err}"


def test_qdial_monotone_endpoints():
    """Q=1 exact; smaller Q generally moves features away from exact."""
    cfg = QNNConfig(n_A=2, n_B=2, depth=1, n_cuts=2)
    obs = default_observables(cfg)
    params = init_params(cfg, seed=3)
    x = np.linspace(-1, 1, cfg.n)
    pc = prepare(cfg, params, x)
    cd = cutting.build_cut(pc)

    f_exact = cutting.reconstruct_features(cd, obs, Q=1.0)
    f_full = full_features(pc, obs)
    assert np.max(np.abs(f_exact - f_full)) < 1e-10

    # Overhead interpolates: Q<1 cheaper than Q=1.
    assert cutting.sampling_overhead(cd, Q=0.3) < cutting.sampling_overhead(cd, Q=1.0)


if __name__ == "__main__":
    test_operator_schmidt_reconstructs_gate()
    test_exact_reconstruction_equals_uncut()
    test_qdial_monotone_endpoints()
    print("cutting tests passed: exact reconstruction == uncut to <1e-10.")
