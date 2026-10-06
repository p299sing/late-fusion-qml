"""
Correctness tests for lfqml.qdial_physical (the Q-dial in the physical QPD basis).

  1. lfqml_to_qiskit reproduces the lfqml statevector exactly (<=1e-8) for random
     params/inputs, several (n_A, n_B, depth, n_cuts) configurations.
  2. The exact per-term values T_i sum to the uncut expectation (<=1e-8) and the
     coefficient 1-norm equals kappa^k = (1+2|sin phi|)^k, overhead = kappa^(2k).
  3. The per-term decomposition T_i is EXACTLY what the library computes when
     reconstruct_expectation_values is called with all other coefficients zeroed
     (same sampler results), and sum_i T_i equals the library's full reconstruction.
  4. Finite-shot reconstruction agrees with exact to shot noise.
  5. The batched PhysicalCutter (parameter binding) equals per-circuit cutting.
  6. Q-dial bookkeeping: Q=0 keeps none, Q=1 keeps all, monotone mass.

Run:  .venv/bin/python -m tests.test_qdial_physical
"""
import numpy as np
from qiskit.quantum_info import Statevector

from lfqml import qsim, cutting
from lfqml.circuits import (QNNConfig, init_params, prepare, default_observables,
                            full_features, full_state, coupling_angle)
from lfqml import qdial_physical as qp


CONFIGS = [(2, 2, 1, 1), (2, 2, 2, 2), (3, 2, 2, 2), (2, 3, 1, 1), (3, 3, 1, 2)]


def _random_model(cfg, rng, trial):
    params = init_params(cfg, seed=trial) + 0.7 * rng.standard_normal(init_params(cfg).shape)
    x = rng.uniform(-np.pi / 2, np.pi / 2, size=cfg.n)
    return params, x


def test_statevector_matches_lfqml():
    rng = np.random.default_rng(7)
    worst = 0.0
    for (nA, nB, depth, cuts) in CONFIGS:
        cfg = QNNConfig(n_A=nA, n_B=nB, depth=depth, n_cuts=cuts, trainable_coupling=True)
        for trial in range(4):
            params, x = _random_model(cfg, rng, trial)
            psi = full_state(prepare(cfg, params, x))
            qc = qp.lfqml_to_qiskit(cfg, params, x)
            sv = Statevector(qc).reverse_qargs().data          # little -> big endian
            # up to global phase
            k = np.argmax(np.abs(psi))
            phase = psi[k] / sv[k]
            err = float(np.max(np.abs(psi - phase * sv)))
            worst = max(worst, err)
            assert err < 1e-8, (nA, nB, depth, cuts, trial, err)
            # and in fact exactly equal (same gate conventions): |phase - 1| ~ 0
            assert abs(phase - 1) < 1e-8
            # uncoupled circuit == finished subcircuits (late-fusion raw features)
            pc = prepare(cfg, params, x)
            loc = np.kron(pc.finish_A(pc.psiA_pre.copy()), pc.finish_B(pc.psiB_pre.copy()))
            sv2 = Statevector(qp.lfqml_to_qiskit(cfg, params, x, include_coupling=False)).reverse_qargs().data
            assert np.max(np.abs(loc - sv2)) < 1e-8
    print(f"  [1] statevector mismatch (max over {len(CONFIGS)}x4 circuits): {worst:.2e}")
    return worst


def test_exact_terms_sum_to_uncut():
    rng = np.random.default_rng(11)
    worst = 0.0
    for (nA, nB, depth, cuts) in CONFIGS:
        cfg = QNNConfig(n_A=nA, n_B=nB, depth=depth, n_cuts=cuts, trainable_coupling=True)
        obs = default_observables(cfg)
        labels = [qp.observable_to_pauli(o, cfg) for o in obs]
        for trial in range(3):
            params, x = _random_model(cfg, rng, trial)
            pc = prepare(cfg, params, x)
            f_uncut = full_features(pc, obs)
            qc = qp.lfqml_to_qiskit(cfg, params, x)
            assert np.max(np.abs(qp.exact_expectations(qc, labels) - f_uncut)) < 1e-8
            terms = qp.physical_terms(qc, qp.partition_labels(cfg), labels, estimator="exact")
            assert terms.T.shape == (6 ** cuts, len(obs))
            err = float(np.max(np.abs(terms.reconstruct() - f_uncut)))
            worst = max(worst, err)
            assert err < 1e-8, (nA, nB, depth, cuts, trial, err)
            phi = coupling_angle(cfg, params)
            kappa = (1 + 2 * abs(np.sin(phi))) ** cuts
            assert abs(terms.total_mass - kappa) < 1e-9
            assert abs(terms.full_overhead - kappa ** 2) < 1e-8
    print(f"  [2] max |sum_i T_i - exact| (exact estimator): {worst:.2e}")
    return worst


def test_terms_match_library_definition():
    """T_i == library reconstruction with all other coefficients zeroed (same results)."""
    cfg = QNNConfig(n_A=2, n_B=2, depth=2, n_cuts=2, trainable_coupling=True)
    rng = np.random.default_rng(3)
    params, x = _random_model(cfg, rng, 0)
    obs = default_observables(cfg)
    labels = [qp.observable_to_pauli(o, cfg) for o in obs]
    qc = qp.lfqml_to_qiskit(cfg, params, x)
    full, Tlib, Tmine = qp.library_reconstruction(qc, qp.partition_labels(cfg), labels,
                                                  shots=2000, seed=5, per_term=True)
    d_term = float(np.max(np.abs(Tlib - Tmine)))
    d_sum = float(np.max(np.abs(Tlib.sum(axis=0) - full)))
    assert d_term < 1e-10, d_term
    assert d_sum < 1e-10, d_sum
    print(f"  [3] per-term definition vs library (zeroed coefficients): {d_term:.2e}; "
          f"sum_i T_i vs library full reconstruction: {d_sum:.2e}")


def test_finite_shots_agree_to_shot_noise():
    cfg = QNNConfig(n_A=2, n_B=2, depth=2, n_cuts=2, trainable_coupling=True)
    rng = np.random.default_rng(5)
    obs = default_observables(cfg)
    labels = [qp.observable_to_pauli(o, cfg) for o in obs]
    errs, kappas = [], []
    for trial in range(3):
        params, x = _random_model(cfg, rng, trial)
        qc = qp.lfqml_to_qiskit(cfg, params, x)
        f_uncut = full_features(prepare(cfg, params, x), obs)
        t = qp.physical_terms(qc, qp.partition_labels(cfg), labels, estimator="shots",
                              shots=4000, seed=trial)
        errs.append(float(np.max(np.abs(t.reconstruct() - f_uncut))))
        kappas.append(t.total_mass)
    # std of the reconstruction ~ kappa^k / sqrt(shots) per term-sum; 4000 shots -> ~0.1 at kappa^k=9
    bound = 5 * max(kappas) / np.sqrt(4000)
    assert max(errs) < bound, (errs, bound)
    print(f"  [4] finite-shot (4000) max |recon - exact| = {max(errs):.3f} "
          f"(kappa^k up to {max(kappas):.2f}; 5-sigma bound {bound:.3f})")


def test_batched_cutter_equals_per_circuit():
    cfg = QNNConfig(n_A=2, n_B=2, depth=2, n_cuts=2, trainable_coupling=True)
    rng = np.random.default_rng(9)
    params, _ = _random_model(cfg, rng, 1)
    obs = default_observables(cfg)
    cutter = qp.PhysicalCutter(cfg, params, obs)
    worst = 0.0
    for trial in range(3):
        x = rng.uniform(-np.pi / 2, np.pi / 2, size=cfg.n)
        t_batched = cutter.terms(x, estimator="exact")
        qc = qp.lfqml_to_qiskit(cfg, params, x)
        t_direct = qp.physical_terms(qc, qp.partition_labels(cfg), cutter.obs_labels, estimator="exact")
        assert np.allclose(t_batched.coeffs, t_direct.coeffs, atol=1e-12)
        worst = max(worst, float(np.max(np.abs(t_batched.T - t_direct.T))))
        # raw local features (shots) agree with lfqml exact to shot noise
    assert worst < 1e-8, worst
    X = rng.uniform(-np.pi / 2, np.pi / 2, size=(5, cfg.n))
    S_ex = np.array([cutting.subcircuit_raw_features(prepare(cfg, params, x)) for x in X])
    S_sh = cutter.raw_local_features_shots(X, shots=20000, seed=1)
    assert np.max(np.abs(S_ex - S_sh)) < 5 / np.sqrt(20000) * 1.0 + 0.0, np.max(np.abs(S_ex - S_sh))
    print(f"  [5] batched (parametric) vs per-circuit cutting: {worst:.2e}; "
          f"raw local shots vs exact: {np.max(np.abs(S_ex - S_sh)):.3f}")


def test_qdial_bookkeeping():
    cfg = QNNConfig(n_A=2, n_B=2, depth=2, n_cuts=2, trainable_coupling=True)
    rng = np.random.default_rng(2)
    params, x = _random_model(cfg, rng, 0)
    obs = default_observables(cfg)
    labels = [qp.observable_to_pauli(o, cfg) for o in obs]
    qc = qp.lfqml_to_qiskit(cfg, params, x)
    t = qp.physical_terms(qc, qp.partition_labels(cfg), labels, estimator="exact")
    f0 = qp.qdial_physical_features(t, 0.0)
    assert f0["n_kept"] == 0 and np.all(f0["features"] == 0) and f0["retained_overhead_sq"] == 1.0
    f1 = qp.qdial_physical_features(t, 1.0)
    assert f1["n_kept"] == 6 ** cfg.n_cuts
    assert np.max(np.abs(f1["features"] - full_features(prepare(cfg, params, x), obs))) < 1e-8
    assert abs(f1["retained_overhead_sq"] - t.full_overhead) < 1e-9
    prev = -1
    for Q in [0.1, 0.3, 0.5, 0.7, 0.9]:
        f = qp.qdial_physical_features(t, Q)
        assert f["retained_mass_ratio"] >= Q - 1e-12
        assert f["n_kept"] >= prev; prev = f["n_kept"]
        assert abs(f["retained_overhead_sq"] - max(1.0, f["retained_mass"] ** 2)) < 1e-12
    print("  [6] Q-dial bookkeeping OK (Q=0 none, Q=1 all, monotone, overhead = mass^2)")


if __name__ == "__main__":
    test_statevector_matches_lfqml()
    test_exact_terms_sum_to_uncut()
    test_terms_match_library_definition()
    test_finite_shots_agree_to_shot_noise()
    test_batched_cutter_equals_per_circuit()
    test_qdial_bookkeeping()
    print("ALL qdial_physical TESTS PASSED")
