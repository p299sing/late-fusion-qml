"""Tests for lfqml.deep_circuits (deep coupled QNN, revision item E1)."""
import numpy as np

from lfqml import qsim, deep_circuits as dc
from lfqml.circuits import QNNConfig, init_params as qnn_init_params, prepare, full_state, full_features
from lfqml.cutting import subcircuit_raw_features


def _rand_x(rng, n):
    return rng.uniform(-np.pi / 2, np.pi / 2, size=n)


def test_zero_coupling_is_product_state():
    # All RZZ angles = 0 -> A and B never interact -> S_A = 0 for any L, k, input.
    rng = np.random.default_rng(1)
    for L in [1, 2, 3, 4]:
        for k in [1, 2]:
            cfg = dc.DeepConfig(n_A=2, n_B=2, L=L, n_cuts=k, coupled=True)
            p = dc.init_params(cfg, seed=L * 10 + k)
            p[cfg.n_local_params:] = 0.0                      # kill every coupling angle
            for _ in range(3):
                x = _rand_x(rng, cfg.n)
                psi = dc.state_single(cfg, p, x)
                assert qsim.cut_entanglement_entropy(psi, cfg.A_qubits, cfg.n) < 1e-9
            # and the uncoupled config gives literally the same state
            ucfg = dc.DeepConfig(n_A=2, n_B=2, L=L, n_cuts=k, coupled=False)
            x = _rand_x(rng, cfg.n)
            assert np.allclose(dc.state_single(cfg, p, x),
                               dc.state_single(ucfg, p[:ucfg.n_params], x), atol=1e-12)


def test_nonzero_coupling_entangles():
    # NB: at the default init (near-identity local layers, every phi = pi/2) an EVEN number of
    # coupling layers composes to RZZ(L*pi/2), which for L=2 is -i Z(x)Z -- a LOCAL unitary --
    # so the initial state is almost a product state.  Use random local angles instead.
    rng = np.random.default_rng(3)
    for L in [1, 2, 3, 4]:
        cfg = dc.DeepConfig(n_A=2, n_B=2, L=L, n_cuts=1)
        p = 0.7 * rng.standard_normal(cfg.n_params)
        p[cfg.n_local_params:] = np.pi / 2                     # maximally entangling couplings
        psi = dc.state_single(cfg, p, np.array([0.3, -0.7, 1.1, 0.2]))
        assert qsim.cut_entanglement_entropy(psi, cfg.A_qubits, cfg.n) > 0.05


def test_batch_matches_single_reference():
    # The vectorised path (used by the trainers) must equal the qsim-primitive path.
    rng = np.random.default_rng(7)
    for L in [1, 3]:
        for coupled in [True, False]:
            cfg = dc.DeepConfig(n_A=2, n_B=2, L=L, n_cuts=2, coupled=coupled)
            p = 0.7 * rng.standard_normal(cfg.n_params)
            X = np.stack([_rand_x(rng, cfg.n) for _ in range(5)])
            psis = dc.states_batch(cfg, p, X)
            for i, x in enumerate(X):
                assert np.allclose(psis[i], dc.state_single(cfg, p, x), atol=1e-12)
            # batched expectation values vs qsim.expval
            obs = dc.observables(cfg)
            F = dc.feature_matrix(cfg, p, X, obs)
            qcfg = cfg.as_qnn_config()
            for i, x in enumerate(X):
                ref = [qsim.expval(psis[i], o.global_dict(qcfg), cfg.n) for o in obs]
                assert np.allclose(F[i], ref, atol=1e-12)


def test_L1_matches_single_layer_model():
    # Layouts coincide at L=1: [pre_A|pre_B|post_A|post_B|phi] == [l0_A|l0_B|l1_A|l1_B|phi],
    # so the same parameter vector + same seed must give the SAME full state, the same
    # B0 readout features, and the same initial parameters as circuits.QNNConfig(depth=1).
    rng = np.random.default_rng(11)
    for k in [1, 2]:
        cfg = dc.DeepConfig(n_A=2, n_B=2, L=1, n_cuts=k, coupled=True)
        qcfg = QNNConfig(n_A=2, n_B=2, depth=1, n_cuts=k, trainable_coupling=True)
        assert cfg.n_params == len(qnn_init_params(qcfg, seed=0))
        assert np.allclose(dc.init_params(cfg, seed=5), qnn_init_params(qcfg, seed=5))
        p = 0.9 * rng.standard_normal(cfg.n_params)
        for _ in range(4):
            x = _rand_x(rng, 4)
            pc = prepare(qcfg, p, x)
            assert np.allclose(dc.state_single(cfg, p, x), full_state(pc), atol=1e-12)
            assert np.allclose(dc.feature_matrix(cfg, p, x[None], dc.observables(cfg))[0],
                               full_features(pc, dc.observables(cfg)), atol=1e-12)
        # uncoupled local-Z features == cutting.subcircuit_raw_features (coupling removed)
        ucfg = dc.DeepConfig(n_A=2, n_B=2, L=1, n_cuts=k, coupled=False)
        uq = QNNConfig(n_A=2, n_B=2, depth=1, n_cuts=k, trainable_coupling=False, coupling_init=0.0)
        pu = p[:ucfg.n_params]
        x = _rand_x(rng, 4)
        assert np.allclose(dc.local_z_features(ucfg, pu, x[None])[0],
                           subcircuit_raw_features(prepare(uq, pu, x), paulis=(qsim.Z,)), atol=1e-12)


if __name__ == "__main__":
    test_zero_coupling_is_product_state()
    test_nonzero_coupling_entangles()
    test_batch_matches_single_reference()
    test_L1_matches_single_layer_model()
    print("all deep_circuits tests passed")
