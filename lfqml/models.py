"""
models.py -- derive every readout baseline from ONE trained circuit.

Baselines (mirrors the plan's comparison ladder, §5.1):
  B0  uncut full circuit          -- trained head on exact features (gold standard)
  B1  full QPD reconstruction     -- exact reconstruction == B0 (sanity: equal)
  B2  late fusion                 -- trained head on RAW per-subcircuit features
  B6  Kawase-style fixed sum      -- parameter-free ensemble sum of expectation values
  B7  entanglement ablation       -- coupling phi := 0 (product circuit), re-read
  Qdial(Q)                        -- [Q-truncated reconstruction ⊕ raw fusion] + head

Everything below the training of B0 is cheap, so we sweep the full Q-dial and all
baselines from the single trained circuit.  This isolates 'readout method' as the
only variable (H1/H3).
"""

from __future__ import annotations

from dataclasses import replace
import numpy as np

from . import cutting, qsim
from .circuits import QNNConfig, prepare, full_features, coupling_angle
from .fusion import LogisticHead, MLPHead
from .metrics import classification_metrics, mean_cut_entanglement
from .train import TrainedQNN


# --------------------------------------------------------------------------- #
# Featurisers
# --------------------------------------------------------------------------- #
def _prepare_all(cfg, theta, X):
    return [prepare(cfg, theta, x) for x in X]


def _exact_features(tq: TrainedQNN, prepared):
    return np.array([full_features(pc, tq.observables) for pc in prepared])


def _recon_features(tq: TrainedQNN, prepared, Q):
    out = []
    for pc in prepared:
        cd = cutting.build_cut(pc)
        out.append(cutting.reconstruct_features(cd, tq.observables, Q=Q))
    return np.array(out)


def _subcircuit_features(prepared):
    return np.array([cutting.subcircuit_raw_features(pc) for pc in prepared])


# --------------------------------------------------------------------------- #
# The evaluation driver
# --------------------------------------------------------------------------- #
def evaluate_baselines(tq: TrainedQNN, X_tr, y_tr, X_te, y_te,
                       Q_grid=(0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0)) -> dict:
    """Return a dict of {baseline: metrics+cost} plus the cut-entanglement."""
    cfg = tq.cfg
    pre_tr = _prepare_all(cfg, tq.params, X_tr)
    pre_te = _prepare_all(cfg, tq.params, X_te)

    results = {}

    # ---- B0 (uncut) and B1 (exact reconstruction) --------------------------
    F0_te = _exact_features(tq, pre_te)
    p0 = qsim_sigmoid(F0_te @ tq.head_w + tq.head_b)
    results["B0_uncut"] = {**classification_metrics(y_te, p0),
                           "cost_overhead": 1.0, "n_subexp": 1}

    # exact reconstruction on test == B0; confirm numerically + report its cost
    Frec_te = _recon_features(tq, pre_te, Q=1.0)
    p1 = qsim_sigmoid(Frec_te @ tq.head_w + tq.head_b)
    cd0 = cutting.build_cut(pre_te[0])
    results["B1_reconstruction"] = {
        **classification_metrics(y_te, p1),
        "max_abs_vs_B0": float(np.max(np.abs(F0_te - Frec_te))),
        "cost_overhead": cutting.sampling_overhead(cd0, Q=1.0),
        "n_subexp": cutting.n_subexperiments(cd0, Q=1.0)}

    # ---- B2 (late fusion): trained head on raw subcircuit features ----------
    # Two fusion heads matter for a FAIR test of "how much is quantum":
    #   * linear head  -> also loses classical nonlinear (product-of-marginals)
    #                     cross-correlations, so its gap over-attributes to 'quantum'.
    #   * MLP head      -> can form products of marginals, so a residual gap over
    #                     reconstruction is genuinely non-factorisable (quantum).
    # H1 is tested against the MLP variant; the linear variant is reported to show
    # the classical-vs-quantum decomposition of the gap.
    S_tr, S_te = _subcircuit_features(pre_tr), _subcircuit_features(pre_te)
    head_lin = LogisticHead().fit(S_tr, y_tr)
    results["B2_fusion_linear"] = {
        **classification_metrics(y_te, head_lin.predict_proba(S_te)),
        "cost_overhead": float(cfg.n_A + cfg.n_B), "n_subexp": cfg.n_A + cfg.n_B}
    head_mlp = MLPHead(hidden=8, seed=0).fit(S_tr, y_tr)
    results["B2_fusion_mlp"] = {
        **classification_metrics(y_te, head_mlp.predict_proba(S_te)),
        "cost_overhead": float(cfg.n_A + cfg.n_B), "n_subexp": cfg.n_A + cfg.n_B}

    # ---- B6 (Kawase-style parameter-free expectation-value sum) -------------
    score_tr = S_tr.sum(axis=1, keepdims=True)       # fixed unweighted sum
    score_te = S_te.sum(axis=1, keepdims=True)
    head6 = LogisticHead().fit(score_tr, y_tr)        # only a global scale+bias
    results["B6_kawase_sum"] = {
        **classification_metrics(y_te, head6.predict_proba(score_te)),
        "cost_overhead": float(cfg.n_A + cfg.n_B), "n_subexp": cfg.n_A + cfg.n_B}

    # ---- B7 (entanglement ablation): coupling angle -> 0 --------------------
    cfg_abl = replace(cfg, trainable_coupling=False, coupling_init=0.0)
    pre_te_abl = _prepare_all(cfg_abl, tq.params, X_te)   # phi forced to 0
    F7 = _exact_features_cfg(tq, cfg_abl, pre_te_abl)
    p7 = qsim_sigmoid(F7 @ tq.head_w + tq.head_b)
    results["B7_ent_ablation"] = {**classification_metrics(y_te, p7),
                                  "cost_overhead": 1.0, "n_subexp": 1}

    # ---- Q-dial sweep -------------------------------------------------------
    # Head is an MLP so the Q=0 endpoint coincides with the fair (nonlinear)
    # fusion baseline, and Q=1 recovers exact reconstruction (== B0).
    qcurve = []
    for Q in Q_grid:
        Fq_tr = np.hstack([_recon_features(tq, pre_tr, Q), S_tr])
        Fq_te = np.hstack([_recon_features(tq, pre_te, Q), S_te])
        hq = MLPHead(hidden=8, seed=0).fit(Fq_tr, y_tr)
        m = classification_metrics(y_te, hq.predict_proba(Fq_te))
        cd = cutting.build_cut(pre_te[0])
        qcurve.append({"Q": Q, **m,
                       "cost_overhead": cutting.sampling_overhead(cd, Q=Q),
                       "n_subexp": cutting.n_subexperiments(cd, Q=Q)})
    results["Qdial"] = qcurve

    # ---- the 'quantumness' readout -----------------------------------------
    results["cut_entanglement"] = mean_cut_entanglement(cfg, pre_te)
    results["gamma_per_cut"] = cutting.build_cut(pre_te[0]).gamma_per_cut
    results["coupling_angle"] = coupling_angle(cfg, tq.params)
    return results


# --------------------------------------------------------------------------- #
# small helpers
# --------------------------------------------------------------------------- #
def qsim_sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def _exact_features_cfg(tq, cfg, prepared):
    """Exact features using a possibly-different cfg (for the ablation)."""
    from .circuits import default_observables
    obs = tq.observables   # same observable set/order as training
    return np.array([full_features(pc, obs) for pc in prepared])
