"""metrics.py -- task metrics and the 'quantumness' readout."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from . import qsim
from .circuits import PreparedCircuit, QNNConfig, full_state


def classification_metrics(y_true, y_prob) -> dict:
    """Accuracy / F1 / AUC from probabilities (threshold 0.5)."""
    y_pred = (np.asarray(y_prob) >= 0.5).astype(int)
    out = {"accuracy": accuracy_score(y_true, y_pred),
           "f1": f1_score(y_true, y_pred, zero_division=0)}
    try:
        out["auc"] = roc_auc_score(y_true, y_prob)
    except ValueError:
        out["auc"] = float("nan")   # single-class edge case
    return out


def mean_cut_entanglement(cfg: QNNConfig, prepared: list[PreparedCircuit]) -> float:
    """Average cut-entanglement entropy across a set of inputs (the H1/H3 quantity).

    This is the entanglement across A:B of the *trained* circuit, averaged over
    the data distribution -- our measurable proxy for 'how much quantumness does
    the model actually use across the cut'.
    """
    vals = []
    for pc in prepared:
        psi = full_state(pc)
        vals.append(qsim.cut_entanglement_entropy(psi, cfg.A_qubits, cfg.n))
    return float(np.mean(vals))
