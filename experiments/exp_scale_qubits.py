"""
exp_scale_qubits.py -- width scale check: does late fusion still track reconstruction
as the QUBIT COUNT grows (not just the cut count)?

For width n = 4, 6, 8 qubits (n_A = n_B = 2, 3, 4), single cut, on a moderately-local
synthetic task, we compare independent late fusion vs reconstruction (B0) accuracy and
report the cut entanglement. Complements exp_scaling.py (which scales the number of cuts)
by scaling the register width -- addressing the "toy 4-qubit" critique.

Outputs: results/scale_qubits_results.json + results/scale_qubits.png
Run:  python -m experiments.exp_scale_qubits
"""
import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from lfqml import qsim, datasets
from lfqml.circuits import QNNConfig, prepare, full_features, full_state
from lfqml.train import train_qnn, train_fusion_independent
from lfqml.models import qsim_sigmoid
from lfqml.metrics import classification_metrics

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS, exist_ok=True)

SEEDS = [0, 1, 2, 3, 4]
ALPHA = 0.3  # moderately local: signal fusion should mostly recover


def run():
    rows = []
    for half in [2, 3, 4]:                     # n = 4, 6, 8 qubits
        cfg = QNNConfig(n_A=half, n_B=half, depth=2, n_cuts=1, trainable_coupling=True)
        n = 2 * half
        recon_a, fusion_a, ents = [], [], []
        for seed in SEEDS:
            X, y = datasets.synthetic_partition(ALPHA, half, half, 200, seed=seed)
            Xtr, ytr, Xte, yte = X[:140], y[:140], X[140:], y[140:]
            tq = train_qnn(cfg, Xtr, ytr, seed=seed, maxiter=45)
            F0 = np.array([full_features(prepare(cfg, tq.params, x), tq.observables) for x in Xte])
            recon_a.append(classification_metrics(
                yte, qsim_sigmoid(F0 @ tq.head_w + tq.head_b))["accuracy"])
            # cut entanglement of the trained circuit (mean over a subsample)
            ents.append(float(np.mean([
                qsim.cut_entanglement_entropy(full_state(prepare(cfg, tq.params, X[i])),
                                              cfg.A_qubits, cfg.n)
                for i in range(0, 60, 4)])))
            tf = train_fusion_independent(cfg, Xtr, ytr, seed=seed,
                                          hidden=8 + 2 * half, maxiter=50)
            fusion_a.append(classification_metrics(yte, tf.predict_proba(Xte))["accuracy"])
        rows.append({"qubits": n, "acc_recon": float(np.mean(recon_a)),
                     "acc_recon_std": float(np.std(recon_a)),
                     "acc_fusion": float(np.mean(fusion_a)),
                     "acc_fusion_std": float(np.std(fusion_a)),
                     "gap": float(np.mean(recon_a) - np.mean(fusion_a)),
                     "cut_entropy": float(np.mean(ents))})
        print(f"n={n}: recon={np.mean(recon_a):.3f} fusion={np.mean(fusion_a):.3f} "
              f"gap={np.mean(recon_a)-np.mean(fusion_a):+.3f} ent={np.mean(ents):.3f}", flush=True)

    with open(os.path.join(RESULTS, "scale_qubits_results.json"), "w") as f:
        json.dump(rows, f, indent=2)
    ns = [r["qubits"] for r in rows]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.errorbar(ns, [r["acc_recon"] for r in rows], yerr=[r["acc_recon_std"] for r in rows],
                marker="^", color="#c0392b", label="reconstruction", capsize=3)
    ax.errorbar(ns, [r["acc_fusion"] for r in rows], yerr=[r["acc_fusion_std"] for r in rows],
                marker="o", color="#2471a3", label="late fusion", capsize=3)
    ax.set_xlabel("qubits (register width)"); ax.set_ylabel("accuracy")
    ax.set_xticks(ns); ax.legend(); ax.set_title(f"Fusion tracks reconstruction with width (alpha={ALPHA})")
    fig.tight_layout(); fig.savefig(os.path.join(RESULTS, "scale_qubits.png"), dpi=130)
    print("DONE -> results/scale_qubits_results.json + scale_qubits.png")


if __name__ == "__main__":
    run()
