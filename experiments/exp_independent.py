"""
exp_independent.py -- does INDEPENDENT training close the residual fusion gap?

The 10-seed H1 run showed a residual gap (~0.17) at high cross-cut dependence when
late fusion re-uses FROZEN B0-subcircuits. Hypothesis: those subcircuits were trained
inside a coupled circuit, so their no-coupling marginals are poor features. If instead
we train the subcircuits JOINTLY with the fusion head, from scratch, with NO coupling
gate ("independent / fusion-native" training), they should learn fusion-friendly
features and close the gap.

Compares, across alpha and seeds:
  * B0 / reconstruction  (upper bound)
  * B2 frozen fusion     (MLP on frozen B0-subcircuit marginals)  -- current method
  * B2 independent fusion (subcircuits + head trained together, no coupling) -- new

Outputs: results/independent_results.json + results/independent.png
Run:  python -m experiments.exp_independent
"""
import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from lfqml import qsim, datasets, cutting
from lfqml.circuits import QNNConfig, prepare, full_features
from lfqml.train import train_qnn, train_fusion_independent
from lfqml.fusion import MLPHead
from lfqml.models import qsim_sigmoid
from lfqml.metrics import classification_metrics

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS, exist_ok=True)

ALPHAS = [0.25, 0.5, 0.75, 1.0]
SEEDS = list(range(10))          # 10 seeds for camera-ready error bars
N, NTR = 220, 150


def run():
    cfg = QNNConfig(n_A=2, n_B=2, depth=2, n_cuts=2, trainable_coupling=True)
    rows = []
    for alpha in ALPHAS:
        for seed in SEEDS:
            X, y = datasets.synthetic_partition(alpha, 2, 2, n_samples=N, seed=seed)
            Xtr, ytr, Xte, yte = X[:NTR], y[:NTR], X[NTR:], y[NTR:]

            # --- B0 / reconstruction upper bound + frozen fusion ---
            tq = train_qnn(cfg, Xtr, ytr, seed=seed, maxiter=90)
            F0 = np.array([full_features(prepare(cfg, tq.params, x), tq.observables) for x in Xte])
            acc_B0 = classification_metrics(yte, qsim_sigmoid(F0 @ tq.head_w + tq.head_b))["accuracy"]
            Str = np.array([cutting.subcircuit_raw_features(prepare(cfg, tq.params, x)) for x in Xtr])
            Ste = np.array([cutting.subcircuit_raw_features(prepare(cfg, tq.params, x)) for x in Xte])
            frozen = MLPHead(hidden=8, seed=seed).fit(Str, ytr)
            acc_frozen = classification_metrics(yte, frozen.predict_proba(Ste))["accuracy"]

            # --- independent / fusion-native training ---
            tf = train_fusion_independent(cfg, Xtr, ytr, seed=seed, hidden=8,
                                          paulis=(qsim.Z,), maxiter=120)
            acc_indep = classification_metrics(yte, tf.predict_proba(Xte))["accuracy"]

            rows.append({"alpha": alpha, "seed": seed, "acc_B0": acc_B0,
                         "acc_frozen": acc_frozen, "acc_indep": acc_indep,
                         "gap_frozen": acc_B0 - acc_frozen,
                         "gap_indep": acc_B0 - acc_indep})
            print(f"alpha={alpha:.2f} seed={seed}  B0={acc_B0:.3f}  "
                  f"frozen={acc_frozen:.3f} (gap {acc_B0-acc_frozen:+.3f})  "
                  f"indep={acc_indep:.3f} (gap {acc_B0-acc_indep:+.3f})", flush=True)

    with open(os.path.join(RESULTS, "independent_results.json"), "w") as f:
        json.dump(rows, f, indent=2)
    make_plot(rows)
    # summary
    print("\n=== mean gaps vs reconstruction ===")
    for a in ALPHAS:
        rs = [r for r in rows if r["alpha"] == a]
        gf = np.mean([r["gap_frozen"] for r in rs]); gi = np.mean([r["gap_indep"] for r in rs])
        print(f"  alpha={a:.2f}  frozen_gap={gf:+.3f}   independent_gap={gi:+.3f}")
    print("DONE -> results/independent_results.json + independent.png")


def make_plot(rows):
    fig, ax = plt.subplots(figsize=(6, 4))
    for key, lab, mk in [("acc_B0", "reconstruction (B0)", "^"),
                         ("acc_frozen", "frozen fusion (B2)", "s"),
                         ("acc_indep", "independent fusion (new)", "o")]:
        xs = ALPHAS
        ms = [np.mean([r[key] for r in rows if r["alpha"] == a]) for a in xs]
        ss = [np.std([r[key] for r in rows if r["alpha"] == a]) for a in xs]
        ax.errorbar(xs, ms, yerr=ss, marker=mk, label=lab, capsize=3)
    ax.set_xlabel("alpha (cross-cut dependence)")
    ax.set_ylabel("test accuracy")
    ax.set_title("Does independent training close the residual fusion gap?")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "independent.png"), dpi=130)


if __name__ == "__main__":
    run()
