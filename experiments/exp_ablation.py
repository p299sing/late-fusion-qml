"""
exp_ablation.py -- G11: ablate the fusion design knobs.

On a hard cross-cut task (synthetic alpha=0.75), vary:
  * fusion feature set: <Z> only / <Z,X> / <Z,X,Y> per subcircuit qubit
  * fusion head capacity: hidden units {4, 8, 16}
Reports independent-fusion accuracy for each, to show which knobs matter.

Outputs: results/ablation_results.json
Run:  python -m experiments.exp_ablation
"""
import json
import os
import numpy as np

from lfqml import qsim, datasets
from lfqml.circuits import QNNConfig
from lfqml.train import train_fusion_independent
from lfqml.metrics import classification_metrics

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS, exist_ok=True)

FEATURES = {"Z": (qsim.Z,), "ZX": (qsim.Z, qsim.X), "ZXY": (qsim.Z, qsim.X, qsim.Y)}
HIDDEN = [4, 8, 16]
SEEDS = [0, 1, 2, 3, 4]


def run():
    cfg = QNNConfig(n_A=2, n_B=2, depth=2, n_cuts=2, trainable_coupling=True)
    rows = []
    for fname, paulis in FEATURES.items():
        for h in HIDDEN:
            accs = []
            for seed in SEEDS:
                X, y = datasets.synthetic_partition(0.75, 2, 2, 240, seed=seed)
                Xtr, ytr, Xte, yte = X[:160], y[:160], X[160:], y[160:]
                tf = train_fusion_independent(cfg, Xtr, ytr, seed=seed, hidden=h,
                                              paulis=paulis, maxiter=55)
                accs.append(classification_metrics(yte, tf.predict_proba(Xte))["accuracy"])
            rows.append({"features": fname, "hidden": h,
                         "acc": float(np.mean(accs)), "std": float(np.std(accs))})
            print(f"features={fname:3s} hidden={h:2d}  acc={np.mean(accs):.3f} "
                  f"+/- {np.std(accs):.3f}", flush=True)
    with open(os.path.join(RESULTS, "ablation_results.json"), "w") as f:
        json.dump(rows, f, indent=2)
    print("DONE -> results/ablation_results.json")


if __name__ == "__main__":
    run()
