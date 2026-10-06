"""
exp_h2.py -- the entanglement-regularizer sweep (H2), and a cleaner H1 test.

Idea: hold the TASK fixed and vary the circuit's cut-entanglement directly, via
the training penalty beta.  This is a cleaner probe than the alpha-sweep because
beta controls entanglement while the task difficulty is held constant.

For each task (an easy and a hard alpha) and each beta:
  * train the QNN with the entanglement penalty,
  * measure accuracy retained (B0), the realised cut-entanglement, the quantum
    fusion gap (Delta_mlp), and the Q-dial knee.

Expected (H2): larger beta -> lower cut-entanglement -> the Pareto frontier shifts
toward cheaper operating points (knee -> smaller Q).  On the HARD task, forcing
low entanglement should cost more accuracy than on the EASY task -- that accuracy
cost is 'the price of the quantumness the task actually needs' (H1, restated).

Outputs: results/h2_results.json and results/h2_*.png
Run:  python -m experiments.exp_h2
"""
import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from lfqml.circuits import QNNConfig
from lfqml import datasets
from lfqml.train import train_qnn
from lfqml.models import evaluate_baselines

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS, exist_ok=True)

TASKS = {"easy(alpha=0.25)": 0.25, "hard(alpha=1.0)": 1.0}
BETAS = [0.0, 0.2, 0.5, 1.0, 2.0]
SEEDS = [0, 1]
N_SAMPLES, N_TRAIN = 180, 120


def knee(qdial, tol=0.02):
    acc1 = [q for q in qdial if q["Q"] >= 1.0][0]["accuracy"]
    for q in sorted(qdial, key=lambda r: r["Q"]):
        if q["accuracy"] >= acc1 - tol:
            return q["Q"]
    return 1.0


def run():
    cfg = QNNConfig(n_A=2, n_B=2, depth=2, n_cuts=2, trainable_coupling=True)
    rows = []
    for tname, alpha in TASKS.items():
        for beta in BETAS:
            for seed in SEEDS:
                X, y = datasets.synthetic_partition(alpha=alpha, n_A=2, n_B=2,
                                                    n_samples=N_SAMPLES, seed=seed)
                Xtr, ytr, Xte, yte = X[:N_TRAIN], y[:N_TRAIN], X[N_TRAIN:], y[N_TRAIN:]
                tq = train_qnn(cfg, Xtr, ytr, seed=seed, maxiter=80, beta=beta)
                res = evaluate_baselines(tq, Xtr, ytr, Xte, yte)
                rows.append({
                    "task": tname, "alpha": alpha, "beta": beta, "seed": seed,
                    "acc_B0": res["B0_uncut"]["accuracy"],
                    "acc_B2_mlp": res["B2_fusion_mlp"]["accuracy"],
                    "cut_ent": res["cut_entanglement"],
                    "delta_mlp": res["B0_uncut"]["accuracy"] - res["B2_fusion_mlp"]["accuracy"],
                    "kneeQ": knee(res["Qdial"]),
                    "Qdial": res["Qdial"],
                })
                print(f"{tname} beta={beta:.1f} seed={seed}  "
                      f"B0={rows[-1]['acc_B0']:.3f} ent={rows[-1]['cut_ent']:.3f} "
                      f"kneeQ={rows[-1]['kneeQ']:.2f}", flush=True)

    with open(os.path.join(RESULTS, "h2_results.json"), "w") as f:
        json.dump(rows, f, indent=2)
    make_plots(rows)
    print("DONE -> results/h2_results.json and plots")


def _agg(rows, task, key_x, key_y):
    rs = [r for r in rows if r["task"] == task]
    xs = sorted(set(r[key_x] for r in rs))
    ms = [np.mean([r[key_y] for r in rs if r[key_x] == x]) for x in xs]
    return np.array(xs), np.array(ms)


def make_plots(rows):
    # entanglement and accuracy vs beta, per task
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for task in TASKS:
        b, e = _agg(rows, task, "beta", "cut_ent")
        axes[0].plot(b, e, marker="o", label=task)
        b, a = _agg(rows, task, "beta", "acc_B0")
        axes[1].plot(b, a, marker="o", label=task)
    axes[0].set_xlabel("beta (entanglement penalty)"); axes[0].set_ylabel("cut-entanglement (bits)")
    axes[0].set_title("H2: beta suppresses cut-entanglement"); axes[0].legend()
    axes[1].set_xlabel("beta"); axes[1].set_ylabel("acc(B0)")
    axes[1].set_title("Accuracy cost of forcing low entanglement"); axes[1].legend()
    fig.tight_layout(); fig.savefig(os.path.join(RESULTS, "h2_beta_sweep.png"), dpi=130)

    # frontier shift: knee-Q vs entanglement
    fig, ax = plt.subplots(figsize=(5, 4))
    for task in TASKS:
        rs = [r for r in rows if r["task"] == task]
        ax.scatter([r["cut_ent"] for r in rs], [r["kneeQ"] for r in rs], label=task)
    ax.set_xlabel("cut-entanglement (bits)"); ax.set_ylabel("Q-dial knee (needed Q)")
    ax.set_title("H1/H3: needed reconstruction fraction vs cut-entanglement")
    ax.legend(); fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "h2_knee_vs_ent.png"), dpi=130)


if __name__ == "__main__":
    run()
