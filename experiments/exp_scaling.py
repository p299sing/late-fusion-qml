"""
exp_scaling.py -- Tier 1: the exponential-vs-linear cost crossover.

As the number of cuts k grows, reconstruction's sampling overhead grows like
gamma^(2k) (EXPONENTIAL) while late fusion stays linear (one run per subcircuit).
We increase k and record, per k: the two overheads and the accuracy each readout
retains.  The headline figure: reconstruction cost curves up exponentially, fusion
cost is flat, and fusion's accuracy tracks reconstruction's.

Uses wider registers so we can place more cross cuts. 6 qubits (3+3).

Outputs: results/scaling_results.json + results/scaling_cost.png
Run:  python -m experiments.exp_scaling
"""
import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from lfqml.circuits import QNNConfig
from lfqml import datasets, cutting
from lfqml.train import train_qnn
from lfqml.models import evaluate_baselines

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS, exist_ok=True)

CUTS = [1, 2, 3]
SEEDS = list(range(10))


def run():
    rows = []
    for k in CUTS:
        for seed in SEEDS:
            cfg = QNNConfig(n_A=3, n_B=3, depth=2, n_cuts=k)
            # a moderately cross-cut-dependent task so cuts matter
            X, y = datasets.synthetic_partition(0.5, n_A=3, n_B=3, n_samples=200, seed=seed)
            Xtr, ytr, Xte, yte = X[:140], y[:140], X[140:], y[140:]
            tq = train_qnn(cfg, Xtr, ytr, seed=seed, maxiter=70)
            res = evaluate_baselines(tq, Xtr, ytr, Xte, yte, Q_grid=(0.0, 0.5, 1.0))
            rows.append({
                "n_cuts": k, "seed": seed,
                "acc_B0": res["B0_uncut"]["accuracy"],
                "acc_B2_mlp": res["B2_fusion_mlp"]["accuracy"],
                "recon_overhead": res["B1_reconstruction"]["cost_overhead"],
                "recon_subexp": res["B1_reconstruction"]["n_subexp"],
                "fusion_overhead": res["B2_fusion_mlp"]["cost_overhead"],
            })
            print(f"k={k} seed={seed}  B0={rows[-1]['acc_B0']:.3f} "
                  f"B2mlp={rows[-1]['acc_B2_mlp']:.3f}  "
                  f"recon_overhead={rows[-1]['recon_overhead']:.1f} "
                  f"fusion_overhead={rows[-1]['fusion_overhead']:.1f}", flush=True)

    with open(os.path.join(RESULTS, "scaling_results.json"), "w") as f:
        json.dump(rows, f, indent=2)
    make_plot(rows)
    print("DONE -> results/scaling_results.json + scaling_cost.png")


def _agg(rows, key):
    ks = sorted(set(r["n_cuts"] for r in rows))
    return ks, [np.mean([r[key] for r in rows if r["n_cuts"] == k]) for k in ks]


def make_plot(rows):
    ks, rec = _agg(rows, "recon_overhead")
    _, fus = _agg(rows, "fusion_overhead")
    _, a0 = _agg(rows, "acc_B0")
    _, a2 = _agg(rows, "acc_B2_mlp")

    fig, ax1 = plt.subplots(figsize=(6.5, 4))
    ax1.plot(ks, rec, marker="o", color="crimson", label="reconstruction overhead (exp)")
    ax1.plot(ks, fus, marker="s", color="steelblue", label="late fusion overhead (linear)")
    ax1.set_yscale("log")
    ax1.set_xlabel("number of cuts k")
    ax1.set_ylabel("sampling overhead (log)")
    ax1.set_xticks(ks)
    ax2 = ax1.twinx()
    ax2.plot(ks, a0, marker="^", ls="--", color="gray", label="acc reconstruction/B0")
    ax2.plot(ks, a2, marker="v", ls="--", color="green", label="acc late fusion")
    ax2.set_ylabel("test accuracy")
    ax2.set_ylim(0, 1.05)
    lines = ax1.get_lines() + ax2.get_lines()
    ax1.legend(lines, [l.get_label() for l in lines], fontsize=7, loc="center left")
    ax1.set_title("Cost crossover: reconstruction explodes, fusion stays flat")
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "scaling_cost.png"), dpi=130)


if __name__ == "__main__":
    run()
