"""
exp_h1.py -- the controlled alpha-sweep behind hypotheses H1/H3.

For each alpha (cross-cut dependence of the label) and several seeds:
  * train the target QNN (B0),
  * evaluate reconstruction (B1), linear + MLP late fusion (B2), Kawase (B6),
    entanglement ablation (B7), and the full Q-dial,
  * record cut-entanglement and the two accuracy gaps:
        Delta_linear = acc(B0) - acc(B2 linear fusion)   [classical + quantum]
        Delta_mlp    = acc(B0) - acc(B2 MLP fusion)       [genuinely quantum]

Outputs: results/h1_results.json and results/h1_*.png

Run:  python -m experiments.exp_h1
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

ALPHAS = [0.0, 0.25, 0.5, 0.75, 1.0]
SEEDS = list(range(10))          # T4: 10 seeds for statistical rigor / error bars
N_SAMPLES = 300
N_TRAIN = 180                    # larger 120-sample test set


def one_run(job):
    alpha, seed = job
    cfg = QNNConfig(n_A=2, n_B=2, depth=2, n_cuts=2, trainable_coupling=True)
    X, y = datasets.synthetic_partition(alpha=alpha, n_A=2, n_B=2,
                                        n_samples=N_SAMPLES, seed=seed)
    Xtr, ytr = X[:N_TRAIN], y[:N_TRAIN]
    Xte, yte = X[N_TRAIN:], y[N_TRAIN:]
    tq = train_qnn(cfg, Xtr, ytr, seed=seed, maxiter=90)
    res = evaluate_baselines(tq, Xtr, ytr, Xte, yte)
    row = {
        "alpha": alpha, "seed": seed,
        "acc_B0": res["B0_uncut"]["accuracy"],
        "acc_B1": res["B1_reconstruction"]["accuracy"],
        "acc_B2_lin": res["B2_fusion_linear"]["accuracy"],
        "acc_B2_mlp": res["B2_fusion_mlp"]["accuracy"],
        "acc_B6": res["B6_kawase_sum"]["accuracy"],
        "acc_B7": res["B7_ent_ablation"]["accuracy"],
        "cut_ent": res["cut_entanglement"],
        "coupling_angle": res["coupling_angle"],
        "Qdial": res["Qdial"],
    }
    row["delta_lin"] = row["acc_B0"] - row["acc_B2_lin"]
    row["delta_mlp"] = row["acc_B0"] - row["acc_B2_mlp"]
    return row


def run():
    from concurrent.futures import ProcessPoolExecutor
    jobs = [(a, s) for a in ALPHAS for s in SEEDS]
    rows = []
    with ProcessPoolExecutor(max_workers=10) as ex:
        for row in ex.map(one_run, jobs):
            rows.append(row)
            print(f"alpha={row['alpha']:.2f} seed={row['seed']}  B0={row['acc_B0']:.3f} "
                  f"B2lin={row['acc_B2_lin']:.3f} B2mlp={row['acc_B2_mlp']:.3f} "
                  f"ent={row['cut_ent']:.3f} phi={row['coupling_angle']:.2f}", flush=True)
    rows.sort(key=lambda r: (r["alpha"], r["seed"]))
    with open(os.path.join(RESULTS, "h1_results.json"), "w") as f:
        json.dump(rows, f, indent=2)
    make_plots(rows)
    print("DONE -> results/h1_results.json and plots")


def _agg(rows, key_x, key_y):
    xs = sorted(set(r[key_x] for r in rows))
    means, stds = [], []
    for x in xs:
        vals = [r[key_y] for r in rows if r[key_x] == x]
        means.append(np.mean(vals)); stds.append(np.std(vals))
    return np.array(xs), np.array(means), np.array(stds)


def make_plots(rows):
    # (1) gaps + entanglement vs alpha
    fig, ax1 = plt.subplots(figsize=(6, 4))
    a, dl, sl = _agg(rows, "alpha", "delta_lin")
    _, dm, sm = _agg(rows, "alpha", "delta_mlp")
    ax1.errorbar(a, dl, yerr=sl, marker="o", label="Δ linear fusion (classical+quantum)")
    ax1.errorbar(a, dm, yerr=sm, marker="s", label="Δ MLP fusion (genuinely quantum)")
    ax1.set_xlabel("alpha (cross-cut dependence of label)")
    ax1.set_ylabel("accuracy gap  acc(B0) - acc(fusion)")
    ax2 = ax1.twinx()
    _, ce, se = _agg(rows, "alpha", "cut_ent")
    ax2.errorbar(a, ce, yerr=se, marker="^", color="green", alpha=0.5,
                 label="cut-entanglement (bits)")
    ax2.set_ylabel("cut-entanglement (bits)", color="green")
    ax1.legend(loc="upper left", fontsize=8)
    ax1.set_title("H1: gaps and cut-entanglement vs task cross-cut dependence")
    fig.tight_layout(); fig.savefig(os.path.join(RESULTS, "h1_gaps_vs_alpha.png"), dpi=130)

    # (2) H1 scatter: Delta_mlp vs cut-entanglement (the key correlation)
    fig, ax = plt.subplots(figsize=(5, 4))
    ce = [r["cut_ent"] for r in rows]
    dm = [r["delta_mlp"] for r in rows]
    ax.scatter(ce, dm, c=[r["alpha"] for r in rows], cmap="viridis")
    ax.set_xlabel("cut-entanglement (bits)")
    ax.set_ylabel("Δ MLP fusion (quantum gap)")
    ax.set_title("H1: does the quantum gap track cut-entanglement?")
    fig.tight_layout(); fig.savefig(os.path.join(RESULTS, "h1_scatter.png"), dpi=130)

    # (3) representative Q-dial Pareto frontier (highest-alpha, seed 0)
    rep = [r for r in rows if r["seed"] == 0][-1]
    qs = rep["Qdial"]
    fig, ax = plt.subplots(figsize=(5, 4))
    costs = [max(q["cost_overhead"], 1e-3) for q in qs]
    accs = [q["accuracy"] for q in qs]
    ax.plot(costs, accs, marker="o")
    for q in qs:
        ax.annotate(f"Q={q['Q']}", (max(q["cost_overhead"], 1e-3), q["accuracy"]),
                    fontsize=7)
    ax.set_xscale("log")
    ax.set_xlabel("sampling overhead (log)  [linear -> exponential]")
    ax.set_ylabel("test accuracy")
    ax.set_title(f"Q-dial Pareto frontier (alpha={rep['alpha']})")
    fig.tight_layout(); fig.savefig(os.path.join(RESULTS, "h1_qdial_frontier.png"), dpi=130)


if __name__ == "__main__":
    run()
