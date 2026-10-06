"""
exp_dense_alpha.py -- W4: dense-alpha validation of the cut-entanglement diagnostic.

The reviewer critique: the diagnostic (S_A ~ knee / fusion gap) rested on 5 alpha points.
Here we sweep a DENSE grid of 13 alpha values x 8 seeds (104 runs) and record, per run:
the full Q-dial curve, the knee under BOTH definitions in the codebase --
  * kneeQ      : smallest Q reaching 95% of the Q=1 accuracy (relative)
  * kneeQ_abs  : smallest Q within 0.02 of the Q=1 accuracy (absolute; analyze_h1 def)
-- the trained-circuit cut-entanglement S_A, and the frozen-fusion gap
delta_mlp = acc(B0) - acc(B2_mlp). We report Spearman correlations with bootstrap
95% CIs under both knee definitions -- the properly powered version of the claim.

Deterministic per (alpha, seed); parallelized over runs. Outputs:
results/dense_alpha_results.json
Run:  python -m experiments.exp_dense_alpha
"""
import json
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from lfqml.circuits import QNNConfig
from lfqml import datasets
from lfqml.train import train_qnn
from lfqml.models import evaluate_baselines

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS, exist_ok=True)
OUT = os.path.join(RESULTS, "dense_alpha_results.json")

ALPHAS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 1.0]
SEEDS = list(range(8))
N_SAMPLES, N_TRAIN = 300, 180
WORKERS = 6


def knee_rel(qcurve, frac=0.95):
    """Smallest Q reaching `frac` of the Q=1 accuracy (relative definition)."""
    accQ1 = [p["accuracy"] for p in qcurve if p["Q"] >= 1.0][0]
    for p in sorted(qcurve, key=lambda p: p["Q"]):
        if p["accuracy"] >= frac * accQ1:
            return p["Q"]
    return 1.0


def knee_abs(qcurve, tol=0.02):
    """Smallest Q within `tol` (absolute) of the Q=1 accuracy (analyze_h1 definition)."""
    accQ1 = [p["accuracy"] for p in qcurve if p["Q"] >= 1.0][0]
    for p in sorted(qcurve, key=lambda p: p["Q"]):
        if p["accuracy"] >= accQ1 - tol:
            return p["Q"]
    return 1.0


def one_run(job):
    alpha, seed = job
    cfg = QNNConfig(n_A=2, n_B=2, depth=2, n_cuts=2, trainable_coupling=True)
    X, y = datasets.synthetic_partition(alpha=alpha, n_A=2, n_B=2,
                                        n_samples=N_SAMPLES, seed=seed)
    Xtr, ytr, Xte, yte = X[:N_TRAIN], y[:N_TRAIN], X[N_TRAIN:], y[N_TRAIN:]
    tq = train_qnn(cfg, Xtr, ytr, seed=seed, maxiter=60)
    res = evaluate_baselines(tq, Xtr, ytr, Xte, yte)
    qcurve = [{"Q": p["Q"], "accuracy": p["accuracy"],
               "cost_overhead": p["cost_overhead"]} for p in res["Qdial"]]
    return {"alpha": alpha, "seed": seed,
            "acc_B0": res["B0_uncut"]["accuracy"],
            "acc_B2_mlp": res["B2_fusion_mlp"]["accuracy"],
            "delta_mlp": res["B0_uncut"]["accuracy"] - res["B2_fusion_mlp"]["accuracy"],
            "cut_ent": res["cut_entanglement"],
            "kneeQ": knee_rel(qcurve),
            "kneeQ_abs": knee_abs(qcurve),
            "qdial": qcurve}


def correlations(rows, knee_key):
    from scipy.stats import spearmanr
    rng = np.random.default_rng(0)
    ent = np.array([r["cut_ent"] for r in rows])
    out = {}
    for name, x in [("knee~S_A", np.array([r[knee_key] for r in rows])),
                    ("gap~S_A", np.array([r["delta_mlp"] for r in rows]))]:
        stats = []
        for _ in range(2000):
            i = rng.integers(0, len(ent), len(ent))
            stats.append(spearmanr(ent[i], x[i]).statistic)
        lo, hi = np.percentile(stats, [2.5, 97.5])
        out[name] = (spearmanr(ent, x).statistic, lo, hi)
    return out


def run():
    jobs = [(a, s) for a in ALPHAS for s in SEEDS]
    rows = []
    with ProcessPoolExecutor(max_workers=WORKERS) as ex:
        for row in ex.map(one_run, jobs):
            rows.append(row)
            print(f"alpha={row['alpha']:.2f} seed={row['seed']}  "
                  f"knee={row['kneeQ']:.2f} knee_abs={row['kneeQ_abs']:.2f} "
                  f"S_A={row['cut_ent']:.3f} gap={row['delta_mlp']:+.3f}", flush=True)
    rows.sort(key=lambda r: (r["alpha"], r["seed"]))
    with open(OUT, "w") as f:
        json.dump(rows, f, indent=2)

    for key in ("kneeQ", "kneeQ_abs"):
        print(f"--- knee definition: {key} ---")
        for name, (r, lo, hi) in correlations(rows, key).items():
            print(f"{name}: spearman={r:.3f}  95% CI [{lo:.3f}, {hi:.3f}]  n={len(rows)}")
    print("DONE -> results/dense_alpha_results.json")


if __name__ == "__main__":
    run()
