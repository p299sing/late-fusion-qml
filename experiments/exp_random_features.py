"""
exp_random_features.py -- reviewer ablation W9: do the TRAINED subcircuits matter,
or is the classical fusion head doing all the work?

Fits the SAME MLP fusion head on features from UNTRAINED (random-parameter)
subcircuits, under protocols that exactly match existing studies so the numbers
are directly comparable to stored results:
  * alpha-sweep: synthetic_partition(alpha, 2, 2, 220, seed), split 150/70,
    matching exp_independent (compare to acc_indep / acc_B0 in
    independent_results.json).
  * breast-cancer: leakage-free PCA-4, 70/30 permutation split, matching
    exp_benchmarks (compare to acc_B2_mlp / acc_B0 in bench_results.json).

Random params = init_params(cfg_nocoupling, seed) -- the exact initialization
training would start from, so the comparison isolates the effect of training
the quantum feature maps.

Outputs: results/random_features_results.json
Run:  python -m experiments.exp_random_features
"""
import json
import os
import numpy as np

from lfqml import datasets, cutting, qsim
from lfqml.circuits import QNNConfig, init_params, prepare
from lfqml.fusion import MLPHead
from lfqml.metrics import classification_metrics

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")
OUT = os.path.join(RESULTS, "random_features_results.json")

ALPHAS = [0.25, 0.5, 0.75, 1.0]
SEEDS = list(range(10))


def _head_acc(cfg, theta, Xtr, ytr, Xte, yte, seed):
    Str = np.array([cutting.subcircuit_raw_features(prepare(cfg, theta, x)) for x in Xtr])
    Ste = np.array([cutting.subcircuit_raw_features(prepare(cfg, theta, x)) for x in Xte])
    head = MLPHead(hidden=8, seed=seed).fit(Str, ytr)
    return classification_metrics(yte, head.predict_proba(Ste))["accuracy"]


def run():
    from dataclasses import replace
    rows = []
    # (i) alpha-sweep, exp_independent protocol
    cfg = QNNConfig(n_A=2, n_B=2, depth=2, n_cuts=2, trainable_coupling=True)
    cfg_fu = replace(cfg, trainable_coupling=False, coupling_init=0.0)
    for alpha in ALPHAS:
        for seed in SEEDS:
            X, y = datasets.synthetic_partition(alpha, 2, 2, n_samples=220, seed=seed)
            Xtr, ytr, Xte, yte = X[:150], y[:150], X[150:], y[150:]
            theta0 = init_params(cfg_fu, seed=seed)
            acc = _head_acc(cfg_fu, theta0, Xtr, ytr, Xte, yte, seed)
            rows.append({"task": "alpha_sweep", "alpha": alpha, "seed": seed,
                         "acc_random_fusion": acc})
            print(f"alpha={alpha:.2f} seed={seed}  random-feat fusion={acc:.3f}", flush=True)
    # (ii) breast-cancer, exp_benchmarks protocol (leakage-free)
    for seed in SEEDS:
        X, y = datasets.breast_cancer_raw()
        rng = np.random.default_rng(seed)
        idx = rng.permutation(len(X)); ntr = int(0.7 * len(X))
        tr, te = idx[:ntr], idx[ntr:]
        Xtr, Xte = datasets.preprocess_split(X[tr], X[te], n_features=4, seed=seed)
        theta0 = init_params(cfg_fu, seed=seed)
        acc = _head_acc(cfg_fu, theta0, Xtr, y[tr], Xte, y[te], seed)
        rows.append({"task": "breast_cancer", "alpha": None, "seed": seed,
                     "acc_random_fusion": acc})
        print(f"breast_cancer seed={seed}  random-feat fusion={acc:.3f}", flush=True)

    with open(OUT, "w") as f:
        json.dump(rows, f, indent=2)

    # comparison against stored trained results
    ind = json.load(open(os.path.join(RESULTS, "independent_results.json")))
    print("\n=== alpha-sweep: random vs trained-independent vs B0 (10 seeds) ===")
    for a in ALPHAS:
        rnd = np.mean([r["acc_random_fusion"] for r in rows
                       if r["task"] == "alpha_sweep" and r["alpha"] == a])
        tr_ = np.mean([r["acc_indep"] for r in ind if r["alpha"] == a])
        b0 = np.mean([r["acc_B0"] for r in ind if r["alpha"] == a])
        print(f"  alpha={a:.2f}  random={rnd:.3f}  trained={tr_:.3f}  B0={b0:.3f}")
    bench = json.load(open(os.path.join(RESULTS, "bench_results.json")))
    bc = [r for r in bench if r["dataset"] == "breast_cancer"]
    rnd = np.mean([r["acc_random_fusion"] for r in rows if r["task"] == "breast_cancer"])
    print(f"\nbreast_cancer  random={rnd:.3f}  "
          f"trained-frozen={np.mean([r['acc_B2_mlp'] for r in bc]):.3f}  "
          f"B0={np.mean([r['acc_B0'] for r in bc]):.3f}")
    print("DONE -> results/random_features_results.json")


if __name__ == "__main__":
    run()
