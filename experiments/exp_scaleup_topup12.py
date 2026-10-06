"""
exp_scaleup_topup12.py -- top up the n=12 scale-up rows from 3 to 5 seeds.

Replicates exp_scaleup's exact per-(k, seed) computation (same data, init, and
maxiter, so seeds 0-2 reproduce the existing rows) with seeds 0..4, parallelized
over k. Writes to a SIDE file so the concurrently running exp_scaleup (n=20) is
untouched; merge into results/scaleup_results.json after both finish.

Run:  python -m experiments.exp_scaleup_topup12
Output: results/scaleup_n12_5seeds.json
"""
import json
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from lfqml.circuits import QNNConfig, prepare, full_features
from lfqml import datasets
from lfqml.train import train_qnn, train_fusion_independent
from lfqml.models import qsim_sigmoid
from lfqml.metrics import classification_metrics

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")
OUT = os.path.join(RESULTS, "scaleup_n12_5seeds.json")

ALPHA = 0.3
HALF, SEEDS, B0_ITER, FU_ITER, NSAMP = 6, [0, 1, 2, 3, 4], 35, 45, 200


def one_k(k):
    cfg = QNNConfig(n_A=HALF, n_B=HALF, depth=2, n_cuts=k, trainable_coupling=True)
    accs_b0, accs_fu = [], []
    for seed in SEEDS:
        X, y = datasets.synthetic_partition(ALPHA, HALF, HALF, NSAMP, seed=seed)
        ntr = int(0.7 * NSAMP)
        Xtr, ytr, Xte, yte = X[:ntr], y[:ntr], X[ntr:], y[ntr:]
        tf = train_fusion_independent(cfg, Xtr, ytr, seed=seed,
                                      hidden=8 + HALF, maxiter=FU_ITER)
        accs_fu.append(classification_metrics(yte, tf.predict_proba(Xte))["accuracy"])
        tq = train_qnn(cfg, Xtr, ytr, seed=seed, maxiter=B0_ITER)
        F0 = np.array([full_features(prepare(cfg, tq.params, x), tq.observables)
                       for x in Xte])
        accs_b0.append(classification_metrics(
            yte, qsim_sigmoid(F0 @ tq.head_w + tq.head_b))["accuracy"])
    row = {"qubits": 2 * HALF, "n_cuts": k, "n_seeds": len(SEEDS),
           "acc_fusion": float(np.mean(accs_fu)),
           "acc_fusion_std": float(np.std(accs_fu)),
           "acc_B0": float(np.mean(accs_b0)),
           "acc_B0_std": float(np.std(accs_b0)),
           "recon_overhead_9k": float(9 ** k)}
    print(f"n=12 k={k}: fusion={row['acc_fusion']:.3f}+/-{row['acc_fusion_std']:.3f}  "
          f"B0={row['acc_B0']:.3f}+/-{row['acc_B0_std']:.3f}", flush=True)
    return row


def run():
    with ProcessPoolExecutor(max_workers=4) as ex:
        rows = list(ex.map(one_k, [1, 2, 3, 4, 5, 6]))
    with open(OUT, "w") as f:
        json.dump(rows, f, indent=2)
    print("DONE -> results/scaleup_n12_5seeds.json")


if __name__ == "__main__":
    run()
