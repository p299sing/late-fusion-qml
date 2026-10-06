"""
exp_scale_transfer.py -- reviewer's #1 request (W1/Q1): does the cut-entanglement
diagnostic TRANSFER ACROSS SCALE? I.e. does S_A measured on a CHEAP small-width
coupled circuit predict the fusion--reconstruction gap at a LARGER, more expensive
width on the same task family?

Design (same task family = synthetic_partition at a fixed alpha, different width):
  * cheap probe  : train the coupled circuit at n=8 (4+4, k=2); record S_A^{n=8}
  * expensive tgt: at n=12 (6+6, k=2) train the uncut reference (B0) and independent
                   late fusion; record gap^{n=12} = acc_B0 - acc_fusion
A spread of task locality (alpha in {0.0, 0.3, 0.6, 0.9}) x seeds gives the range of
gaps needed for a correlation. We report Spearman(S_A^{n=8}, gap^{n=12}) over all
(alpha, seed) runs and the per-alpha rank ordering. Even a null/non-monotone result
is an honest, reportable finding (the diagnostic would then be within-scale only).

Outputs: results/scale_transfer_results.json
Run:  python -m experiments.exp_scale_transfer
"""
import json
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from scipy.stats import spearmanr

from lfqml import datasets
from lfqml.circuits import QNNConfig, prepare, full_features
from lfqml.train import train_qnn, train_fusion_independent
from lfqml.models import qsim_sigmoid
from lfqml.metrics import classification_metrics, mean_cut_entanglement

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")
OUT = os.path.join(RESULTS, "scale_transfer_results.json")

ALPHAS = [0.0, 0.3, 0.6, 0.9]
SEEDS = [0, 1, 2]
NS = 200


def one_run(job):
    alpha, seed = job
    # cheap probe: S_A of the trained coupled circuit at n=8 (4+4, k=2)
    cfg8 = QNNConfig(n_A=4, n_B=4, depth=2, n_cuts=2, trainable_coupling=True)
    X8, y8 = datasets.synthetic_partition(alpha, 4, 4, NS, seed=seed)
    tr = int(0.7 * NS)
    tq8 = train_qnn(cfg8, X8[:tr], y8[:tr], seed=seed, maxiter=45)
    sa8 = mean_cut_entanglement(cfg8, [prepare(cfg8, tq8.params, x) for x in X8[tr:]])

    # expensive target: fusion-reconstruction gap at n=12 (6+6, k=2)
    cfg12 = QNNConfig(n_A=6, n_B=6, depth=2, n_cuts=2, trainable_coupling=True)
    X, y = datasets.synthetic_partition(alpha, 6, 6, NS, seed=seed)
    Xtr, ytr, Xte, yte = X[:tr], y[:tr], X[tr:], y[tr:]
    tq = train_qnn(cfg12, Xtr, ytr, seed=seed, maxiter=35)
    F0 = np.array([full_features(prepare(cfg12, tq.params, x), tq.observables) for x in Xte])
    acc_b0 = classification_metrics(yte, qsim_sigmoid(F0 @ tq.head_w + tq.head_b))["accuracy"]
    tf = train_fusion_independent(cfg12, Xtr, ytr, seed=seed, hidden=8 + 6, maxiter=45)
    acc_fus = classification_metrics(yte, tf.predict_proba(Xte))["accuracy"]
    gap12 = acc_b0 - acc_fus
    return {"alpha": alpha, "seed": seed, "S_A_n8": float(sa8),
            "acc_b0_n12": float(acc_b0), "acc_fusion_n12": float(acc_fus),
            "gap_n12": float(gap12)}


def run():
    jobs = [(a, s) for a in ALPHAS for s in SEEDS]
    with ProcessPoolExecutor(max_workers=6) as ex:
        rows = list(ex.map(one_run, jobs))
    rows.sort(key=lambda r: (r["alpha"], r["seed"]))
    with open(OUT, "w") as f:
        json.dump(rows, f, indent=2)

    for r in rows:
        print(f"alpha={r['alpha']:.1f} seed={r['seed']}  S_A(n=8)={r['S_A_n8']:.3f}  "
              f"gap(n=12)={r['gap_n12']:+.3f}", flush=True)
    sa = np.array([r["S_A_n8"] for r in rows])
    gap = np.array([r["gap_n12"] for r in rows])
    rho, p = spearmanr(sa, gap)
    print(f"\nSpearman(S_A@n8, gap@n12) = {rho:.3f}  (p={p:.3f}, n={len(rows)})")
    print("per-alpha means (rank check):")
    for a in ALPHAS:
        rs = [r for r in rows if r["alpha"] == a]
        print(f"  alpha={a:.1f}: S_A(n=8)={np.mean([r['S_A_n8'] for r in rs]):.3f}  "
              f"gap(n=12)={np.mean([r['gap_n12'] for r in rs]):+.3f}")
    print("DONE -> results/scale_transfer_results.json")


if __name__ == "__main__":
    run()
