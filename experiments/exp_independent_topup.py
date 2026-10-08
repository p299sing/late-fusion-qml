"""
exp_independent_topup.py -- add seeds 10..19 to the uncoupled-vs-frozen-vs-reconstruction alpha
sweep of exp_independent.py (same data, config, maxiter and heads), so that the single-alpha
equivalence tests have n=20 pairs instead of 10 (several cells were "inconclusive" at n=10).
Parallel over (alpha, seed) with 8 processes; writes results/independent_results_topup.json.
Run:  OMP_NUM_THREADS=1 python -m experiments.exp_independent_topup
Merge: the analysis reads both files when present.
"""
import json, os
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from lfqml import qsim, datasets, cutting
from lfqml.circuits import QNNConfig, prepare, full_features
from lfqml.train import train_qnn, train_fusion_independent
from lfqml.fusion import MLPHead
from lfqml.models import qsim_sigmoid
from lfqml.metrics import classification_metrics

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")
OUT = os.path.join(RESULTS, "independent_results_topup.json")
ALPHAS = [0.25, 0.5, 0.75, 1.0]; SEEDS = list(range(10, 20)); N, NTR = 220, 150

def one(job):
    alpha, seed = job
    cfg = QNNConfig(n_A=2, n_B=2, depth=2, n_cuts=2, trainable_coupling=True)
    X, y = datasets.synthetic_partition(alpha, 2, 2, n_samples=N, seed=seed)
    Xtr, ytr, Xte, yte = X[:NTR], y[:NTR], X[NTR:], y[NTR:]
    tq = train_qnn(cfg, Xtr, ytr, seed=seed, maxiter=90)
    F0 = np.array([full_features(prepare(cfg, tq.params, x), tq.observables) for x in Xte])
    acc_B0 = classification_metrics(yte, qsim_sigmoid(F0 @ tq.head_w + tq.head_b))["accuracy"]
    Str = np.array([cutting.subcircuit_raw_features(prepare(cfg, tq.params, x)) for x in Xtr])
    Ste = np.array([cutting.subcircuit_raw_features(prepare(cfg, tq.params, x)) for x in Xte])
    frozen = MLPHead(hidden=8, seed=seed).fit(Str, ytr)
    acc_frozen = classification_metrics(yte, frozen.predict_proba(Ste))["accuracy"]
    tf = train_fusion_independent(cfg, Xtr, ytr, seed=seed, hidden=8, paulis=(qsim.Z,), maxiter=120)
    acc_indep = classification_metrics(yte, tf.predict_proba(Xte))["accuracy"]
    return {"alpha": alpha, "seed": seed, "acc_B0": acc_B0, "acc_frozen": acc_frozen, "acc_indep": acc_indep,
            "gap_frozen": acc_B0 - acc_frozen, "gap_indep": acc_B0 - acc_indep}

if __name__ == "__main__":
    jobs = [(a, s) for a in ALPHAS for s in SEEDS]
    with ProcessPoolExecutor(8) as ex:
        rows = list(ex.map(one, jobs))
    json.dump(rows, open(OUT, "w"), indent=2)
    for a in ALPHAS:
        rs = [r for r in rows if r["alpha"] == a]
        print(f"alpha={a}: indep-B0 gap {np.mean([r['gap_indep'] for r in rs]):+.3f} (n={len(rs)})")
    print("DONE ->", OUT)
