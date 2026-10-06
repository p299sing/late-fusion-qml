"""
exp_shots_total.py -- reviewer W5(ii): the shot-robustness comparison at EQUAL
TOTAL measurement budget (circuit executions), not per-expectation shots.

Fusion needs 2 measurement settings (Z on every qubit of each finished
subcircuit); reconstruction needs n_subexp quasiprobability subexperiments per
expectation family. At a total budget B:
    fusion per-setting shots   s_f = B / 2          -> feature std 1/sqrt(s_f)
    recon  per-subexp shots    s_r = B / n_subexp   -> expectation std gamma/sqrt(s_r)
Same trained models and protocol as exp_shots (breast-cancer PCA-4 leakage-free,
2 cuts, 10 seeds, 30 noise trials).

Outputs: results/shots_total_results.json
Run:  python -m experiments.exp_shots_total
"""
import json
import os
import numpy as np

from lfqml.circuits import QNNConfig, prepare
from lfqml import datasets, cutting
from lfqml.train import train_qnn
from lfqml.fusion import MLPHead
from lfqml.models import qsim_sigmoid
from lfqml.metrics import classification_metrics

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")
OUT = os.path.join(RESULTS, "shots_total_results.json")

TOTALS = [64, 200, 640, 2000, 6400, 20000, 64000, 200000]
NOISE_TRIALS = 30
SEEDS = list(range(10))
FUSION_SETTINGS = 2


def run():
    cfg = QNNConfig(n_A=2, n_B=2, depth=2, n_cuts=2)
    rows = []
    for seed in SEEDS:
        X, y = datasets.breast_cancer_raw()
        rng = np.random.default_rng(seed)
        idx = rng.permutation(len(X)); ntr = int(0.7 * len(X))
        tr, te = idx[:ntr], idx[ntr:]
        Xtr, Xte = datasets.preprocess_split(X[tr], X[te], n_features=4, seed=seed)
        tq = train_qnn(cfg, Xtr, y[tr], seed=seed, maxiter=90)

        pre_tr = [prepare(cfg, tq.params, x) for x in Xtr]
        pre_te = [prepare(cfg, tq.params, x) for x in Xte]
        Frec_te = np.array([cutting.reconstruct_features(cutting.build_cut(pc),
                            tq.observables, Q=1.0) for pc in pre_te])
        S_tr = np.array([cutting.subcircuit_raw_features(pc) for pc in pre_tr])
        S_te = np.array([cutting.subcircuit_raw_features(pc) for pc in pre_te])
        fusion_head = MLPHead(hidden=8, seed=seed).fit(S_tr, y[tr])

        cd0 = cutting.build_cut(pre_te[0])
        gamma_total = cd0.gamma_per_cut ** cd0.n_cuts
        n_subexp = cutting.n_subexperiments(cd0, Q=1.0)

        for B in TOTALS:
            s_f = B / FUSION_SETTINGS
            s_r = B / n_subexp
            noise_rng = np.random.default_rng(1000 + seed)
            acc_rec, acc_fus = [], []
            for _ in range(NOISE_TRIALS):
                Nrec = Frec_te + noise_rng.normal(0, gamma_total / np.sqrt(s_r), Frec_te.shape)
                p_rec = qsim_sigmoid(Nrec @ tq.head_w + tq.head_b)
                acc_rec.append(classification_metrics(y[te], p_rec)["accuracy"])
                Nfus = S_te + noise_rng.normal(0, 1.0 / np.sqrt(s_f), S_te.shape)
                acc_fus.append(classification_metrics(y[te],
                               fusion_head.predict_proba(Nfus))["accuracy"])
            rows.append({"seed": seed, "total_budget": B, "n_subexp": int(n_subexp),
                         "gamma_total": gamma_total,
                         "acc_recon": float(np.mean(acc_rec)),
                         "acc_fusion": float(np.mean(acc_fus))})
            print(f"seed={seed} B={B:7d}  recon={rows[-1]['acc_recon']:.3f} "
                  f"fusion={rows[-1]['acc_fusion']:.3f}", flush=True)

    with open(OUT, "w") as f:
        json.dump(rows, f, indent=2)
    print("\n=== equal-total-budget means ===")
    for B in TOTALS:
        rs = [r for r in rows if r["total_budget"] == B]
        print(f"B={B:7d}  recon={np.mean([r['acc_recon'] for r in rs]):.3f}  "
              f"fusion={np.mean([r['acc_fusion'] for r in rs]):.3f}")
    print("DONE -> results/shots_total_results.json")


if __name__ == "__main__":
    run()
