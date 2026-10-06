"""
exp_proxy_diag.py -- reviewer C3a: does a CHEAPLY-trained coupled circuit's S_A
already predict the Q-dial knee, mitigating the diagnostic's training circularity?

The fully-trained S_A predicts the knee (rho=0.59); S_A at initialization carries
no signal (|rho|<0.1). Here we test the intermediate point: train the coupled
circuit for only PROXY_ITER L-BFGS iterations (a small fraction of the full 60)
and correlate that S_A with the knee stored in dense_alpha_results.json.

Outputs: results/proxy_diag_results.json
Run:  python -m experiments.exp_proxy_diag
"""
import json
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from scipy.stats import spearmanr

from lfqml import datasets
from lfqml.circuits import QNNConfig, prepare
from lfqml.metrics import mean_cut_entanglement
from lfqml.train import train_qnn

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")
OUT = os.path.join(RESULTS, "proxy_diag_results.json")
N_SAMPLES, N_TRAIN = 300, 180
PROXY_ITER = 5


def one_run(job):
    alpha, seed = job
    cfg = QNNConfig(n_A=2, n_B=2, depth=2, n_cuts=2, trainable_coupling=True)
    X, y = datasets.synthetic_partition(alpha=alpha, n_A=2, n_B=2,
                                        n_samples=N_SAMPLES, seed=seed)
    tq = train_qnn(cfg, X[:N_TRAIN], y[:N_TRAIN], seed=seed, maxiter=PROXY_ITER)
    ent = mean_cut_entanglement(cfg, [prepare(cfg, tq.params, x) for x in X[N_TRAIN:]])
    return {"alpha": alpha, "seed": seed, "cut_ent_proxy": ent}


def run():
    dense = json.load(open(os.path.join(RESULTS, "dense_alpha_results.json")))
    jobs = [(r["alpha"], r["seed"]) for r in dense]
    with ProcessPoolExecutor(max_workers=6) as ex:
        rows = list(ex.map(one_run, jobs))
    km = {(r["alpha"], r["seed"]): r for r in dense}
    for r in rows:
        r["kneeQ_abs"] = km[(r["alpha"], r["seed"])]["kneeQ_abs"]
        r["cut_ent_trained"] = km[(r["alpha"], r["seed"])]["cut_ent"]
    with open(OUT, "w") as f:
        json.dump(rows, f, indent=2)

    ent = np.array([r["cut_ent_proxy"] for r in rows])
    knee = np.array([r["kneeQ_abs"] for r in rows])
    entT = np.array([r["cut_ent_trained"] for r in rows])
    rng = np.random.default_rng(0)

    def boot(a, b):
        stats = []
        for _ in range(2000):
            i = rng.integers(0, len(a), len(a))
            stats.append(spearmanr(a[i], b[i]).statistic)
        lo, hi = np.percentile(stats, [2.5, 97.5])
        return spearmanr(a, b).statistic, lo, hi

    r1, lo1, hi1 = boot(ent, knee)
    r2, lo2, hi2 = boot(ent, entT)
    print(f"PROXY (maxiter={PROXY_ITER}) S_A ~ knee: spearman={r1:.3f} CI [{lo1:.3f},{hi1:.3f}]")
    print(f"proxy vs fully-trained S_A:  spearman={r2:.3f} CI [{lo2:.3f},{hi2:.3f}]")
    print("DONE -> results/proxy_diag_results.json")


if __name__ == "__main__":
    run()
