"""
exp_untrained_diag.py -- reviewer question W6 (diagnostic circularity): does the
cut entanglement of the UNTRAINED coupled circuit already predict the Q-dial knee,
so the diagnostic does not require training the coupled model first?

For every (alpha, seed) run of the dense-alpha grid, rebuild the coupled circuit
at its INITIAL parameters (init_params(cfg, seed) -- exactly where train_qnn
starts) and compute S_A over the same test split; correlate with the knee stored
in dense_alpha_results.json (0.02-absolute definition, kneeQ_abs).

No training involved -- runs in seconds. Outputs: results/untrained_diag_results.json
Run:  python -m experiments.exp_untrained_diag
"""
import json
import os
import numpy as np
from scipy.stats import spearmanr

from lfqml import datasets
from lfqml.circuits import QNNConfig, init_params, prepare
from lfqml.metrics import mean_cut_entanglement

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")
OUT = os.path.join(RESULTS, "untrained_diag_results.json")
N_SAMPLES, N_TRAIN = 300, 180


def run():
    dense = json.load(open(os.path.join(RESULTS, "dense_alpha_results.json")))
    cfg = QNNConfig(n_A=2, n_B=2, depth=2, n_cuts=2, trainable_coupling=True)
    rows = []
    for r in dense:
        alpha, seed = r["alpha"], r["seed"]
        X, y = datasets.synthetic_partition(alpha=alpha, n_A=2, n_B=2,
                                            n_samples=N_SAMPLES, seed=seed)
        Xte = X[N_TRAIN:]
        theta0 = init_params(cfg, seed=seed)
        ent0 = mean_cut_entanglement(cfg, [prepare(cfg, theta0, x) for x in Xte])
        rows.append({"alpha": alpha, "seed": seed, "cut_ent_untrained": ent0,
                     "cut_ent_trained": r["cut_ent"],
                     "kneeQ_abs": r["kneeQ_abs"], "kneeQ": r["kneeQ"],
                     "delta_mlp": r["delta_mlp"]})
    with open(OUT, "w") as f:
        json.dump(rows, f, indent=2)

    ent0 = np.array([r["cut_ent_untrained"] for r in rows])
    entT = np.array([r["cut_ent_trained"] for r in rows])
    rng = np.random.default_rng(0)

    def boot(a, b):
        stats = []
        for _ in range(2000):
            i = rng.integers(0, len(a), len(a))
            stats.append(spearmanr(a[i], b[i]).statistic)
        lo, hi = np.percentile(stats, [2.5, 97.5])
        return spearmanr(a, b).statistic, lo, hi

    for name, x in [("knee_abs", np.array([r["kneeQ_abs"] for r in rows])),
                    ("gap", np.array([r["delta_mlp"] for r in rows]))]:
        r0, lo0, hi0 = boot(ent0, x)
        print(f"UNTRAINED S_A ~ {name}: spearman={r0:.3f}  CI [{lo0:.3f}, {hi0:.3f}]  n={len(rows)}")
    rt, lot, hit = boot(ent0, entT)
    print(f"untrained vs trained S_A: spearman={rt:.3f}  CI [{lot:.3f}, {hit:.3f}]")
    print("DONE -> results/untrained_diag_results.json")


if __name__ == "__main__":
    run()
