"""
exp_scale_oracle.py -- reviewer C2a: an empirical accuracy ceiling for the scale-up tasks.

At n=16/20 the exact quantum reference is untrainable, so fusion's 0.78-0.83 is
unanchored. Here we estimate the achievable ceiling with high-capacity CLASSICAL
models trained on the RAW synthetic features at large sample size -- an empirical
stand-in for the Bayes rate of the data-generating process (the task is classical
by construction, so a strong classical learner upper-bounds what any pipeline can
extract from these features).

Outputs: results/scale_oracle_results.json
Run:  python -m experiments.exp_scale_oracle
"""
import json
import os
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.neural_network import MLPClassifier

from lfqml import datasets

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")
OUT = os.path.join(RESULTS, "scale_oracle_results.json")

ALPHA = 0.3
N_BIG = 10000
SEEDS = [0, 1, 2, 3, 4]


def run():
    rows = []
    for half in [8, 10]:                       # n = 16, 20
        for seed in SEEDS:
            X, y = datasets.synthetic_partition(ALPHA, half, half, N_BIG, seed=seed)
            ntr = int(0.7 * N_BIG)
            Xtr, ytr, Xte, yte = X[:ntr], y[:ntr], X[ntr:], y[ntr:]
            accs = {}
            gbt = HistGradientBoostingClassifier(random_state=seed).fit(Xtr, ytr)
            accs["gbt"] = float(gbt.score(Xte, yte))
            mlp = MLPClassifier((128, 64), max_iter=800, random_state=seed).fit(Xtr, ytr)
            accs["mlp"] = float(mlp.score(Xte, yte))
            # same-sample-size classical (the 160-sample regime fusion actually sees)
            Xs, ys = datasets.synthetic_partition(ALPHA, half, half, 160, seed=seed)
            m2 = MLPClassifier((32, 16), max_iter=1500, random_state=seed).fit(Xs[:112], ys[:112])
            accs["mlp_smallN"] = float(m2.score(Xs[112:], ys[112:]))
            rows.append({"qubits": 2 * half, "seed": seed, **accs})
            print(f"n={2*half} seed={seed}  gbt={accs['gbt']:.3f} mlp={accs['mlp']:.3f} "
                  f"smallN={accs['mlp_smallN']:.3f}", flush=True)
    with open(OUT, "w") as f:
        json.dump(rows, f, indent=2)
    for n in [16, 20]:
        rs = [r for r in rows if r["qubits"] == n]
        for k in ["gbt", "mlp", "mlp_smallN"]:
            v = [r[k] for r in rs]
            print(f"n={n} {k}: {np.mean(v):.3f}±{np.std(v):.3f}")
    print("DONE -> results/scale_oracle_results.json")


if __name__ == "__main__":
    run()
