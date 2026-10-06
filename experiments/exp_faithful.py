"""
exp_faithful.py -- G4 (harder image data) + G7 (faithful prior-art baselines).

Compares on Fashion-MNIST, MNIST (PCA-4 binary) and a hard synthetic task:
  * reconstruction (B0)                 -- upper bound
  * independent late fusion (ours)      -- trained subcircuits + MLP fusion head
  * faithful Kawase (2312.13650)        -- subcircuits + parameter-free expectation sum
  * best tuned classical (SVM/MLP)      -- the Bowles anchor

Outputs: results/faithful_results.json
Run:  python -m experiments.exp_faithful
"""
import json
import os
import numpy as np
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score

from lfqml import qsim, datasets
from lfqml.circuits import QNNConfig, prepare, full_features
from lfqml.train import train_qnn, train_fusion_independent, train_kawase
from lfqml.models import qsim_sigmoid
from lfqml.metrics import classification_metrics

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS, exist_ok=True)

SEEDS = list(range(10))


def classical_best(Xtr, ytr, Xte, yte):
    """Tuned classical anchor with NO test leakage (audit fix M1): model selection by
    CV on the TRAIN split only, then refit on train and score on the SAME held-out
    test split the quantum models use."""
    cands = [SVC(kernel="rbf", C=5, gamma="scale"),
             MLPClassifier((32, 16), max_iter=1500, random_state=0)]
    cv = [cross_val_score(m, Xtr, ytr, cv=4).mean() for m in cands]
    best = cands[int(np.argmax(cv))]
    best.fit(Xtr, ytr)
    return best.score(Xte, yte)


def run():
    # image tasks: raw pixels; standardize + PCA-4 + angle scaling fit on TRAIN only
    specs = [
        ("fashion_mnist", lambda s: datasets.fashion_mnist_binary_raw((0, 6), 130, s), True),
        ("mnist", lambda s: datasets.mnist_binary_raw((3, 6), 130, s), True),
        ("synthetic_a0.75", lambda s: datasets.synthetic_partition(0.75, 2, 2, 240, seed=s), False),
    ]
    cfg = QNNConfig(n_A=2, n_B=2, depth=2, n_cuts=2, trainable_coupling=True)
    rows = []
    for name, load, is_image in specs:
        for seed in SEEDS:
            X, y = load(seed)
            rng = np.random.default_rng(seed)
            idx = rng.permutation(len(X)); ntr = int(0.7 * len(X))
            tr, te = idx[:ntr], idx[ntr:]
            if is_image:
                Xtr, Xte = datasets.preprocess_split(X[tr], X[te], n_features=4,
                                                     standardize=True, seed=seed)
            else:
                Xtr, Xte = X[tr], X[te]
            ytr, yte = y[tr], y[te]

            tq = train_qnn(cfg, Xtr, ytr, seed=seed, maxiter=55)
            F0 = np.array([full_features(prepare(cfg, tq.params, x), tq.observables) for x in Xte])
            acc_recon = classification_metrics(yte, qsim_sigmoid(F0 @ tq.head_w + tq.head_b))["accuracy"]

            tf = train_fusion_independent(cfg, Xtr, ytr, seed=seed, maxiter=60)
            acc_fusion = classification_metrics(yte, tf.predict_proba(Xte))["accuracy"]

            tk = train_kawase(cfg, Xtr, ytr, seed=seed, maxiter=60)
            acc_kawase = classification_metrics(yte, tk.predict_proba(Xte))["accuracy"]

            acc_classical = classical_best(Xtr, ytr, Xte, yte)

            rows.append({"dataset": name, "seed": seed, "acc_recon": acc_recon,
                         "acc_fusion": acc_fusion, "acc_kawase": acc_kawase,
                         "acc_classical": acc_classical})
            print(f"{name:16s} seed={seed}  recon={acc_recon:.3f} fusion={acc_fusion:.3f} "
                  f"kawase={acc_kawase:.3f} classical={acc_classical:.3f}", flush=True)

    with open(os.path.join(RESULTS, "faithful_results.json"), "w") as f:
        json.dump(rows, f, indent=2)
    print("\n=== means over seeds ===")
    for name, _, _ in specs:
        rs = [r for r in rows if r["dataset"] == name]
        m = lambda k: np.mean([r[k] for r in rs])
        print(f"  {name:16s} recon={m('acc_recon'):.3f} fusion={m('acc_fusion'):.3f} "
              f"kawase={m('acc_kawase'):.3f} classical={m('acc_classical'):.3f}")
    print("DONE -> results/faithful_results.json")


if __name__ == "__main__":
    run()
