"""
exp_multiclass.py -- G12: late fusion is not limited to binary labels.

K-way softmax fusion head over independently-trained subcircuit features, on:
  * Iris (3 classes)
  * MNIST 0/1/2 (3 classes, PCA-4)
Compares independent late fusion (ours) vs best tuned classical (SVM/MLP) for context.

Outputs: results/multiclass_results.json
Run:  python -m experiments.exp_multiclass
"""
import json
import os
import numpy as np
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score

from lfqml import datasets
from lfqml.circuits import QNNConfig
from lfqml.train import train_fusion_multiclass

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS, exist_ok=True)

SEEDS = [0, 1, 2, 3, 4]


def classical_best(Xtr, ytr, Xte, yte):
    """Tuned classical anchor with NO test leakage (audit fix M1): CV model selection
    on the TRAIN split only, refit on train, score on the SAME test split."""
    cands = [SVC(kernel="rbf", C=5, gamma="scale"),
             MLPClassifier((32, 16), max_iter=1500, random_state=0)]
    cv = [cross_val_score(m, Xtr, ytr, cv=4).mean() for m in cands]
    best = cands[int(np.argmax(cv))]
    best.fit(Xtr, ytr)
    return best.score(Xte, yte)


def run():
    # raw loaders; preprocessing (standardize/PCA for MNIST, angle scaling) fit on TRAIN only
    specs = [
        ("iris3", 3, lambda s: datasets.iris_multiclass_raw(), False),
        ("mnist012", 3, lambda s: datasets.mnist_multiclass_raw((0, 1, 2), 110, s), True),
    ]
    cfg = QNNConfig(n_A=2, n_B=2, depth=2, n_cuts=2, trainable_coupling=False)
    rows = []
    for name, K, load, is_image in specs:
        for seed in SEEDS:
            X, y = load(seed)
            rng = np.random.default_rng(seed)
            idx = rng.permutation(len(X)); ntr = int(0.7 * len(X))
            tr, te = idx[:ntr], idx[ntr:]
            Xtr, Xte = datasets.preprocess_split(X[tr], X[te],
                                                 n_features=4 if is_image else None,
                                                 standardize=is_image, seed=seed)
            tf = train_fusion_multiclass(cfg, Xtr, y[tr], K, seed=seed, hidden=12, maxiter=75)
            acc = float(np.mean(tf.predict(Xte) == y[te]))
            acc_cl = classical_best(Xtr, y[tr], Xte, y[te])
            rows.append({"dataset": name, "n_classes": K, "seed": seed,
                         "acc_fusion": acc, "acc_classical": acc_cl})
            print(f"{name:10s} seed={seed} K={K}  fusion={acc:.3f} classical={acc_cl:.3f}",
                  flush=True)

    with open(os.path.join(RESULTS, "multiclass_results.json"), "w") as f:
        json.dump(rows, f, indent=2)
    print("\n=== means ===")
    for name, K, _, _ in specs:
        rs = [r for r in rows if r["dataset"] == name]
        m = lambda k: np.mean([r[k] for r in rs])
        print(f"  {name:10s} (K={K}) fusion={m('acc_fusion'):.3f} classical={m('acc_classical'):.3f}")
    print("DONE -> results/multiclass_results.json")


if __name__ == "__main__":
    run()
