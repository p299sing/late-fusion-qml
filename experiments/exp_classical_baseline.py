"""
exp_classical_baseline.py -- the missing three-way anchor: PURE CLASSICAL ML.

Answers the 'why not just classical ML?' question by running tuned classical
models (SVM-RBF, MLP) on the RAW features of the same datasets, so we can put
   classical ML   vs   quantum (B0/reconstruction)   vs   late fusion
on the same axis. This is the Bowles (arXiv:2403.07059) fairness anchor.

Run:  python -m experiments.exp_classical_baseline
"""
import numpy as np
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV

from lfqml import datasets


def best_classical(X, y, seed=0):
    """Tuned SVM-RBF and MLP; return the better 5-fold CV accuracy."""
    svm = GridSearchCV(SVC(kernel="rbf"),
                       {"C": [0.5, 1, 5, 10], "gamma": ["scale", 0.5, 1, 2]}, cv=5)
    svm.fit(X, y)
    mlp = MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=2000, random_state=seed)
    mlp_acc = cross_val_score(mlp, X, y, cv=5).mean()
    return {"svm": svm.best_score_, "mlp": mlp_acc, "best": max(svm.best_score_, mlp_acc)}


def main():
    specs = [
        ("synthetic a=0.5", lambda s: datasets.synthetic_partition(0.5, 2, 2, 300, seed=s)),
        ("synthetic a=1.0", lambda s: datasets.synthetic_partition(1.0, 2, 2, 300, seed=s)),
        ("moons", lambda s: datasets.moons(300, seed=s)),
        ("circles", lambda s: datasets.circles(300, seed=s)),
        ("iris", lambda s: datasets.iris_binary(4, seed=s)),
        ("breast_cancer", lambda s: datasets.breast_cancer(4, seed=s)),
    ]
    print(f"{'dataset':18s}{'SVM':>8}{'MLP':>8}{'best_classical':>16}")
    for name, load in specs:
        accs = [best_classical(*load(s), seed=s) for s in [0, 1, 2]]
        svm = np.mean([a["svm"] for a in accs])
        mlp = np.mean([a["mlp"] for a in accs])
        best = np.mean([a["best"] for a in accs])
        print(f"{name:18s}{svm:8.3f}{mlp:8.3f}{best:16.3f}")


if __name__ == "__main__":
    main()
