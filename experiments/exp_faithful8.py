"""
exp_faithful8.py -- reviewer C7 (task diversity): the image comparison at PCA-8
(double the feature dimension, 4+4 qubit registers) with the leakage-free pipeline.

Same protocol as exp_faithful (train-only preprocessing, train-only CV for the
classical anchor, identical splits) but n_features=8 and QNNConfig(4,4).

Outputs: results/faithful8_results.json
Run:  python -m experiments.exp_faithful8
"""
import json
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score

from lfqml import datasets
from lfqml.circuits import QNNConfig, prepare, full_features
from lfqml.train import train_qnn, train_fusion_independent, train_kawase
from lfqml.models import qsim_sigmoid
from lfqml.metrics import classification_metrics

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")
OUT = os.path.join(RESULTS, "faithful8_results.json")

SEEDS = list(range(10))
SPECS = [("fashion_mnist", (0, 6)), ("mnist", (3, 6))]


def classical_best(Xtr, ytr, Xte, yte):
    cands = [SVC(kernel="rbf", C=5, gamma="scale"),
             MLPClassifier((32, 16), max_iter=1500, random_state=0)]
    cv = [cross_val_score(m, Xtr, ytr, cv=4).mean() for m in cands]
    best = cands[int(np.argmax(cv))]
    best.fit(Xtr, ytr)
    return best.score(Xte, yte)


def one_run(job):
    spec_i, seed = job
    name, classes = SPECS[spec_i]
    raw = (datasets.fashion_mnist_binary_raw if name == "fashion_mnist"
           else datasets.mnist_binary_raw)
    X, y = raw(classes, 130, seed)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X)); ntr = int(0.7 * len(X))
    tr, te = idx[:ntr], idx[ntr:]
    Xtr, Xte = datasets.preprocess_split(X[tr], X[te], n_features=8,
                                         standardize=True, seed=seed)
    ytr, yte = y[tr], y[te]
    cfg = QNNConfig(n_A=4, n_B=4, depth=2, n_cuts=2, trainable_coupling=True)

    tq = train_qnn(cfg, Xtr, ytr, seed=seed, maxiter=55)
    F0 = np.array([full_features(prepare(cfg, tq.params, x), tq.observables) for x in Xte])
    acc_recon = classification_metrics(yte, qsim_sigmoid(F0 @ tq.head_w + tq.head_b))["accuracy"]
    tf = train_fusion_independent(cfg, Xtr, ytr, seed=seed, maxiter=60)
    acc_fusion = classification_metrics(yte, tf.predict_proba(Xte))["accuracy"]
    tk = train_kawase(cfg, Xtr, ytr, seed=seed, maxiter=60)
    acc_kawase = classification_metrics(yte, tk.predict_proba(Xte))["accuracy"]
    acc_classical = classical_best(Xtr, ytr, Xte, yte)
    return {"dataset": name, "seed": seed, "acc_recon": acc_recon,
            "acc_fusion": acc_fusion, "acc_kawase": acc_kawase,
            "acc_classical": acc_classical}


def run():
    jobs = [(i, s) for i in range(len(SPECS)) for s in SEEDS]
    rows = []
    with ProcessPoolExecutor(max_workers=8) as ex:
        for r in ex.map(one_run, jobs):
            rows.append(r)
            print(f"{r['dataset']:14s} seed={r['seed']}  recon={r['acc_recon']:.3f} "
                  f"fusion={r['acc_fusion']:.3f} kawase={r['acc_kawase']:.3f} "
                  f"classical={r['acc_classical']:.3f}", flush=True)
    rows.sort(key=lambda r: (r["dataset"], r["seed"]))
    with open(OUT, "w") as f:
        json.dump(rows, f, indent=2)
    for name, _ in SPECS:
        rs = [r for r in rows if r["dataset"] == name]
        m = lambda k: np.mean([r[k] for r in rs])
        print(f"  {name:14s} recon={m('acc_recon'):.3f} fusion={m('acc_fusion'):.3f} "
              f"kawase={m('acc_kawase'):.3f} classical={m('acc_classical'):.3f}")
    print("DONE -> results/faithful8_results.json")


if __name__ == "__main__":
    run()
