"""
exp_shots.py -- Tier 1: fusion is MORE ROBUST than reconstruction under finite shots.

Real devices estimate each expectation from a finite number of measurement runs
("shots"), which injects statistical noise ~ 1/sqrt(shots).  Reconstruction combines
subcircuit results with a SIGNED quasiprobability sum whose variance is amplified by
the gamma^2 factor (gamma = per-cut 1-norm, multiplicative over cuts).  So under finite
shots:
    * a reconstructed expectation has noise std ~ gamma_total / sqrt(shots)
    * a late-fusion (per-subcircuit) expectation has noise std ~ 1 / sqrt(shots)
i.e. reconstruction is not just exponentially more EXPENSIVE, it is also noisier at a
fixed shot budget.  We sweep shots and compare test accuracy of the two readouts.

Outputs: results/shots_results.json + results/shots_robustness.png
Run:  python -m experiments.exp_shots
"""
import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from lfqml.circuits import QNNConfig, prepare, full_features
from lfqml import datasets, cutting
from lfqml.train import train_qnn
from lfqml.fusion import MLPHead
from lfqml.models import qsim_sigmoid
from lfqml.metrics import classification_metrics

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS, exist_ok=True)

SHOTS = [32, 100, 300, 1000, 3000, 10000, 30000]
NOISE_TRIALS = 30            # average over independent shot-noise realisations
SEEDS = list(range(10))


def run():
    cfg = QNNConfig(n_A=2, n_B=2, depth=2, n_cuts=2)
    rows = []
    for seed in SEEDS:
        # split FIRST, then fit PCA + angle scaling on the train split only
        X, y = datasets.breast_cancer_raw()
        rng = np.random.default_rng(seed)
        idx = rng.permutation(len(X)); ntr = int(0.7 * len(X))
        tr, te = idx[:ntr], idx[ntr:]
        Xtr, Xte = datasets.preprocess_split(X[tr], X[te], n_features=4, seed=seed)
        tq = train_qnn(cfg, Xtr, y[tr], seed=seed, maxiter=90)

        pre_tr = [prepare(cfg, tq.params, x) for x in Xtr]
        pre_te = [prepare(cfg, tq.params, x) for x in Xte]
        # exact features
        Frec_te = np.array([cutting.reconstruct_features(cutting.build_cut(pc),
                            tq.observables, Q=1.0) for pc in pre_te])
        S_tr = np.array([cutting.subcircuit_raw_features(pc) for pc in pre_tr])
        S_te = np.array([cutting.subcircuit_raw_features(pc) for pc in pre_te])
        fusion_head = MLPHead(hidden=8, seed=seed).fit(S_tr, y[tr])

        cd0 = cutting.build_cut(pre_te[0])
        gamma_total = cd0.gamma_per_cut ** cd0.n_cuts    # multiplicative over cuts

        for shots in SHOTS:
            noise_rng = np.random.default_rng(1000 + seed)
            acc_rec, acc_fus = [], []
            for _ in range(NOISE_TRIALS):
                # reconstruction: variance amplified by gamma_total
                Nrec = Frec_te + noise_rng.normal(0, gamma_total / np.sqrt(shots), Frec_te.shape)
                p_rec = qsim_sigmoid(Nrec @ tq.head_w + tq.head_b)
                acc_rec.append(classification_metrics(y[te], p_rec)["accuracy"])
                # fusion: plain 1/sqrt(shots) noise
                Nfus = S_te + noise_rng.normal(0, 1.0 / np.sqrt(shots), S_te.shape)
                acc_fus.append(classification_metrics(y[te], fusion_head.predict_proba(Nfus))["accuracy"])
            rows.append({"seed": seed, "shots": shots, "gamma_total": gamma_total,
                         "acc_recon": float(np.mean(acc_rec)),
                         "acc_fusion": float(np.mean(acc_fus))})
            print(f"seed={seed} shots={shots:6d}  recon={rows[-1]['acc_recon']:.3f} "
                  f"fusion={rows[-1]['acc_fusion']:.3f}  (gamma_tot={gamma_total:.2f})",
                  flush=True)

    with open(os.path.join(RESULTS, "shots_results.json"), "w") as f:
        json.dump(rows, f, indent=2)
    make_plot(rows)
    print("DONE -> results/shots_results.json + shots_robustness.png")


def make_plot(rows):
    shots = sorted(set(r["shots"] for r in rows))
    rec = [np.mean([r["acc_recon"] for r in rows if r["shots"] == s]) for s in shots]
    fus = [np.mean([r["acc_fusion"] for r in rows if r["shots"] == s]) for s in shots]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(shots, rec, marker="o", label="reconstruction (gamma^2 noise amplification)")
    ax.plot(shots, fus, marker="s", label="late fusion (no amplification)")
    ax.set_xscale("log")
    ax.set_xlabel("shots per expectation (log)")
    ax.set_ylabel("test accuracy")
    ax.set_title("Fusion is more shot-noise robust than reconstruction")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "shots_robustness.png"), dpi=130)


if __name__ == "__main__":
    run()
