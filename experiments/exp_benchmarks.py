"""
exp_benchmarks.py -- does the primary finding hold on standard QML datasets?

Runs the full baseline ladder on moons / circles / iris / breast-cancer and asks:
does nonlinear late fusion (B2 MLP) match full reconstruction (B1) at linear cost,
and where does the Q-dial knee sit?

Outputs: results/bench_results.json  (+ printed table)
Run:  python -m experiments.exp_benchmarks
"""
import json
import os
import numpy as np

from lfqml.circuits import QNNConfig
from lfqml import datasets
from lfqml.train import train_qnn
from lfqml.models import evaluate_baselines

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS, exist_ok=True)

# (name, raw loader, PCA dim (None = keep), QNNConfig) -- widths match feature dim.
# Preprocessing (PCA + angle scaling) is fit on the TRAIN split only (leakage-free).
SPECS = [
    ("moons",  lambda s: datasets.moons_raw(300, seed=s),  None, QNNConfig(1, 1, depth=2, n_cuts=1)),
    ("circles", lambda s: datasets.circles_raw(300, seed=s), None, QNNConfig(1, 1, depth=2, n_cuts=1)),
    ("iris",   lambda s: datasets.iris_binary_raw(),       None, QNNConfig(2, 2, depth=2, n_cuts=2)),
    ("breast_cancer", lambda s: datasets.breast_cancer_raw(), 4, QNNConfig(2, 2, depth=2, n_cuts=2)),
]
SEEDS = list(range(10))          # 10 seeds for camera-ready error bars


def knee(qdial, tol=0.02):
    acc1 = [q for q in qdial if q["Q"] >= 1.0][0]["accuracy"]
    for q in sorted(qdial, key=lambda r: r["Q"]):
        if q["accuracy"] >= acc1 - tol:
            return q["Q"]
    return 1.0


def one_run(job):
    spec_i, seed = job
    name, load, n_pca, cfg = SPECS[spec_i]
    X, y = load(seed)
    # split 70/30 FIRST, then fit preprocessing on the train split only
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    ntr = int(0.7 * len(X))
    tr, te = idx[:ntr], idx[ntr:]
    Xtr, Xte = datasets.preprocess_split(X[tr], X[te], n_features=n_pca, seed=seed)
    tq = train_qnn(cfg, Xtr, y[tr], seed=seed, maxiter=90)
    res = evaluate_baselines(tq, Xtr, y[tr], Xte, y[te])
    return {
        "dataset": name, "seed": seed, "n_features": Xtr.shape[1],
        "acc_B0": res["B0_uncut"]["accuracy"],
        "acc_B1": res["B1_reconstruction"]["accuracy"],
        "acc_B2_lin": res["B2_fusion_linear"]["accuracy"],
        "acc_B2_mlp": res["B2_fusion_mlp"]["accuracy"],
        "acc_B6": res["B6_kawase_sum"]["accuracy"],
        "acc_B7": res["B7_ent_ablation"]["accuracy"],
        # F1 / AUC for the main-text table (reviewer request)
        "f1_B0": res["B0_uncut"]["f1"], "auc_B0": res["B0_uncut"]["auc"],
        "f1_B2_mlp": res["B2_fusion_mlp"]["f1"], "auc_B2_mlp": res["B2_fusion_mlp"]["auc"],
        "f1_B6": res["B6_kawase_sum"]["f1"], "auc_B6": res["B6_kawase_sum"]["auc"],
        "f1_B7": res["B7_ent_ablation"]["f1"], "auc_B7": res["B7_ent_ablation"]["auc"],
        "cut_ent": res["cut_entanglement"],
        "kneeQ": knee(res["Qdial"]),
        "recon_overhead": res["B1_reconstruction"]["cost_overhead"],
    }


def run():
    from concurrent.futures import ProcessPoolExecutor
    jobs = [(i, s) for i in range(len(SPECS)) for s in SEEDS]
    rows = []
    with ProcessPoolExecutor(max_workers=6) as ex:
        for r in ex.map(one_run, jobs):
            rows.append(r)
            print(f"{r['dataset']:14s} seed={r['seed']}  B0={r['acc_B0']:.3f} "
                  f"B1={r['acc_B1']:.3f} B2mlp={r['acc_B2_mlp']:.3f} "
                  f"B7={r['acc_B7']:.3f} kneeQ={r['kneeQ']:.2f} "
                  f"ent={r['cut_ent']:.3f}", flush=True)
    rows.sort(key=lambda r: (r["dataset"], r["seed"]))

    with open(os.path.join(RESULTS, "bench_results.json"), "w") as f:
        json.dump(rows, f, indent=2)

    # aggregate table
    print("\n=== Benchmark summary (mean over seeds) ===")
    print(f"{'dataset':14s} {'B0':>6} {'B1':>6} {'B2mlp':>6} {'B2lin':>6} "
          f"{'B6':>6} {'B7':>6} {'kneeQ':>6} {'reconOv':>8}")
    for name, _, _, _ in SPECS:
        rs = [r for r in rows if r["dataset"] == name]
        def m(k): return np.mean([r[k] for r in rs])
        print(f"{name:14s} {m('acc_B0'):6.3f} {m('acc_B1'):6.3f} {m('acc_B2_mlp'):6.3f} "
              f"{m('acc_B2_lin'):6.3f} {m('acc_B6'):6.3f} {m('acc_B7'):6.3f} "
              f"{m('kneeQ'):6.2f} {m('recon_overhead'):8.1f}")
    print("DONE -> results/bench_results.json")


if __name__ == "__main__":
    run()
