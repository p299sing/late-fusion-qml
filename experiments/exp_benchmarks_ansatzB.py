"""
exp_benchmarks_ansatzB.py -- ansatz-robustness check for the benchmark ladder.

Same four datasets, split (70/30, preprocessing fit on train only), seeds (0..9)
and baseline ladder (models.evaluate_baselines) as exp_benchmarks.py, but with
ansatz "B" (RX,RY + CNOT ladder) and a FIXED maximally entangling coupling
phi = pi/2 (trainable_coupling=False) -> physical QPD overhead 9 per cut.

Outputs: results/bench_ansatzB_results.json
Run:  OMP_NUM_THREADS=1 python -m experiments.exp_benchmarks_ansatzB
"""
import json
import os
import time
import numpy as np

from lfqml.circuits import QNNConfig
from lfqml import datasets
from lfqml.train import train_qnn
from lfqml.models import evaluate_baselines
from experiments.exp_benchmarks import knee

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS, exist_ok=True)
OUT = os.path.join(RESULTS, "bench_ansatzB_results.json")

B_KW = dict(ansatz="B", trainable_coupling=False, coupling_init=np.pi / 2)
SPECS = [
    ("moons",   lambda s: datasets.moons_raw(300, seed=s),   None, QNNConfig(1, 1, depth=2, n_cuts=1, **B_KW)),
    ("circles", lambda s: datasets.circles_raw(300, seed=s), None, QNNConfig(1, 1, depth=2, n_cuts=1, **B_KW)),
    ("iris",    lambda s: datasets.iris_binary_raw(),        None, QNNConfig(2, 2, depth=2, n_cuts=2, **B_KW)),
    ("breast_cancer", lambda s: datasets.breast_cancer_raw(), 4,  QNNConfig(2, 2, depth=2, n_cuts=2, **B_KW)),
]
SEEDS = list(range(10))
PHYS_OVERHEAD_PER_CUT = float((1 + 2 * abs(np.sin(np.pi / 2))) ** 2)   # = 9


def one_run(job):
    spec_i, seed = job
    t0 = time.time()
    name, load, n_pca, cfg = SPECS[spec_i]
    X, y = load(seed)
    rng = np.random.default_rng(seed)            # identical split to exp_benchmarks
    idx = rng.permutation(len(X))
    ntr = int(0.7 * len(X))
    tr, te = idx[:ntr], idx[ntr:]
    Xtr, Xte = datasets.preprocess_split(X[tr], X[te], n_features=n_pca, seed=seed)
    tq = train_qnn(cfg, Xtr, y[tr], seed=seed, maxiter=90)
    res = evaluate_baselines(tq, Xtr, y[tr], Xte, y[te])
    return {
        "dataset": name, "seed": seed, "ansatz": "B", "phi": float(cfg.coupling_init),
        "n_features": Xtr.shape[1],
        "acc_B0": res["B0_uncut"]["accuracy"],
        "acc_B1": res["B1_reconstruction"]["accuracy"],
        "acc_B2_lin": res["B2_fusion_linear"]["accuracy"],
        "acc_B2_mlp": res["B2_fusion_mlp"]["accuracy"],
        "acc_B6": res["B6_kawase_sum"]["accuracy"],
        "acc_B7": res["B7_ent_ablation"]["accuracy"],
        "f1_B0": res["B0_uncut"]["f1"], "auc_B0": res["B0_uncut"]["auc"],
        "f1_B2_mlp": res["B2_fusion_mlp"]["f1"], "auc_B2_mlp": res["B2_fusion_mlp"]["auc"],
        "cut_ent": res["cut_entanglement"],
        "kneeQ": knee(res["Qdial"]),
        "recon_max_abs_vs_B0": res["B1_reconstruction"]["max_abs_vs_B0"],
        "recon_overhead_schmidt": res["B1_reconstruction"]["cost_overhead"],
        "phys_overhead_per_cut": PHYS_OVERHEAD_PER_CUT,
        "phys_overhead": PHYS_OVERHEAD_PER_CUT ** cfg.n_cuts,
        "train_loss_B0": tq.history["loss"],
        "runtime_s": time.time() - t0,
    }


def run():
    from concurrent.futures import ProcessPoolExecutor
    t0 = time.time()
    jobs = [(i, s) for i in range(len(SPECS)) for s in SEEDS]
    rows = []
    with ProcessPoolExecutor(max_workers=6) as ex:
        for r in ex.map(one_run, jobs):
            rows.append(r)
            print(f"{r['dataset']:14s} seed={r['seed']}  B0={r['acc_B0']:.3f} "
                  f"B1={r['acc_B1']:.3f} B2mlp={r['acc_B2_mlp']:.3f} B6={r['acc_B6']:.3f} "
                  f"B7={r['acc_B7']:.3f} kneeQ={r['kneeQ']:.2f} ent={r['cut_ent']:.3f} "
                  f"recon_err={r['recon_max_abs_vs_B0']:.1e} [{r['runtime_s']:.0f}s]", flush=True)
    rows.sort(key=lambda r: (r["dataset"], r["seed"]))
    with open(OUT, "w") as f:
        json.dump(rows, f, indent=2)

    print("\n=== ansatz B benchmark summary (mean over seeds) ===")
    print(f"{'dataset':14s} {'B0':>6} {'B1':>6} {'B2mlp':>6} {'B2lin':>6} {'B6':>6} {'B7':>6} "
          f"{'kneeQ':>6} {'S_A':>6} {'physOv':>7}")
    for name, _, _, _ in SPECS:
        rs = [r for r in rows if r["dataset"] == name]
        def m(k): return np.mean([r[k] for r in rs])
        print(f"{name:14s} {m('acc_B0'):6.3f} {m('acc_B1'):6.3f} {m('acc_B2_mlp'):6.3f} "
              f"{m('acc_B2_lin'):6.3f} {m('acc_B6'):6.3f} {m('acc_B7'):6.3f} "
              f"{m('kneeQ'):6.2f} {m('cut_ent'):6.3f} {m('phys_overhead'):7.1f}")
    print(f"wall time: {time.time() - t0:.0f}s")
    print(f"DONE -> {OUT}")
    try:
        from experiments.analyze_ansatzB import analyze_bench
        analyze_bench()
    except Exception as e:
        print("analysis skipped:", e)


if __name__ == "__main__":
    run()
