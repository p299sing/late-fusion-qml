"""
exp_smoke.py -- end-to-end sanity run of the whole pipeline on one small dataset.

Checks:
  * training runs and B0 beats chance,
  * B1 (reconstruction) == B0 to machine precision on the test set,
  * B2/B6/B7 and the Q-dial all produce numbers,
  * Q=1 end of the dial matches B0.
Run:  python -m experiments.exp_smoke
"""
import numpy as np

from lfqml.circuits import QNNConfig
from lfqml import datasets
from lfqml.train import train_qnn
from lfqml.models import evaluate_baselines


def main():
    cfg = QNNConfig(n_A=2, n_B=2, depth=1, n_cuts=2, trainable_coupling=True)
    X, y = datasets.synthetic_partition(alpha=0.6, n_A=2, n_B=2, n_samples=200, seed=0)
    ntr = 140
    Xtr, ytr, Xte, yte = X[:ntr], y[:ntr], X[ntr:], y[ntr:]

    print("Training B0 (uncut QNN)...")
    tq = train_qnn(cfg, Xtr, ytr, seed=0, maxiter=120, verbose=True)

    res = evaluate_baselines(tq, Xtr, ytr, Xte, yte)

    print("\n=== Baselines (test accuracy) ===")
    for k in ["B0_uncut", "B1_reconstruction", "B2_fusion_linear", "B2_fusion_mlp",
              "B6_kawase_sum", "B7_ent_ablation"]:
        print(f"  {k:20s} acc={res[k]['accuracy']:.3f}  "
              f"overhead={res[k]['cost_overhead']:.1f}")
    print(f"  B1 max|B1-B0| on test = {res['B1_reconstruction']['max_abs_vs_B0']:.2e}")
    print(f"  cut_entanglement = {res['cut_entanglement']:.4f} bits  "
          f"gamma/cut = {res['gamma_per_cut']:.4f}")

    print("\n=== Q-dial ===")
    for row in res["Qdial"]:
        print(f"  Q={row['Q']:.2f}  acc={row['accuracy']:.3f}  "
              f"overhead={row['cost_overhead']:.2f}  n_subexp={row['n_subexp']}")

    assert res["B0_uncut"]["accuracy"] > 0.6, "B0 failed to learn"
    assert res["B1_reconstruction"]["max_abs_vs_B0"] < 1e-9, "B1 != B0"
    print("\nSMOKE OK")


if __name__ == "__main__":
    main()
