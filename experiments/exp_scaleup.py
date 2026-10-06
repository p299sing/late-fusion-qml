"""
exp_scaleup.py -- W2: fusion keeps training at scales where reconstruction is infeasible.

Reviewer attack: "the exponential-vs-linear thesis is shown only where the exponential is
affordable (k<=3, n<=8)."  Answer: late fusion never builds the joint state -- each
subcircuit is HALF width -- so we can train fusion at n=12..20 qubits and k=6..10 cuts,
where reconstruction's physical sampling overhead 9^k reaches 5*10^5 .. 3.5*10^9 (far
beyond any realistic shot budget), and check accuracy against the exact uncut model (B0)
wherever B0 is still simulable:

  n=12 (6+6), k in {1,..,6} : fusion vs B0 (statevector 4096-dim, still trainable)
  n=16 (8+8), k=8           : fusion only -- training the uncut reference exceeded 35
  n=20 (10+10), k=10          CPU-hours without converging at n=16 (finite-difference
                              gradients over 65536-dim statevectors), which IS the
                              point: even the simulator-side reference outgrows the
                              budget, while fusion's half-width subcircuits train in
                              minutes. Accuracy anchoring comes from n=12 (fusion==B0).

Task: synthetic_partition(alpha=0.3) -- moderately local, fusion's valid regime
(the boundary experiments cover where fusion must fail).

Writes results incrementally after every config. Outputs: results/scaleup_results.json
Run:  python -m experiments.exp_scaleup
"""
import json
import os
import numpy as np

from lfqml.circuits import QNNConfig, prepare, full_features
from lfqml import datasets
from lfqml.train import train_qnn, train_fusion_independent
from lfqml.models import qsim_sigmoid
from lfqml.metrics import classification_metrics

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS, exist_ok=True)
OUT = os.path.join(RESULTS, "scaleup_results.json")

ALPHA = 0.3
# (half_width, cuts, seeds, do_B0, b0_maxiter, fusion_maxiter, n_samples)
CONFIGS = (
    [(6, k, [0, 1, 2], True, 35, 45, 200) for k in [1, 2, 3, 4, 5, 6]]
    + [(8, 8, [0, 1, 2, 3, 4], False, 0, 45, 160)]
    + [(10, 10, [0, 1, 2, 3, 4], False, 0, 45, 160)]
)


def _save(rows):
    with open(OUT, "w") as f:
        json.dump(rows, f, indent=2)


def run():
    # resume-aware: skip (qubits, n_cuts) configs already in the JSON
    rows = json.load(open(OUT)) if os.path.exists(OUT) else []
    done = {(r["qubits"], r["n_cuts"]) for r in rows}
    for half, k, seeds, do_b0, b0_iter, fu_iter, nsamp in CONFIGS:
        if (2 * half, k) in done:
            print(f"n={2*half} k={k}: already done, skipping", flush=True)
            continue
        n = 2 * half
        cfg = QNNConfig(n_A=half, n_B=half, depth=2, n_cuts=k, trainable_coupling=True)
        accs_b0, accs_fu = [], []
        for seed in seeds:
            X, y = datasets.synthetic_partition(ALPHA, half, half, nsamp, seed=seed)
            ntr = int(0.7 * nsamp)
            Xtr, ytr, Xte, yte = X[:ntr], y[:ntr], X[ntr:], y[ntr:]

            tf = train_fusion_independent(cfg, Xtr, ytr, seed=seed,
                                          hidden=8 + half, maxiter=fu_iter)
            accs_fu.append(classification_metrics(yte, tf.predict_proba(Xte))["accuracy"])

            if do_b0:
                tq = train_qnn(cfg, Xtr, ytr, seed=seed, maxiter=b0_iter)
                F0 = np.array([full_features(prepare(cfg, tq.params, x), tq.observables)
                               for x in Xte])
                accs_b0.append(classification_metrics(
                    yte, qsim_sigmoid(F0 @ tq.head_w + tq.head_b))["accuracy"])

        row = {"qubits": n, "n_cuts": k, "n_seeds": len(seeds),
               "acc_fusion": float(np.mean(accs_fu)),
               "acc_fusion_std": float(np.std(accs_fu)),
               "acc_B0": float(np.mean(accs_b0)) if accs_b0 else None,
               "acc_B0_std": float(np.std(accs_b0)) if accs_b0 else None,
               "recon_overhead_9k": float(9 ** k)}
        rows.append(row); _save(rows)
        b0s = f"{row['acc_B0']:.3f}" if accs_b0 else "  --  (infeasible)"
        print(f"n={n:2d} k={k:2d}: fusion={row['acc_fusion']:.3f}+/-"
              f"{row['acc_fusion_std']:.3f}  B0={b0s}  9^k={9**k:.1e}", flush=True)
    print("DONE -> results/scaleup_results.json")


if __name__ == "__main__":
    run()
