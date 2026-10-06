"""
exp_scaleup14.py -- QMI revision item E4: extend the exact accuracy reference one step past n=12.
Reviewer R2-W3: the n=16/20 fusion cells have no uncut/reconstruction reference. We add n=14 (7+7)
with the same computation as exp_scaleup / exp_scaleup_topup12 (alpha=0.3, 70/30 split, B0
maxiter 35, fusion maxiter 45, hidden 8+HALF), at k in {1, 4, 7}, seeds 0..2, parallel over
(k, seed) with 4 workers (OMP_NUM_THREADS=1). Each (k, seed) row is written as soon as it
finishes, so partial results are usable. Reference training at n=14 uses finite-difference
gradients over 16384-dim statevectors; expect hours per row.
Run:  OMP_NUM_THREADS=1 python -m experiments.exp_scaleup14
Output: results/scaleup_n14.json
"""
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

from lfqml.circuits import QNNConfig, prepare, full_features
from lfqml import datasets
from lfqml.train import train_qnn, train_fusion_independent
from lfqml.models import qsim_sigmoid
from lfqml.metrics import classification_metrics

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")
OUT = os.path.join(RESULTS, "scaleup_n14.json")
ALPHA = 0.3
HALF, KS, SEEDS, B0_ITER, FU_ITER, NSAMP = 7, [1, 4, 7], [0, 1, 2], 35, 45, 200


def one(job):
    k, seed = job
    t0 = time.time()
    cfg = QNNConfig(n_A=HALF, n_B=HALF, depth=2, n_cuts=k, trainable_coupling=True)
    X, y = datasets.synthetic_partition(ALPHA, HALF, HALF, NSAMP, seed=seed)
    ntr = int(0.7 * NSAMP)
    Xtr, ytr, Xte, yte = X[:ntr], y[:ntr], X[ntr:], y[ntr:]
    tf = train_fusion_independent(cfg, Xtr, ytr, seed=seed, hidden=8 + HALF, maxiter=FU_ITER)
    acc_fu = classification_metrics(yte, tf.predict_proba(Xte))["accuracy"]
    t1 = time.time()
    tq = train_qnn(cfg, Xtr, ytr, seed=seed, maxiter=B0_ITER)
    F0 = np.array([full_features(prepare(cfg, tq.params, x), tq.observables) for x in Xte])
    acc_b0 = classification_metrics(yte, qsim_sigmoid(F0 @ tq.head_w + tq.head_b))["accuracy"]
    return {"qubits": 2 * HALF, "n_cuts": k, "seed": seed, "acc_fusion": float(acc_fu),
            "acc_B0": float(acc_b0), "recon_overhead_9k": float(9 ** k),
            "fusion_time_s": t1 - t0, "b0_time_s": time.time() - t1,
            "b0_iters": int(getattr(tq, "history", {}).get("nit", B0_ITER)) if hasattr(tq, "history") else B0_ITER}


def run():
    rows = json.load(open(OUT)) if os.path.exists(OUT) else []
    done = {(r["n_cuts"], r["seed"]) for r in rows}
    jobs = [(k, s) for k in KS for s in SEEDS if (k, s) not in done]
    print(f"{len(jobs)} jobs (n={2*HALF})", flush=True)
    with ProcessPoolExecutor(max_workers=4) as ex:
        futs = {ex.submit(one, j): j for j in jobs}
        for f in as_completed(futs):
            r = f.result(); rows.append(r)
            json.dump(rows, open(OUT, "w"), indent=2)
            print(f"n=14 k={r['n_cuts']} seed={r['seed']}: fusion={r['acc_fusion']:.3f} B0={r['acc_B0']:.3f} "
                  f"(fusion {r['fusion_time_s']/60:.1f} min, B0 {r['b0_time_s']/60:.1f} min)", flush=True)
    print("DONE -> results/scaleup_n14.json")


if __name__ == "__main__":
    run()
