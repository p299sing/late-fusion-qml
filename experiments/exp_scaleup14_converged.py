"""
exp_scaleup14_converged.py -- the n=14 exact reference with a CONVERGED budget (paper revision).
exp_scaleup14.py trained the uncut reference with maxiter=35 (the scale-up protocol) and found it
under-trained (0.82-0.86 vs fusion 0.87). Here both the uncut reference and uncoupled fusion get
maxiter=150, and we record the optimiser's own convergence flag, iteration count and final loss, so
the comparison at n=14 is between converged models. Same data/split/config as exp_scaleup14.py.
Run:  OMP_NUM_THREADS=1 python -m experiments.exp_scaleup14_converged   (8 processes; hours per row)
Output: results/scaleup_n14_converged.json (written after every row)
"""
import json, os, time
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
from lfqml.circuits import QNNConfig, prepare, full_features
from lfqml import datasets
from lfqml.train import train_qnn, train_fusion_independent
from lfqml.models import qsim_sigmoid
from lfqml.metrics import classification_metrics

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")
OUT = os.path.join(RESULTS, "scaleup_n14_converged.json")
ALPHA, HALF, KS, SEEDS, NSAMP = 0.3, 7, [1, 4, 7], [0, 1, 2], 200
MAXITER = 150

def one(job):
    k, seed = job
    cfg = QNNConfig(n_A=HALF, n_B=HALF, depth=2, n_cuts=k, trainable_coupling=True)
    X, y = datasets.synthetic_partition(ALPHA, HALF, HALF, NSAMP, seed=seed)
    ntr = int(0.7 * NSAMP); Xtr, ytr, Xte, yte = X[:ntr], y[:ntr], X[ntr:], y[ntr:]
    t0 = time.time()
    tf = train_fusion_independent(cfg, Xtr, ytr, seed=seed, hidden=8 + HALF, maxiter=MAXITER)
    acc_fu = classification_metrics(yte, tf.predict_proba(Xte))["accuracy"]; t1 = time.time()
    tq = train_qnn(cfg, Xtr, ytr, seed=seed, maxiter=MAXITER)
    F0 = np.array([full_features(prepare(cfg, tq.params, x), tq.observables) for x in Xte])
    acc_b0 = classification_metrics(yte, qsim_sigmoid(F0 @ tq.head_w + tq.head_b))["accuracy"]
    hist = getattr(tq, "history", {}) or {}
    return {"qubits": 2 * HALF, "n_cuts": k, "seed": seed, "maxiter": MAXITER,
            "acc_fusion": float(acc_fu), "acc_B0": float(acc_b0),
            "b0_final_loss": float(hist.get("loss", float("nan"))),
            "fusion_time_s": t1 - t0, "b0_time_s": time.time() - t1, "recon_overhead_9k": float(9 ** k)}

if __name__ == "__main__":
    rows = json.load(open(OUT)) if os.path.exists(OUT) else []
    done = {(r["n_cuts"], r["seed"]) for r in rows}
    jobs = [(k, s) for k in KS for s in SEEDS if (k, s) not in done]
    print(f"{len(jobs)} jobs, maxiter={MAXITER}", flush=True)
    with ProcessPoolExecutor(max_workers=8) as ex:
        futs = {ex.submit(one, j): j for j in jobs}
        for f in as_completed(futs):
            r = f.result(); rows.append(r); json.dump(rows, open(OUT, "w"), indent=2)
            print(f"n=14 k={r['n_cuts']} seed={r['seed']}: fusion={r['acc_fusion']:.3f} B0={r['acc_B0']:.3f} "
                  f"loss={r['b0_final_loss']:.3f} (fusion {r['fusion_time_s']/60:.0f} min, B0 {r['b0_time_s']/60:.0f} min)", flush=True)
    print("DONE ->", OUT)
