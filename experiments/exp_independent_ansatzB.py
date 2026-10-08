"""
exp_independent_ansatzB.py -- ansatz-robustness check for the independent-fusion result.

Same alpha sweep as exp_independent.py (alpha in {0.25,0.5,0.75,1.0}, seeds 0..9,
N=220 / 150 train, n_A=n_B=2, depth=2, n_cuts=2, train_qnn maxiter=90, frozen fusion
MLPHead(hidden=8), independent fusion hidden=8 / paulis=(Z,) / maxiter=120) but with
the SECOND hardware-efficient ansatz family:

    ansatz "B": RX,RY per qubit + CNOT ladder (no wrap), with a FIXED maximally
    entangling coupling phi = pi/2 (trainable_coupling=False), so the physical QPD
    overhead is exactly (1 + 2|sin phi|)^2 = 9 per cut -> 9^k.

The uncoupled (independent) subcircuits also use ansatz "B" (train_fusion_independent
keeps cfg.ansatz; it only forces phi=0).  Also records the trained circuit's cut
entanglement S_A (key `cut_ent`, test-set mean, as elsewhere).

Outputs: results/independent_ansatzB_results.json
Run:  OMP_NUM_THREADS=1 python -m experiments.exp_independent_ansatzB
"""
import json
import os
import time
import numpy as np

from lfqml import qsim, datasets, cutting
from lfqml.circuits import QNNConfig, prepare, full_features
from lfqml.train import train_qnn, train_fusion_independent
from lfqml.fusion import MLPHead
from lfqml.models import qsim_sigmoid
from lfqml.metrics import classification_metrics, mean_cut_entanglement

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS, exist_ok=True)
OUT = os.path.join(RESULTS, "independent_ansatzB_results.json")

ALPHAS = [0.25, 0.5, 0.75, 1.0]
SEEDS = list(range(10))
N, NTR = 220, 150
CFG = QNNConfig(n_A=2, n_B=2, depth=2, n_cuts=2, ansatz="B",
                trainable_coupling=False, coupling_init=np.pi / 2)
PHYS_OVERHEAD_PER_CUT = float((1 + 2 * abs(np.sin(CFG.coupling_init))) ** 2)   # = 9


def one_run(job):
    alpha, seed = job
    t0 = time.time()
    cfg = CFG
    X, y = datasets.synthetic_partition(alpha, 2, 2, n_samples=N, seed=seed)
    Xtr, ytr, Xte, yte = X[:NTR], y[:NTR], X[NTR:], y[NTR:]

    # --- B0 / reconstruction upper bound + frozen fusion (identical protocol to exp_independent)
    tq = train_qnn(cfg, Xtr, ytr, seed=seed, maxiter=90)
    pre_te = [prepare(cfg, tq.params, x) for x in Xte]
    F0 = np.array([full_features(pc, tq.observables) for pc in pre_te])
    acc_B0 = classification_metrics(yte, qsim_sigmoid(F0 @ tq.head_w + tq.head_b))["accuracy"]
    Str = np.array([cutting.subcircuit_raw_features(prepare(cfg, tq.params, x)) for x in Xtr])
    Ste = np.array([cutting.subcircuit_raw_features(pc) for pc in pre_te])
    frozen = MLPHead(hidden=8, seed=seed).fit(Str, ytr)
    acc_frozen = classification_metrics(yte, frozen.predict_proba(Ste))["accuracy"]
    cut_ent = mean_cut_entanglement(cfg, pre_te)            # trained S_A

    # --- independent / fusion-native training (ansatz B subcircuits, no coupling)
    tf = train_fusion_independent(cfg, Xtr, ytr, seed=seed, hidden=8,
                                  paulis=(qsim.Z,), maxiter=120)
    assert tf.cfg.ansatz == "B"
    acc_indep = classification_metrics(yte, tf.predict_proba(Xte))["accuracy"]

    return {"alpha": alpha, "seed": seed, "ansatz": "B", "phi": float(cfg.coupling_init),
            "acc_B0": acc_B0, "acc_frozen": acc_frozen, "acc_indep": acc_indep,
            "gap_frozen": acc_B0 - acc_frozen, "gap_indep": acc_B0 - acc_indep,
            "cut_ent": cut_ent, "train_loss_B0": tq.history["loss"],
            "phys_overhead_per_cut": PHYS_OVERHEAD_PER_CUT,
            "runtime_s": time.time() - t0}


def run():
    from concurrent.futures import ProcessPoolExecutor
    t0 = time.time()
    jobs = [(a, s) for a in ALPHAS for s in SEEDS]
    rows = []
    with ProcessPoolExecutor(max_workers=6) as ex:
        for r in ex.map(one_run, jobs):
            rows.append(r)
            print(f"alpha={r['alpha']:.2f} seed={r['seed']}  B0={r['acc_B0']:.3f}  "
                  f"frozen={r['acc_frozen']:.3f} (gap {r['gap_frozen']:+.3f})  "
                  f"indep={r['acc_indep']:.3f} (gap {r['gap_indep']:+.3f})  "
                  f"S_A={r['cut_ent']:.3f}  [{r['runtime_s']:.0f}s]", flush=True)
    rows.sort(key=lambda r: (r["alpha"], r["seed"]))
    with open(OUT, "w") as f:
        json.dump(rows, f, indent=2)

    print("\n=== ansatz B: mean gaps vs reconstruction ===")
    for a in ALPHAS:
        rs = [r for r in rows if r["alpha"] == a]
        print(f"  alpha={a:.2f}  B0={np.mean([r['acc_B0'] for r in rs]):.3f}  "
              f"frozen_gap={np.mean([r['gap_frozen'] for r in rs]):+.3f}  "
              f"independent_gap={np.mean([r['gap_indep'] for r in rs]):+.3f}  "
              f"S_A={np.mean([r['cut_ent'] for r in rs]):.3f}")
    print(f"physical overhead per cut (phi=pi/2): {PHYS_OVERHEAD_PER_CUT:.1f}")
    print(f"wall time: {time.time() - t0:.0f}s")
    print(f"DONE -> {OUT}")
    try:
        from experiments.analyze_ansatzB import analyze_independent
        analyze_independent()
    except Exception as e:          # analysis is a convenience; never lose the results
        print("analysis skipped:", e)


if __name__ == "__main__":
    run()
