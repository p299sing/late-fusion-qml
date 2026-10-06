"""
exp_marshall.py -- W5: head-to-head with Marshall-style LEARNED truncated reconstruction.

Marshall, Gyurik, Dunjko (Quantum 7, 1078 (2023), arXiv:2203.13739) cut a large circuit
and LEARN a truncated subset of the reconstruction sum, trained jointly through the cut.
Faithful analog on our machinery:

  * circuit trained JOINTLY through the cut (the same trained coupled circuit tq as B0);
  * readout keeps the top-T reconstruction terms by coefficient mass and LEARNS a weight
    per (term, observable) on the actual truncated-sum summands
    Re[ c*_nu c_mu <A_nu|O_A|A_mu><B_nu|O_B|B_mu> ]  -- with all-ones weights this IS the
    truncated sum, so the learned readout dominates the fixed truncation by construction;
  * compared against our Q-dial readout at the MATCHED subexperiment budget (n_subexp=T),
    and against pure fusion (T=0 gives the fusion features only).

Sweep: alpha in {0, 0.5, 0.75, 1.0} x 5 seeds x budgets T in {1, 2, 4, 8, 16(all)}.
Outputs: results/marshall_results.json
Run:  python -m experiments.exp_marshall
"""
import json
import os
import numpy as np

from lfqml.circuits import QNNConfig
from lfqml import datasets, cutting, qsim
from lfqml.circuits import prepare
from lfqml.train import train_qnn
from lfqml.fusion import LogisticHead, MLPHead
from lfqml.metrics import classification_metrics

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS, exist_ok=True)

ALPHAS = [0.0, 0.5, 0.75, 1.0]
SEEDS = list(range(10))
BUDGETS = [1, 2, 4, 8, 16]          # nb^2 = 16 terms total at 2 cuts (nb = 4 branches)


def term_summands(cd, observables, T):
    """The top-T truncated-sum summands per observable (the quantities Marshall's
    learned combination weighs): Re[c*_nu c_mu <A_nu|O_A|A_mu><B_nu|O_B|B_mu>]."""
    nb = len(cd.coeffs)
    mass = cutting._term_weights(cd).ravel()
    keep = np.argsort(mass)[::-1][:T]                     # top-T (nu,mu) indices
    C = np.outer(np.conj(cd.coeffs), cd.coeffs).ravel()
    feats = []
    for obs in observables:
        MA = np.zeros((nb, nb), dtype=complex)
        MB = np.zeros((nb, nb), dtype=complex)
        for nu in range(nb):
            for mu in range(nb):
                MA[nu, mu] = qsim.matrix_element(cd.A_branches[nu], cd.A_branches[mu],
                                                 obs.A_part, cd.n_A)
                MB[nu, mu] = qsim.matrix_element(cd.B_branches[nu], cd.B_branches[mu],
                                                 obs.B_part, cd.n_B)
        terms = (C * (MA * MB).ravel())[keep]
        feats.extend(list(np.real(terms)))
    return np.array(feats)


def qdial_acc_at_budget(qcurve, T):
    """Q-dial accuracy at the largest n_subexp <= T (matched budget)."""
    ok = [p for p in qcurve if p["n_subexp"] <= T]
    return max(ok, key=lambda p: p["n_subexp"])["accuracy"] if ok else None


def run():
    cfg = QNNConfig(n_A=2, n_B=2, depth=2, n_cuts=2, trainable_coupling=True)
    rows = []
    for alpha in ALPHAS:
        for seed in SEEDS:
            X, y = datasets.synthetic_partition(alpha, 2, 2, 300, seed=seed)
            Xtr, ytr, Xte, yte = X[:180], y[:180], X[180:], y[180:]
            tq = train_qnn(cfg, Xtr, ytr, seed=seed, maxiter=60)
            pre_tr = [prepare(cfg, tq.params, x) for x in Xtr]
            pre_te = [prepare(cfg, tq.params, x) for x in Xte]
            cds_tr = [cutting.build_cut(pc) for pc in pre_tr]
            cds_te = [cutting.build_cut(pc) for pc in pre_te]

            # Q-dial reference curve on the SAME trained circuit
            from lfqml.models import evaluate_baselines
            res = evaluate_baselines(tq, Xtr, ytr, Xte, yte)
            qcurve = res["Qdial"]

            # raw fusion features: the strengthened Marshall+raw variant receives
            # the same extra inputs the Q-dial gets (reviewer fairness check)
            Rtr = np.array([cutting.subcircuit_raw_features(pc) for pc in pre_tr])
            Rte = np.array([cutting.subcircuit_raw_features(pc) for pc in pre_te])

            for T in BUDGETS:
                Ftr = np.array([term_summands(cd, tq.observables, T) for cd in cds_tr])
                Fte = np.array([term_summands(cd, tq.observables, T) for cd in cds_te])
                head = LogisticHead().fit(Ftr, ytr)
                accM = classification_metrics(yte, head.predict_proba(Fte))["accuracy"]
                headR = LogisticHead().fit(np.hstack([Ftr, Rtr]), ytr)
                accMR = classification_metrics(
                    yte, headR.predict_proba(np.hstack([Fte, Rte])))["accuracy"]
                accQ = qdial_acc_at_budget(qcurve, T)
                rows.append({"alpha": alpha, "seed": seed, "budget_T": T,
                             "acc_marshall": accM, "acc_marshall_raw": accMR,
                             "acc_qdial": accQ,
                             "acc_fusion": res["B2_fusion_mlp"]["accuracy"],
                             "acc_B0": res["B0_uncut"]["accuracy"]})
            print(f"alpha={alpha:.2f} seed={seed}  " + "  ".join(
                f"T={r['budget_T']}:M={r['acc_marshall']:.2f}/Q={r['acc_qdial']:.2f}"
                for r in rows[-len(BUDGETS):]), flush=True)
        with open(os.path.join(RESULTS, "marshall_results.json"), "w") as f:
            json.dump(rows, f, indent=2)

    print("\n=== means over seeds (per alpha, budget) ===")
    for a in ALPHAS:
        for T in BUDGETS:
            rs = [r for r in rows if r["alpha"] == a and r["budget_T"] == T]
            print(f"alpha={a:.2f} T={T:2d}: marshall={np.mean([r['acc_marshall'] for r in rs]):.3f} "
                  f"qdial={np.mean([r['acc_qdial'] for r in rs]):.3f} "
                  f"fusion={np.mean([r['acc_fusion'] for r in rs]):.3f}")
    print("DONE -> results/marshall_results.json")


if __name__ == "__main__":
    run()
