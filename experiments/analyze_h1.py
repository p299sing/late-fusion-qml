"""
analyze_h1.py -- summarise results/h1_results.json.

Prints per-alpha aggregates and the H1 correlation between the *quantum* gap
(Delta_mlp) and cut-entanglement, and reports the Q-dial knee.

Run:  python -m experiments.analyze_h1
"""
import json
import os
import numpy as np
from scipy.stats import pearsonr, spearmanr

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")


def knee(qdial, tol=0.02):
    """Smallest Q whose accuracy is within tol of the Q=1 (exact) accuracy."""
    acc1 = [q for q in qdial if q["Q"] >= 1.0][0]["accuracy"]
    for q in sorted(qdial, key=lambda r: r["Q"]):
        if q["accuracy"] >= acc1 - tol:
            return q["Q"], q["cost_overhead"]
    return 1.0, acc1


def main():
    with open(os.path.join(RESULTS, "h1_results.json")) as f:
        rows = json.load(f)

    print(f"{'alpha':>6} {'B0':>6} {'B1':>6} {'B2lin':>6} {'B2mlp':>6} "
          f"{'B6':>6} {'B7':>6} {'d_lin':>6} {'d_mlp':>6} {'ent':>6} {'kneeQ':>6}")
    alphas = sorted(set(r["alpha"] for r in rows))
    for a in alphas:
        rs = [r for r in rows if r["alpha"] == a]
        def m(k): return np.mean([r[k] for r in rs])
        knees = [knee(r["Qdial"])[0] for r in rs]
        print(f"{a:6.2f} {m('acc_B0'):6.3f} {m('acc_B1'):6.3f} {m('acc_B2_lin'):6.3f} "
              f"{m('acc_B2_mlp'):6.3f} {m('acc_B6'):6.3f} {m('acc_B7'):6.3f} "
              f"{m('delta_lin'):6.3f} {m('delta_mlp'):6.3f} {m('cut_ent'):6.3f} "
              f"{np.mean(knees):6.2f}")

    ent = np.array([r["cut_ent"] for r in rows])
    dmlp = np.array([r["delta_mlp"] for r in rows])
    dlin = np.array([r["delta_lin"] for r in rows])
    print("\nH1 correlations (across all runs):")
    print(f"  Delta_mlp  vs cut-entanglement : pearson={pearsonr(ent, dmlp)[0]:+.3f} "
          f"spearman={spearmanr(ent, dmlp)[0]:+.3f}")
    print(f"  Delta_lin  vs cut-entanglement : pearson={pearsonr(ent, dlin)[0]:+.3f} "
          f"spearman={spearmanr(ent, dlin)[0]:+.3f}")
    print(f"  Delta_mlp  vs alpha            : "
          f"pearson={pearsonr([r['alpha'] for r in rows], dmlp)[0]:+.3f}")


if __name__ == "__main__":
    main()
