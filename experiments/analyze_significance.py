"""
analyze_significance.py -- paired per-seed statistics for every headline comparison
(reviewer W4). Pure analysis over existing results/*.json; no retraining.

For each comparison we report the paired mean gap, a bootstrap 95% CI on the
paired gap, a two-sided Wilcoxon signed-rank p-value (exact for small n), and a
TOST equivalence verdict at a PREDEFINED margin DELTA=0.02 accuracy (QMI revision,
reviewer R3-W5: "without a predefined equivalence margin the evidence establishes
closeness, not equivalence").
Note the epistemics: for "A matches B" claims the relevant evidence is a tight
CI around zero; for "A beats B" claims it is a CI excluding zero / small p.

Outputs: results/significance_results.json (+ printed table)
Run:  python -m experiments.analyze_significance
"""
import json
import os
import numpy as np
from scipy.stats import wilcoxon

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")
OUT = os.path.join(RESULTS, "significance_results.json")
DELTA = 0.02   # predefined equivalence margin (accuracy); TOST at alpha=0.05 via the 90% CI


def _load(name):
    return json.load(open(os.path.join(RESULTS, name)))


def paired(a, b, label):
    """a, b: per-seed arrays (same order). Returns dict of paired stats of (a - b)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = a - b
    rng = np.random.default_rng(0)
    boots = [np.mean(d[rng.integers(0, len(d), len(d))]) for _ in range(10000)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    if np.allclose(d, 0):
        p = 1.0
    else:
        p = float(wilcoxon(a, b, zero_method="wilcox").pvalue)
    # --- predefined equivalence margin (reviewer request) ---
    # TOST at alpha=0.05 <=> the 90% bootstrap CI of the paired gap lies inside [-DELTA, +DELTA].
    lo90, hi90 = np.percentile(boots, [5.0, 95.0])
    equivalent = bool(lo90 > -DELTA and hi90 < DELTA)
    different = bool(lo > 0 or hi < 0)            # 95% CI excludes zero
    if equivalent and not different:
        verdict = "equivalent"
    elif equivalent and different:
        verdict = "equivalent (small significant offset)"   # |gap| significant but < DELTA
    elif different:
        verdict = "deficit" if np.mean(d) < 0 else "advantage"
    else:
        verdict = "inconclusive"
    return {"comparison": label, "n_pairs": len(d), "mean_gap": float(np.mean(d)),
            "ci95": [float(lo), float(hi)], "ci90": [float(lo90), float(hi90)],
            "wilcoxon_p": p, "delta": DELTA, "tost_equivalent": equivalent, "verdict": verdict}


def by_seed(rows, key, filt):
    sel = sorted([r for r in rows if filt(r)], key=lambda r: r["seed"])
    return [r[key] for r in sel]


def run():
    out = []

    bench = _load("bench_results.json")
    for ds in ["moons", "circles", "iris", "breast_cancer"]:
        f = lambda r, ds=ds: r["dataset"] == ds
        out.append(paired(by_seed(bench, "acc_B2_mlp", f), by_seed(bench, "acc_B0", f),
                          f"bench/{ds}: fusion - B0"))
        out.append(paired(by_seed(bench, "acc_B2_mlp", f), by_seed(bench, "acc_B6", f),
                          f"bench/{ds}: fusion - Kawase"))

    ind = _load("independent_results.json")
    for a in [0.25, 0.5, 0.75, 1.0]:
        f = lambda r, a=a: r["alpha"] == a
        out.append(paired(by_seed(ind, "acc_indep", f), by_seed(ind, "acc_B0", f),
                          f"independent/alpha={a}: fusion - reconstruction"))
    # pooled over the sweep (40 pairs)
    sel = sorted(ind, key=lambda r: (r["alpha"], r["seed"]))
    out.append(paired([r["acc_indep"] for r in sel], [r["acc_B0"] for r in sel],
                      "independent/pooled: fusion - reconstruction"))

    faith = _load("faithful_results.json")
    for ds in ["fashion_mnist", "mnist", "synthetic_a0.75"]:
        f = lambda r, ds=ds: r["dataset"] == ds
        out.append(paired(by_seed(faith, "acc_fusion", f), by_seed(faith, "acc_recon", f),
                          f"faithful/{ds}: fusion - reconstruction"))
        out.append(paired(by_seed(faith, "acc_fusion", f), by_seed(faith, "acc_kawase", f),
                          f"faithful/{ds}: fusion - Kawase"))
        out.append(paired(by_seed(faith, "acc_fusion", f), by_seed(faith, "acc_classical", f),
                          f"faithful/{ds}: fusion - classical"))

    f8 = _load("faithful8_results.json")
    for ds in ["fashion_mnist", "mnist"]:
        f = lambda r, ds=ds: r["dataset"] == ds
        out.append(paired(by_seed(f8, "acc_fusion", f), by_seed(f8, "acc_recon", f),
                          f"faithful-PCA8/{ds}: fusion - reconstruction"))

    shots = _load("shots_results.json")
    for s in [32, 100, 1000]:
        f = lambda r, s=s: r["shots"] == s
        out.append(paired(by_seed(shots, "acc_fusion", f), by_seed(shots, "acc_recon", f),
                          f"shots={s}: fusion - reconstruction"))

    mar = _load("marshall_results.json")
    for T in [1, 2, 4, 8, 16]:
        sel = sorted([r for r in mar if r["budget_T"] == T],
                     key=lambda r: (r["alpha"], r["seed"]))
        out.append(paired([r["acc_qdial"] for r in sel], [r["acc_marshall"] for r in sel],
                          f"marshall/T={T} (pooled alpha): Qdial - Marshall"))

    mc = _load("multiclass_results.json")
    for ds in ["iris3", "mnist012"]:
        f = lambda r, ds=ds: r["dataset"] == ds
        out.append(paired(by_seed(mc, "acc_fusion", f), by_seed(mc, "acc_classical", f),
                          f"multiclass/{ds}: fusion - classical"))

    with open(OUT, "w") as fjson:
        json.dump(out, fjson, indent=2)
    w = max(len(o["comparison"]) for o in out)
    for o in out:
        print(f"{o['comparison']:<{w}}  n={o['n_pairs']:2d}  gap={o['mean_gap']:+.3f}  "
              f"CI[{o['ci95'][0]:+.3f},{o['ci95'][1]:+.3f}]  p={o['wilcoxon_p']:.4f}  "
              f"TOST(d={DELTA}): {o['verdict']}")
    print("DONE -> results/significance_results.json")


if __name__ == "__main__":
    run()
