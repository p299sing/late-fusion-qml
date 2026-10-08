"""
analyze_ansatzB.py -- side-by-side (ansatz A vs ansatz B) paired statistics.

For each alpha (independent sweep) and each dataset (benchmarks) prints mean +- std
of B0, frozen fusion and uncoupled fusion (independent / B2_mlp), the paired gap
uncoupled - B0 with a 10 000-resample bootstrap 95% CI and the TOST verdict at
DELTA = 0.02, reusing `paired` from experiments/analyze_significance.py.

  A: results/independent_results.json, results/bench_results.json
  B: results/independent_ansatzB_results.json, results/bench_ansatzB_results.json

Outputs: results/ansatzB_significance.json (+ printed tables)
Run:  python -m experiments.analyze_ansatzB
"""
import json
import os
import numpy as np

from experiments.analyze_significance import paired, DELTA

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")
OUT = os.path.join(RESULTS, "ansatzB_significance.json")
ALPHAS = [0.25, 0.5, 0.75, 1.0]
DATASETS = ["moons", "circles", "iris", "breast_cancer"]


def _load(name):
    p = os.path.join(RESULTS, name)
    return json.load(open(p)) if os.path.exists(p) else None


def _ms(rows, k):
    v = np.array([r[k] for r in rows], float)
    return f"{v.mean():.3f}+-{v.std():.3f}"


def _block(rows, key_group, groups, k_b0, k_frozen, k_unc, label, extra=()):
    """Per-group table + paired(uncoupled - B0) stats for one ansatz."""
    out = []
    for g in groups:
        rs = sorted([r for r in rows if r[key_group] == g], key=lambda r: r["seed"])
        if not rs:
            continue
        st = paired([r[k_unc] for r in rs], [r[k_b0] for r in rs], f"{label}/{g}: uncoupled - B0")
        st.update({"group": g, "n": len(rs), "B0": _ms(rs, k_b0),
                   "frozen": _ms(rs, k_frozen) if k_frozen else "-",
                   "uncoupled": _ms(rs, k_unc)})
        for k in extra:
            st[k] = _ms(rs, k) if k in rs[0] else "-"
        out.append(st)
    return out


def _print_side_by_side(title, A, B, extra=()):
    print(f"\n=== {title}: ansatz A (RY,RZ+CZ ring) vs ansatz B (RX,RY+CNOT ladder, phi=pi/2) ===")
    hdr = (f"{'group':>8} {'ans':>3} {'n':>2} {'B0':>13} {'frozen':>13} {'uncoupled':>13} "
           f"{'gap(unc-B0)':>12} {'95% CI':>17} {f'TOST d={DELTA}':>34}")
    for k in extra:
        hdr += f" {k:>13}"
    print(hdr)
    groups = [s["group"] for s in (A or B)]
    for g in groups:
        for tag, S in (("A", A), ("B", B)):
            if S is None:
                print(f"{str(g):>8} {tag:>3}  (no results yet)")
                continue
            s = [s for s in S if s["group"] == g]
            if not s:
                continue
            s = s[0]
            line = (f"{str(g):>8} {tag:>3} {s['n']:>2} {s['B0']:>13} {s['frozen']:>13} {s['uncoupled']:>13} "
                    f"{s['mean_gap']:>+12.3f} [{s['ci95'][0]:+.3f},{s['ci95'][1]:+.3f}] "
                    f"{s['verdict']:>34}")
            for k in extra:
                line += f" {s.get(k, '-'):>13}"
            print(line)


def analyze_independent(save=True):
    A = _load("independent_results.json")
    B = _load("independent_ansatzB_results.json")
    sA = _block(A, "alpha", ALPHAS, "acc_B0", "acc_frozen", "acc_indep", "A/independent") if A else None
    sB = _block(B, "alpha", ALPHAS, "acc_B0", "acc_frozen", "acc_indep", "B/independent",
                extra=("cut_ent",)) if B else None
    _print_side_by_side("independent-fusion alpha sweep", sA, sB, extra=("cut_ent",))
    # pooled over the sweep (40 pairs)
    pooled = {}
    for tag, rows in (("A", A), ("B", B)):
        if rows:
            sel = sorted(rows, key=lambda r: (r["alpha"], r["seed"]))
            pooled[tag] = paired([r["acc_indep"] for r in sel], [r["acc_B0"] for r in sel],
                                 f"{tag}/independent/pooled: uncoupled - B0")
            p = pooled[tag]
            print(f"  pooled {tag}: n={p['n_pairs']} gap={p['mean_gap']:+.3f} "
                  f"CI[{p['ci95'][0]:+.3f},{p['ci95'][1]:+.3f}] -> {p['verdict']}")
    return {"independent_A": sA, "independent_B": sB, "independent_pooled": pooled}


def analyze_bench(save=True):
    A = _load("bench_results.json")
    B = _load("bench_ansatzB_results.json")
    ex = ("acc_B6", "acc_B7", "kneeQ", "cut_ent")
    sA = _block(A, "dataset", DATASETS, "acc_B0", None, "acc_B2_mlp", "A/bench", extra=ex) if A else None
    sB = _block(B, "dataset", DATASETS, "acc_B0", None, "acc_B2_mlp", "B/bench", extra=ex) if B else None
    _print_side_by_side("benchmarks (uncoupled = B2_mlp frozen late fusion)", sA, sB, extra=ex)
    return {"bench_A": sA, "bench_B": sB}


def run():
    out = {}
    out.update(analyze_independent())
    out.update(analyze_bench())
    with open(OUT, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nDONE -> {OUT}")


if __name__ == "__main__":
    run()
