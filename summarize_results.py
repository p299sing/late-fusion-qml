"""
summarize_results.py -- print paper-ready numbers from results/*.json in one place.

Run after experiments finish to fill RESULTS.md tables and the paper's \\CUT* macros:
    python summarize_results.py
Only prints sections whose JSON exists, so it is safe to run mid-suite.
"""
import json
import os
import statistics as st

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def load(name):
    p = os.path.join(RES, name)
    return json.load(open(p)) if os.path.exists(p) else None


def mean(rows, key):
    xs = [r[key] for r in rows if key in r]
    return st.mean(xs) if xs else float("nan")


def sec(title):
    print("\n" + "=" * 68 + f"\n{title}\n" + "-" * 68)


def ablation():
    rows = load("ablation_results.json")
    if not rows:
        return
    sec("ABLATION (G11)  features x hidden")
    feats = sorted({r["features"] for r in rows}, key=lambda f: len(f))
    hs = sorted({r["hidden"] for r in rows})
    print("feat  " + "  ".join(f"h={h:<2}" for h in hs))
    for f in feats:
        cells = []
        for h in hs:
            m = [r for r in rows if r["features"] == f and r["hidden"] == h]
            cells.append(f"{m[0]['acc']:.3f}" if m else "  -  ")
        print(f"{f:<5} " + "  ".join(cells))


def faithful():
    rows = load("faithful_results.json")
    if not rows:
        return
    sec("FAITHFUL (G4/G7)  means over seeds")
    for ds in dict.fromkeys(r["dataset"] for r in rows):
        rs = [r for r in rows if r["dataset"] == ds]
        print(f"{ds:16s} recon={mean(rs,'acc_recon'):.3f} fusion={mean(rs,'acc_fusion'):.3f} "
              f"kawase={mean(rs,'acc_kawase'):.3f} classical={mean(rs,'acc_classical'):.3f} "
              f"(n_seeds={len(rs)})")


def multiclass():
    rows = load("multiclass_results.json")
    if not rows:
        return
    sec("MULTICLASS (G12)  means over seeds")
    for ds in dict.fromkeys(r["dataset"] for r in rows):
        rs = [r for r in rows if r["dataset"] == ds]
        print(f"{ds:10s} (K={rs[0]['n_classes']}) fusion={mean(rs,'acc_fusion'):.3f} "
              f"classical={mean(rs,'acc_classical'):.3f} (n_seeds={len(rs)})")


def scale_qubits():
    rows = load("scale_qubits_results.json")
    if not rows:
        return
    sec("SCALE-QUBITS (width scaling)")
    for r in rows:
        print(f"n={r['qubits']}: recon={r['acc_recon']:.3f} fusion={r['acc_fusion']:.3f} "
              f"gap={r['gap']:+.3f} cut_entropy={r['cut_entropy']:.3f}")


def noise():
    rows = load("noise_results.json")
    if not rows:
        return
    sec("DEVICE NOISE (G5)  recon vs fusion-feature error")
    for nm in dict.fromkeys(r.get("noise", "depolarizing") for r in rows):
        for r in [r for r in rows if r.get("noise", "depolarizing") == nm]:
            print(f"[{nm:12s}] k={r['n_cuts']}: recon={r['recon_noise_err']:.4f}"
                  f"+/-{r.get('recon_noise_std', 0):.4f}  fusion={r['fusion_noise_err']:.4f}")


def multilayer():
    rows = load("multilayer_results.json")
    if not rows:
        return
    sec("MULTILAYER (G3)")
    for r in rows:
        print(f"L={r['layers']}: cut_entropy={r['cut_entropy']:.3f}  "
              f"fusion_error={r['fusion_error']:.3f}")


def qiskit():
    rows = load("qiskit_results.json")
    if not rows:
        return
    sec("QISKIT (G2; k>=4 at reduced shots)")
    for r in rows:
        extra = f"  [shots={r['shots']}]" if "shots" in r else ""
        print(f"k={r['n_cuts']}: overhead={r['real_overhead']:.0f}  "
              f"max|recon-exact|={r['max_recon_error']:.4f}{extra}")


def scaleup():
    rows = load("scaleup_results.json")
    if not rows:
        return
    sec("SCALE-UP (W2)  fusion where reconstruction is infeasible")
    for r in rows:
        b0 = f"{r['acc_B0']:.3f}" if r.get("acc_B0") is not None else "--(infeasible)"
        print(f"n={r['qubits']:2d} k={r['n_cuts']:2d}: fusion={r['acc_fusion']:.3f}"
              f"+/-{r['acc_fusion_std']:.3f}  B0={b0}  9^k={r['recon_overhead_9k']:.1e}")


def tfim():
    d = load("tfim_results.json")
    if not d:
        return
    sec("TFIM (W1)  quantum-native phase / criticality")
    for task in dict.fromkeys(r["task"] for r in d["rows"]):
        rs = [r for r in d["rows"] if r["task"] == task]
        print(f"{task:12s} fusion={mean(rs,'acc_fusion'):.3f} joint={mean(rs,'acc_joint'):.3f} "
              f"gap={mean(rs,'gap'):+.3f}  S_A={mean(rs,'mean_cut_entropy'):.3f}")
    sa = d["entropy_curve"]["S_A"]
    g = d["entropy_curve"]["g"]
    print(f"S_A(g) peak: {max(sa):.3f} at g={g[sa.index(max(sa))]:.2f}")


if __name__ == "__main__":
    for fn in [ablation, faithful, multiclass, scale_qubits, noise, multilayer, qiskit,
               scaleup, tfim]:
        fn()
    print()
