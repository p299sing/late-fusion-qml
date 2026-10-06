"""
exp_classical_split.py -- QMI revision item A4 (answers AAAI reviewers R1-W2, R2-W1,
R2-Q1, R3-W6 and R3-S1 with one experiment).

Two questions, one run:
  (1) R3-S1: a MATCHED split-feature CLASSICAL fusion baseline. Classical local encoders
      see exactly the register-wise feature partition the subcircuits see (x_A | x_B), each
      emits n_j bounded features (same width as the subcircuit's Z-readout), and a fusion
      head of the same capacity as the quantum pipeline's (tanh MLP, hidden=8) combines
      them -- trained end-to-end, exactly Algorithm 1 with classical encoders.
  (2) Train-free locality proxy: the gap
          g_cls = acc(classical, full input) - acc(classical split-feature fusion)
      needs NO quantum circuit and NO coupled training. If it rank-correlates with the
      Q-dial knee / fusion gap as well as (or better than) the trained-circuit cut
      entanglement S_A does, it is an a-priori diagnostic available before any quantum
      training -- the capability reviewers said S_A lacks.

Runs on the SAME 104 (alpha, seed) cells and splits as results/dense_alpha_results.json, so
the correlations are paired per run. Also runs the alpha-sweep seeds used by exp_independent
(alpha in {0.25,0.5,0.75,1.0}, 10 seeds, N=220/150) to give a baseline row for Table 3.

Outputs: results/classical_split_results.json
Run:  python -m experiments.exp_classical_split
"""
import json
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from scipy.optimize import minimize
from scipy.stats import spearmanr, rankdata
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV

from lfqml import datasets

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS, exist_ok=True)
DENSE = os.path.join(RESULTS, "dense_alpha_results.json")
OUT = os.path.join(RESULTS, "classical_split_results.json")
WORKERS = 8
N_A = N_B = 2


# --------------------------------------------------------------------------- #
# classical late fusion with matched partition + capacity (Alg. 1, classical)
# --------------------------------------------------------------------------- #
def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


class ClassicalSplitFusion:
    """x_A -> enc_A -> m_A (n_A tanh units); x_B -> enc_B -> m_B; head MLP(hidden) -> p.
    Encoders are one-hidden-layer tanh nets (hidden=enc_hidden) with n_j outputs, so the
    fusion head sees exactly as many features per register as the quantum pipeline
    (one Z expectation per qubit, bounded in [-1,1])."""

    def __init__(self, n_A, n_B, enc_hidden=8, head_hidden=8, l2=1e-3, seed=0):
        self.n_A, self.n_B, self.eh, self.hh, self.l2, self.seed = n_A, n_B, enc_hidden, head_hidden, l2, seed
        self.params = None

    def _shapes(self):
        nA, nB, eh, hh = self.n_A, self.n_B, self.eh, self.hh
        d = nA + nB
        return [("WA1", (nA, eh)), ("bA1", (eh,)), ("WA2", (eh, nA)), ("bA2", (nA,)),
                ("WB1", (nB, eh)), ("bB1", (eh,)), ("WB2", (eh, nB)), ("bB2", (nB,)),
                ("W1", (d, hh)), ("b1", (hh,)), ("W2", (hh,)), ("b2", (1,))]

    def _unpack(self, z):
        out, i = {}, 0
        for name, shp in self._shapes():
            k = int(np.prod(shp)); out[name] = z[i:i + k].reshape(shp); i += k
        return out

    def _forward(self, z, X):
        P = self._unpack(z)
        xA, xB = X[:, :self.n_A], X[:, self.n_A:]
        mA = np.tanh(np.tanh(xA @ P["WA1"] + P["bA1"]) @ P["WA2"] + P["bA2"])   # local, no coupling
        mB = np.tanh(np.tanh(xB @ P["WB1"] + P["bB1"]) @ P["WB2"] + P["bB2"])
        F = np.hstack([mA, mB])
        h = np.tanh(F @ P["W1"] + P["b1"])
        return _sigmoid(h @ P["W2"] + P["b2"][0])

    def fit(self, X, y, maxiter=400):
        n = sum(int(np.prod(s)) for _, s in self._shapes())
        rng = np.random.default_rng(self.seed)
        eps = 1e-9

        def loss(z):
            p = self._forward(z, X)
            return -np.mean(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps)) + self.l2 * float(z @ z)

        best = None
        for restart in range(3):                       # a few restarts; still seconds
            z0 = 0.3 * rng.standard_normal(n)
            res = minimize(loss, z0, method="L-BFGS-B", options={"maxiter": maxiter})
            if best is None or res.fun < best.fun:
                best = res
        self.params = best.x
        return self

    def predict(self, X):
        return (self._forward(self.params, X) > 0.5).astype(int)


def best_classical_full(Xtr, ytr, Xte, yte, seed=0):
    """Tuned SVM-RBF and MLP on the FULL input (the paper's existing classical anchor);
    model selection by CV on the training split only; report the better test accuracy."""
    svm = GridSearchCV(SVC(kernel="rbf"),
                       {"C": [0.5, 1, 5, 10], "gamma": ["scale", 0.5, 1, 2]}, cv=5).fit(Xtr, ytr)
    mlp = MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=2000, random_state=seed).fit(Xtr, ytr)
    acc_svm = float((svm.predict(Xte) == yte).mean())
    acc_mlp = float((mlp.predict(Xte) == yte).mean())
    # pick by CV score (no test peeking)
    from sklearn.model_selection import cross_val_score
    cv_mlp = cross_val_score(MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=2000,
                                           random_state=seed), Xtr, ytr, cv=5).mean()
    chosen = acc_svm if svm.best_score_ >= cv_mlp else acc_mlp
    return {"svm": acc_svm, "mlp": acc_mlp, "full": chosen}


def one_cell(job):
    alpha, seed, n_samples, n_train, tag = job
    X, y = datasets.synthetic_partition(alpha=alpha, n_A=N_A, n_B=N_B,
                                        n_samples=n_samples, seed=seed)
    Xtr, ytr, Xte, yte = X[:n_train], y[:n_train], X[n_train:], y[n_train:]
    full = best_classical_full(Xtr, ytr, Xte, yte, seed=seed)
    split = ClassicalSplitFusion(N_A, N_B, seed=seed).fit(Xtr, ytr)
    acc_split = float((split.predict(Xte) == yte).mean())
    # per-register-only classical models (how much each half knows on its own)
    svA = SVC(kernel="rbf").fit(Xtr[:, :N_A], ytr); svB = SVC(kernel="rbf").fit(Xtr[:, N_A:], ytr)
    return {"tag": tag, "alpha": alpha, "seed": seed,
            "acc_cls_full": full["full"], "acc_cls_svm": full["svm"], "acc_cls_mlp": full["mlp"],
            "acc_cls_split": acc_split,
            "acc_cls_A_only": float((svA.predict(Xte[:, :N_A]) == yte).mean()),
            "acc_cls_B_only": float((svB.predict(Xte[:, N_A:]) == yte).mean()),
            "gap_cls": full["full"] - acc_split}


# --------------------------------------------------------------------------- #
def spearman_ci(a, b, n_boot=2000, seed=0):
    a, b = np.asarray(a, float), np.asarray(b, float)
    rho = spearmanr(a, b).correlation
    p = spearmanr(a, b).pvalue
    rng = np.random.default_rng(seed)
    bs = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(a), len(a))
        bs.append(spearmanr(a[idx], b[idx]).correlation)
    lo, hi = np.nanpercentile(bs, [2.5, 97.5])
    return {"rho": float(rho), "p": float(p), "ci95": [float(lo), float(hi)], "n": int(len(a))}


def partial_spearman(a, b, ctrl):
    """Rank-based partial correlation of a and b controlling for ctrl (alpha)."""
    ra, rb, rc = rankdata(a), rankdata(b), rankdata(ctrl)
    def resid(v):
        A = np.vstack([rc, np.ones_like(rc)]).T
        coef, *_ = np.linalg.lstsq(A, v, rcond=None)
        return v - A @ coef
    return float(spearmanr(resid(ra), resid(rb)).correlation)


def main():
    dense = json.load(open(DENSE))
    dense_jobs = [(r["alpha"], r["seed"], 300, 180, "dense") for r in dense]
    sweep_jobs = [(a, s, 220, 150, "sweep") for a in (0.25, 0.5, 0.75, 1.0) for s in range(10)]
    jobs = dense_jobs + sweep_jobs
    print(f"{len(jobs)} cells on {WORKERS} workers", flush=True)
    with ProcessPoolExecutor(WORKERS) as ex:
        rows = list(ex.map(one_cell, jobs))

    # ---- pair with the quantum results on the dense grid ----
    key = {(r["alpha"], r["seed"]): r for r in dense}
    paired = []
    for r in rows:
        if r["tag"] != "dense":
            continue
        q = key[(r["alpha"], r["seed"])]
        paired.append({**r, "cut_ent": q["cut_ent"], "kneeQ": q["kneeQ"], "kneeQ_abs": q["kneeQ_abs"],
                       "delta_mlp": q["delta_mlp"], "acc_B0": q["acc_B0"], "acc_B2_mlp": q["acc_B2_mlp"]})
    g = [p["gap_cls"] for p in paired]; S = [p["cut_ent"] for p in paired]
    al = [p["alpha"] for p in paired]
    stats = {}
    for target in ("kneeQ_abs", "kneeQ", "delta_mlp"):
        t = [p[target] for p in paired]
        stats[target] = {
            "proxy_gap_cls": {**spearman_ci(g, t), "partial_alpha": partial_spearman(g, t, al)},
            "S_A_trained":   {**spearman_ci(S, t), "partial_alpha": partial_spearman(S, t, al)},
        }
    stats["gap_cls_vs_S_A"] = spearman_ci(g, S)
    stats["gap_cls_vs_alpha"] = spearman_ci(g, al)

    # ---- per-alpha table for the sweep (Table 3 row material) ----
    sweep = {}
    for a in (0.25, 0.5, 0.75, 1.0):
        rs = [r for r in rows if r["tag"] == "sweep" and r["alpha"] == a]
        sweep[str(a)] = {k: [float(np.mean([r[k] for r in rs])), float(np.std([r[k] for r in rs]))]
                         for k in ("acc_cls_full", "acc_cls_split", "acc_cls_A_only", "acc_cls_B_only", "gap_cls")}

    out = {"protocol": {"dense_cells": len(paired), "sweep_cells": len(rows) - len(paired),
                        "encoder": "tanh MLP hidden=8 -> n_j tanh outputs per register",
                        "head": "tanh MLP hidden=8 (matches MLPHead)", "full_model": "tuned SVM-RBF / MLP(32,16) chosen by 5-fold CV on train"},
           "correlations_dense_grid": stats, "sweep_table": sweep, "rows": rows, "paired": paired}
    json.dump(out, open(OUT, "w"), indent=1)

    print("\n=== Train-free proxy (classical split gap) vs trained S_A, dense grid (n=%d) ===" % len(paired))
    for target, d in stats.items():
        if not isinstance(d, dict) or "proxy_gap_cls" not in d:
            continue
        p, s = d["proxy_gap_cls"], d["S_A_trained"]
        print(f"{target:11s} proxy rho={p['rho']:+.2f} CI[{p['ci95'][0]:+.2f},{p['ci95'][1]:+.2f}] partial(alpha)={p['partial_alpha']:+.2f}"
              f"   |   S_A rho={s['rho']:+.2f} CI[{s['ci95'][0]:+.2f},{s['ci95'][1]:+.2f}] partial(alpha)={s['partial_alpha']:+.2f}")
    print("gap_cls vs S_A: rho=%+.2f ; gap_cls vs alpha: rho=%+.2f" % (stats["gap_cls_vs_S_A"]["rho"], stats["gap_cls_vs_alpha"]["rho"]))
    print("\n=== Sweep (10 seeds): classical full / classical split-fusion / A-only / B-only / gap ===")
    for a, d in sweep.items():
        print(f"alpha={a}: " + "  ".join(f"{k.replace('acc_cls_','')}={v[0]:.3f}±{v[1]:.3f}" for k, v in d.items()))


if __name__ == "__main__":
    main()
