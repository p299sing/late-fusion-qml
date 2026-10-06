"""
exp_deep_coupling.py -- E1: does TRAINED late fusion lose accuracy as the coupled model gets deeper?

Reviewer R2: the coupling-depth study (Fig. 6(b), exp_multilayer.py) measures cut entropy and the
unrecoverable connected correlation on UNTRAINED random circuits, so "the relationship between
increasing circuit depth, entanglement, and actual fusion accuracy remains only indirectly
demonstrated".  This experiment trains the models.

For every (L, k, alpha, seed):
  * DEEP COUPLED reference  -- L coupling layers of k RZZ(phi_l) cross gates interleaved with local
    layers, readout <Z_q> + <Z_a Z_b>, linear head, trained jointly (lfqml.deep_circuits).  Exact
    reconstruction equals this uncut model identically, at QPD cost 9^(kL); we never actually cut it.
  * FROZEN fusion           -- MLP(hidden=8) on the trained coupled circuit's register-local marginal
    features <Z_q> = Tr(rho_A Z_q), Tr(rho_B Z_q)  (no cross-cut correlator).
    (also recorded: `acc_frozen_cutfree`, the same head on the trained circuit with every RZZ
    removed at inference -- the way cutting.subcircuit_raw_features builds B2 for L=1)
  * UNCOUPLED fusion        -- same circuit with every RZZ removed (A (x) B product), <Z_q> features,
    MLP(hidden=8), trained end-to-end (mirrors train.train_fusion_independent).
  * diagnostics of the trained coupled model: mean test cut entropy S_A, trained coupling angles,
    and the unrecovered connected correlation |<Z_a Z_b> - <Z_a><Z_b>| over test inputs / pairs.

Data: datasets.synthetic_partition(alpha, 2, 2, n_samples=220, seed), 150 train / 70 test
(as exp_independent.py).  maxiter: coupled 90, fusion 120 (as exp_independent.py).

Outputs: results/deep_coupling_results.json
Run:  OMP_NUM_THREADS=1 python -m experiments.exp_deep_coupling [--procs 4] [--seeds 5]
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
import json
import time
from multiprocessing import Pool

import numpy as np
from scipy.stats import spearmanr

from lfqml import datasets, deep_circuits as dc
from lfqml.fusion import MLPHead
from lfqml.metrics import classification_metrics

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS, exist_ok=True)
OUT = os.path.join(RESULTS, "deep_coupling_results.json")

LS = [1, 2, 3, 4]
KS = [1, 2]
ALPHAS = [0.25, 0.5, 0.75, 1.0]
N, NTR = 220, 150
MAXITER_COUPLED, MAXITER_FUSION = 90, 120
HIDDEN = 8


def _acc(y, p):
    return float(classification_metrics(y, p)["accuracy"])


def run_cell(cell):
    L, k, alpha, seed = cell
    t0 = time.time()
    X, y = datasets.synthetic_partition(alpha, 2, 2, n_samples=N, seed=seed)
    Xtr, ytr, Xte, yte = X[:NTR], y[:NTR], X[NTR:], y[NTR:]
    cfg = dc.DeepConfig(n_A=2, n_B=2, L=L, n_cuts=k, coupled=True)

    # --- deep coupled reference (== exact reconstruction) ---
    tq = dc.train_deep_coupled(cfg, Xtr, ytr, seed=seed, maxiter=MAXITER_COUPLED)
    acc_c = _acc(yte, tq.predict_proba(Xte))
    acc_c_tr = _acc(ytr, tq.predict_proba(Xtr))

    # --- frozen fusion: MLP head on the coupled state's register-local marginals ---
    Ftr = dc.local_z_features(cfg, tq.params, Xtr)
    Fte = dc.local_z_features(cfg, tq.params, Xte)
    frozen = MLPHead(hidden=HIDDEN, seed=seed).fit(Ftr, ytr)
    acc_f = _acc(yte, frozen.predict_proba(Fte))
    # variant: couplings removed at inference (product circuit with the trained local angles)
    p0 = tq.params.copy(); p0[cfg.n_local_params:] = 0.0
    Ftr0 = dc.local_z_features(cfg, p0, Xtr); Fte0 = dc.local_z_features(cfg, p0, Xte)
    frozen0 = MLPHead(hidden=HIDDEN, seed=seed).fit(Ftr0, ytr)
    acc_f0 = _acc(yte, frozen0.predict_proba(Fte0))

    # --- uncoupled fusion trained end-to-end ---
    tf = dc.train_deep_fusion(cfg, Xtr, ytr, seed=seed, hidden=HIDDEN, maxiter=MAXITER_FUSION)
    acc_u = _acc(yte, tf.predict_proba(Xte))
    acc_u_tr = _acc(ytr, tf.predict_proba(Xtr))

    # --- quantumness diagnostics of the trained coupled model (test inputs) ---
    S = dc.cut_entropies(cfg, tq.params, Xte)
    phis = dc.coupling_angles(cfg, tq.params)
    phis_wrapped = (phis + np.pi) % (2 * np.pi) - np.pi           # RZZ(phi) is 2pi-periodic
    conn = dc.connected_correlations(cfg, tq.params, Xte)         # (n_test, k)

    return {"L": L, "k": k, "alpha": alpha, "seed": seed,
            "acc_coupled": acc_c, "acc_coupled_train": acc_c_tr,
            "loss_coupled": tq.history["loss"], "nit_coupled": tq.history["nit"],
            "acc_frozen": acc_f, "acc_frozen_cutfree": acc_f0,
            "acc_uncoupled": acc_u, "acc_uncoupled_train": acc_u_tr,
            "gap_frozen": acc_c - acc_f, "gap_uncoupled": acc_c - acc_u,
            "S_A": float(S.mean()), "S_A_std": float(S.std()),
            "phis": [float(v) for v in phis], "mean_abs_phi": float(np.mean(np.abs(phis_wrapped))),
            "mean_abs_sin_phi": float(np.mean(np.abs(np.sin(phis)))),
            "conn_corr": float(conn.mean()),
            "n_params_coupled": int(cfg.n_params), "n_params_uncoupled": int(cfg.n_local_params),
            "time_s": time.time() - t0}


def summarise(rows):
    lines = []
    summary = {}
    hdr = (f"{'k':>2} {'L':>2} | {'acc_coup':>8} {'acc_froz':>8} {'acc_unc':>8} | "
           f"{'S_A':>6} {'|phi|':>6} {'|sin|':>6} {'conn':>6} | {'gap_frozen':>14} {'gap_uncoupled':>14} | n")
    lines.append(hdr); lines.append("-" * len(hdr))
    for k in KS:
        for L in LS:
            rs = [r for r in rows if r["k"] == k and r["L"] == L]
            if not rs:
                continue
            m = lambda key: float(np.mean([r[key] for r in rs]))
            s = lambda key: float(np.std([r[key] for r in rs]))
            summary[f"k{k}_L{L}"] = {key: m(key) for key in
                                     ["acc_coupled", "acc_coupled_train", "acc_frozen", "acc_frozen_cutfree",
                                      "acc_uncoupled", "acc_uncoupled_train", "S_A", "mean_abs_phi",
                                      "mean_abs_sin_phi", "conn_corr", "gap_frozen", "gap_uncoupled"]}
            summary[f"k{k}_L{L}"].update({"gap_frozen_std": s("gap_frozen"),
                                          "gap_uncoupled_std": s("gap_uncoupled"), "n_cells": len(rs)})
            lines.append(f"{k:>2} {L:>2} | {m('acc_coupled'):8.3f} {m('acc_frozen'):8.3f} {m('acc_uncoupled'):8.3f} | "
                         f"{m('S_A'):6.3f} {m('mean_abs_phi'):6.3f} {m('mean_abs_sin_phi'):6.3f} {m('conn_corr'):6.3f} | "
                         f"{m('gap_frozen'):+7.3f} ± {s('gap_frozen'):5.3f} {m('gap_uncoupled'):+7.3f} ± {s('gap_uncoupled'):5.3f} | {len(rs)}")
    # per-alpha breakdown of the uncoupled gap vs L (pooled over k)
    lines.append("\nmean gap_uncoupled by alpha (rows) x L (cols), pooled over k:")
    lines.append("alpha | " + " ".join(f"L={L:<5}" for L in LS))
    for a in ALPHAS:
        vals = [np.mean([r["gap_uncoupled"] for r in rows if r["alpha"] == a and r["L"] == L]) for L in LS]
        lines.append(f"{a:5.2f} | " + " ".join(f"{v:+6.3f}" for v in vals))
    lines.append("mean S_A by alpha (rows) x L (cols), pooled over k:")
    for a in ALPHAS:
        vals = [np.mean([r["S_A"] for r in rows if r["alpha"] == a and r["L"] == L]) for L in LS]
        lines.append(f"{a:5.2f} | " + " ".join(f"{v:6.3f}" for v in vals))
    # correlations across all cells
    corr = {}
    for xk, yk in [("S_A", "gap_uncoupled"), ("S_A", "gap_frozen"), ("conn_corr", "gap_uncoupled"),
                   ("conn_corr", "gap_frozen"), ("L", "gap_uncoupled"), ("L", "S_A"), ("L", "acc_coupled")]:
        rho, p = spearmanr([r[xk] for r in rows], [r[yk] for r in rows])
        corr[f"{xk}_vs_{yk}"] = {"spearman_rho": float(rho), "p": float(p), "n": len(rows)}
        lines.append(f"Spearman({xk}, {yk}) over {len(rows)} cells: rho={rho:+.3f}  p={p:.2g}")
    # same on (k, L, alpha) cell means (removes seed noise)
    keys = sorted({(r["k"], r["L"], r["alpha"]) for r in rows})
    mS = [np.mean([r["S_A"] for r in rows if (r["k"], r["L"], r["alpha"]) == kk]) for kk in keys]
    mG = [np.mean([r["gap_uncoupled"] for r in rows if (r["k"], r["L"], r["alpha"]) == kk]) for kk in keys]
    rho, p = spearmanr(mS, mG)
    corr["S_A_vs_gap_uncoupled_cellmeans"] = {"spearman_rho": float(rho), "p": float(p), "n": len(keys)}
    lines.append(f"Spearman(S_A, gap_uncoupled) over {len(keys)} (k,L,alpha) means: rho={rho:+.3f}  p={p:.2g}")
    lines.append("\ncoupled-model trainability (mean over k, alpha, seeds):")
    for L in LS:
        rs = [r for r in rows if r["L"] == L]
        lines.append(f"  L={L}: test acc {np.mean([r['acc_coupled'] for r in rs]):.3f}  "
                     f"train acc {np.mean([r['acc_coupled_train'] for r in rs]):.3f}  "
                     f"final loss {np.mean([r['loss_coupled'] for r in rs]):.3f}  "
                     f"uncoupled test acc {np.mean([r['acc_uncoupled'] for r in rs]):.3f}  "
                     f"frozen test acc {np.mean([r['acc_frozen'] for r in rs]):.3f}")
    return summary, corr, "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--seeds", type=int, default=5)
    args = ap.parse_args()
    seeds = list(range(args.seeds))
    cells = [(L, k, a, s) for L in LS for k in KS for a in ALPHAS for s in seeds]
    cells.sort(key=lambda c: -c[0])                      # heaviest (deep) cells first
    protocol = {"L": LS, "k": KS, "alphas": ALPHAS, "seeds": seeds, "n_samples": N, "n_train": NTR,
                "maxiter_coupled": MAXITER_COUPLED, "maxiter_fusion": MAXITER_FUSION,
                "fusion_hidden": HIDDEN, "fusion_features": "Z per qubit",
                "coupled_readout": "Z per qubit + ZZ per cross pair, linear logistic head",
                "frozen_features": "local Z of the trained COUPLED state (reduced-state marginals); "
                                   "acc_frozen_cutfree = couplings zeroed at inference",
                "coupling_angles": "one trainable phi_l per coupling layer, shared across the k pairs "
                                   "(as the single-layer model); init pi/2",
                "procs": args.procs,
                "deviations": ([] if args.seeds == 5 else
                               [f"{args.seeds} seeds instead of the planned 5 (runtime allowed it)"])}
    t0 = time.time()
    rows = []
    with Pool(args.procs) as pool:
        for r in pool.imap_unordered(run_cell, cells, chunksize=1):
            rows.append(r)
            print(f"[{len(rows):3d}/{len(cells)}] L={r['L']} k={r['k']} alpha={r['alpha']:.2f} seed={r['seed']}  "
                  f"coup={r['acc_coupled']:.3f} froz={r['acc_frozen']:.3f} unc={r['acc_uncoupled']:.3f}  "
                  f"S_A={r['S_A']:.3f} conn={r['conn_corr']:.3f}  ({r['time_s']:.1f}s)", flush=True)
            if len(rows) % 20 == 0:
                with open(OUT, "w") as f:
                    json.dump({"protocol": protocol, "rows": rows, "partial": True}, f, indent=1)
    rows.sort(key=lambda r: (r["k"], r["L"], r["alpha"], r["seed"]))
    summary, corr, text = summarise(rows)
    protocol["wall_time_s"] = time.time() - t0
    with open(OUT, "w") as f:
        json.dump({"protocol": protocol, "summary_by_kL": summary, "correlations": corr,
                   "rows": rows}, f, indent=1)
    print("\n" + text)
    print(f"\nDONE in {protocol['wall_time_s']/60:.1f} min -> {OUT}")


if __name__ == "__main__":
    main()
