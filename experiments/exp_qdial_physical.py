"""
exp_qdial_physical.py -- the Q-dial in the PHYSICAL quasiprobability basis (B1-B4).

Answers the reviewers' objections to the Schmidt-basis Q-dial:
  * every retained term is a product of two PHYSICALLY EXECUTABLE subexperiment
    expectation values produced by qiskit-addon-cutting (no transition elements);
  * the cost axis is the Monte-Carlo sampling overhead (sum_kept |a_i|)^2, not the
    1-norm mass; the old convention is recorded alongside for a side-by-side plot;
  * the endpoints are tied to the headline models: Q=0 is frozen late fusion (MLP on
    raw local features), Q=1 is exact reconstruction; the FIXED-head B0/B1 point and
    independent (uncoupled) fusion are recorded as reference lines.

Two panels per (alpha, seed): EXACT subexperiment values and a FINITE-SHOT run
(Aer SamplerV2, SHOTS per subexperiment) in which train AND test features -- the
partial reconstruction and the raw local features -- are all estimated from shots.

Outputs: results/qdial_physical_results.json
Run:  OMP_NUM_THREADS=1 .venv/bin/python -m experiments.exp_qdial_physical
"""
import json
import os
import sys
import time
import numpy as np

os.environ.setdefault("OMP_NUM_THREADS", "1")

from lfqml import qsim, datasets, cutting
from lfqml.circuits import QNNConfig, prepare
from lfqml.train import train_qnn, train_fusion_independent
from lfqml.models import evaluate_baselines
from lfqml.fusion import MLPHead
from lfqml.metrics import classification_metrics
from lfqml import qdial_physical as qp

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS, exist_ok=True)

ALPHAS = [0.0, 0.25, 0.5, 0.75, 1.0]
SEEDS = list(range(5))
SHOT_SEEDS = list(range(3))        # finite-shot panel (Aer is ~3 s/sample); exact uses all SEEDS
N_SAMPLES, N_TRAIN = 300, 180      # same split as exp_h1
Q_GRID = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0]
SHOTS = 4000
KNEE_TOL = 0.02
MAX_WORKERS = 4


def _acc(y, p):
    return float(classification_metrics(y, p)["accuracy"])


def _knee(curve, acc_key="accuracy"):
    """Smallest Q whose accuracy is within KNEE_TOL of the Q=1 accuracy."""
    top = [c for c in curve if c["Q"] >= 1.0][0][acc_key]
    for c in sorted(curve, key=lambda c: c["Q"]):
        if c[acc_key] >= top - KNEE_TOL:
            return c["Q"]
    return 1.0


def _qdial_curve(coeffs, T_tr, T_te, S_tr, S_te, ytr, yte):
    """Physical Q-dial: head = MLP on [partial reconstruction r_O(Q) ++ raw local features]."""
    curve = []
    for Q in Q_GRID:
        R_tr = qp.qdial_curve_from_terms(coeffs, T_tr, Q)
        R_te = qp.qdial_curve_from_terms(coeffs, T_te, Q)
        F_tr = np.hstack([R_tr, S_tr]); F_te = np.hstack([R_te, S_te])
        head = MLPHead(hidden=8, seed=0).fit(F_tr, ytr)
        cost = qp.qdial_cost(coeffs, Q)
        curve.append({**cost, "accuracy": _acc(yte, head.predict_proba(F_te))})
    return curve


def one_run(job):
    alpha, seed, do_shots = job
    t0 = time.time()
    cfg = QNNConfig(n_A=2, n_B=2, depth=2, n_cuts=2, trainable_coupling=True)
    X, y = datasets.synthetic_partition(alpha, 2, 2, n_samples=N_SAMPLES, seed=seed)
    Xtr, ytr, Xte, yte = X[:N_TRAIN], y[:N_TRAIN], X[N_TRAIN:], y[N_TRAIN:]

    # --- the ONE trained circuit (B0) and the old Schmidt-convention baselines ---
    tq = train_qnn(cfg, Xtr, ytr, seed=seed, maxiter=90)
    base = evaluate_baselines(tq, Xtr, ytr, Xte, yte, Q_grid=tuple(Q_GRID))

    pre_tr = [prepare(cfg, tq.params, x) for x in Xtr]
    pre_te = [prepare(cfg, tq.params, x) for x in Xte]
    S_tr = np.array([cutting.subcircuit_raw_features(pc) for pc in pre_tr])
    S_te = np.array([cutting.subcircuit_raw_features(pc) for pc in pre_te])

    # --- physical QPD cut of the trained circuit (qiskit-addon-cutting) ---
    cutter = qp.PhysicalCutter(cfg, tq.params, tq.observables)
    T_tr = cutter.term_matrix(Xtr, estimator="exact")
    T_te = cutter.term_matrix(Xte, estimator="exact")
    # verification: sum_i T_i == uncut expectation (lfqml) on every test sample
    from lfqml.circuits import full_features
    F0_te = np.array([full_features(pc, tq.observables) for pc in pre_te])
    max_sum_err = float(np.max(np.abs(T_te.sum(axis=1) - F0_te)))
    exact_curve = _qdial_curve(cutter.coeffs, T_tr, T_te, S_tr, S_te, ytr, yte)

    # --- finite-shot ('physically executed') panel ---
    shots_curve, shots_recon_err = None, None
    if do_shots:
        rng = np.random.default_rng(1000 + seed)
        T_tr_s = cutter.term_matrix(Xtr, estimator="shots", shots=SHOTS, seed=int(rng.integers(1 << 30)))
        T_te_s = cutter.term_matrix(Xte, estimator="shots", shots=SHOTS, seed=int(rng.integers(1 << 30)))
        S_tr_s = cutter.raw_local_features_shots(Xtr, shots=SHOTS, seed=int(rng.integers(1 << 30)))
        S_te_s = cutter.raw_local_features_shots(Xte, shots=SHOTS, seed=int(rng.integers(1 << 30)))
        shots_recon_err = float(np.sqrt(np.mean((T_te_s.sum(axis=1) - F0_te) ** 2)))
        shots_curve = _qdial_curve(cutter.coeffs, T_tr_s, T_te_s, S_tr_s, S_te_s, ytr, yte)

    # --- reference points ---
    tf = train_fusion_independent(cfg, Xtr, ytr, seed=seed, hidden=8, paulis=(qsim.Z,), maxiter=120)
    acc_indep = _acc(yte, tf.predict_proba(Xte))
    acc_Q1_exact = [c for c in exact_curve if c["Q"] >= 1.0][0]["accuracy"]
    acc_Q0_exact = [c for c in exact_curve if c["Q"] == 0.0][0]["accuracy"]

    old_curve = [{"Q": c["Q"], "accuracy": c["accuracy"], "retained_mass": c["cost_overhead"],
                  "n_kept": c["n_subexp"]} for c in base["Qdial"]]

    row = {
        "alpha": alpha, "seed": seed,
        "coupling_angle": float(base["coupling_angle"]),
        "cut_entanglement": float(base["cut_entanglement"]),
        "kappa_per_cut": float(cutter.kappa_per_cut),
        "n_terms": int(len(cutter.coeffs)),
        "total_mass": float(np.sum(np.abs(cutter.coeffs))),            # kappa^k
        "full_overhead_sq": float(np.sum(np.abs(cutter.coeffs)) ** 2),  # physical overhead
        "gamma_schmidt_per_cut": float(base["gamma_per_cut"]),
        "max_abs_sumT_minus_exact": max_sum_err,
        "shots_recon_rmse": shots_recon_err,
        # reference accuracies
        "acc_B0_fixed_head": float(base["B0_uncut"]["accuracy"]),      # (i) fixed trained head
        "acc_B1_fixed_head": float(base["B1_reconstruction"]["accuracy"]),
        "acc_Q1_mlp_exact": acc_Q1_exact,                               # (ii) retrained head
        "acc_Q0_frozen_fusion": acc_Q0_exact,                           # (iii) == B2 MLP fusion
        "acc_B2_mlp_models": float(base["B2_fusion_mlp"]["accuracy"]),  # cross-check of (iii)
        "acc_independent_fusion": acc_indep,                            # (iv)
        # curves
        "physical_exact": exact_curve,
        "physical_shots": shots_curve,
        "schmidt_old": old_curve,
        # knees
        "knee_physical_exact": _knee(exact_curve),
        "knee_physical_shots": _knee(shots_curve) if shots_curve else None,
        "knee_schmidt_old": _knee(old_curve),
        "runtime_s": time.time() - t0,
    }
    return row


def summarize(rows, out=sys.stdout):
    P = lambda *a: print(*a, file=out)
    P("\n==================== physical Q-dial summary ====================")
    for alpha in ALPHAS:
        rs = [r for r in rows if r["alpha"] == alpha]
        if not rs:
            continue
        rs_sh = [r for r in rs if r["physical_shots"]]
        P(f"\n--- alpha = {alpha:.2f}   ({len(rs)} seeds exact, {len(rs_sh)} seeds finite-shot) ---")
        P(f"  phi = {np.mean([r['coupling_angle'] for r in rs]):+.3f} +- {np.std([r['coupling_angle'] for r in rs]):.3f}   "
          f"kappa^k (physical 1-norm) = {np.mean([r['total_mass'] for r in rs]):.3f}   "
          f"physical overhead kappa^2k = {np.mean([r['full_overhead_sq'] for r in rs]):.2f}   "
          f"Schmidt gamma_S^2k = {np.mean([r['gamma_schmidt_per_cut']**4 for r in rs]):.2f}")
        P(f"  reference acc: B0/B1 fixed head {np.mean([r['acc_B0_fixed_head'] for r in rs]):.3f} | "
          f"Q=1 MLP head {np.mean([r['acc_Q1_mlp_exact'] for r in rs]):.3f} | "
          f"Q=0 frozen fusion {np.mean([r['acc_Q0_frozen_fusion'] for r in rs]):.3f} | "
          f"independent fusion {np.mean([r['acc_independent_fusion'] for r in rs]):.3f}")
        P(f"  {'Q':>4} | {'acc exact':>9} {'acc shots':>9} | {'n_kept':>6} {'mass':>6} {'ovh=mass^2':>10} | "
          f"{'old acc':>7} {'old mass':>8} {'old n':>5}")
        for qi, Q in enumerate(Q_GRID):
            ex = np.mean([r["physical_exact"][qi]["accuracy"] for r in rs])
            sh = np.mean([r["physical_shots"][qi]["accuracy"] for r in rs_sh]) if rs_sh else float("nan")
            nk = np.mean([r["physical_exact"][qi]["n_kept"] for r in rs])
            ms = np.mean([r["physical_exact"][qi]["retained_mass"] for r in rs])
            ov = np.mean([r["physical_exact"][qi]["retained_overhead_sq"] for r in rs])
            oa = np.mean([r["schmidt_old"][qi]["accuracy"] for r in rs])
            om = np.mean([r["schmidt_old"][qi]["retained_mass"] for r in rs])
            on = np.mean([r["schmidt_old"][qi]["n_kept"] for r in rs])
            P(f"  {Q:>4.1f} | {ex:9.3f} {sh:9.3f} | {nk:6.1f} {ms:6.3f} {ov:10.3f} | {oa:7.3f} {om:8.3f} {on:5.1f}")
        kn_ex = [r["knee_physical_exact"] for r in rs]
        kn_sh = [r["knee_physical_shots"] for r in rs_sh]
        kn_old = [r["knee_schmidt_old"] for r in rs]
        P(f"  knee Q (within {KNEE_TOL} of Q=1): physical-exact mean {np.mean(kn_ex):.2f} {kn_ex} | "
          f"physical-shots mean {np.mean(kn_sh) if kn_sh else float('nan'):.2f} {kn_sh} | "
          f"Schmidt-old mean {np.mean(kn_old):.2f} {kn_old}")
        # cost at knee under each convention
        def cost_at(r, curve_key, Qk, key):
            return [c for c in r[curve_key] if c["Q"] == Qk][0][key]
        P(f"  cost at knee: physical (mass^2) {np.mean([cost_at(r, 'physical_exact', r['knee_physical_exact'], 'retained_overhead_sq') for r in rs]):.2f} "
          f"vs its 1-norm mass {np.mean([cost_at(r, 'physical_exact', r['knee_physical_exact'], 'retained_mass') for r in rs]):.2f}; "
          f"old Schmidt 1-norm at old knee {np.mean([cost_at(r, 'schmidt_old', r['knee_schmidt_old'], 'retained_mass') for r in rs]):.2f}")
    P(f"\n  verification: max |sum_i T_i - exact| over all runs/test samples = "
      f"{max(r['max_abs_sumT_minus_exact'] for r in rows):.2e}; "
      f"finite-shot reconstruction RMSE (Q=1, {SHOTS} shots) = "
      f"{np.mean([r['shots_recon_rmse'] for r in rows if r['shots_recon_rmse'] is not None]):.3f}")


def run():
    from concurrent.futures import ProcessPoolExecutor
    jobs = [(a, s, s in SHOT_SEEDS) for a in ALPHAS for s in SEEDS]
    jobs.sort(key=lambda j: (not j[2], j[0], j[1]))      # long (shot) jobs first
    rows = []
    out_path = os.path.join(RESULTS, "qdial_physical_results.json")
    protocol = {
        "cfg": {"n_A": 2, "n_B": 2, "depth": 2, "n_cuts": 2, "trainable_coupling": True},
        "alphas": ALPHAS, "seeds_exact": SEEDS, "seeds_finite_shot": SHOT_SEEDS,
        "n_samples": N_SAMPLES, "n_train": N_TRAIN, "n_test": N_SAMPLES - N_TRAIN,
        "train_qnn_maxiter": 90, "independent_fusion": {"hidden": 8, "paulis": ["Z"], "maxiter": 120},
        "head": "MLPHead(hidden=8, seed=0) refit per Q on [physical partial reconstruction ++ raw local <Z>]",
        "Q_grid": Q_GRID, "knee_tolerance": KNEE_TOL,
        "cutting": "qiskit-addon-cutting partition_problem + generate_cutting_experiments(num_samples=inf); "
                   "RZZ(phi) -> 6-term Mitarai-Fujii QPD per cut; 6^k=36 terms; sum|a_i| = (1+2|sin phi|)^k",
        "term_definition": "T_i(O) = a_i * E^A_i(O_A) * E^B_i(O_B), identical to reconstruct_expectation_values "
                           "with all other coefficients zeroed (tests/test_qdial_physical.py)",
        "qdial_rule": "sort terms by |a_i| desc; keep until sum_kept|a_i| >= Q * sum_all|a_i| (Q=0 none, Q=1 all)",
        "cost_new": "retained_overhead_sq = (sum_kept |a_i|)^2, floored at 1 (Monte-Carlo sampling overhead)",
        "cost_old": "retained_mass = sum_kept |a_i| (1-norm; the earlier convention) -- recorded for comparison; "
                    "schmidt_old = models.evaluate_baselines()['Qdial'] (operator-Schmidt (nu,mu) terms)",
        "exact_estimator": "statevector branching over the QPD mid-circuit measurements (infinite-shot value)",
        "finite_shot_estimator": f"Aer SamplerV2, {SHOTS} shots per subexperiment, train AND test features "
                                 f"(partial reconstruction and raw local <Z>) estimated from shots; "
                                 f"finite-shot panel restricted to seeds {SHOT_SEEDS} for runtime "
                                 f"(Aer simulates mid-circuit-measurement circuits shot by shot)",
        "parallelism": f"ProcessPoolExecutor(max_workers={MAX_WORKERS}), OMP_NUM_THREADS=1",
    }
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as ex:
        for row in ex.map(one_run, jobs):
            rows.append(row)
            print(f"alpha={row['alpha']:.2f} seed={row['seed']} phi={row['coupling_angle']:+.2f} "
                  f"B0={row['acc_B0_fixed_head']:.3f} Q1mlp={row['acc_Q1_mlp_exact']:.3f} "
                  f"Q0={row['acc_Q0_frozen_fusion']:.3f} indep={row['acc_independent_fusion']:.3f} "
                  f"knee(phys/old)={row['knee_physical_exact']}/{row['knee_schmidt_old']} "
                  f"shots={'y' if row['physical_shots'] else 'n'} sumT_err={row['max_abs_sumT_minus_exact']:.1e} "
                  f"[{row['runtime_s']/60:.1f} min]", flush=True)
            rows_sorted = sorted(rows, key=lambda r: (r["alpha"], r["seed"]))
            with open(out_path, "w") as f:
                json.dump({"protocol": protocol, "rows": rows_sorted}, f, indent=1)
    rows.sort(key=lambda r: (r["alpha"], r["seed"]))
    summarize(rows)
    print(f"DONE -> {out_path}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--summary":
        with open(os.path.join(RESULTS, "qdial_physical_results.json")) as f:
            summarize(json.load(f)["rows"])
    else:
        run()
