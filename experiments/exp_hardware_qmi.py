"""
exp_hardware_qmi.py -- live-hardware follow-ups for the QMI revision (plan items F2 and F3).

Mode `budget` (F2, reviewer R2-Q3 "how sensitive is the 7x gap to calibration and transpilation"):
    the equal-total-budget comparison of exp_hardware_budget.py (k=1,2 on 2+2 qubits, k=3 on
    3+3; B shots per readout split evenly over each readout's circuits), repeated on a chosen
    backend / day with SEVERAL transpiler seeds, and now logging what the first run did not:
    backend name, calibration (properties last-update) timestamp, transpiler seed, job ids,
    wall-clock time. Run it on a second backend and/or a second day to answer the question.

Mode `dial` (F3, reviewer R1-Q3 "what happens if I implement 0<Q<1 on Qiskit/hardware"):
    one trained coupled model (alpha=0.5, seed 0, k=2) is cut with qiskit-addon-cutting; for each
    requested Q only the KEPT QPD subexperiments are executed on the device for N_TEST held-out
    inputs (plus the two local fusion circuits), the head fitted on exact train features is
    applied, and test accuracy is recorded next to the exact-simulation accuracy of the same
    head. Q=0 costs 2 circuits per input; Q=0.3 ~ 5 kept terms x (groups) per side.

Both modes have --dry-run (Aer, no QPU) so the code path is validated before spending minutes.

Run:
  python -m experiments.exp_hardware_qmi budget --backend ibm_torino --instances 3 --transpile-seeds 0 1 --tag day2
  python -m experiments.exp_hardware_qmi dial   --backend ibm_fez --Q 0.0 0.3 --n-test 40 --shots 1000
  python -m experiments.exp_hardware_qmi budget --dry-run
Outputs: results/hardware_qmi_budget.json , results/hardware_qmi_dial.json (appended per row)
"""
import argparse
import json
import os
import time
from datetime import datetime, timezone

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import SparsePauliOp
from qiskit_addon_cutting import (partition_problem, generate_cutting_experiments,
                                  reconstruct_expectation_values)

from lfqml.qiskit_cut import build_qnn_circuit, exact_expectations
from experiments.exp_hardware_budget import _observables, _counts_to_z

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")
OUT_BUDGET = os.path.join(RESULTS, "hardware_qmi_budget.json")
OUT_DIAL = os.path.join(RESULTS, "hardware_qmi_dial.json")
CONFIGS = [(2, 2, 1), (2, 2, 2), (3, 3, 3)]     # (n_A, n_B, k)
SEED = 7                                        # same instances as exp_hardware_budget


def _connect(backend_name, dry_run):
    if dry_run:
        from qiskit_aer import AerSimulator
        from qiskit_aer.primitives import SamplerV2
        return AerSimulator(), SamplerV2(), "aer-dry-run", None
    from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
    service = QiskitRuntimeService()
    backend = (service.least_busy(operational=True, simulator=False)
               if backend_name == "least_busy" else service.backend(backend_name))
    try:
        cal = backend.properties().last_update_date
        cal = cal.isoformat() if hasattr(cal, "isoformat") else str(cal)
    except Exception:
        cal = None
    return backend, SamplerV2(mode=backend), backend.name, cal


def _append(path, row):
    rows = json.load(open(path)) if os.path.exists(path) else []
    rows.append(row)
    json.dump(rows, open(path, "w"), indent=2)


def _job_id(job):
    try:
        return job.job_id()
    except Exception:
        return None


# --------------------------------------------------------------------------- #
def run_budget(args):
    backend, sampler, label, cal = _connect(args.backend, args.dry_run)
    print(f"target {label}  calibration {cal}  budget {args.budget}  seeds {args.transpile_seeds}", flush=True)
    done = {(r["tag"], r["backend"], r["transpile_seed"], r["k"], r["instance"])
            for r in (json.load(open(OUT_BUDGET)) if os.path.exists(OUT_BUDGET) else [])}
    for tseed in args.transpile_seeds:
        for inst in range(args.instances):
            rng = np.random.default_rng(SEED + inst)
            for n_A, n_B, k in CONFIGS:
                if args.k and k not in args.k:
                    continue
                if (args.tag, label, tseed, k, inst) in done:
                    print(f"[{args.tag}] tseed={tseed} k={k} inst={inst}: already recorded, skipping", flush=True); continue
                n = n_A + n_B
                obs_labels = _observables(n)
                params = rng.uniform(-np.pi, np.pi, 2 * (2 * n_A) + 2 * (2 * n_B))
                t0 = time.time(); jobs = []
                qc, labels = build_qnn_circuit(n_A, n_B, 2, k, params)
                exact = exact_expectations(qc, obs_labels)
                part = partition_problem(circuit=qc, partition_labels=labels,
                                         observables=SparsePauliOp(obs_labels).paulis)
                subexp, coeffs = generate_cutting_experiments(
                    circuits=part.subcircuits, observables=part.subobservables, num_samples=np.inf)
                n_circ = sum(len(c) for c in subexp.values())
                shots_rec = max(1, args.budget // n_circ)
                results = {}
                for lab, circs in subexp.items():
                    tcircs = transpile(circs, backend=backend, optimization_level=3, seed_transpiler=tseed)
                    job = sampler.run(tcircs, shots=shots_rec); jobs.append(_job_id(job))
                    results[lab] = job.result()
                recon = np.array(reconstruct_expectation_values(results, coeffs, part.subobservables))
                err_recon = float(np.mean(np.abs(recon - exact)))
                # fusion: two half-width circuits
                qf, _ = build_qnn_circuit(n_A, n_B, 2, 0, params)
                exact_f = exact_expectations(qf, obs_labels)
                shots_fus = args.budget // 2
                z = np.zeros(n); subs = []
                for reg in (list(range(n_A)), list(range(n_A, n))):
                    qs = QuantumCircuit(len(reg))
                    for inst_ in qf.data:
                        qargs = [qf.find_bit(q).index for q in inst_.qubits]
                        if all(q in reg for q in qargs):
                            qs.append(inst_.operation, [reg.index(q) for q in qargs])
                    qs.measure_all(); subs.append((reg, qs))
                tsubs = transpile([c for _, c in subs], backend=backend, optimization_level=3, seed_transpiler=tseed)
                job = sampler.run(tsubs, shots=shots_fus); jobs.append(_job_id(job))
                res_f = job.result()
                for (reg, _), r in zip(subs, res_f):
                    zr = _counts_to_z(r.data.meas.get_counts(), len(reg))
                    for i, q in enumerate(reg):
                        z[q] = zr[i]
                err_fusion = float(np.mean(np.abs(z - exact_f)))
                row = {"mode": "budget", "tag": args.tag, "backend": label, "calibration_last_update": cal,
                       "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                       "transpile_seed": tseed, "optimization_level": 3,
                       "k": k, "n_qubits": n, "instance": inst, "budget_total": args.budget,
                       "n_subexperiments": n_circ, "shots_per_subexp": shots_rec, "shots_per_fusion_circ": shots_fus,
                       "err_recon": err_recon, "err_fusion": err_fusion,
                       "ratio": err_recon / max(err_fusion, 1e-12), "job_ids": [j for j in jobs if j],
                       "wall_s": time.time() - t0}
                _append(OUT_BUDGET, row)
                print(f"[{args.tag}] tseed={tseed} k={k} inst={inst}: recon={err_recon:.4f} ({n_circ}x{shots_rec}sh) "
                      f"fusion={err_fusion:.4f} (2x{shots_fus}sh) ratio={row['ratio']:.1f}x  jobs={len(row['job_ids'])}", flush=True)


def batched_device_terms(pc, X, keep, sampler, backend, shots, tseed, job_log, max_circuits_per_job=300):
    """All kept QPD subexperiments of all inputs in as few device jobs as possible.

    Returns T (n_inputs, n_terms, n_obs) with non-kept terms = 0, computed with the same
    post-processing as lfqml.qdial_physical.evaluate_subexperiments.
    """
    from qiskit_addon_cutting.utils.observable_grouping import ObservableCollection
    from lfqml.qdial_physical import _group_values_from_pub, _subsystem_expvals
    labels = list(pc.sub.keys())
    so = {l: ObservableCollection(pc.pp.subobservables[l]) for l in labels}
    ng = {l: len(so[l].groups) for l in labels}
    n_terms = len(pc.coeffs); kept_idx = np.flatnonzero(keep)
    circs, index = [], []                       # index[j] = (i_input, label, term, group)
    for i, x in enumerate(X):
        bound = pc._bound(x)
        for l in labels:
            for t in kept_idx:
                for g in range(ng[l]):
                    circs.append(bound[l][t * ng[l] + g]); index.append((i, l, int(t), g))
    tcircs = transpile(circs, backend=backend, optimization_level=3, seed_transpiler=tseed)
    data = [None] * len(tcircs)
    for start in range(0, len(tcircs), max_circuits_per_job):
        chunk = tcircs[start:start + max_circuits_per_job]
        job = sampler.run(chunk, shots=shots); jid = _job_id(job)
        if jid: job_log.append(jid)
        res = job.result()
        for j in range(len(chunk)):
            data[start + j] = res[j].data
    T = np.zeros((len(X), n_terms, len(pc.obs_labels)))
    gv = {}
    for j, (i, l, t, g) in enumerate(index):
        gv[(i, l, t, g)] = _group_values_from_pub(data[j], so[l].groups[g])
    for i in range(len(X)):
        E = {}
        for l in labels:
            rows = [[gv[(i, l, t, g)] if keep[t] else np.zeros(len(so[l].groups[g].pauli_bitmasks))
                     for g in range(ng[l])] for t in range(n_terms)]
            E[l] = _subsystem_expvals(rows, so[l], pc.pp.subobservables[l])
        T[i] = pc.coeffs[:, None] * np.prod([E[l] for l in labels], axis=0)
    print(f"  device circuits for this Q: {len(circs)} ({len(X)} inputs x {len(kept_idx)} kept terms x sides/groups)", flush=True)
    return T


# --------------------------------------------------------------------------- #
def run_dial(args):
    from lfqml import datasets, cutting
    from lfqml.circuits import QNNConfig, prepare
    from lfqml.train import train_qnn
    from lfqml.fusion import MLPHead
    from lfqml.qdial_physical import PhysicalCutter, qdial_kept_mask, qdial_cost, evaluate_subexperiments, _bitarray_to_ints
    backend, sampler, label, cal = _connect(args.backend, args.dry_run)
    cfg = QNNConfig(n_A=2, n_B=2, depth=2, n_cuts=2, trainable_coupling=True)
    X, y = datasets.synthetic_partition(alpha=args.alpha, n_A=2, n_B=2, n_samples=300, seed=args.seed)
    Xtr, ytr, Xte, yte = X[:180], y[:180], X[180:180 + args.n_test], y[180:180 + args.n_test]
    tq = train_qnn(cfg, Xtr, ytr, seed=args.seed, maxiter=90)
    pc = PhysicalCutter(cfg, tq.params, tq.observables)
    # exact train features (no QPU): head fitted once per Q on exact partial reconstruction + local Z
    pre_tr = [prepare(cfg, tq.params, x) for x in Xtr]
    S_tr = np.array([cutting.subcircuit_raw_features(p) for p in pre_tr])
    T_tr = pc.term_matrix(Xtr, estimator="exact")                      # (n, terms, obs)
    pre_te = [prepare(cfg, tq.params, x) for x in Xte]
    S_te_exact = np.array([cutting.subcircuit_raw_features(p) for p in pre_te])
    T_te_exact = pc.term_matrix(Xte, estimator="exact")
    print(f"target {label} cal {cal}; model alpha={args.alpha} seed={args.seed} phi={pc.phi:+.3f} "
          f"kappa^k={pc.coeffs.__abs__().sum():.2f}; n_test={len(Xte)} shots={args.shots}", flush=True)
    done_Q = {r["Q"] for r in (json.load(open(OUT_DIAL)) if os.path.exists(OUT_DIAL) else [])
              if r.get("tag") == args.tag and r.get("backend") == label and r.get("seed") == args.seed and r.get("alpha") == args.alpha}
    for Q in args.Q:
        if Q in done_Q:
            print(f"[{args.tag}] Q={Q}: already recorded for {label}, skipping", flush=True); continue
        t0 = time.time(); jobs = []
        keep = qdial_kept_mask(pc.coeffs, Q); cost = qdial_cost(pc.coeffs, Q)
        def feats(T, S):
            r = (T[:, keep, :]).sum(axis=1) if keep.any() else np.zeros((len(S), T.shape[2]))
            return np.hstack([r, S])
        head = MLPHead(hidden=8, seed=0).fit(feats(T_tr, S_tr), ytr)
        acc_exact = float(((head.predict_proba(feats(T_te_exact, S_te_exact)) > 0.5).astype(int) == yte).mean())
        # ---- device: kept subexperiments per test input + the two local circuits
        T_dev = np.zeros_like(T_te_exact)
        if keep.any():
            T_dev = batched_device_terms(pc, Xte, keep, sampler, backend, args.shots, args.transpile_seed, jobs,
                                         max_circuits_per_job=args.max_circuits_per_job)
        local = [pc._bind(pc._local_qc, x) for x in Xte]
        tlocal = transpile(local, backend=backend, optimization_level=3, seed_transpiler=args.transpile_seed)
        job = sampler.run(tlocal, shots=args.shots); jobs.append(_job_id(job)); res = job.result()
        S_dev = np.zeros((len(Xte), cfg.n))
        for i in range(len(Xte)):
            ints = _bitarray_to_ints(res[i].data.meas.array)
            for q in range(cfg.n):
                S_dev[i, q] = 1.0 - 2.0 * np.mean((ints >> q) & 1)
        acc_dev = float(((head.predict_proba(feats(T_dev, S_dev)) > 0.5).astype(int) == yte).mean())
        row = {"mode": "dial", "tag": args.tag, "backend": label, "calibration_last_update": cal,
               "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
               "transpile_seed": args.transpile_seed, "alpha": args.alpha, "seed": args.seed, "phi": float(pc.phi),
               "Q": Q, "n_kept": int(keep.sum()), "n_terms": int(len(keep)), "retained_overhead_sq": cost.get("retained_overhead_sq"),
               "shots": args.shots, "n_test": len(Xte), "acc_exact_features": acc_exact, "acc_device_features": acc_dev,
               "rmse_T_device_vs_exact": float(np.sqrt(np.mean((T_dev - T_te_exact) ** 2))) if keep.any() else None,
               "rmse_local_device_vs_exact": float(np.sqrt(np.mean((S_dev - S_te_exact) ** 2))),
               "job_ids": [j for j in jobs if j], "wall_s": time.time() - t0}
        _append(OUT_DIAL, row)
        print(f"[{args.tag}] Q={Q}: kept {row['n_kept']}/{row['n_terms']} overhead {row['retained_overhead_sq']:.1f}  "
              f"acc exact-features {acc_exact:.3f} vs device-features {acc_dev:.3f}  jobs={len(row['job_ids'])}  "
              f"{row['wall_s']/60:.1f} min", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="mode", required=True)
    b = sub.add_parser("budget"); b.add_argument("--backend", default="least_busy"); b.add_argument("--instances", type=int, default=3)
    b.add_argument("--transpile-seeds", type=int, nargs="+", default=[0]); b.add_argument("--budget", type=int, default=8000)
    b.add_argument("--k", type=int, nargs="*", default=None); b.add_argument("--tag", default="day2"); b.add_argument("--dry-run", action="store_true")
    d = sub.add_parser("dial"); d.add_argument("--backend", default="least_busy"); d.add_argument("--Q", type=float, nargs="+", default=[0.0, 0.3])
    d.add_argument("--n-test", type=int, default=40); d.add_argument("--shots", type=int, default=1000); d.add_argument("--alpha", type=float, default=0.5)
    d.add_argument("--seed", type=int, default=0); d.add_argument("--transpile-seed", type=int, default=0); d.add_argument("--tag", default="dial")
    d.add_argument("--dry-run", action="store_true"); d.add_argument("--max-circuits-per-job", type=int, default=300)
    args = ap.parse_args()
    run_budget(args) if args.mode == "budget" else run_dial(args)
