"""
exp_hardware_budget.py -- live-device comparison at EQUAL TOTAL measurement budget
(the deployment-relevant accounting; complements exp_hardware.py's full-shots run).

A fixed total budget of B_TOTAL shots is split evenly across each readout's
circuits: fusion has 2 measurement settings (one per half-width subcircuit), so
each gets B_TOTAL/2 shots; QPD reconstruction at k cuts has 2*6^k subexperiments,
so each gets B_TOTAL/(2*6^k) shots. Estimator error |est - exact| is compared.

Configs: k=1,2 on 2+2 qubits; k=3 on 3+3 qubits (a 2+2 boundary supports at most
2 cuts). Per instance: k=1 -> 12 circuits @666, k=2 -> 72 @111, k=3 -> 432 @18,
fusion -> 2 @4000. Three instances per k ~= 1.6k circuits, mostly tiny shot
counts -- a few QPU-minutes.

Run:  python -m experiments.exp_hardware_budget [--backend least_busy] [--dry-run]
Outputs: results/hardware_budget_results.json
"""
import argparse
import json
import os

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import SparsePauliOp
from qiskit_addon_cutting import (partition_problem, generate_cutting_experiments,
                                  reconstruct_expectation_values)

from lfqml.qiskit_cut import build_qnn_circuit, exact_expectations

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")
OUT = os.path.join(RESULTS, "hardware_budget_results.json")

B_TOTAL = 8000
CONFIGS = [(2, 2, 1), (2, 2, 2), (3, 3, 3)]     # (n_A, n_B, k)
N_INSTANCES = 3
SEED = 7


def _observables(n):
    return [("I" * q) + "Z" + ("I" * (n - q - 1)) for q in range(n)][::-1]


def _counts_to_z(counts, n):
    total = sum(counts.values())
    z = np.zeros(n)
    for bits, c in counts.items():
        b = bits.replace(" ", "")[::-1]
        for q in range(n):
            z[q] += c * (1.0 if b[q] == "0" else -1.0)
    return z / total


def run(backend_name: str, dry_run: bool):
    if dry_run:
        from qiskit_aer import AerSimulator
        from qiskit_aer.primitives import SamplerV2
        backend = AerSimulator()
        sampler = SamplerV2()
        label = "aer-dry-run"
    else:
        from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
        service = QiskitRuntimeService()
        backend = (service.least_busy(operational=True, simulator=False)
                   if backend_name == "least_busy" else service.backend(backend_name))
        sampler = SamplerV2(mode=backend)
        label = backend.name
    print(f"target: {label}  (budget {B_TOTAL} shots per readout)", flush=True)

    rows = []
    if os.path.exists(OUT):
        rows = json.load(open(OUT))
    done = {(r["k"], r["instance"]) for r in rows if r["backend"] == label}
    for inst in range(N_INSTANCES):
        rng = np.random.default_rng(SEED + inst)
        for n_A, n_B, k in CONFIGS:
            if (k, inst) in done:
                print(f"k={k} instance={inst}: done, skipping", flush=True)
                continue
            n = n_A + n_B
            obs_labels = _observables(n)
            params = rng.uniform(-np.pi, np.pi, 2 * (2 * n_A) + 2 * (2 * n_B))

            # reconstruction: budget split across all subexperiments
            qc, labels = build_qnn_circuit(n_A, n_B, 2, k, params)
            exact = exact_expectations(qc, obs_labels)
            part = partition_problem(circuit=qc, partition_labels=labels,
                                     observables=SparsePauliOp(obs_labels).paulis)
            subexp, coeffs = generate_cutting_experiments(
                circuits=part.subcircuits, observables=part.subobservables,
                num_samples=np.inf)
            n_circ = sum(len(c) for c in subexp.values())
            shots_rec = max(1, B_TOTAL // n_circ)
            results = {}
            for lab, circs in subexp.items():
                tcircs = transpile(circs, backend=backend, optimization_level=3)
                results[lab] = sampler.run(tcircs, shots=shots_rec).result()
            recon = np.array(reconstruct_expectation_values(results, coeffs,
                                                            part.subobservables))
            err_recon = float(np.mean(np.abs(recon - exact)))

            # fusion: budget split across the 2 subcircuit settings
            qf, _ = build_qnn_circuit(n_A, n_B, 2, 0, params)
            exact_f = exact_expectations(qf, obs_labels)
            shots_fus = B_TOTAL // 2
            z = np.zeros(n)
            subs = []
            for reg in (list(range(n_A)), list(range(n_A, n))):
                qs = QuantumCircuit(len(reg))
                for inst_ in qf.data:
                    qargs = [qf.find_bit(q).index for q in inst_.qubits]
                    if all(q in reg for q in qargs):
                        qs.append(inst_.operation, [reg.index(q) for q in qargs])
                qs.measure_all()
                subs.append((reg, qs))
            tsubs = transpile([c for _, c in subs], backend=backend,
                              optimization_level=3)
            res_f = sampler.run(tsubs, shots=shots_fus).result()
            for (reg, _), r in zip(subs, res_f):
                zr = _counts_to_z(r.data.meas.get_counts(), len(reg))
                for i, q in enumerate(reg):
                    z[q] = zr[i]
            err_fusion = float(np.mean(np.abs(z - exact_f)))

            row = {"backend": label, "k": k, "n_qubits": n, "instance": inst,
                   "budget_total": B_TOTAL, "n_subexperiments": n_circ,
                   "shots_per_subexp": shots_rec, "shots_per_fusion_circ": shots_fus,
                   "err_recon": err_recon, "err_fusion": err_fusion,
                   "ratio": err_recon / max(err_fusion, 1e-12)}
            rows.append(row)
            with open(OUT, "w") as f:
                json.dump(rows, f, indent=2)
            print(f"k={k} inst={inst}: recon={err_recon:.4f} ({n_circ}x{shots_rec}sh)  "
                  f"fusion={err_fusion:.4f} (2x{shots_fus}sh)  ratio={row['ratio']:.1f}x",
                  flush=True)

    print("\n=== equal-budget summary ===")
    for k in sorted(set(r["k"] for r in rows)):
        rs = [r for r in rows if r["k"] == k and r["backend"] == label]
        if not rs:
            continue
        er = np.array([r["err_recon"] for r in rs])
        ef = np.array([r["err_fusion"] for r in rs])
        print(f"k={k} ({len(rs)} inst): recon={er.mean():.4f}+/-{er.std():.4f}  "
              f"fusion={ef.mean():.4f}+/-{ef.std():.4f}  pooled-ratio={er.mean()/ef.mean():.1f}x")
    print("DONE -> results/hardware_budget_results.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="least_busy")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    run(args.backend, args.dry_run)
