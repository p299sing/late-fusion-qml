"""
exp_hardware.py -- OPTIONAL real-hardware validation (NOT run for the submission;
the paper honestly reports simulation-only noise results).

The real-device analogue of Fig. 3(b) / exp_noise: on a REAL IBM backend, compare
  * reconstruction error: |QPD-reconstructed <Z_q> - exact <Z_q>|  (k cuts), vs
  * fusion-feature error: |device <Z_q> - exact <Z_q>| of the same circuit with
    the coupling removed (the two subcircuits run as one uncoupled circuit).
No training happens on hardware and no parameter transfer is needed (random
seeded parameters, as in exp_noise/exp_qiskit), so the run is cheap and robust.

QPU budget: k=1 uses ~6 subexperiment circuits + 1 fusion circuit; k=2 ~36 + 1.
At 4000 shots each this is well under 10 QPU-minutes on an Eagle/Heron backend
(queue time dominates wall clock) -- comfortably inside a 100-minute allocation.

One-time setup (requires YOUR IBM token; interactive):
    python -c "from qiskit_ibm_runtime import QiskitRuntimeService as S; \
               S.save_account(channel='ibm_quantum_platform', token='<TOKEN>')"

Run:  python -m experiments.exp_hardware --dry-run          # full pipeline on Aer, no QPU
      python -m experiments.exp_hardware --backend ibm_brisbane   # real device

Outputs: results/hardware_results.json
"""
import argparse
import json
import os

import numpy as np
from qiskit import transpile
from qiskit.quantum_info import SparsePauliOp
from qiskit_addon_cutting import (partition_problem, generate_cutting_experiments,
                                  reconstruct_expectation_values)

from lfqml.qiskit_cut import build_qnn_circuit, exact_expectations

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")
OUT = os.path.join(RESULTS, "hardware_results.json")

N_A = N_B = 2
SHOTS = 4000
CUTS = [1, 2]
SEED = 7          # base seed; instance i uses SEED + i


def _observables(n):
    return [("I" * q) + "Z" + ("I" * (n - q - 1)) for q in range(n)][::-1]


def _counts_to_z(counts, n):
    """Per-qubit <Z> from a counts dict (bitstrings little-endian per qiskit)."""
    total = sum(counts.values())
    z = np.zeros(n)
    for bits, c in counts.items():
        b = bits.replace(" ", "")[::-1]          # qubit q = b[q]
        for q in range(n):
            z[q] += c * (1.0 if b[q] == "0" else -1.0)
    return z / total


def run(backend_name: str, dry_run: bool, n_instances: int = 1, cuts=None,
        merge: bool = False):
    if dry_run:
        from qiskit_aer import AerSimulator
        from qiskit_aer.primitives import SamplerV2
        backend = AerSimulator()
        sampler = SamplerV2(default_shots=SHOTS)
        label = "aer-dry-run"
    else:
        from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
        service = QiskitRuntimeService()
        if backend_name == "least_busy":
            backend = service.least_busy(operational=True, simulator=False)
        else:
            backend = service.backend(backend_name)
        sampler = SamplerV2(mode=backend)
        label = backend.name
    print(f"target: {label}", flush=True)

    n = N_A + N_B
    obs_labels = _observables(n)
    rows = []
    if merge and os.path.exists(OUT):
        rows = json.load(open(OUT))
    done = {(r["k"], r.get("instance", 0)) for r in rows}
    for inst in range(n_instances):
      rng = np.random.default_rng(SEED + inst)
      for k in (cuts or CUTS):
        if (k, inst) in done:
            print(f"k={k} instance={inst}: already done, skipping", flush=True)
            continue
        params = rng.uniform(-np.pi, np.pi, 2 * (2 * N_A) + 2 * (2 * N_B))

        # --- reconstruction on the device (QPD subexperiments) ---------------
        qc, labels = build_qnn_circuit(N_A, N_B, 2, k, params)
        exact = exact_expectations(qc, obs_labels)
        obs = SparsePauliOp(obs_labels)
        part = partition_problem(circuit=qc, partition_labels=labels,
                                 observables=obs.paulis)
        overhead = float(np.prod([b.overhead for b in part.bases]))
        subexp, coeffs = generate_cutting_experiments(
            circuits=part.subcircuits, observables=part.subobservables,
            num_samples=np.inf)
        results = {}
        n_circ = 0
        for lab, circs in subexp.items():
            tcircs = transpile(circs, backend=backend, optimization_level=3)
            n_circ += len(tcircs)
            results[lab] = sampler.run(tcircs, shots=SHOTS).result()
        recon = np.array(reconstruct_expectation_values(results, coeffs,
                                                        part.subobservables))
        err_recon = float(np.mean(np.abs(recon - exact)))

        # --- fusion features on the device: DEPLOYMENT-FAITHFUL probe --------
        # each register runs as its OWN small circuit (that is the whole point of
        # fusion: half-width subcircuits on small devices), so the transpiler can
        # place each 2-qubit circuit independently -- exactly like the QPD
        # subexperiments get to.
        qf, _ = build_qnn_circuit(N_A, N_B, 2, 0, params)   # no coupling
        exact_f = exact_expectations(qf, obs_labels)
        z = np.zeros(n)
        sub_circuits = []
        for reg_qubits in (list(range(N_A)), list(range(N_A, n))):
            from qiskit import QuantumCircuit
            qs = QuantumCircuit(len(reg_qubits))
            for inst_ in qf.data:                     # project the register's gates
                qargs = [qf.find_bit(q).index for q in inst_.qubits]
                if all(q in reg_qubits for q in qargs):
                    qs.append(inst_.operation, [reg_qubits.index(q) for q in qargs])
            qs.measure_all()
            sub_circuits.append((reg_qubits, qs))
        tsubs = transpile([c for _, c in sub_circuits], backend=backend,
                          optimization_level=3)
        res_f = sampler.run(tsubs, shots=SHOTS).result()
        for (reg_qubits, _), r in zip(sub_circuits, res_f):
            zr = _counts_to_z(r.data.meas.get_counts(), len(reg_qubits))
            for i, q in enumerate(reg_qubits):
                z[q] = zr[i]
        err_fusion = float(np.mean(np.abs(z - exact_f)))

        row = {"backend": label, "k": k, "instance": inst, "shots": SHOTS,
               "n_subexperiments": n_circ, "qpd_overhead": overhead,
               "err_recon": err_recon, "err_fusion": err_fusion,
               "ratio": err_recon / max(err_fusion, 1e-12)}
        rows.append(row)
        with open(OUT, "w") as f:          # incremental save (queue-safe)
            json.dump(rows, f, indent=2)
        print(f"k={k} instance={inst}: recon_err={err_recon:.4f} ({n_circ} circuits, "
              f"overhead {overhead:.0f})  fusion_err={err_fusion:.4f}  "
              f"ratio={row['ratio']:.1f}x", flush=True)

    print("\n=== per-k summary over instances ===")
    for k in sorted(set(r["k"] for r in rows)):
        rs = [r for r in rows if r["k"] == k]
        er = np.array([r["err_recon"] for r in rs]); ef = np.array([r["err_fusion"] for r in rs])
        print(f"k={k} ({len(rs)} inst): recon={er.mean():.4f}+/-{er.std():.4f}  "
              f"fusion={ef.mean():.4f}+/-{ef.std():.4f}  mean-ratio={(er/ef).mean():.2f}x  "
              f"pooled-ratio={er.mean()/ef.mean():.2f}x", flush=True)
    print("DONE -> results/hardware_results.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="least_busy",
                    help="backend name, or 'least_busy' (default) to auto-select")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--instances", type=int, default=1)
    ap.add_argument("--cuts", type=int, nargs="+", default=None)
    ap.add_argument("--merge", action="store_true",
                    help="keep existing rows in the JSON and skip completed (k, instance)")
    args = ap.parse_args()
    run(args.backend, args.dry_run, n_instances=args.instances, cuts=args.cuts,
        merge=args.merge)
