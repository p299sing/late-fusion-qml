"""
qiskit_cut.py -- G2: reproduce the cut/reconstruct pipeline through the REAL
`qiskit-addon-cutting` library (physical quasiprobability decomposition), so the
core results are not artifacts of our custom operator-Schmidt simulator.

Provides:
  * build_qnn_circuit(...)  -> a 2-register hardware-efficient QNN as a QuantumCircuit
  * reconstruct_via_qiskit(...) -> reconstructed expectation values + real sampling overhead
  * exact_expectations(...) -> ground-truth values (Aer statevector Estimator)
The verification experiment (experiments/exp_qiskit.py) checks reconstruction matches
exact and that the overhead grows exponentially with the number of cuts.
"""
from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit_addon_cutting import (partition_problem, generate_cutting_experiments,
                                  reconstruct_expectation_values)


def build_qnn_circuit(n_A, n_B, depth, n_cuts, params, phi=np.pi / 2, x=None):
    """Two-register hardware-efficient QNN. Cross gates (RZZ) join the boundary
    qubits and are what the cutter will cut. Matches lfqml.circuits in spirit."""
    n = n_A + n_B
    qc = QuantumCircuit(n)
    x = np.zeros(n) if x is None else x
    idx = 0

    def local_block(qubits):
        nonlocal idx
        for q in qubits:
            qc.ry(params[idx], q); idx += 1
            qc.rz(params[idx], q); idx += 1
        for i in range(len(qubits) - 1):
            qc.cz(qubits[i], qubits[i + 1])

    A = list(range(n_A)); B = list(range(n_A, n))
    for q in range(n):                      # angle encoding
        qc.ry(float(x[q]), q)
    local_block(A); local_block(B)          # pre-coupling local layers
    for j in range(n_cuts):                 # coupling layer (the cuts)
        qc.rzz(phi, n_A - 1 - j, n_A + j)
    local_block(A); local_block(B)          # post-coupling local layers
    return qc, "".join(["A"] * n_A + ["B"] * n_B)


def _param_count(n_A, n_B, depth):
    # matches build_qnn_circuit usage: two local blocks per side, RY+RZ per qubit
    return 2 * (2 * n_A) + 2 * (2 * n_B)


def exact_expectations(qc, observables):
    """Ground-truth <O> via exact statevector."""
    sv = Statevector(qc)
    return np.array([sv.expectation_value(SparsePauliOp(o)).real for o in observables])


def _aer_sampler(shots, noise_model=None):
    from qiskit_aer.primitives import SamplerV2
    if noise_model is None:
        return SamplerV2(default_shots=shots)
    return SamplerV2(default_shots=shots,
                     options={"backend_options": {"noise_model": noise_model}})


def reconstruct_via_qiskit(qc, partition_labels, observables, shots=100_000,
                           noise_model=None, basis_gates=None):
    """Cut with qiskit-addon-cutting, reconstruct <O> via the documented Sampler
    workflow, return (values, overhead). Pass an Aer noise_model for device-noise
    runs (G5): reconstruction's signed sum amplifies device noise by gamma^2.
    Pass basis_gates (e.g. a fake backend's basis) to transpile subexperiments to
    that basis first -- REQUIRED for a fake-backend noise model, whose per-gate
    errors only fire on its native gates; omit it for our depolarizing model, which
    is defined directly on the ansatz gates (ry/cz/rzz).
    """
    obs = SparsePauliOp(observables) if isinstance(observables, list) else observables
    partitioned = partition_problem(circuit=qc, partition_labels=partition_labels,
                                    observables=obs.paulis)
    overhead = float(np.prod([b.overhead for b in partitioned.bases]))
    subexperiments, coefficients = generate_cutting_experiments(
        circuits=partitioned.subcircuits, observables=partitioned.subobservables,
        num_samples=np.inf)
    sampler = _aer_sampler(shots, noise_model)
    results = {}
    for label, circs in subexperiments.items():
        if basis_gates is not None:
            from qiskit import transpile
            circs = transpile(circs, basis_gates=basis_gates, optimization_level=1)
        results[label] = sampler.run(circs, shots=shots).result()
    reconstructed = reconstruct_expectation_values(results, coefficients,
                                                   partitioned.subobservables)
    return np.array(reconstructed), overhead


def fake_backend_noise(name="FakeManilaV2"):
    """A REAL device-calibrated Aer noise model from an IBM fake backend (G5).
    More credible than a hand-rolled model: uses measured T1/T2, gate and readout
    errors from a real 5-qubit IBM device. Falls back to None if unavailable."""
    try:
        from qiskit_aer.noise import NoiseModel
        import qiskit_ibm_runtime.fake_provider as fp
        backend = getattr(fp, name)()
        return NoiseModel.from_backend(backend)
    except Exception:
        return None


def simple_depolarizing_noise(p1=0.005, p2=0.02, p_ro=0.02):
    """A self-contained device-like noise model: depolarizing on 1q/2q gates +
    readout error. Stand-in for an Aer fake-backend noise model (G5)."""
    from qiskit_aer.noise import (NoiseModel, depolarizing_error,
                                  ReadoutError)
    nm = NoiseModel()
    nm.add_all_qubit_quantum_error(depolarizing_error(p1, 1), ["ry", "rz", "h", "x", "sx"])
    nm.add_all_qubit_quantum_error(depolarizing_error(p2, 2), ["cz", "cx", "rzz"])
    nm.add_all_qubit_readout_error(ReadoutError([[1 - p_ro, p_ro], [p_ro, 1 - p_ro]]))
    return nm
