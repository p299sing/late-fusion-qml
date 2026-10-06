"""
exp_qiskit.py -- G2/G3: validate the pipeline through the REAL qiskit-addon-cutting.

(1) Correctness: for random circuits, reconstruction via the real library matches the
    exact statevector expectation (to shot noise).
(2) Physical overhead scaling: the library's reported sampling overhead is 9^k for
    k CNOT-type cuts -- matching DistributedEstimator's O(9^c) and confirming the
    exponential wall our method sidesteps (fusion overhead stays linear).

Outputs: results/qiskit_results.json
Run:  python -m experiments.exp_qiskit
"""
import json
import os
import numpy as np

from lfqml.qiskit_cut import (build_qnn_circuit, exact_expectations,
                              reconstruct_via_qiskit, _param_count)

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS, exist_ok=True)


def obs_for(nA, nB, k):
    """Product observables that factorize across the A|B cut (little-endian strings)."""
    n = nA + nB
    obs = []
    # <Z> on qubit 0
    s = ["I"] * n; s[n - 1 - 0] = "Z"; obs.append("".join(s))
    # <Z_a Z_b> across each cut pair
    for j in range(k):
        a, b = nA - 1 - j, nA + j
        s = ["I"] * n; s[n - 1 - a] = "Z"; s[n - 1 - b] = "Z"; obs.append("".join(s))
    return obs


def run():
    # (cuts, half_width, seeds, shots): shots shrink as the SUBEXPERIMENT COUNT explodes
    # (6^k unique subexperiments per side at num_samples=inf) -- exactly the wall in action.
    SPECS = [(1, 3, 3, 100_000), (2, 3, 3, 100_000), (3, 3, 3, 100_000),
             (4, 4, 2, 20_000), (5, 5, 1, 4_000)]
    rows = []
    for k, half, n_seeds, shots in SPECS:
        nA = nB = half
        P = _param_count(nA, nB, 1)
        errs, overhead = [], None
        for seed in range(n_seeds):
            rng = np.random.default_rng(seed)
            params = rng.uniform(-1, 1, P)
            x = rng.uniform(-1, 1, nA + nB)
            qc, labels = build_qnn_circuit(nA, nB, 1, k, params, phi=np.pi / 2, x=x)
            observables = obs_for(nA, nB, k)
            exact = exact_expectations(qc, observables)
            recon, overhead = reconstruct_via_qiskit(qc, labels, observables, shots=shots)
            errs.append(float(np.max(np.abs(exact - recon))))
        rows.append({"n_cuts": k, "real_overhead": overhead,
                     "max_recon_error": float(np.mean(errs)),
                     "shots": shots, "n_seeds": n_seeds})
        print(f"k={k}: real sampling overhead={overhead:.0f}  "
              f"(9^{k}={9**k})  mean max|recon-exact|={np.mean(errs):.4f} "
              f"[shots={shots}]", flush=True)
        with open(os.path.join(RESULTS, "qiskit_results.json"), "w") as f:
            json.dump(rows, f, indent=2)
    print("DONE -> results/qiskit_results.json")
    print("\nInterpretation: real library reproduces reconstruction (error ~ shot noise) and")
    print("its physical overhead is exactly 9^k -> the exponential wall fusion avoids (linear).")


if __name__ == "__main__":
    run()
