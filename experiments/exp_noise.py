"""
exp_noise.py -- G5: reconstruction accumulates DEVICE noise across cuts; fusion does not.

Under a depolarizing + readout Aer noise model, reconstruction combines exponentially many
noisy subexperiments with signed coefficients, so its error grows with the number of cuts.
A late-fusion feature is a single subcircuit measurement whose device-noise error is FIXED
(independent of how many cuts the full model has). We report both vs the number of cuts.

Outputs: results/noise_results.json + results/noise.png
Run:  python -m experiments.exp_noise
"""
import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from qiskit.quantum_info import Statevector, SparsePauliOp

from lfqml.qiskit_cut import (build_qnn_circuit, exact_expectations,
                              reconstruct_via_qiskit, simple_depolarizing_noise,
                              fake_backend_noise, _param_count)

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS, exist_ok=True)


def subcircuit_noise_error(nA, nB, params, x, nm, shots=20000, basis_gates=None):
    """Error of a single A-subcircuit <Z0> measurement under device noise (fixed, no cuts).

    The probe replicates the REAL fusion A-subcircuit depth from build_qnn_circuit:
    encoding + pre-coupling local block (RY,RZ + CZ chain) + post-coupling local
    block -- the same gate count a fusion feature actually pays, so the comparison
    with reconstruction is depth-matched (audit fix C1). Param layout follows
    build_qnn_circuit: [A-pre | B-pre | A-post | B-post], 2 params/qubit/block.
    basis_gates transpiles to the device basis so a fake-backend noise model
    actually fires (its errors are per-native-gate); omit for the depolarizing model.
    """
    from qiskit import QuantumCircuit, transpile
    from qiskit_aer.primitives import EstimatorV2
    qc = QuantumCircuit(nA)

    def a_block(offset):
        idx = offset
        for q in range(nA):
            qc.ry(params[idx], q); idx += 1
            qc.rz(params[idx], q); idx += 1
        for i in range(nA - 1):
            qc.cz(i, i + 1)

    for q in range(nA):
        qc.ry(float(x[q]), q)                 # encoding
    a_block(0)                                # A pre-coupling block
    a_block(2 * nA + 2 * nB)                  # A post-coupling block (skip B-pre params)
    obs = SparsePauliOp("I" * (nA - 1) + "Z")
    exact = Statevector(qc).expectation_value(obs).real
    if basis_gates is not None:
        qc = transpile(qc, basis_gates=basis_gates, optimization_level=1)
    est = EstimatorV2(options={"backend_options": {"noise_model": nm}})
    noisy = est.run([(qc, obs)]).result()[0].data.evs
    return abs(float(noisy) - exact)


def _row(nm, k, seeds=5, basis_gates=None):
    nA = nB = 3
    P = _param_count(nA, nB, 1)
    rerr, ferr = [], []
    for seed in range(seeds):
        rng = np.random.default_rng(seed)
        params = rng.uniform(-1, 1, P); x = rng.uniform(-1, 1, nA + nB)
        qc, labels = build_qnn_circuit(nA, nB, 1, k, params, phi=np.pi / 2, x=x)
        n = nA + nB; s = ["I"] * n; s[n - 1] = "Z"; obs = ["".join(s)]
        exact = exact_expectations(qc, obs)[0]
        noisy, _ = reconstruct_via_qiskit(qc, labels, obs, shots=20000,
                                          noise_model=nm, basis_gates=basis_gates)
        rerr.append(abs(noisy[0] - exact))
        ferr.append(subcircuit_noise_error(nA, nB, params, x[:nA], nm,
                                           basis_gates=basis_gates))
    return float(np.mean(rerr)), float(np.std(rerr)), float(np.mean(ferr))


def run():
    # two noise models: hand-rolled depolarizing (native gates), and a REAL IBM
    # fake-backend model (needs basis-gate transpilation to actually fire).
    models = [("depolarizing", simple_depolarizing_noise(), None)]
    fake = fake_backend_noise("FakeManilaV2")
    if fake is not None:
        models.append(("fake_manila", fake, list(fake.basis_gates)))

    rows = []
    for mname, nm, bg in models:
        try:
            for k in [1, 2, 3]:
                rmean, rstd, fmean = _row(nm, k, basis_gates=bg)
                rows.append({"noise": mname, "n_cuts": k, "recon_noise_err": rmean,
                             "recon_noise_std": rstd, "fusion_noise_err": fmean})
                print(f"[{mname}] k={k}: recon_noise_err={rmean:.4f}+/-{rstd:.4f}  "
                      f"fusion_noise_err={fmean:.4f}", flush=True)
        except Exception as e:  # a fake-backend basis mismatch must not lose other models
            print(f"[{mname}] SKIPPED ({type(e).__name__}: {e})", flush=True)

    with open(os.path.join(RESULTS, "noise_results.json"), "w") as f:
        json.dump(rows, f, indent=2)
    rows = [r for r in rows if r["noise"] == "depolarizing"]  # figure uses depolarizing model
    ks = [r["n_cuts"] for r in rows]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(ks, [r["recon_noise_err"] for r in rows], "o-", color="#c0392b", label="reconstruction")
    ax.plot(ks, [r["fusion_noise_err"] for r in rows], "s-", color="#2471a3", label="late fusion (fixed)")
    ax.set_xlabel("number of cuts k"); ax.set_ylabel("|estimate - exact| under device noise")
    ax.set_xticks(ks); ax.legend()
    ratio = np.mean([r["recon_noise_err"] / max(r["fusion_noise_err"], 1e-9) for r in rows])
    ax.set_title(f"Device noise: reconstruction ~{ratio:.1f}x a fusion feature's error")
    fig.tight_layout(); fig.savefig(os.path.join(RESULTS, "noise.png"), dpi=130)
    print("DONE -> results/noise_results.json + noise.png")


if __name__ == "__main__":
    run()
