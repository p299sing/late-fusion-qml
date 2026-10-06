"""
exp_trainability.py -- G8: subcircuits dodge barren plateaus (a fusion advantage).

Barren plateaus: for a random hardware-efficient ansatz, the variance of a cost
gradient decays EXPONENTIALLY with the number of qubits [McClean et al. 2018].
Circuit cutting runs each subcircuit on HALF the qubits, so each subcircuit's
gradient variance is exponentially LARGER than the full circuit's -> subcircuits
are easier to train. We measure Var[d<Z0>/dtheta] over random initialisations vs
qubit count, and read off the full-circuit vs subcircuit (half-width) operating points.

Outputs: results/trainability_results.json + results/trainability.png
Run:  python -m experiments.exp_trainability
"""
import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from lfqml import qsim

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS, exist_ok=True)

QUBITS = [2, 4, 6, 8, 10]
DEPTH = 20            # deep enough to approach 2-design -> barren plateau regime
N_SAMPLES = 200       # random parameter initialisations
SHIFT = np.pi / 2


def random_ansatz_state(n, depth, params):
    """Hardware-efficient ansatz: RY(theta) per qubit + CZ ring, repeated `depth`."""
    psi = qsim.zero_state(n)
    idx = 0
    for _ in range(depth):
        for q in range(n):
            psi = qsim.apply_1q(psi, qsim.ry(params[idx]), q, n); idx += 1
        for q in range(n - 1):
            psi = qsim.apply_2q(psi, qsim.CZ, q, q + 1, n)
        if n > 2:
            psi = qsim.apply_2q(psi, qsim.CZ, n - 1, 0, n)
    return psi


def cost(n, depth, params):
    psi = random_ansatz_state(n, depth, params)
    return qsim.expval(psi, {0: qsim.Z}, n)      # <Z_0>


def grad_variance(n, depth, n_samples, seed=0):
    """Var over random inits of d<Z0>/dtheta_k for a fixed middle parameter k."""
    rng = np.random.default_rng(seed)
    n_params = depth * n
    k = n_params // 2                             # a central parameter
    grads = []
    for _ in range(n_samples):
        p = rng.uniform(0, 2 * np.pi, size=n_params)
        pp, pm = p.copy(), p.copy()
        pp[k] += SHIFT; pm[k] -= SHIFT
        grads.append(0.5 * (cost(n, depth, pp) - cost(n, depth, pm)))
    return float(np.var(grads))


def run():
    rows = []
    for n in QUBITS:
        v = grad_variance(n, DEPTH, N_SAMPLES, seed=0)
        rows.append({"qubits": n, "grad_var": v})
        print(f"qubits={n:2d}  Var[grad]={v:.3e}", flush=True)
    with open(os.path.join(RESULTS, "trainability_results.json"), "w") as f:
        json.dump(rows, f, indent=2)

    ns = [r["qubits"] for r in rows]
    vs = [r["grad_var"] for r in rows]
    # exponential fit slope (log-linear)
    slope = np.polyfit(ns, np.log(vs), 1)[0]
    print(f"\nlog Var[grad] slope = {slope:.3f} per qubit  (barren plateau if < 0)")

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.semilogy(ns, vs, marker="o", color="crimson", label="gradient variance")
    # annotate: a full n-qubit model vs its n/2-qubit subcircuits
    ax.set_xlabel("number of qubits")
    ax.set_ylabel("Var[ ∂⟨Z₀⟩/∂θ ]  (log)")
    ax.set_title(f"Barren plateau: gradient variance decays exp. (slope {slope:.2f}/qubit)\n"
                 "subcircuits use half the qubits → exponentially larger gradients")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "trainability.png"), dpi=130)
    print("DONE -> results/trainability_results.json + trainability.png")


if __name__ == "__main__":
    run()
