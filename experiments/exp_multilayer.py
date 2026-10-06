"""
exp_multilayer.py -- G3: what multiple coupling LAYERS do to fusion.

Our controlled single-coupling-layer study is one point on a spectrum. Here we build circuits
with L = 1..4 coupling layers (RZZ across the cut, interleaved with random local layers)
and measure, with NO training (pure statevector, fast):
  * cut entanglement entropy S(rho_A)        -- the quantumness the cut carries
  * connected cross-cut correlation that marginal fusion features cannot recover:
        | <Z_a Z_b>  -  <Z_a><Z_b> |
Since a fusion head sees only per-qubit expectations, it can form the product <Z_a><Z_b> but
never the true <Z_a Z_b> when they differ; this residual is exactly the information late fusion
discards. It grows monotonically with coupling depth (entanglement) -- the boundary
Proposition 1 predicts -- while reconstruction (qiskit, validated separately) stays exact for
any L at 9^k cost.

Outputs: results/multilayer_results.json + results/multilayer.png
Run:  python -m experiments.exp_multilayer
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

N_A = N_B = 2
N = N_A + N_B
A = list(range(N_A))          # qubits 0,1
B = list(range(N_A, N))       # qubits 2,3
SEEDS = list(range(40))       # cheap -> average many random circuits


def random_local_layer(psi, rng):
    for q in range(N):
        psi = qsim.apply_1q(psi, qsim.ry(rng.uniform(0, 2 * np.pi)), q, N)
        psi = qsim.apply_1q(psi, qsim.rz(rng.uniform(0, 2 * np.pi)), q, N)
    return psi


def coupling_layer(psi, phi):
    # one RZZ across the A:B boundary (qubit 1 <-> qubit 2)
    return qsim.apply_2q(psi, qsim.rzz(phi), N_A - 1, N_A, N)


def run():
    rows = []
    for L in [1, 2, 3, 4]:
        ents, errs = [], []
        for seed in SEEDS:
            rng = np.random.default_rng(1000 * L + seed)
            psi = qsim.zero_state(N)
            psi = random_local_layer(psi, rng)
            for _ in range(L):
                psi = coupling_layer(psi, np.pi / 2)
                psi = random_local_layer(psi, rng)
            ents.append(qsim.cut_entanglement_entropy(psi, A, N))
            # cross-cut observable Z on qubit 1 (in A) and Z on qubit 2 (in B)
            zz = qsim.expval(psi, {1: qsim.Z, 2: qsim.Z}, N)
            za = qsim.expval(psi, {1: qsim.Z}, N)
            zb = qsim.expval(psi, {2: qsim.Z}, N)
            errs.append(abs(zz - za * zb))              # fusion product error
        rows.append({"layers": L, "cut_entropy": float(np.mean(ents)),
                     "fusion_error": float(np.mean(errs)),
                     "fusion_error_std": float(np.std(errs))})
        print(f"L={L}: cut_entropy={np.mean(ents):.3f} bits  "
              f"fusion_error={np.mean(errs):.3f}", flush=True)

    with open(os.path.join(RESULTS, "multilayer_results.json"), "w") as f:
        json.dump(rows, f, indent=2)
    Ls = [r["layers"] for r in rows]
    fig, ax1 = plt.subplots(figsize=(6, 4))
    ax1.plot(Ls, [r["cut_entropy"] for r in rows], "o-", color="#8e44ad", label="cut entropy")
    ax1.set_xlabel("coupling layers $L$"); ax1.set_ylabel("cut entanglement (bits)")
    ax1.set_xticks(Ls)
    ax2 = ax1.twinx()
    ax2.plot(Ls, [r["fusion_error"] for r in rows], "s--", color="#c0392b",
             label="fusion product error")
    ax2.set_ylabel(r"$|\langle Z_aZ_b\rangle-\langle Z_a\rangle\langle Z_b\rangle|$")
    ax1.set_title("Fusion degrades with coupling depth (recon stays exact)")
    fig.tight_layout(); fig.savefig(os.path.join(RESULTS, "multilayer.png"), dpi=130)
    print("DONE -> results/multilayer_results.json + multilayer.png")


if __name__ == "__main__":
    run()
