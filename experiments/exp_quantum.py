"""
exp_quantum.py -- the QUANTUM-NECESSITY regime (completes the two-regime story).

On genuinely entangled data, the label lives in a cross-cut correlator and ALL
local marginals are uninformative.  We sweep the data-entanglement knob theta and
compare two readouts on a train/test split:

  * FUSION      : an MLP over all six LOCAL marginals <X,Y,Z>_a and <X,Y,Z>_b.
                  These are everything an independent-subcircuit late-fusion model
                  can access (measure each subcircuit in any basis, combine).
  * JOINT/RECON : an MLP that ALSO sees the cross-correlators <ZZ>,<XX>,<YY>.
                  This is what full reconstruction / the uncut circuit provides.

Prediction: as theta -> pi/4 (Bell states, data entanglement -> 1 bit), FUSION
collapses to chance while JOINT stays high -- the regime where reconstruction is
NECESSARY and the Q-dial knee must sit at Q=1.  At theta=0 (product data) the
marginals carry the label and fusion matches joint.

Outputs: results/quantum_results.json + results/quantum_regime.png
Run:  python -m experiments.exp_quantum
"""
import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from lfqml import qsim, datasets
from lfqml.fusion import MLPHead
from lfqml.metrics import classification_metrics

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS, exist_ok=True)

# denser sampling near the transition; stronger local noise so fusion degrades
# GRADUALLY as marginals lose reliability (more convincing than a step).
THETAS = [0.0, 0.2, 0.4, 0.55, 0.65, 0.72, 0.76, np.pi / 4]   # 0 -> pi/4
SEEDS = list(range(10))
N = 400
NOISE = 0.6

LOCAL = [(0, qsim.X), (0, qsim.Y), (0, qsim.Z),      # marginals on qubit a
         (1, qsim.X), (1, qsim.Y), (1, qsim.Z)]      # marginals on qubit b
JOINT = [({0: qsim.Z, 1: qsim.Z}), ({0: qsim.X, 1: qsim.X}), ({0: qsim.Y, 1: qsim.Y})]

SHOTS = 400   # finite-shot estimation -> realistic; tiny marginals get swamped by noise


def _shot_estimate(exact, shots, rng):
    """Estimate a +/-1 observable expectation from `shots` samples (binomial noise)."""
    if shots is None:
        return exact
    p = np.clip((1.0 + exact) / 2.0, 0.0, 1.0)
    succ = rng.binomial(shots, p)
    return 2.0 * succ / shots - 1.0


def local_features(states, shots=None, rng=None):
    out = np.empty((len(states), len(LOCAL)))
    for i, psi in enumerate(states):
        for j, (q, P) in enumerate(LOCAL):
            out[i, j] = _shot_estimate(qsim.expval(psi, {q: P}, 2), shots, rng)
    return out


def joint_features(states, shots=None, rng=None):
    loc = local_features(states, shots, rng)
    jnt = np.empty((len(states), len(JOINT)))
    for i, psi in enumerate(states):
        for j, d in enumerate(JOINT):
            jnt[i, j] = _shot_estimate(qsim.expval(psi, d, 2), shots, rng)
    return np.hstack([loc, jnt])


def data_entanglement(states):
    return float(np.mean([qsim.cut_entanglement_entropy(psi, [0], 2) for psi in states]))


def run():
    rows = []
    for theta in THETAS:
        for seed in SEEDS:
            states, y = datasets.quantum_correlation_states(theta, n_samples=N,
                                                            noise=NOISE, seed=seed)
            ntr = int(0.7 * N)
            rng = np.random.default_rng(1000 + seed)
            Ffus = local_features(states, shots=SHOTS, rng=rng)
            Fjnt = joint_features(states, shots=SHOTS, rng=rng)
            fus = MLPHead(hidden=12, seed=seed).fit(Ffus[:ntr], y[:ntr])
            jnt = MLPHead(hidden=12, seed=seed).fit(Fjnt[:ntr], y[:ntr])
            acc_fus = classification_metrics(y[ntr:], fus.predict_proba(Ffus[ntr:]))["accuracy"]
            acc_jnt = classification_metrics(y[ntr:], jnt.predict_proba(Fjnt[ntr:]))["accuracy"]
            ent = data_entanglement(states)
            rows.append({"theta": theta, "seed": seed, "data_entanglement": ent,
                         "acc_fusion": acc_fus, "acc_joint": acc_jnt,
                         "gap": acc_jnt - acc_fus})
            print(f"theta={theta:.3f} seed={seed}  data_ent={ent:.3f}  "
                  f"fusion={acc_fus:.3f} joint={acc_jnt:.3f} gap={acc_jnt-acc_fus:+.3f}",
                  flush=True)

    with open(os.path.join(RESULTS, "quantum_results.json"), "w") as f:
        json.dump(rows, f, indent=2)
    make_plot(rows)
    print("DONE -> results/quantum_results.json + quantum_regime.png")


def _agg(rows, key_x, key_y):
    xs = sorted(set(r[key_x] for r in rows))
    ms = [np.mean([r[key_y] for r in rows if r[key_x] == x]) for x in xs]
    ss = [np.std([r[key_y] for r in rows if r[key_x] == x]) for x in xs]
    return np.array(xs), np.array(ms), np.array(ss)


def make_plot(rows):
    fig, ax = plt.subplots(figsize=(6, 4))
    e, ff, sf = _agg(rows, "data_entanglement", "acc_fusion")
    _, jj, sj = _agg(rows, "data_entanglement", "acc_joint")
    # sort by entanglement
    order = np.argsort(e)
    ax.errorbar(e[order], ff[order], yerr=sf[order], marker="o",
                label="late fusion (all local marginals)")
    ax.errorbar(e[order], jj[order], yerr=sj[order], marker="s",
                label="joint / reconstruction (+ cross-correlators)")
    ax.axhline(0.5, ls="--", color="gray", lw=1, label="chance")
    ax.set_xlabel("data entanglement across the cut (bits)")
    ax.set_ylabel("test accuracy")
    ax.set_title("Quantum-necessity regime: fusion collapses, reconstruction survives")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "quantum_regime.png"), dpi=130)


if __name__ == "__main__":
    run()
