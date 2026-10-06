"""
exp_tfim.py -- W1: a QUANTUM-NATIVE task where "classical ML on raw features" is undefined.

Inputs are ground states of the transverse-field Ising model (TFIM) on an n-spin chain,
    H(g) = -sum_i Z_i Z_{i+1} - g sum_i X_i - h sum_i Z_i     (h small: symmetry breaking)
obtained by exact diagonalization. Because the input is a quantum state (not a feature
vector), the only meaningful comparison is BETWEEN QUANTUM READOUTS: what a cut,
fusion-style pipeline can measure (single-qubit + within-half observables) versus what
reconstruction preserves (adds the cross-cut correlators). Two tasks on the same states:

  PHASE : ferromagnet (g<1) vs paramagnet (g>1), |g-1|>=0.25.  The order parameters
          (magnetization <Z>, transverse <X>) are LOCAL -> fusion should match joint.
          (A pilot "criticality band" task was also solved perfectly by BOTH readouts --
          any label that is a function of g alone is locally recoverable, since <X>
          tracks g -- so it is omitted as uninformative.)
  BOND  : is the cross-cut coupling physically present?  Class 1 = full TFIM chain;
          class 0 = same chain with the middle bond (m-1,m) DELETED (same g
          distribution).  In class 0 the state is a product across the cut (S_A = 0
          exactly) and <Z_{m-1}Z_m> factorizes; in class 1 the connected cross-cut
          correlation is nonzero.  The label IS the cross-cut correlation -- the
          canonical fusion-hard target -- though boundary spins leak some local signal
          (one fewer bond changes their local environment), which we report honestly.

Features (finite shots, like exp_quantum): LOCAL = <Z_q>,<X_q> for all q + nearest-neighbor
<Z_q Z_{q+1}> WITHIN each half; JOINT = LOCAL + the cross-cut correlators
<Z_{m-1}Z_m>, <X_{m-1}X_m>, <Z_{m-2}Z_{m+1}> (m = cut position). Heads: MLP (equal capacity).

Outputs: results/tfim_results.json + results/tfim.png
Run:  python -m experiments.exp_tfim
"""
import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from lfqml import qsim
from lfqml.fusion import MLPHead
from lfqml.metrics import classification_metrics

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS, exist_ok=True)

N = 8                    # spins; cut at m = N//2 (4|4)
M = N // 2
H_LONG = 0.05            # small longitudinal field: breaks Z2, gives <Z> != 0 in FM phase
SHOTS = 400
SEEDS = [0, 1, 2, 3, 4]
N_PER_CLASS = 120


# --------------------------------------------------------------------------- #
# TFIM ground states by exact diagonalization (dense; 2^8 = 256 dims)
# --------------------------------------------------------------------------- #
def _kron_ops():
    """Dense Z_i, X_i and Z_i Z_{i+1} terms for the n-spin chain (big-endian like qsim)."""
    I2 = np.eye(2)
    Zs, Xs = [], []
    for q in range(N):
        oz, ox = np.array([[1.0]]), np.array([[1.0]])
        for j in range(N):
            oz = np.kron(oz, qsim.Z if j == q else I2)
            ox = np.kron(ox, qsim.X if j == q else I2)
        Zs.append(oz); Xs.append(ox)
    return Zs, Xs


_ZS, _XS = _kron_ops()


def tfim_ground_state(g, broken_bond=False):
    """Ground state of the TFIM chain; broken_bond=True deletes the (m-1,m) coupling,
    making the state an exact product across the A|B cut (S_A = 0)."""
    Hm = np.zeros((2 ** N, 2 ** N), dtype=complex)   # qsim gate matrices are complex
    for i in range(N - 1):
        if broken_bond and i == M - 1:
            continue
        Hm -= _ZS[i] @ _ZS[i + 1]
    for i in range(N):
        Hm -= g * _XS[i] + H_LONG * _ZS[i]
    vals, vecs = np.linalg.eigh(Hm)
    return vecs[:, 0].astype(complex)


def cut_entropy(psi):
    return qsim.cut_entanglement_entropy(psi, list(range(M)), N)


# --------------------------------------------------------------------------- #
# Features (finite-shot estimates of expectations)
# --------------------------------------------------------------------------- #
def _shot(exact, rng):
    p = np.clip((1.0 + exact) / 2.0, 0.0, 1.0)
    return 2.0 * rng.binomial(SHOTS, p) / SHOTS - 1.0


def local_features(psi, rng):
    f = []
    for q in range(N):                                   # single-qubit Z, X
        f.append(_shot(qsim.expval(psi, {q: qsim.Z}, N), rng))
        f.append(_shot(qsim.expval(psi, {q: qsim.X}, N), rng))
    for a, b in [(i, i + 1) for i in range(M - 1)] + \
                [(i, i + 1) for i in range(M, N - 1)]:   # NN ZZ within each half
        f.append(_shot(qsim.expval(psi, {a: qsim.Z, b: qsim.Z}, N), rng))
    return f


def joint_features(psi, rng):
    f = local_features(psi, rng)
    for d in [{M - 1: qsim.Z, M: qsim.Z}, {M - 1: qsim.X, M: qsim.X},
              {M - 2: qsim.Z, M + 1: qsim.Z}]:           # cross-cut correlators
        f.append(_shot(qsim.expval(psi, d, N), rng))
    return f


# --------------------------------------------------------------------------- #
def sample_gs(task, rng):
    """Sample (g, broken_bond, label) triples for a task."""
    if task == "phase":     # FM (g<0.75) = 1  vs  PM (g>1.25) = 0; bond always intact
        g1 = rng.uniform(0.20, 0.75, N_PER_CLASS)
        g0 = rng.uniform(1.25, 1.80, N_PER_CLASS)
        gs = np.concatenate([g1, g0])
        broken = np.zeros(len(gs), bool)
        ys = np.concatenate([np.ones(len(g1), int), np.zeros(len(g0), int)])
    else:                   # bond: coupled (label 1) vs middle bond deleted (label 0),
        gs = rng.uniform(0.40, 1.60, 2 * N_PER_CLASS)   # same g distribution per class
        broken = np.concatenate([np.zeros(N_PER_CLASS, bool), np.ones(N_PER_CLASS, bool)])
        ys = (~broken).astype(int)
    order = rng.permutation(len(gs))
    return gs[order], broken[order], ys[order]


def run():
    rows = []
    # cache ground states per (g, broken) to avoid re-diagonalizing across seeds
    cache = {}

    def state(g, broken=False):
        key = (round(float(g), 4), bool(broken))
        if key not in cache:
            cache[key] = tfim_ground_state(key[0], broken_bond=key[1])
        return cache[key]

    for task in ["phase", "bond"]:
        for seed in SEEDS:
            rng = np.random.default_rng(seed)
            gs, broken, ys = sample_gs(task, rng)
            states = [state(g, b) for g, b in zip(gs, broken)]
            Floc = np.array([local_features(p, rng) for p in states])
            Fjnt = np.array([joint_features(p, rng) for p in states])
            ntr = int(0.7 * len(ys))
            fus = MLPHead(hidden=16, seed=seed).fit(Floc[:ntr], ys[:ntr])
            jnt = MLPHead(hidden=16, seed=seed).fit(Fjnt[:ntr], ys[:ntr])
            a_f = classification_metrics(ys[ntr:], fus.predict_proba(Floc[ntr:]))["accuracy"]
            a_j = classification_metrics(ys[ntr:], jnt.predict_proba(Fjnt[ntr:]))["accuracy"]
            ent1 = float(np.mean([cut_entropy(p) for p, yy in zip(states, ys) if yy == 1]))
            ent0 = float(np.mean([cut_entropy(p) for p, yy in zip(states, ys) if yy == 0]))
            rows.append({"task": task, "seed": seed, "acc_fusion": a_f, "acc_joint": a_j,
                         "gap": a_j - a_f, "mean_cut_entropy": (ent0 + ent1) / 2,
                         "S_A_class1": ent1, "S_A_class0": ent0})
            print(f"{task:6s} seed={seed}  fusion={a_f:.3f} joint={a_j:.3f} "
                  f"gap={a_j-a_f:+.3f}  S_A(y=1)={ent1:.3f} S_A(y=0)={ent0:.3f}", flush=True)

    # diagnostic curve: S_A(g) peaks at criticality
    g_grid = np.linspace(0.2, 1.8, 33)
    s_curve = [cut_entropy(state(g)) for g in g_grid]
    with open(os.path.join(RESULTS, "tfim_results.json"), "w") as f:
        json.dump({"rows": rows,
                   "entropy_curve": {"g": list(map(float, g_grid)),
                                     "S_A": list(map(float, s_curve))}}, f, indent=2)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.5))
    for i, task in enumerate(["phase", "bond"]):
        rs = [r for r in rows if r["task"] == task]
        ax1.bar(i - 0.17, np.mean([r["acc_fusion"] for r in rs]), 0.34, color="#2471a3",
                label="fusion" if i == 0 else None)
        ax1.bar(i + 0.17, np.mean([r["acc_joint"] for r in rs]), 0.34, color="#c0392b",
                label="joint/recon" if i == 0 else None)
    ax1.set_xticks([0, 1]); ax1.set_xticklabels(["phase", "bond detection"])
    ax1.set_ylabel("accuracy"); ax1.legend(); ax1.set_title("TFIM tasks")
    ax2.plot(g_grid, s_curve, "-", color="#8e44ad")
    ax2.axvline(1.0, ls=":", color="k", lw=0.8)
    ax2.set_xlabel("$g$"); ax2.set_ylabel("$S_A$ (bits)"); ax2.set_title("cut entanglement")
    fig.tight_layout(); fig.savefig(os.path.join(RESULTS, "tfim.png"), dpi=130)
    print("DONE -> results/tfim_results.json + tfim.png")


if __name__ == "__main__":
    run()
