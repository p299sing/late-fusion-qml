# CODE_FLOW.md — how the late-fusion QML codebase fits together

This document explains the architecture so you can modify any piece confidently.
It describes how the library, experiments and result files fit together.

> **TL;DR of the design choice.** The scientific core runs on a small, transparent
> **NumPy state-vector simulator with a hand-implemented gate cut**, *not* Qiskit.
> This is deliberate: it makes exact reconstruction verifiable to machine precision,
> makes the Q-dial a one-line truncation, and gives exact cut-entanglement — none of
> which Qiskit hands us cheaply. Qiskit / `qiskit-addon-cutting` is a **later,
> separate path** for camera-ready benchmark credibility (see "Integration points").

---

## 1. Directory layout

```
_late_fusion/
├── docs/ARCHITECTURE.md             # this file
├── CODE_FLOW.md                     # this file
├── lfqml/                           # the library
│   ├── qsim.py         # state-vector simulator: gates, expvals, entropy
│   ├── circuits.py     # target QNN + A|B partition ("cut architecture")
│   ├── cutting.py      # exact reconstruction, Q-dial, subcircuit features
│   ├── fusion.py       # classical readout heads (Logistic, MLP)
│   ├── train.py        # trainers: QNN(B0), independent fusion, Kawase sum,
│   │                   #   + multiclass softmax fusion (train_fusion_multiclass)
│   ├── datasets.py     # datasets incl. alpha-sweep, images, multiclass loaders
│   ├── metrics.py      # accuracy/F1/AUC + cut-entanglement
│   ├── models.py       # derive ALL baselines (B0/B1/B2/B6/B7 + Q-dial)
│   └── qiskit_cut.py   # G2/G5: REAL qiskit-addon-cutting pipeline + Aer noise
├── experiments/        # each writes results/<name>_*.json (+ some PNGs)
│   ├── exp_smoke.py exp_h1.py exp_quantum.py exp_benchmarks.py exp_h2.py
│   ├── exp_classical_baseline.py exp_independent.py exp_shots.py exp_scaling.py
│   ├── exp_goldilocks.py exp_trainability.py             # T-series + G8
│   ├── exp_qiskit.py     # G2 real-library validation (overhead = 9^k)
│   ├── exp_noise.py      # G5 device-noise: recon vs fusion feature error
│   ├── exp_multilayer.py # G3 coupling-depth sweep (entropy + unrecoverable corr)
│   ├── exp_faithful.py   # G4/G7 image data + faithful Kawase/classical baselines
│   │                     #   (classical baseline is LEAKAGE-FREE: CV on train only,
│   │                     #    scored on the same held-out split — audit fix M1)
│   ├── exp_multiclass.py # G12 softmax multiclass fusion (Iris-3, MNIST-012)
│   ├── exp_ablation.py   # G11 feature-set × head-width ablation
│   ├── exp_scaleup.py    # W2 fusion at n=12..20, k=1..10 (recon 9^k infeasible)
│   └── exp_tfim.py       # W1 TFIM ground states: quantum-native phase/criticality
├── scripts/migrate_qdial_cost.py  # one-time Q-dial cost migration (audit fix C2)
├── make_figures.py     # regenerate ALL 9 paper figures from results/*.json
├── paper/              # main.tex + sections/*.tex + references.bib + figures/
├── tests/
│   ├── test_qsim.py    # simulator sanity (Bell entropy = 1 bit, etc.)
│   └── test_cutting.py # THE load-bearing test: reconstruction == uncut (<1e-10)
├── results/            # JSON + PNG outputs (created on first run)
└── .venv/              # virtualenv (numpy/scipy/sklearn/matplotlib)
```

Run anything from the repo root with the venv active:
```bash
source .venv/bin/activate
python -m tests.test_qsim
python -m tests.test_cutting      # verifies the physics is exact
python -m experiments.exp_smoke   # ~2 min end-to-end sanity
python -m experiments.exp_h1      # ~20 min; writes results/h1_*.png + json
```

---

## 2. The data-flow, end to end

```
             features x  (first n_A -> register A, rest -> register B)
                 │
   circuits.prepare(cfg, theta, x)
                 │  encode + PRE-coupling local layers (register-local)
                 ▼
        PreparedCircuit
        ├── psiA_pre  (product component on A)   ── the pre-coupling state is
        ├── psiB_pre  (product component on B)      EXACTLY |psi_A> ⊗ |psi_B>
        ├── finish_A / finish_B  (POST-coupling local layers)
        └── phi  (coupling RZZ angle)
                 │
      ┌──────────┴───────────────────────────────────────────────┐
      ▼ UNCUT (B0)                                                 ▼ CUT
 circuits.full_state ── apply coupling layer on the joint state    cutting.build_cut
 circuits.full_features ── <O_i> of the whole circuit              │  operator-Schmidt of
      │                                                            │  each cross gate ->
      │                                                            │  branch states |A_mu>,|B_mu>
      ▼                                                            ▼
 train.train_qnn  ── jointly fit theta + linear head on B0    cutting.reconstruct_features(Q)
 (this is the ONLY optimisation; everything else is a readout)     ├── Q=1 -> exact (== B0)
      │                                                            └── Q<1 -> partial (Q-dial)
      ▼                                                        cutting.subcircuit_raw_features
 models.evaluate_baselines(tq, train/test) ───────────────────────┘   (inputs to late fusion)
      │
      ▼
 dict of {B0,B1,B2_linear,B2_mlp,B6,B7, Qdial[...], cut_entanglement, ...}
```

**Key invariant (why the science is trustworthy):** because everything before the
coupling layer is register-local, the pre-coupling state is an exact A⊗B product;
because everything after is local and the readout observables factorize as O_A⊗O_B,
the gate-cut reconstruction is **exact**. `tests/test_cutting.py` asserts
`reconstruction == uncut` to `<1e-10`. If you break the invariant (see §5), that test
will fail — which is the point.

---

## 3. Module responsibilities (what to edit for what)

| You want to change… | Edit | Notes |
|---|---|---|
| Gate set / add a gate | `qsim.py` | add a 2x2 or 4x4 matrix + (if 2q) it works with `apply_2q` |
| Ansatz depth/width, encoding, # cuts, entangler | `circuits.py` (`QNNConfig`, `local_var_layer`, `coupling_unitary_4x4`) | keep cross-gates ONLY in the coupling layer |
| Readout observable set | `circuits.default_observables` | must stay product (O_A⊗O_B) to remain reconstructable |
| How reconstruction / Q-dial truncation works | `cutting.py` (`reconstruct_observable`, `_term_weights`) | Q-dial ranks (nu,mu) terms by coefficient mass |
| What late fusion "sees" | `cutting.subcircuit_raw_features` | currently <Z> per subcircuit qubit; add <X>,<Y> via `paulis=` |
| Fusion head type (linear/MLP/attention) | `fusion.py` | add new head with `.fit`/`.predict_proba` |
| Which baselines are computed | `models.evaluate_baselines` | B0/B1/B2_lin/B2_mlp/B6/B7 + Q-dial |
| Datasets / the alpha knob | `datasets.py` | `synthetic_partition(alpha, ...)` is the H1/H3 generator |
| Cost model (gamma) | `cutting.sampling_overhead` | Q=1: gamma_S^(2k) (Schmidt 1-norm — a LOWER BOUND on the physical 9^k, see qiskit_cut); Q<1: ACTUAL retained 1-norm mass, floored at 1 (audit fix C2; old JSONs migrated by `scripts/migrate_qdial_cost.py`) |
| An experiment | `experiments/*.py` | copy `exp_h1.py` as a template |

---

## 4. The baselines (mirror of plan §5.1)

| Key in results dict | Meaning |
|---|---|
| `B0_uncut` | gold standard: trained head on exact features of the full circuit |
| `B1_reconstruction` | exact QPD reconstruction; **must equal B0** (`max_abs_vs_B0` reported) |
| `B2_fusion_linear` | late fusion, **linear** head over raw subcircuit `<Z>` |
| `B2_fusion_mlp` | late fusion, **MLP** head — the FAIR test (can form products of marginals) |
| `B6_kawase_sum` | Kawase-style parameter-free sum of expectation values (closest prior art) |
| `B7_ent_ablation` | coupling angle forced to 0 → product circuit → how much is genuinely quantum |
| `Qdial` | list over Q of {accuracy, cost_overhead, n_subexp} — the Pareto frontier |
| `cut_entanglement` | mean A:B entanglement entropy of the trained circuit (the H1/H3 quantity) |

**Why two fusion heads?** A linear head cannot multiply two marginals, so even a
*classical* nonlinear cross-correlation (e.g. `<Z_a Z_b> = <Z_a><Z_b>` for a product
state) shows up as a "gap". The MLP head *can* form such products, so its residual
gap over reconstruction is **genuinely non-factorisable (quantum)**. H1 is tested
against `B2_fusion_mlp`; the linear-vs-MLP difference decomposes the gap into
classical vs quantum parts. (This subtlety was found empirically in the smoke run —
see the plan's discussion.)

---

## 5. Scope & how to extend

**Current scope (by design, for exactness & speed):**
- A single **cross-coupling layer** (k cross gates on boundary qubits). This is what
  makes reconstruction exactly factorisable and verifiable.
- Small qubit counts (≤ ~10) — dense state vectors, statevector (noiseless).
- Product readout observables.

**To extend to multiple coupling layers / general circuits:** the clean
single-layer factorisation in `cutting.build_cut` no longer holds; you must
propagate the Pauli/Schmidt decomposition through each coupling layer (a transfer-
matrix style contraction) or switch to the Qiskit path (§6). Keep `test_cutting.py`
green as your correctness guard while doing this.

**To add shot noise / device noise:** replace exact `expval`/`matrix_element` in the
featurisers with sampled estimates (add a `shots=` path in `qsim`), or use the Aer
fake-backend path from the plan's protocol (§5.3a). Report accuracy at 1024/8192
shots and under `NoiseModel.from_backend(FakeSherbrooke)`.

---

## 6. Integration points (camera-ready path)

For benchmark credibility the paper will want a run through the real library:
- Build the same two-register QNN as a Qiskit `QuantumCircuit`.
- Cut with `qiskit-addon-cutting`: `partition_problem` / `cut_wires` →
  `generate_cutting_experiments` → Sampler → `reconstruct_expectation_values`.
- Our `models.evaluate_baselines` stays the same; only the featurisers
  (`_recon_features`, `_subcircuit_features`) get a Qiskit-backed implementation.
- Keep the NumPy path as the fast oracle to cross-check the Qiskit path.

This is intentionally **not** on the critical path for the H1/H3/Q-dial science.

---

## 7. Performance / scaling up

- Training (`train.py`) uses L-BFGS with **numerical** gradients — fine for ~20–40
  params. For bigger circuits, swap in a **parameter-shift** gradient (each angle
  appears once, so the shift rule is exact) and Adam. Hook: replace the `minimize`
  call and provide `jac=`.
- `evaluate_baselines` rebuilds the cut per sample; for large sweeps, cache
  `build_cut` results or vectorise the branch construction.
- The Q-dial reuses `build_cut`, so sweeping Q is nearly free once B1 exists.

---

## 8. Reproducibility

- All randomness is seeded (`init_params(seed)`, dataset `seed`, head `seed`).
- `experiments/exp_h1.py` writes `results/h1_results.json` (every number) plus PNGs.
- Pin versions: numpy 2.5, scipy 1.18, scikit-learn 1.9, matplotlib 3.11 (see `.venv`).
  A `requirements.txt` can be frozen with `pip freeze > requirements.txt`.
