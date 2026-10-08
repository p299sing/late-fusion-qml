# Late fusion of uncoupled quantum subcircuits

Code, result files and figure scripts for

> **How Much Reconstruction Does Quantum Machine Learning Need? Late Fusion of Uncoupled Quantum Subcircuits.**
> Prabhjot Singh, Adel N. Toosi, Rajkumar Buyya. Submitted to *Quantum Machine Intelligence*, 2026.

Circuit cutting lets a quantum neural network run as small subcircuits, but reassembling its outputs by quasiprobability reconstruction costs a number of subexperiments exponential in the number of cuts. This repository implements and evaluates **late fusion** (subcircuits trained with no cross-cut coupling, combined by a small classical head), the **reconstruction dial** between fusion and full reconstruction (in the physical quasiprobability basis via `qiskit-addon-cutting`), and the **cut-entanglement diagnostic**, against exact reconstruction, prior readouts, and tuned classical baselines.

## Layout

| path | contents |
|---|---|
| `lfqml/` | library: exact statevector simulator (`qsim.py`), two-register QNN and operator-Schmidt cut (`circuits.py`, `cutting.py`), physical QPD dial (`qdial_physical.py`), deep-coupling circuits (`deep_circuits.py`), trainers and heads (`train.py`, `fusion.py`, `models.py`), datasets, metrics |
| `experiments/` | one script per study, `python -m experiments.<name>`; each writes `results/<name>_results.json` |
| `results/` | the per-seed result files behind every figure and table in the paper |
| `make_figures.py` | regenerates every figure in `figures/` from `results/` |
| `summarize_results.py` | prints the headline numbers used in the text |
| `tests/` | unit tests (`python -m tests.<name>`): cut reconstruction reproduces the uncut circuit to 1e-10; Qiskit translation and physical dial to 1e-15 |
| `docs/ARCHITECTURE.md` | how the pieces fit together and how to extend them |

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export OMP_NUM_THREADS=1          # the experiments parallelise over processes; keep BLAS single-threaded
```

Python 3.13, NumPy 2.5, SciPy 1.18, scikit-learn 1.9, Qiskit 2.5, Qiskit-Aer 0.17, qiskit-addon-cutting 0.10 were used for the paper.

## Reproducing the paper

| paper element | command |
|---|---|
| Table 2 (standard datasets, F1/AUC) | `python -m experiments.exp_benchmarks` |
| Fig. 2 (uncoupled vs frozen fusion, alpha sweep) | `python -m experiments.exp_independent` |
| Fig. 3 (shot and device noise) | `python -m experiments.exp_shots`, `exp_shots_total`, `exp_noise` |
| Fig. 1(b) (physical 9^k overhead) | `python -m experiments.exp_qiskit` |
| live hardware (needs an IBM account) | `python -m experiments.exp_hardware`, `exp_hardware_budget`; multi-backend sensitivity and the dial on a device: `exp_hardware_qmi budget` / `exp_hardware_qmi dial` → `hardware_qmi_budget.json`, `hardware_qmi_dial.json` |
| physical reconstruction dial (Fig. 4) | `python -m experiments.exp_qdial_physical` |
| diagnostic grid (104 runs) | `python -m experiments.exp_dense_alpha` |
| classical split-feature fusion / train-free proxy | `python -m experiments.exp_classical_split` |
| entangled-data boundary (Fig. 5) | `python -m experiments.exp_quantum` |
| TFIM | `python -m experiments.exp_tfim` |
| scaling and the trained deep-coupling study | `python -m experiments.exp_scaleup`, `exp_scaleup14`, `exp_scale_qubits`, `exp_deep_coupling` |
| paired statistics and TOST equivalence (appendix D) | `python -m experiments.analyze_significance` |
| second circuit family, ansatz B (robustness check, Results 5.1) | `python -m experiments.exp_independent_ansatzB`, `exp_benchmarks_ansatzB`, then `python -m experiments.analyze_ansatzB` |
| all figures | `python make_figures.py` |

Every experiment threads an explicit seed through data generation, parameter initialisation and heads; result files are rewritten in place, so delete a file to recompute it.

## Hardware runs

`exp_hardware*.py` need a saved IBM Quantum account (`QiskitRuntimeService.save_account(...)`). They log backend name, calibration timestamp, transpiler seed and job IDs with every row.

## License

MIT. Please cite the paper (see `CITATION.cff`) if you use this code.
