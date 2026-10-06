"""
qdial_physical.py -- the Q-dial in the PHYSICAL quasiprobability (QPD) basis.

Why this module exists (paper revision items B1-B4)
---------------------------------------------------
The original Q-dial (cutting.py) truncates the operator-Schmidt double sum
    <O> = sum_{nu,mu} conj(c_nu) c_mu <A_nu|O_A|A_mu> <B_nu|O_B|B_mu>
by the 1-norm mass |c_nu c_mu|.  Reviewers objected that (a) the off-diagonal
(nu != mu) transition elements have no physical estimator, (b) the 1-norm mass is
not a sampling budget (Monte-Carlo overhead is the SQUARE of the 1-norm), and
(c) the endpoints are not the paper's headline models.

Here the SAME trained circuit is cut with qiskit-addon-cutting.  Each RZZ(phi)
cut is replaced by the Mitarai-Fujii quasiprobability decomposition into 6 local
operations (identity, Z x Z, and four measure-and-rotate channels), so k cuts give
6^k terms

    <O> = sum_i a_i  E^A_i(O_A) E^B_i(O_B),      sum_i |a_i| = kappa^k,
    kappa = 1 + 2|sin phi|,   physical sampling overhead = (sum_i |a_i|)^2.

Every term is the product of two expectation values of PHYSICALLY EXECUTABLE
subexperiments (no transition elements), exactly as the library's
`reconstruct_expectation_values` combines them.  The physical Q-dial keeps the
largest-|a_i| terms until their 1-norm reaches a fraction Q of kappa^k and reports
the retained Monte-Carlo overhead (sum_kept |a_i|)^2.

Public API
----------
  lfqml_to_qiskit(cfg, params, x)        -> QuantumCircuit (bit-exact w.r.t. lfqml)
  partition_labels(cfg)                  -> "AA..BB.."
  observable_to_pauli(obs, cfg)          -> little-endian Pauli label
  physical_terms(qc, labels, observables, estimator="exact"|"shots", shots=...)
                                         -> PhysicalTerms (a_i, E^A_i, E^B_i, T_i)
  qdial_kept_mask(coeffs, Q)             -> boolean mask of retained terms
  qdial_physical_features(terms, Q)      -> dict(features, overheads, n_kept, ...)
  PhysicalCutter(tq)                     -> batched (parametric) version for a
                                            trained model: terms for many inputs
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import PauliList, Statevector, SparsePauliOp
from qiskit_addon_cutting import partition_problem, generate_cutting_experiments
from qiskit_addon_cutting.utils.observable_grouping import ObservableCollection

from . import qsim
from .circuits import QNNConfig, Observable, _slice, coupling_angle


# --------------------------------------------------------------------------- #
# 1. lfqml circuit -> qiskit circuit (bit-exact)
# --------------------------------------------------------------------------- #
def _local_block_qiskit(qc: QuantumCircuit, block, qubits, depth: int):
    """Mirror of circuits.local_var_layer: per layer RY,RZ on each qubit (in
    qubit order), then a CZ chain (q, q+1), closed into a ring only if n_reg > 2."""
    n_reg = len(qubits)
    idx = 0
    for _ in range(depth):
        for q in qubits:
            qc.ry(block[idx], q); idx += 1
            qc.rz(block[idx], q); idx += 1
        for k in range(n_reg - 1):
            qc.cz(qubits[k], qubits[k + 1])
        if n_reg > 2:
            qc.cz(qubits[-1], qubits[0])


def lfqml_to_qiskit(cfg: QNNConfig, params: np.ndarray, x,
                    include_coupling: bool = True) -> QuantumCircuit:
    """Build the qiskit circuit whose statevector equals circuits.full_state(prepare(cfg, params, x)).

    Gate conventions coincide exactly (RY, RZ, CZ, RZZ(phi) = exp(-i phi/2 Z Z)),
    so the match is exact, not merely up to phase.  Qubit q of lfqml is qubit q of
    the qiskit circuit; only the *basis-index* convention differs (lfqml big-endian,
    qiskit little-endian), i.e. Statevector(qc).reverse_qargs().data == lfqml psi.

    `x` may be a numeric vector or a qiskit ParameterVector (for batched binding).
    """
    s = _slice(cfg)
    n = cfg.n
    qc = QuantumCircuit(n)
    A, B = cfg.A_qubits, cfg.B_qubits
    # angle encoding (A then B; register-local so the order is immaterial)
    for q in A:
        qc.ry(x[q], q)
    _local_block_qiskit(qc, params[slice(*s["pre_A"])], A, cfg.depth)
    for q in B:
        qc.ry(x[q], q)
    _local_block_qiskit(qc, params[slice(*s["pre_B"])], B, cfg.depth)
    # coupling layer -- the gates that get cut
    if include_coupling:
        phi = coupling_angle(cfg, params)
        for (a, b) in cfg.cross_pairs:
            qc.rzz(phi, a, b)
    # post-coupling local blocks
    _local_block_qiskit(qc, params[slice(*s["post_A"])], A, cfg.depth)
    _local_block_qiskit(qc, params[slice(*s["post_B"])], B, cfg.depth)
    return qc


def partition_labels(cfg: QNNConfig) -> str:
    return "A" * cfg.n_A + "B" * cfg.n_B


def _pauli_char(P: np.ndarray) -> str:
    for lab, M in zip(qsim.PAULI_LABELS, qsim.PAULIS):
        if np.allclose(P, M):
            return lab
    raise ValueError("observable factor is not a Pauli matrix")


def observable_to_pauli(obs: Observable, cfg: QNNConfig) -> str:
    """lfqml product observable -> little-endian Pauli label (char n-1-q <-> qubit q)."""
    chars = ["I"] * cfg.n
    for q, P in obs.global_dict(cfg).items():
        chars[cfg.n - 1 - q] = _pauli_char(P)
    return "".join(chars)


def lfqml_statevector(psi_lfqml: np.ndarray) -> Statevector:
    """Convert an lfqml (big-endian) amplitude vector to a qiskit Statevector."""
    n = int(round(np.log2(psi_lfqml.size)))
    return Statevector(psi_lfqml.reshape([2] * n).transpose(list(range(n))[::-1]).reshape(-1))


# --------------------------------------------------------------------------- #
# 2. Per-term evaluation of the library's subexperiments
# --------------------------------------------------------------------------- #
@dataclass
class PhysicalTerms:
    """All 6^k QPD terms of one cut circuit for a list of observables.

    coeffs[i]            a_i  (signed; sum|a_i| = kappa^k)
    E[label][i, o]       expectation value of subexperiment i on subsystem `label`
                         for observable o (includes the (-1)^{qpd parity} factor)
    T[i, o]              a_i * prod_label E[label][i, o]  -- the term value; sum_i T[i,o]
                         is exactly reconstruct_expectation_values' output
    """
    coeffs: np.ndarray
    E: dict
    T: np.ndarray
    observables: list          # Pauli labels (little-endian, full system)
    estimator: str
    shots: int | None

    @property
    def total_mass(self) -> float:          # sum_i |a_i| = kappa^k
        return float(np.sum(np.abs(self.coeffs)))

    @property
    def full_overhead(self) -> float:       # physical sampling overhead (kappa^k)^2
        return self.total_mass ** 2

    def reconstruct(self) -> np.ndarray:    # full (Q=1) reconstruction
        return self.T.sum(axis=0)


def _popcount(a: np.ndarray) -> np.ndarray:
    return np.bitwise_count(a.astype(np.uint64)).astype(np.int64)


def _bitarray_to_ints(arr: np.ndarray) -> np.ndarray:
    """(shots, nbytes) uint8, big-endian bytes -> int64 per shot (as int.from_bytes(..., 'big'))."""
    arr = np.asarray(arr)
    if arr.ndim == 1:
        arr = arr[:, None]
    nbytes = arr.shape[1]
    out = np.zeros(arr.shape[0], dtype=np.int64)
    for j in range(nbytes):
        out = (out << 8) | arr[:, j].astype(np.int64)
    return out


def _group_values_from_pub(data_pub, cog) -> np.ndarray:
    """Vectorised twin of qiskit_addon_cutting's _process_outcome_v2 averaged over shots."""
    obs_ints = _bitarray_to_ints(data_pub.observable_measurements.array)
    qpd_ints = _bitarray_to_ints(data_pub.qpd_measurements.array)
    qpd_factor = 1 - 2 * (_popcount(qpd_ints) & 1)
    vals = np.zeros(len(cog.pauli_bitmasks))
    for m, mask in enumerate(cog.pauli_bitmasks):
        obs = 1 - 2 * (_popcount(obs_ints & int(mask)) & 1)
        vals[m] = np.mean(qpd_factor * obs)
    return vals


def _group_values_exact(qc_sub: QuantumCircuit, cog) -> np.ndarray:
    """Exact (infinite-shot) value of a subexperiment: E[(-1)^{qpd parity} * O_meas].

    Branches the statevector at every QPD mid-circuit measurement (unnormalised
    projected states carry the branch probability; the sign carries the parity),
    then evaluates the Z-products the library would read out of the
    `observable_measurements` register.
    """
    n = qc_sub.num_qubits
    branches = [(1.0, Statevector.from_int(0, 2 ** n))]
    clbit_to_qubit = {}
    for inst in qc_sub.data:
        op = inst.operation
        if op.name == "measure":
            cl = inst.clbits[0]
            reg, pos = qc_sub.find_bit(cl).registers[0]
            q = qc_sub.find_bit(inst.qubits[0]).index
            if reg.name == "qpd_measurements":
                new = []
                for sgn, sv in branches:
                    t = sv.data.reshape([2] * n)
                    ax = n - 1 - q                       # qiskit: qubit q is axis n-1-q
                    t0 = t.copy(); t1 = t.copy()
                    idx1 = [slice(None)] * n; idx1[ax] = 1
                    idx0 = [slice(None)] * n; idx0[ax] = 0
                    t0[tuple(idx1)] = 0.0                # project on |0>
                    t1[tuple(idx0)] = 0.0                # project on |1>
                    new.append((sgn, Statevector(t0.reshape(-1))))
                    new.append((-sgn, Statevector(t1.reshape(-1))))
                branches = new
            elif reg.name == "observable_measurements":
                clbit_to_qubit[pos] = q
            else:                                        # pragma: no cover
                raise ValueError(f"unexpected classical register {reg.name}")
        elif op.name in ("barrier", "delay"):
            continue
        elif op.name == "reset":                         # pragma: no cover
            raise NotImplementedError("qubit re-use (reset) not supported here")
        else:
            qargs = [qc_sub.find_bit(qb).index for qb in inst.qubits]
            branches = [(sgn, sv.evolve(op, qargs=qargs)) for sgn, sv in branches]
    vals = np.zeros(len(cog.pauli_bitmasks))
    for m, mask in enumerate(cog.pauli_bitmasks):
        qubits = [clbit_to_qubit[j] for j in range(int(mask).bit_length()) if (int(mask) >> j) & 1]
        lab = ["I"] * n
        for q in qubits:
            lab[n - 1 - q] = "Z"
        Zop = SparsePauliOp("".join(lab))
        vals[m] = float(np.real(sum(sgn * sv.expectation_value(Zop) for sgn, sv in branches)))
    return vals


def _subsystem_expvals(group_vals_by_term, so: ObservableCollection, subobs: PauliList) -> np.ndarray:
    """(n_terms, n_obs): mirror of the library's `np.mean([...so.lookup[subobservable]])`."""
    n_terms = len(group_vals_by_term)
    out = np.zeros((n_terms, len(subobs)))
    for i in range(n_terms):
        for k, sub in enumerate(subobs):
            out[i, k] = np.mean([group_vals_by_term[i][m][nn] for m, nn in so.lookup[sub]])
    return out


def _aer_sampler(shots: int, seed=None):
    from qiskit_aer.primitives import SamplerV2
    bo = {"max_parallel_threads": 1, "max_parallel_experiments": 1,
          "shot_branching_enable": True, "shot_branching_sampling_enable": True}
    if seed is not None:
        bo["seed_simulator"] = int(seed)
    return SamplerV2(default_shots=shots, options={"backend_options": bo})


def evaluate_subexperiments(subexperiments: dict, coefficients, subobservables: dict,
                            estimator: str = "exact", shots: int = 4000, seed=None,
                            observables=None, sampler=None, backend=None,
                            transpile_seed=None, job_log: list | None = None,
                            keep_mask=None) -> PhysicalTerms:
    """Evaluate every library subexperiment and assemble the per-term matrix T.

    subexperiments / coefficients / subobservables are exactly the objects
    returned by partition_problem + generate_cutting_experiments(num_samples=inf).

    Hardware / external execution (QMI revision item F3): pass `sampler` (any
    SamplerV2-compatible primitive, e.g. qiskit_ibm_runtime.SamplerV2(mode=backend))
    and `backend` (circuits are then ISA-transpiled for it at optimization level 3
    with `transpile_seed`). Job ids are appended to `job_log` when the job exposes
    them. `keep_mask` (bool array over terms) restricts execution to the retained
    dial terms -- the non-executed terms get T_i = 0 -- so a single intermediate-Q
    point costs only the kept subexperiments.
    """
    coeffs = np.array([c[0] for c in coefficients], dtype=float)
    n_terms = len(coeffs)
    E = {}
    if sampler is None and estimator == "shots":
        sampler = _aer_sampler(shots, seed)
    keep = np.ones(n_terms, dtype=bool) if keep_mask is None else np.asarray(keep_mask, dtype=bool)
    for label, circs in subexperiments.items():
        so = ObservableCollection(subobservables[label])
        groups = so.groups
        ng = len(groups)
        assert len(circs) == n_terms * ng
        if estimator == "exact":
            gv = [[_group_values_exact(circs[i * ng + k], cog) if keep[i] else np.zeros(len(cog.pauli_bitmasks))
                   for k, cog in enumerate(groups)] for i in range(n_terms)]
        elif estimator == "shots":
            idx = [i * ng + k for i in range(n_terms) if keep[i] for k in range(ng)]
            run_circs = [circs[j] for j in idx]
            if backend is not None:
                from qiskit import transpile
                run_circs = transpile(run_circs, backend=backend, optimization_level=3,
                                      seed_transpiler=transpile_seed)
            job = sampler.run(run_circs, shots=shots)
            if job_log is not None and hasattr(job, "job_id"):
                try:
                    job_log.append(job.job_id())
                except Exception:
                    pass
            res = job.result()
            pos = {j: r for j, r in zip(idx, range(len(idx)))}
            gv = [[_group_values_from_pub(res[pos[i * ng + k]].data, cog) if keep[i]
                   else np.zeros(len(cog.pauli_bitmasks))
                   for k, cog in enumerate(groups)] for i in range(n_terms)]
        else:
            raise ValueError(estimator)
        E[label] = _subsystem_expvals(gv, so, subobservables[label])
    T = coeffs[:, None] * np.prod([E[l] for l in E], axis=0)
    return PhysicalTerms(coeffs=coeffs, E=E, T=T,
                         observables=list(observables) if observables is not None else None,
                         estimator=estimator, shots=(shots if estimator == "shots" else None))


def physical_terms(qc: QuantumCircuit, labels: str, observables, estimator: str = "exact",
                   shots: int = 4000, seed=None) -> PhysicalTerms:
    """Cut `qc` with qiskit-addon-cutting and return every term a_i E^A_i E^B_i.

    observables: list of little-endian Pauli labels (or a PauliList).
    estimator:   "exact"  -> infinite-shot value of each subexperiment (statevector
                             branching over the QPD measurements);
                 "shots"  -> Aer SamplerV2 with `shots` per subexperiment.
    """
    obs = observables if isinstance(observables, PauliList) else PauliList(list(observables))
    pp = partition_problem(circuit=qc, partition_labels=labels, observables=obs)
    sub, coeffs = generate_cutting_experiments(pp.subcircuits, pp.subobservables,
                                               num_samples=np.inf)
    return evaluate_subexperiments(sub, coeffs, pp.subobservables, estimator=estimator,
                                   shots=shots, seed=seed, observables=[str(p) for p in obs])


def library_reconstruction(qc: QuantumCircuit, labels: str, observables, shots: int = 4000,
                           seed=None, per_term: bool = False):
    """Reference path through the library's own reconstruct_expectation_values.

    With per_term=True also returns the (n_terms, n_obs) matrix obtained by calling
    reconstruct_expectation_values once per term with all OTHER coefficients zeroed
    -- the definition of T_i used in the paper -- on the SAME sampler results.
    """
    from qiskit_addon_cutting import reconstruct_expectation_values
    from qiskit_addon_cutting.qpd import WeightType
    obs = observables if isinstance(observables, PauliList) else PauliList(list(observables))
    pp = partition_problem(circuit=qc, partition_labels=labels, observables=obs)
    sub, coeffs = generate_cutting_experiments(pp.subcircuits, pp.subobservables,
                                               num_samples=np.inf)
    sampler = _aer_sampler(shots, seed)
    results = {lab: sampler.run(circs, shots=shots).result() for lab, circs in sub.items()}
    full = np.array(reconstruct_expectation_values(results, coeffs, pp.subobservables))
    if not per_term:
        return full, None, None
    n_terms = len(coeffs)
    Tlib = np.zeros((n_terms, len(obs)))
    for i in range(n_terms):
        ci = [(c[0], c[1]) if j == i else (0.0, WeightType.EXACT) for j, c in enumerate(coeffs)]
        Tlib[i] = reconstruct_expectation_values(results, ci, pp.subobservables)
    # my per-term values from the identical results
    E = {}
    for label, circs in sub.items():
        so = ObservableCollection(pp.subobservables[label])
        gv = [[_group_values_from_pub(results[label][i * len(so.groups) + k].data, cog)
               for k, cog in enumerate(so.groups)] for i in range(n_terms)]
        E[label] = _subsystem_expvals(gv, so, pp.subobservables[label])
    a = np.array([c[0] for c in coeffs])
    Tmine = a[:, None] * np.prod([E[l] for l in E], axis=0)
    return full, Tlib, Tmine


def exact_expectations(qc: QuantumCircuit, observables) -> np.ndarray:
    sv = Statevector(qc)
    return np.array([np.real(sv.expectation_value(SparsePauliOp(o))) for o in observables])


# --------------------------------------------------------------------------- #
# 3. The physical Q-dial
# --------------------------------------------------------------------------- #
def qdial_kept_mask(coeffs: np.ndarray, Q: float) -> np.ndarray:
    """Keep the largest-|a_i| terms until sum_kept |a_i| >= Q * sum_all |a_i|.

    Q = 0 keeps nothing; Q >= 1 keeps everything.  Ties in |a_i| are broken by the
    library's own order (stable sort), so the kept set is deterministic.
    """
    mass = np.abs(np.asarray(coeffs, dtype=float))
    keep = np.zeros(len(mass), dtype=bool)
    if Q >= 1.0:
        keep[:] = True
        return keep
    order = np.argsort(-mass, kind="stable")
    total, acc = mass.sum(), 0.0
    for idx in order:
        if acc >= Q * total - 1e-12:         # threshold met BEFORE adding: Q=0 keeps none
            break
        keep[idx] = True
        acc += mass[idx]
    return keep


def qdial_cost(coeffs: np.ndarray, Q: float) -> dict:
    """Cost bookkeeping of the retained term set under BOTH conventions."""
    mass = np.abs(np.asarray(coeffs, dtype=float))
    keep = qdial_kept_mask(coeffs, Q)
    kept = float(mass[keep].sum())
    total = float(mass.sum())
    return {
        "Q": float(Q),
        "n_kept": int(keep.sum()),
        "n_terms": int(len(mass)),
        # NEW (physical) convention: Monte-Carlo sampling overhead of the kept terms
        "retained_overhead_sq": max(1.0, kept ** 2),
        "full_overhead_sq": total ** 2,
        # OLD convention: 1-norm mass (what the earlier Q-dial labelled 'overhead')
        "retained_mass": kept,
        "retained_mass_ratio": kept / total if total > 0 else 0.0,
        "total_mass": total,
    }


def qdial_physical_features(terms: PhysicalTerms, Q: float) -> dict:
    """Partial physical reconstruction r_O(Q) = sum_{kept} T_i(O) + cost bookkeeping."""
    keep = qdial_kept_mask(terms.coeffs, Q)
    out = qdial_cost(terms.coeffs, Q)
    out["features"] = terms.T[keep].sum(axis=0) if keep.any() else np.zeros(terms.T.shape[1])
    out["kept_mask"] = keep
    return out


# --------------------------------------------------------------------------- #
# 4. Batched evaluation for a trained model (parametric subexperiments)
# --------------------------------------------------------------------------- #
class PhysicalCutter:
    """Cut a trained lfqml model ONCE (the decomposition depends only on phi), then
    bind the encoding angles of each input into the library's subexperiments.

    The subexperiments are exactly those generate_cutting_experiments produces for
    the concrete circuit of each input (verified in tests/test_qdial_physical.py);
    binding parameters merely avoids re-running the cutter 300 times per model.
    """

    def __init__(self, cfg: QNNConfig, params: np.ndarray, observables: list[Observable]):
        self.cfg, self.params = cfg, np.asarray(params, dtype=float)
        self.obs_labels = [observable_to_pauli(o, cfg) for o in observables]
        self.xv = ParameterVector("x", cfg.n)
        qc = lfqml_to_qiskit(cfg, self.params, self.xv)
        self.pp = partition_problem(circuit=qc, partition_labels=partition_labels(cfg),
                                    observables=PauliList(self.obs_labels))
        self.sub, self.coefficients = generate_cutting_experiments(
            self.pp.subcircuits, self.pp.subobservables, num_samples=np.inf)
        self.coeffs = np.array([c[0] for c in self.coefficients], dtype=float)
        self.phi = coupling_angle(cfg, self.params)
        self.kappa_per_cut = 1.0 + 2.0 * abs(np.sin(self.phi))
        # uncoupled circuit for the raw local (late-fusion) features
        self._local_qc = lfqml_to_qiskit(cfg, self.params, self.xv, include_coupling=False)
        self._local_qc.measure_all()

    def _bind(self, qc, x):
        # each subcircuit carries only its own register's encoding parameters
        binding = {p: float(x[int(p.index)]) for p in qc.parameters}
        return qc.assign_parameters(binding)

    def _bound(self, x):
        return {lab: [self._bind(c, x) for c in circs] for lab, circs in self.sub.items()}

    def terms(self, x, estimator="exact", shots=4000, seed=None) -> PhysicalTerms:
        return evaluate_subexperiments(self._bound(x), self.coefficients, self.pp.subobservables,
                                       estimator=estimator, shots=shots, seed=seed,
                                       observables=self.obs_labels)

    def term_matrix(self, X, estimator="exact", shots=4000, seed=None) -> np.ndarray:
        """(n_samples, n_terms, n_obs) term values for every input."""
        rng = np.random.default_rng(seed)
        out = []
        for x in X:
            s = int(rng.integers(0, 2 ** 31 - 1)) if seed is not None else None
            out.append(self.terms(x, estimator=estimator, shots=shots, seed=s).T)
        return np.array(out)

    def raw_local_features_shots(self, X, shots=4000, seed=None) -> np.ndarray:
        """<Z_q> of the UNCOUPLED circuit from `shots` Z-basis shots (late-fusion inputs).

        Same quantity as cutting.subcircuit_raw_features, estimated physically.
        """
        sampler = _aer_sampler(shots, seed)
        circs = [self._bind(self._local_qc, x) for x in X]
        res = sampler.run(circs, shots=shots).result()
        feats = np.zeros((len(X), self.cfg.n))
        for i in range(len(X)):
            ints = _bitarray_to_ints(res[i].data.meas.array)
            for q in range(self.cfg.n):
                feats[i, q] = 1.0 - 2.0 * np.mean((ints >> q) & 1)
        return feats


def qdial_curve_from_terms(coeffs, Tmat, Q) -> np.ndarray:
    """(n_samples, n_obs) partial reconstruction at level Q from a term matrix."""
    keep = qdial_kept_mask(coeffs, Q)
    if not keep.any():
        return np.zeros((Tmat.shape[0], Tmat.shape[2]))
    return Tmat[:, keep, :].sum(axis=1)
