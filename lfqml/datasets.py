"""
datasets.py -- classification datasets, scaled to rotation angles for encoding.

The star of the show is `synthetic_partition`, the CONTROLLED generator behind
hypotheses H1/H3.  A single knob `alpha` tunes how much the label depends on a
*joint* (cross-register) function versus *separable* per-register functions:

    score = (1 - alpha) * (u + v)        # separable  -> late-fusion friendly
            + alpha    * (C * u * v)     # joint parity -> needs cross-cut info
    label = 1[score > 0]

  * u = w_A . x_A ,  v = w_B . x_B  are per-register linear scores.
  * alpha = 0: additive/separable -- each register is independently informative,
    so late fusion should match reconstruction (small gap, low needed entanglement).
  * alpha = 1: label = sign(u)*sign(v), an XOR/parity across the cut -- neither
    register's marginal is informative, so late fusion must fail and reconstruction
    (which restores the cross-cut correlation) should win (large gap, high entanglement).

Sweeping alpha traces the predicted accuracy-gap-vs-cut-entanglement curve.

All datasets return (X, y) with X scaled into [-pi/2, pi/2] per feature (the
`qml-benchmarks` convention) and y in {0, 1}.  The first n_A features feed
register A, the remaining n_B features feed register B.
"""

from __future__ import annotations

import numpy as np
from sklearn.datasets import (make_moons, make_circles, load_iris, load_breast_cancer,
                              fetch_openml)
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, StandardScaler


ANGLE = np.pi / 2   # features scaled to [-ANGLE, ANGLE]


def _scale_to_angles(X: np.ndarray) -> np.ndarray:
    """Map each feature to [-pi/2, pi/2]."""
    scaler = MinMaxScaler(feature_range=(-ANGLE, ANGLE))
    return scaler.fit_transform(X)


def preprocess_split(Xtr_raw, Xte_raw, n_features: int | None = None,
                     standardize: bool = False, seed: int = 0):
    """Leakage-free preprocessing: fit (optional StandardScaler ->) (optional PCA ->)
    angle MinMax on the TRAIN split only, then transform both splits.

    Replaces the fit-on-full-dataset pipeline for every real-data experiment; the
    synthetic generator needs none of this (its features are i.i.d. uniform in
    [-1, 1] by construction).
    """
    Xtr = np.asarray(Xtr_raw, dtype=float)
    Xte = np.asarray(Xte_raw, dtype=float)
    if standardize:
        ss = StandardScaler().fit(Xtr)
        Xtr, Xte = ss.transform(Xtr), ss.transform(Xte)
    if n_features is not None and n_features < Xtr.shape[1]:
        pca = PCA(n_components=n_features, random_state=seed).fit(Xtr)
        Xtr, Xte = pca.transform(Xtr), pca.transform(Xte)
    scaler = MinMaxScaler(feature_range=(-ANGLE, ANGLE)).fit(Xtr)
    # test features are clipped to the encoding range (rotation angles are only
    # meaningful on [-pi/2, pi/2]; the clip is defined by the train fit alone)
    return scaler.transform(Xtr), np.clip(scaler.transform(Xte), -ANGLE, ANGLE)


# ---- raw (unpreprocessed) loaders for the leakage-free pipeline ------------ #
def moons_raw(n_samples: int = 300, noise: float = 0.15, seed: int = 0):
    return make_moons(n_samples=n_samples, noise=noise, random_state=seed)


def circles_raw(n_samples: int = 300, noise: float = 0.12, seed: int = 0):
    return make_circles(n_samples=n_samples, noise=noise, factor=0.5, random_state=seed)


def iris_binary_raw(classes=(0, 1)):
    data = load_iris()
    X, y = data.data, data.target
    mask = np.isin(y, classes)
    return X[mask], (y[mask] == classes[1]).astype(int)


def iris_multiclass_raw():
    data = load_iris()
    return data.data, data.target


def breast_cancer_raw():
    data = load_breast_cancer()
    return data.data, data.target


def _openml_image_raw(name, classes, n_per_class, seed):
    """Subsample raw pixels (no scaling/PCA); binary label = second class."""
    if name not in _OPENML_CACHE:
        X, y = fetch_openml(name, version=1, return_X_y=True, as_frame=False)
        _OPENML_CACHE[name] = (X, y.astype(int))
    X, y = _OPENML_CACHE[name]
    rng = np.random.default_rng(seed)
    idx = []
    for c in classes:
        ci = np.where(y == c)[0]
        idx.append(rng.choice(ci, size=min(n_per_class, len(ci)), replace=False))
    idx = np.concatenate(idx)
    return X[idx].astype(float), (y[idx] == classes[1]).astype(int)


def fashion_mnist_binary_raw(classes=(0, 6), n_per_class=250, seed=0):
    return _openml_image_raw("Fashion-MNIST", classes, n_per_class, seed)


def mnist_binary_raw(classes=(3, 6), n_per_class=250, seed=0):
    return _openml_image_raw("mnist_784", classes, n_per_class, seed)


def mnist_multiclass_raw(classes=(0, 1, 2), n_per_class=200, seed=0):
    """Raw-pixel MNIST K-class subsample, labels remapped to 0..K-1."""
    if "mnist_784" not in _OPENML_CACHE:
        X, y = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False)
        _OPENML_CACHE["mnist_784"] = (X, y.astype(int))
    X, y = _OPENML_CACHE["mnist_784"]
    rng = np.random.default_rng(seed)
    idx, lab = [], []
    for new, c in enumerate(classes):
        ci = np.where(y == c)[0]
        pick = rng.choice(ci, size=min(n_per_class, len(ci)), replace=False)
        idx.append(pick); lab.append(np.full(len(pick), new))
    return X[np.concatenate(idx)].astype(float), np.concatenate(lab)


def synthetic_partition(alpha: float, n_A: int = 2, n_B: int = 2,
                        n_samples: int = 300, coupling_strength: float = 2.0,
                        seed: int = 0):
    """Controlled separable<->parity dataset. See module docstring."""
    rng = np.random.default_rng(seed)
    n = n_A + n_B
    X = rng.uniform(-1.0, 1.0, size=(n_samples, n))
    wA = rng.standard_normal(n_A)
    wB = rng.standard_normal(n_B)
    u = X[:, :n_A] @ wA
    v = X[:, n_A:] @ wB
    # standardise u, v so the two terms are comparable across alpha
    u = (u - u.mean()) / (u.std() + 1e-9)
    v = (v - v.mean()) / (v.std() + 1e-9)
    score = (1 - alpha) * (u + v) + alpha * (coupling_strength * u * v)
    y = (score > 0).astype(int)
    return _scale_to_angles(X), y


def moons(n_samples: int = 300, noise: float = 0.15, seed: int = 0):
    X, y = make_moons(n_samples=n_samples, noise=noise, random_state=seed)
    return _scale_to_angles(X), y


def circles(n_samples: int = 300, noise: float = 0.12, seed: int = 0):
    X, y = make_circles(n_samples=n_samples, noise=noise, factor=0.5, random_state=seed)
    return _scale_to_angles(X), y


def iris_binary(n_features: int = 4, classes=(0, 1), seed: int = 0):
    data = load_iris()
    X, y = data.data, data.target
    mask = np.isin(y, classes)
    X, y = X[mask], y[mask]
    y = (y == classes[1]).astype(int)
    if n_features < X.shape[1]:
        X = PCA(n_components=n_features, random_state=seed).fit_transform(X)
    return _scale_to_angles(X), y


def iris_multiclass(n_features: int = 4, seed: int = 0):
    """Full 3-class Iris, features scaled to angles (G12 multiclass demo)."""
    data = load_iris()
    X, y = data.data, data.target
    if n_features < X.shape[1]:
        X = PCA(n_components=n_features, random_state=seed).fit_transform(X)
    return _scale_to_angles(X), y


def mnist_multiclass(classes=(0, 1, 2), n_features=4, n_per_class=200, seed=0):
    """MNIST K-class (default 0/1/2), PCA-reduced, labels remapped to 0..K-1 (G12)."""
    X, y = _OPENML_CACHE.get("mnist_784", (None, None))
    if X is None:
        X, y = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False)
        y = y.astype(int); _OPENML_CACHE["mnist_784"] = (X, y)
    rng = np.random.default_rng(seed)
    idx, lab = [], []
    for new, c in enumerate(classes):
        ci = np.where(y == c)[0]
        pick = rng.choice(ci, size=min(n_per_class, len(ci)), replace=False)
        idx.append(pick); lab.append(np.full(len(pick), new))
    idx = np.concatenate(idx); yy = np.concatenate(lab)
    Xs = StandardScaler().fit_transform(X[idx].astype(float))
    Xr = PCA(n_components=n_features, random_state=seed).fit_transform(Xs)
    return _scale_to_angles(Xr), yy


def breast_cancer(n_features: int = 4, seed: int = 0):
    data = load_breast_cancer()
    X, y = data.data, data.target
    X = PCA(n_components=n_features, random_state=seed).fit_transform(X)
    return _scale_to_angles(X), y


# --------------------------------------------------------------------------- #
# Downscaled image datasets (the standard QML "harder benchmark")
# --------------------------------------------------------------------------- #
_OPENML_CACHE = {}


def _openml_image(name, classes, n_features, n_per_class, seed):
    key = name
    if key not in _OPENML_CACHE:
        X, y = fetch_openml(name, version=1, return_X_y=True, as_frame=False)
        _OPENML_CACHE[key] = (X, y.astype(int))
    X, y = _OPENML_CACHE[key]
    rng = np.random.default_rng(seed)
    idx = []
    for c in classes:
        ci = np.where(y == c)[0]
        idx.append(rng.choice(ci, size=min(n_per_class, len(ci)), replace=False))
    idx = np.concatenate(idx)
    Xs = StandardScaler().fit_transform(X[idx].astype(float))
    Xr = PCA(n_components=n_features, random_state=seed).fit_transform(Xs)
    yy = (y[idx] == classes[1]).astype(int)
    return _scale_to_angles(Xr), yy


def fashion_mnist_binary(classes=(0, 6), n_features=4, n_per_class=250, seed=0):
    """Fashion-MNIST two-class (default T-shirt vs Shirt, a hard pair), PCA-reduced."""
    return _openml_image("Fashion-MNIST", classes, n_features, n_per_class, seed)


def mnist_binary(classes=(3, 6), n_features=4, n_per_class=250, seed=0):
    """MNIST two-class (default 3 vs 6, the QML-standard pair), PCA-reduced."""
    return _openml_image("mnist_784", classes, n_features, n_per_class, seed)


# --------------------------------------------------------------------------- #
# QUANTUM-DATA task: the signal lives in a cross-cut correlator, not marginals.
# --------------------------------------------------------------------------- #
def quantum_correlation_states(theta: float, n_samples: int = 300,
                               noise: float = 0.25, seed: int = 0):
    """Two-qubit entangled-state classification (the 'reconstruction is necessary' regime).

    For label y the state lives in the ZZ = +1 sector (y=1) or ZZ = -1 sector (y=0):
        y=1: cos(theta)|00> + sin(theta)|11>      (<Z_a Z_b> = +1 always)
        y=0: cos(theta)|01> + sin(theta)|10>      (<Z_a Z_b> = -1 always)
    theta is the DATA-ENTANGLEMENT knob:
        theta=0     -> product states; the local marginal <Z_b> carries y  (fusion wins)
        theta=pi/4  -> Bell states; ALL local marginals are 0 for both classes,
                       so the label is only in the cross-correlator <Z_a Z_b>.
                       No local measurement in ANY basis can classify -> fusion MUST fail;
                       only a joint (reconstruction/full-circuit) measurement succeeds.
    A small random local RY noise makes it a nontrivial learning problem.

    Returns (states, y) where states is an (n_samples, 4) complex array of 2-qubit
    state vectors (basis |00>,|01>,|10>,|11>).
    """
    from . import qsim
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, size=n_samples)
    states = np.zeros((n_samples, 4), dtype=complex)
    c, s = np.cos(theta), np.sin(theta)
    for i in range(n_samples):
        psi = np.zeros(4, dtype=complex)
        if y[i] == 1:
            psi[0], psi[3] = c, s                # |00>, |11>
        else:
            psi[1], psi[2] = c, s                # |01>, |10>
        # per-sample local noise: small RY on each qubit (2-qubit state)
        ea, eb = rng.normal(0, noise, size=2)
        psi = qsim.apply_1q(psi, qsim.ry(ea), 0, 2)
        psi = qsim.apply_1q(psi, qsim.ry(eb), 1, 2)
        psi /= np.linalg.norm(psi)
        states[i] = psi
    return states, y
