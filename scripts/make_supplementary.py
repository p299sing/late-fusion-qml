"""
make_supplementary.py -- build the ANONYMIZED supplementary zip for OpenReview.

Includes: code (lfqml/, experiments/, tests/, scripts/), result JSONs, figure/summary
scripts, requirements.txt, an anonymous reviewer README, and the compiled technical
appendix PDF. Excludes: narrative project docs (EXPLAINER/CODE_FLOW/RESULTS/plan --
they reference the authors' prior work in the first person), logs, venv, caches.

Safety: greps the staged tree for identifying strings and ABORTS if any are found.

Run:  python scripts/make_supplementary.py
Output: paper/submission/supplementary.zip
"""
import os
import re
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STAGE = os.path.join(ROOT, "paper", "submission", "_supp_stage")
OUT = os.path.join(ROOT, "paper", "submission", "supplementary.zip")

IDENTIFYING = re.compile(
    r"prabhjot|singh|toosi|buyya|melbourne|unimelb|psingh|claude", re.IGNORECASE)

README = """# Supplementary Code -- "How Much Reconstruction Does Quantum Machine Learning Need?"

Anonymized code and results for review. All experiments run on CPU (no GPU).

## Setup
    python -m venv .venv && source .venv/bin/activate
    pip install -r requirements.txt

## Verify the physics first (fast)
    python -m tests.test_qsim        # simulator sanity
    python -m tests.test_cutting     # reconstruction == uncut to <1e-10

## Reproduce (each writes results/<name>_results.json; figures regenerate from JSONs)
    python -m experiments.exp_h1            # alpha-sweep (Fig. 2's data; 10 seeds)
    python -m experiments.exp_independent   # independent vs frozen fusion (Fig. 2)
    python -m experiments.exp_scaling       # cost crossover (Fig. 1a; 10 seeds)
    python -m experiments.exp_qiskit        # real-library 9^k validation (Fig. 1b)
    python -m experiments.exp_shots         # shot robustness (Fig. 3a; 10 seeds)
    python -m experiments.exp_noise         # device noise, two models (Fig. 3b)
    python -m experiments.exp_quantum       # entangled-data boundary (Fig. 5; 10 seeds)
    python -m experiments.exp_tfim          # TFIM phase + bond detection (5 seeds)
    python -m experiments.exp_scaleup       # fusion at n=12..20, k<=10 (Fig. 6a)
    python -m experiments.exp_multilayer    # coupling-depth sweep (Fig. 6b)
    python -m experiments.exp_dense_alpha   # diagnostic correlations, n=104 (+CIs)
    python -m experiments.exp_marshall      # learned-truncation head-to-head
    python -m experiments.exp_faithful      # image data + faithful baselines (Table 3)
    python -m experiments.exp_multiclass    # 3-class softmax fusion
    python -m experiments.exp_ablation      # feature-set x head-width grid
    python -m experiments.exp_trainability  # barren-plateau gradient variance
    python -m experiments.exp_benchmarks    # standard datasets (Table 2)
    python -m experiments.exp_scale_qubits  # width scaling n=4..8
    python -m experiments.exp_random_features  # untrained-subcircuit ablation
    python -m experiments.exp_untrained_diag   # S_A at initialization vs knee
    python -m experiments.exp_shots_total   # equal-total-budget shot comparison
    python -m experiments.analyze_significance  # paired stats (appendix Table C)

    python make_figures.py                  # regenerates every paper figure from JSONs
    python summarize_results.py             # prints all paper-ready numbers

The provided results/*.json are the exact files behind every number in the paper.
Heavy experiments are resume-aware where noted in their docstrings; run them
sequentially (they are CPU-bound). See appendix.pdf for the full protocol table.
"""


def main():
    if os.path.exists(STAGE):
        shutil.rmtree(STAGE)
    os.makedirs(STAGE)

    # code + tests + scripts (the submission tooling itself does not ship)
    for d in ["lfqml", "experiments", "tests", "scripts"]:
        shutil.copytree(os.path.join(ROOT, d), os.path.join(STAGE, d),
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc",
                                                      "make_supplementary.py",
                                                      # legacy exploratory script (full-data
                                                      # CV protocol; superseded by the
                                                      # train-only classical anchors inside
                                                      # exp_faithful / exp_multiclass)
                                                      "exp_classical_baseline.py"))
    for f in ["make_figures.py", "summarize_results.py", "requirements.txt"]:
        shutil.copy(os.path.join(ROOT, f), STAGE)

    # result JSONs (the exact data behind the paper) + the compiled appendix
    os.makedirs(os.path.join(STAGE, "results"))
    for f in sorted(os.listdir(os.path.join(ROOT, "results"))):
        if f.endswith(".json"):
            shutil.copy(os.path.join(ROOT, "results", f), os.path.join(STAGE, "results", f))
    appendix = os.path.join(ROOT, "paper", "appendix.pdf")
    if os.path.exists(appendix):
        shutil.copy(appendix, STAGE)

    with open(os.path.join(STAGE, "README.md"), "w") as f:
        f.write(README)

    # anonymity check (text files only)
    bad = []
    for dirpath, _, files in os.walk(STAGE):
        for fn in files:
            if fn.endswith((".py", ".md", ".txt", ".json")):
                p = os.path.join(dirpath, fn)
                text = open(p, errors="ignore").read()
                for m in IDENTIFYING.finditer(text):
                    bad.append((os.path.relpath(p, STAGE), m.group(0)))
    if bad:
        for p, w in bad[:20]:
            print(f"IDENTIFYING STRING: {p}: {w!r}")
        sys.exit("ABORTED: anonymity check failed")

    if os.path.exists(OUT):
        os.remove(OUT)
    subprocess.run(["zip", "-qr", OUT, "."], cwd=STAGE, check=True)
    shutil.rmtree(STAGE)
    size = os.path.getsize(OUT) / 1e6
    print(f"OK -> {OUT} ({size:.1f} MB), anonymity check passed")


if __name__ == "__main__":
    main()
