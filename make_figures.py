"""
make_figures.py -- regenerate ALL paper figures from results/*.json.

One function per figure so you can tweak layout/colors/sizes independently.
Writes PNG (and PDF) into paper/figures/ (two-column AAAI sizes). Run:  python make_figures.py
Journal (Quantum Machine Intelligence, single column, 372pt text width) sizes:
    python make_figures.py --journal        -> paper/figures_qmi/
In journal mode every single-plot figure is 3.6 x 2.45 in and every panel figure spans the
5.15 in text width; all are included at natural size, so lettering is 8-9 pt on the page.
Edit STYLE below for global look; edit each fig_*() for per-figure layout.
"""
import json
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
OUT = os.path.join(HERE, "paper", "figures")
os.makedirs(OUT, exist_ok=True)

# ---- global style (edit here) --------------------------------------------- #
STYLE = {
    "figsize": (3.4, 2.6),      # single-column width (inches)
    "dpi": 300,
    "recon_color": "#c0392b",   # reconstruction (red)
    "fusion_color": "#2471a3",  # fusion (blue)
    "accent": "#196f3d",        # green (>=4.5:1 on white)
    "gray": "#566573",          # (>=4.5:1 on white)
    "fontsize": 8,
}
JOURNAL = ("--journal" in sys.argv) or os.environ.get("LFQML_FIG_MODE", "").lower() == "journal"
if JOURNAL:
    OUT = os.path.join(HERE, "paper", "figures_qmi"); os.makedirs(OUT, exist_ok=True)
    STYLE["fontsize"] = 9
W1 = 3.6 if JOURNAL else 3.4      # single-plot figure width (in); journal: included at natural size
H1 = 2.45                         # uniform single-plot height in journal mode
WP = 5.15 if JOURNAL else 7.0     # full-text-width panel figures
SMALL = STYLE["fontsize"] - 1     # legends, annotations (8 pt in journal mode)
def _fs(h=None):
    """single-plot figsize: uniform in journal mode, legacy height otherwise."""
    return (W1, H1 if (JOURNAL or h is None) else h)
def _fsp(h_journal, h_aaai=None):
    """panel figsize spanning the text width."""
    return (WP, h_journal if JOURNAL else (h_aaai or h_journal))
STYLE["figsize"] = _fs(2.6)
# Typeface for the Fig. 1 schematic only. "Comic Neue" (OFL, files in paper/fonts/) is the
# open-source Comic Sans equivalent; set to None to fall back to the default sans-serif.
PIPELINE_FONT = None   # default sans-serif, as in every other figure; "Comic Neue" was tried and reverted
plt.rcParams.update({
    # AAAI forbids Type 3 fonts, even inside graphics -> embed TrueType (Type 42)
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "font.size": STYLE["fontsize"], "axes.titlesize": STYLE["fontsize"],
    "axes.labelsize": STYLE["fontsize"], "legend.fontsize": STYLE["fontsize"] - 1,
    "xtick.labelsize": STYLE["fontsize"] - 1, "ytick.labelsize": STYLE["fontsize"] - 1,
    "axes.spines.top": False, "axes.spines.right": False,
})


def _load(name):
    with open(os.path.join(RES, name)) as f:
        return json.load(f)


def _save(fig, name, w_pad=None, tight=True):
    if tight:
        fig.tight_layout(pad=0.4, **({"w_pad": w_pad} if w_pad else {}))
    fig.savefig(os.path.join(OUT, name + ".png"), dpi=STYLE["dpi"], bbox_inches="tight")
    fig.savefig(os.path.join(OUT, name + ".pdf"), bbox_inches="tight")
    plt.close(fig)
    print("  wrote", name)


def _agg(rows, xkey, ykey):
    xs = sorted(set(r[xkey] for r in rows))
    m = [np.mean([r[ykey] for r in rows if r[xkey] == x]) for x in xs]
    s = [np.std([r[ykey] for r in rows if r[xkey] == x]) for x in xs]
    return np.array(xs), np.array(m), np.array(s)


# --------------------------------------------------------------------------- #
def fig_cost_crossover():
    rows = _load("scaling_results.json")
    ks, rec, _ = _agg(rows, "n_cuts", "recon_overhead")
    _, fus, _ = _agg(rows, "n_cuts", "fusion_overhead")
    _, a0, _ = _agg(rows, "n_cuts", "acc_B0")
    _, a2, _ = _agg(rows, "n_cuts", "acc_B2_mlp")
    fig, ax1 = plt.subplots(figsize=STYLE["figsize"])
    ax1.plot(ks, rec, "o-", color=STYLE["recon_color"], label="reconstruction")
    ax1.plot(ks, fus, "s-", color=STYLE["fusion_color"], label="late fusion")
    ax1.set_yscale("log"); ax1.set_xlabel("number of cuts $k$")
    ax1.set_ylabel("sampling overhead"); ax1.set_xticks(ks)
    ax2 = ax1.twinx()
    ax2.plot(ks, a0, "^--", color=STYLE["gray"], lw=1, label="acc (recon)")
    ax2.plot(ks, a2, "v--", color=STYLE["accent"], lw=1, label="acc (fusion)")
    ax2.set_ylabel("accuracy"); ax2.set_ylim(0, 1.05); ax2.spines["top"].set_visible(False)
    lines = ax1.get_lines() + ax2.get_lines()
    ax1.legend(lines, [l.get_label() for l in lines], fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=4, frameon=False)
    _save(fig, "cost_crossover")


def fig_independent():
    rows = _load("independent_results.json")
    fig, ax = plt.subplots(figsize=_fs(1.95))
    for key, lab, mk, c in [("acc_B0", "reconstruction", "^", STYLE["recon_color"]),
                            ("acc_frozen", "frozen fusion", "s", STYLE["gray"]),
                            ("acc_indep", "independent fusion", "o", STYLE["fusion_color"])]:
        xs, m, s = _agg(rows, "alpha", key)
        ax.errorbar(xs, m, yerr=s, marker=mk, color=c, label=lab, capsize=2, ms=4)
    ax.set_xlabel(r"$\alpha$ (cross-cut dependence)"); ax.set_ylabel("accuracy")
    ax.legend(); _save(fig, "independent")


def fig_shot_robustness():
    rows = _load("shots_results.json")
    xs, rec, _ = _agg(rows, "shots", "acc_recon")
    _, fus, _ = _agg(rows, "shots", "acc_fusion")
    fig, ax = plt.subplots(figsize=STYLE["figsize"])
    ax.plot(xs, rec, "o-", color=STYLE["recon_color"], label="reconstruction ($\\gamma^2$ noise)")
    ax.plot(xs, fus, "s-", color=STYLE["fusion_color"], label="late fusion")
    ax.set_xscale("log"); ax.set_xlabel("shots per expectation"); ax.set_ylabel("accuracy")
    ax.legend(); _save(fig, "shot_robustness")


def fig_regime_boundary():
    rows = _load("quantum_results.json")
    for r in rows:                       # group seeds by rounded entanglement so each
        r["data_entanglement"] = round(r["data_entanglement"], 3)   # point is a 10-seed mean
    xs, ff, sf = _agg(rows, "data_entanglement", "acc_fusion")
    _, jj, sj = _agg(rows, "data_entanglement", "acc_joint")
    o = np.argsort(xs)
    fig, ax = plt.subplots(figsize=_fs(1.95))
    ax.errorbar(xs[o], ff[o], yerr=sf[o], marker="s", color=STYLE["fusion_color"],
                label="late fusion (local)", capsize=2, ms=4)
    ax.errorbar(xs[o], jj[o], yerr=sj[o], marker="^", color=STYLE["recon_color"],
                label="joint / reconstruction", capsize=2, ms=4)
    ax.axhline(0.5, ls=":", color="k", lw=0.8)
    ax.set_xlabel("data entanglement across cut (bits)"); ax.set_ylabel("accuracy")
    ax.legend(); _save(fig, "regime_boundary")


def fig_trainability():
    rows = _load("trainability_results.json")
    ns = [r["qubits"] for r in rows]; vs = [r["grad_var"] for r in rows]
    slope = np.polyfit(ns, np.log(vs), 1)[0]
    fig, ax = plt.subplots(figsize=STYLE["figsize"])
    ax.semilogy(ns, vs, "o-", color=STYLE["recon_color"])
    ax.set_xlabel("number of qubits"); ax.set_ylabel(r"Var$[\partial\langle Z_0\rangle/\partial\theta]$")
    ax.set_title(f"barren plateau (slope {slope:.2f}/qubit)", fontsize=STYLE["fontsize"] - 1)
    _save(fig, "trainability")


def fig_qdial_frontier():
    """Q-dial Pareto frontier (accuracy vs retained cost) for low/mid/high alpha."""
    rows = _load("h1_results.json")
    alphas = sorted(set(r["alpha"] for r in rows))
    picks = [alphas[0], alphas[len(alphas) // 2], alphas[-1]]
    fig, ax = plt.subplots(figsize=_fs(1.78))
    colors = [STYLE["fusion_color"], STYLE["accent"], STYLE["recon_color"]]
    for a, c in zip(picks, colors):
        curves = [r["Qdial"] for r in rows if r["alpha"] == a]
        Qs = [q["Q"] for q in curves[0]]
        cost = [np.mean([cv[i]["cost_overhead"] for cv in curves]) for i in range(len(Qs))]
        acc = [np.mean([cv[i]["accuracy"] for cv in curves]) for i in range(len(Qs))]
        std = [np.std([cv[i]["accuracy"] for cv in curves]) for i in range(len(Qs))]
        ax.errorbar(cost, acc, yerr=std, marker="o", ms=3, color=c, capsize=2,
                    lw=1, label=f"$\\alpha={a}$")
    ax.set_xscale("log"); ax.set_xlabel("retained sampling overhead (floored at 1)")
    ax.set_ylabel("accuracy"); ax.legend()
    _save(fig, "qdial_frontier")


def fig_device_noise():
    """G5: reconstruction's device-noise error vs a fusion feature's (fixed) across cuts."""
    rows = _load("noise_results.json")
    rows = [r for r in rows if r.get("noise", "depolarizing") == "depolarizing"]
    ks = [r["n_cuts"] for r in rows]
    fig, ax = plt.subplots(figsize=STYLE["figsize"])
    ax.plot(ks, [r["recon_noise_err"] for r in rows], "o-", color=STYLE["recon_color"],
            label="reconstruction")
    ax.plot(ks, [r["fusion_noise_err"] for r in rows], "s-", color=STYLE["fusion_color"],
            label="fusion feature (fixed)")
    ax.set_xlabel("number of cuts $k$"); ax.set_ylabel(r"$|\hat O-O|$ under device noise")
    ax.set_xticks(ks); ax.set_ylim(0, None); ax.legend()
    _save(fig, "device_noise")


def fig_qiskit_overhead():
    """G2: real qiskit-addon-cutting overhead is exactly 9^k, reconstruction ~ shot noise."""
    rows = _load("qiskit_results.json")
    ks = [r["n_cuts"] for r in rows]
    fig, ax = plt.subplots(figsize=STYLE["figsize"])
    ax.semilogy(ks, [r["real_overhead"] for r in rows], "o-", color=STYLE["recon_color"],
                label="real overhead")
    ax.semilogy(ks, [9 ** k for k in ks], ":", color="k", lw=0.8, label="$9^k$")
    ax.set_xlabel("number of cuts $k$"); ax.set_ylabel("sampling overhead $\\gamma^2$")
    ax.set_xticks(ks); ax.legend()
    ax.set_title("qiskit-addon-cutting: overhead $= 9^k$", fontsize=STYLE["fontsize"] - 1)
    _save(fig, "qiskit_overhead")


def fig_multilayer():
    """G3: cut entropy and the connected cross-cut correlation grow with coupling depth."""
    rows = _load("multilayer_results.json")
    Ls = [r["layers"] for r in rows]
    fig, ax1 = plt.subplots(figsize=STYLE["figsize"])
    ax1.plot(Ls, [r["cut_entropy"] for r in rows], "o-", color="#8e44ad", label="cut entropy")
    ax1.set_xlabel("coupling layers $L$"); ax1.set_ylabel("cut entanglement (bits)")
    ax1.set_xticks(Ls)
    ax2 = ax1.twinx()
    ax2.plot(Ls, [r["fusion_error"] for r in rows], "s--", color=STYLE["recon_color"],
             label="unrecoverable corr.")
    ax2.set_ylabel(r"$|\langle Z_aZ_b\rangle-\langle Z_a\rangle\langle Z_b\rangle|$")
    ax2.spines["top"].set_visible(False)
    lines = ax1.get_lines() + ax2.get_lines()
    ax1.legend(lines, [l.get_label() for l in lines], fontsize=7, loc="lower right")
    _save(fig, "multilayer")


def fig_scale_qubits():
    """AAAI scale check: fusion tracks reconstruction as register width grows to 8 qubits."""
    rows = _load("scale_qubits_results.json")
    ns = [r["qubits"] for r in rows]
    fig, ax = plt.subplots(figsize=STYLE["figsize"])
    ax.errorbar(ns, [r["acc_recon"] for r in rows], yerr=[r.get("acc_recon_std", 0) for r in rows],
                marker="^", color=STYLE["recon_color"], label="reconstruction", capsize=2, ms=4)
    ax.errorbar(ns, [r["acc_fusion"] for r in rows], yerr=[r.get("acc_fusion_std", 0) for r in rows],
                marker="o", color=STYLE["fusion_color"], label="late fusion", capsize=2, ms=4)
    ax.set_xlabel("qubits (register width)"); ax.set_ylabel("accuracy")
    ax.set_xticks(ns); ax.legend(); _save(fig, "scale_qubits")


def fig_overhead_panel():
    """(a) cost crossover on our simulator + (b) real qiskit-addon-cutting 9^k, one figure*."""
    fig, (ax1, ax3) = plt.subplots(1, 2, figsize=_fsp(2.4, 2.3))
    rows = _load("scaling_results.json")
    ks, rec, _ = _agg(rows, "n_cuts", "recon_overhead")
    _, fus, _ = _agg(rows, "n_cuts", "fusion_overhead")
    _, a0, _ = _agg(rows, "n_cuts", "acc_B0")
    _, a2, _ = _agg(rows, "n_cuts", "acc_B2_mlp")
    ax1.plot(ks, rec, "o-", color=STYLE["recon_color"], label="reconstruction")
    ax1.plot(ks, fus, "s-", color=STYLE["fusion_color"], label="late fusion")
    ax1.set_yscale("log"); ax1.set_xlabel("number of cuts $k$")
    ax1.set_ylabel("sampling overhead"); ax1.set_xticks(ks); ax1.set_title("(a) cost crossover")
    ax2 = ax1.twinx()
    ax2.plot(ks, a0, "^--", color=STYLE["gray"], lw=1, label="acc (recon)")
    ax2.plot(ks, a2, "v--", color=STYLE["accent"], lw=1, label="acc (fusion)")
    ax2.set_ylabel("accuracy"); ax2.set_ylim(0, 1.05); ax2.spines["top"].set_visible(False)
    lines = ax1.get_lines() + ax2.get_lines()
    ax1.legend(lines, [l.get_label() for l in lines], fontsize=SMALL, loc="upper center", bbox_to_anchor=(0.5, -0.26), ncol=2 if JOURNAL else 4, frameon=False)
    rows = _load("qiskit_results.json")
    ks = [r["n_cuts"] for r in rows]
    ax3.semilogy(ks, [r["real_overhead"] for r in rows], "o-", color=STYLE["recon_color"],
                 label="real overhead")
    ax3.semilogy(ks, [9 ** k for k in ks], ":", color="k", lw=0.8, label="$9^k$")
    ax3.set_xlabel("number of cuts $k$"); ax3.set_ylabel("physical overhead $\\gamma^2$")
    ax3.set_xticks(ks); ax3.legend(loc="lower right")
    ax3.set_title("(b) qiskit-addon-cutting: overhead $= 9^k$")
    _save(fig, "overhead_panel", w_pad=3.5 if JOURNAL else None)


def fig_noise_panel():
    """(a) finite-shot robustness + (b) device-noise error, one figure*."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=_fsp(2.2, 1.72))
    rows = _load("shots_results.json")
    xs, rec, _ = _agg(rows, "shots", "acc_recon")
    _, fus, _ = _agg(rows, "shots", "acc_fusion")
    ax1.plot(xs, rec, "o-", color=STYLE["recon_color"], label="reconstruction ($\\gamma^2$ noise)")
    ax1.plot(xs, fus, "s-", color=STYLE["fusion_color"], label="late fusion")
    ax1.set_xscale("log"); ax1.set_xlabel("shots per expectation"); ax1.set_ylabel("accuracy")
    ax1.legend(); ax1.set_title("(a) finite shots")
    rows = [r for r in _load("noise_results.json")
            if r.get("noise", "depolarizing") == "depolarizing"]
    ks = [r["n_cuts"] for r in rows]
    ax2.plot(ks, [r["recon_noise_err"] for r in rows], "o-", color=STYLE["recon_color"],
             label="reconstruction")
    ax2.plot(ks, [r["fusion_noise_err"] for r in rows], "s-", color=STYLE["fusion_color"],
             label="fusion feature")
    ax2.set_xlabel("number of cuts $k$"); ax2.set_ylabel(r"$|\hat O-O|$ device noise")
    ax2.set_xticks(ks); ax2.set_ylim(0, None); ax2.legend(); ax2.set_title("(b) device noise")
    _save(fig, "noise_panel", w_pad=2.5 if JOURNAL else None)


def fig_scaling_panel():
    """(a) scale-up frontier + (b) coupling-depth sweep + (c) TFIM S_A(g), one figure*."""
    # This panel is placed at 0.90\textwidth, so everything here renders at 0.9x:
    # AXLBL 6.5 -> 5.9pt, ticks 6.0 -> 5.4pt, legend 5.0 -> 4.5pt on paper.
    # Keep AXLBL > tick size so the hierarchy reads correctly.
    AXLBL, TICKLBL, LEGSZ = STYLE["fontsize"], STYLE["fontsize"] - 0.5, SMALL
    if JOURNAL:   # (a) spans the text width; (b) and (c) share the second row at equal size
        fig = plt.figure(figsize=(WP, 4.4))
        gs = fig.add_gridspec(2, 2, height_ratios=[1.15, 1], hspace=0.55, wspace=0.85)
        ax1 = fig.add_subplot(gs[0, :]); ax2 = fig.add_subplot(gs[1, 0]); ax4 = fig.add_subplot(gs[1, 1])
    else:
        fig, (ax1, ax2, ax4) = plt.subplots(1, 3, figsize=(7.0, 2.2),
                                            gridspec_kw={"width_ratios": [1.5, 1.2, 1]})
    rows = _load("scaleup_results.json")
    try:   # n=14 exact-reference extension (exp_scaleup14): one aggregated point at k=7
        try:
            n14_rows = _load("scaleup_n14_converged.json")   # 150-iteration reference (preferred)
        except FileNotFoundError:
            n14_rows = _load("scaleup_n14.json")
        n14 = [r for r in n14_rows if r["n_cuts"] == 7]
        if n14:
            rows.append({"qubits": 14, "n_cuts": 7, "n_seeds": len(n14),
                         "acc_fusion": float(np.mean([r["acc_fusion"] for r in n14])),
                         "acc_fusion_std": float(np.std([r["acc_fusion"] for r in n14])),
                         "acc_B0": float(np.mean([r["acc_B0"] for r in n14])),
                         "acc_B0_std": float(np.std([r["acc_B0"] for r in n14])),
                         "recon_overhead_9k": float(9 ** 7)})
            rows.sort(key=lambda r: (r["n_cuts"], r["qubits"]))
    except FileNotFoundError:
        pass
    ks = [r["n_cuts"] for r in rows]
    (l_ovr,) = ax1.semilogy(ks, [r["recon_overhead_9k"] for r in rows], ":",
                            color=STYLE["recon_color"], lw=1, label="recon overhead $9^k$")
    ax1.axhspan(1e6, 2e10, color=STYLE["recon_color"], alpha=0.08)
    ax1.text(1.1, 3e9, "reconstruction infeasible", fontsize=SMALL, color=STYLE["recon_color"])
    ax1.set_xlabel("number of cuts $k$"); ax1.set_ylabel("sampling overhead")
    ax1.set_ylim(1, 2e10); ax1.set_xticks(sorted(set(ks)))
    ax1.set_title("(a) fusion at scale")
    ax1b = ax1.twinx()
    b0 = [(r["n_cuts"], r["acc_B0"]) for r in rows if r.get("acc_B0") is not None]
    (l_b0,) = ax1b.plot([k for k, _ in b0], [a for _, a in b0], "^--", color=STYLE["gray"],
                        lw=1, ms=4, label="uncut model (B0)")
    err_fu = ax1b.errorbar(ks, [r["acc_fusion"] for r in rows],
                           yerr=[r["acc_fusion_std"] for r in rows], marker="o", ls="-",
                           color=STYLE["fusion_color"], ms=4, capsize=2, label="late fusion")
    for r in rows:
        if r["n_cuts"] in (6, 7, 8, 10):
            # k=7, k=8 and k=10 labels go ABOVE their markers (clear of the
            # lower-right legend); the offset also clears the errorbar cap
            dy = 9 if r["n_cuts"] in (7, 8, 10) else -11
            # k=10 sits on the right spine: right-align it so the label does not
            # spill over the twin (accuracy) axis ticks.
            last = r["n_cuts"] == max(ks)
            # white halo: the dotted 9^k overhead curve passes through these labels
            ax1b.annotate(f"$n{{=}}{r['qubits']}$", (r["n_cuts"], r["acc_fusion"]),
                          textcoords="offset points", xytext=(3 if last else 0, dy),
                          fontsize=SMALL, ha="right" if last else "center",
                          bbox=dict(facecolor="white", edgecolor="none",
                                    boxstyle="square,pad=0.12", alpha=0.85))
    ax1b.set_ylabel("accuracy"); ax1b.set_ylim(0.4, 1.05)
    ax1b.spines["top"].set_visible(False)
    handles = [l_ovr, l_b0, err_fu]
    if JOURNAL:
        ax1.legend(handles, [h.get_label() for h in handles], fontsize=LEGSZ, loc="lower right", ncol=1, frameon=True, framealpha=0.9, edgecolor="none")
    else:
        ax1.legend(handles, [h.get_label() for h in handles], fontsize=LEGSZ, loc="upper center", bbox_to_anchor=(0.5, -0.30), ncol=2, frameon=False, columnspacing=0.8, handletextpad=0.4)
    rows = _load("multilayer_results.json")
    Ls = [r["layers"] for r in rows]
    ax2.plot(Ls, [r["cut_entropy"] for r in rows], "o-", color="#8e44ad", label="cut entropy")
    ax2.set_xlabel("coupling layers $L$"); ax2.set_ylabel("cut entanglement (bits)")
    ax2.set_xticks(Ls); ax2.set_title("(b) coupling depth")
    ax3 = ax2.twinx()
    ax3.plot(Ls, [r["fusion_error"] for r in rows], "s--", color=STYLE["recon_color"],
             label="unrecovered corr.")
    ax3.set_ylabel(r"$|\langle Z_aZ_b\rangle-\langle Z_a\rangle\langle Z_b\rangle|$")
    ax3.spines["top"].set_visible(False)
    lines = ax2.get_lines() + ax3.get_lines()
    ax2.legend(lines, [l.get_label() for l in lines], fontsize=SMALL, loc="upper center", bbox_to_anchor=(0.5, -0.30 if not JOURNAL else -0.32), ncol=1, frameon=False)
    d = _load("tfim_results.json")
    curve = d["entropy_curve"]
    ax4.plot(curve["g"], curve["S_A"], "-", color="#8e44ad", lw=1)
    ax4.axvline(1.0, ls=":", color="k", lw=0.6)
    ax4.set_xlabel("$g$"); ax4.set_ylabel("$S_A$ (bits)")
    ax4.set_title("(c) TFIM diagnostic")
    for _ax in (ax1, ax1b, ax2, ax3, ax4):
        _ax.xaxis.label.set_size(AXLBL)
        _ax.yaxis.label.set_size(AXLBL)
        _ax.tick_params(labelsize=TICKLBL)
    _save(fig, "scaling_panel")


def fig_scaleup():
    """W2: fusion keeps training where reconstruction's 9^k overhead is infeasible."""
    rows = _load("scaleup_results.json")
    fig, ax1 = plt.subplots(figsize=STYLE["figsize"])
    ks = [r["n_cuts"] for r in rows]
    ax1.semilogy(ks, [r["recon_overhead_9k"] for r in rows], ":", color=STYLE["recon_color"],
                 lw=1, label="recon overhead $9^k$")
    ax1.axhspan(1e6, 2e10, color=STYLE["recon_color"], alpha=0.08)
    ax1.text(1.2, 3e7, "reconstruction infeasible", fontsize=7, color=STYLE["recon_color"])
    ax1.set_xlabel("number of cuts $k$"); ax1.set_ylabel("sampling overhead")
    ax1.set_ylim(1, 2e10); ax1.set_xticks(sorted(set(ks)))
    ax2 = ax1.twinx()
    b0 = [(r["n_cuts"], r["acc_B0"]) for r in rows if r.get("acc_B0") is not None]
    ax2.plot([k for k, _ in b0], [a for _, a in b0], "^--", color=STYLE["gray"], lw=1,
             ms=4, label="uncut model (B0)")
    ax2.errorbar(ks, [r["acc_fusion"] for r in rows], yerr=[r["acc_fusion_std"] for r in rows],
                 marker="o", ls="-", color=STYLE["fusion_color"], ms=4, capsize=2,
                 label="late fusion")
    for r in rows:  # annotate the qubit count at each frontier point
        if r["n_cuts"] in (6, 8, 10):
            ax2.annotate(f"$n{{=}}{r['qubits']}$", (r["n_cuts"], r["acc_fusion"]),
                         textcoords="offset points", xytext=(0, -11), fontsize=7,
                         ha="center")
    ax2.set_ylabel("accuracy"); ax2.set_ylim(0.4, 1.05); ax2.spines["top"].set_visible(False)
    lines = ax1.get_lines() + ax2.get_lines()
    ax1.legend(lines, [l.get_label() for l in lines], fontsize=7, loc="lower left")
    _save(fig, "scaleup")


def fig_tfim():
    """W1: quantum-native TFIM tasks + the S_A(g) diagnostic curve."""
    d = _load("tfim_results.json")
    rows, curve = d["rows"], d["entropy_curve"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(3.4, 1.7),
                                   gridspec_kw={"width_ratios": [1.1, 1]})
    for i, task in enumerate(["phase", "bond"]):
        rs = [r for r in rows if r["task"] == task]
        ax1.bar(i - 0.17, np.mean([r["acc_fusion"] for r in rs]), 0.34,
                color=STYLE["fusion_color"], label="fusion" if i == 0 else None)
        ax1.bar(i + 0.17, np.mean([r["acc_joint"] for r in rs]), 0.34,
                color=STYLE["recon_color"], label="joint" if i == 0 else None)
    ax1.set_xticks([0, 1]); ax1.set_xticklabels(["phase", "bond"], fontsize=7)
    ax1.set_ylim(0.5, 1.02); ax1.set_ylabel("accuracy"); ax1.legend(fontsize=5)
    ax2.plot(curve["g"], curve["S_A"], "-", color="#8e44ad", lw=1)
    ax2.axvline(1.0, ls=":", color="k", lw=0.6)
    ax2.set_xlabel("$g$"); ax2.set_ylabel("$S_A$ (bits)")
    _save(fig, "tfim")


def fig_qdial_physical():
    """Q-dial in the PHYSICAL QPD basis (qiskit-addon-cutting): accuracy vs retained physical
    sampling overhead (sum_kept|a_i|)^2, exact estimator (solid) and 4000-shot Aer (dashed),
    with the uncoupled-fusion reference (dotted) and fixed-head reconstruction (star)."""
    d = _load("qdial_physical_results.json"); rows = d["rows"]
    picks = [0.0, 0.5, 1.0]
    colors = [STYLE["fusion_color"], STYLE["accent"], STYLE["recon_color"]]
    fig, ax = plt.subplots(figsize=_fs(2.1))
    for a, c in zip(picks, colors):
        rs = [r for r in rows if r["alpha"] == a]
        Qs = [q["Q"] for q in rs[0]["physical_exact"]]
        cost = [np.mean([r["physical_exact"][i]["retained_overhead_sq"] for r in rs]) for i in range(len(Qs))]
        acc = [np.mean([r["physical_exact"][i]["accuracy"] for r in rs]) for i in range(len(Qs))]
        std = [np.std([r["physical_exact"][i]["accuracy"] for r in rs]) for i in range(len(Qs))]
        ax.errorbar(cost, acc, yerr=std, marker="o", ms=3, color=c, capsize=2, lw=1,
                    label=f"$\\alpha={a}$ (exact)")
        rs_s = [r for r in rs if r.get("physical_shots")]
        if rs_s:
            cost_s = [np.mean([r["physical_shots"][i]["retained_overhead_sq"] for r in rs_s]) for i in range(len(Qs))]
            acc_s = [np.mean([r["physical_shots"][i]["accuracy"] for r in rs_s]) for i in range(len(Qs))]
            ax.plot(cost_s, acc_s, ls="--", marker="^", ms=3, color=c, lw=0.8, alpha=0.8)
        indep = np.mean([r["acc_independent_fusion"] for r in rs])
        ax.axhline(indep, color=c, ls=":", lw=0.9)
        ax.plot([cost[-1]], [np.mean([r["acc_B0_fixed_head"] for r in rs])], marker="*", ms=7, color=c, ls="none")
    ax.plot([], [], ls="--", marker="^", ms=3, color="gray", lw=0.8, label="4000 shots")
    ax.plot([], [], ls=":", color="gray", lw=0.9, label="uncoupled fusion")
    ax.plot([], [], marker="*", ms=7, color="gray", ls="none", label="fixed-head recon.")
    ax.set_xscale("log"); ax.set_xlabel("retained physical sampling overhead $(\\sum_{\\mathrm{kept}}|a_i|)^2$")
    ax.set_ylabel("accuracy"); ax.set_ylim(0.6, 1.0)
    ax.legend(ncol=2, fontsize=SMALL - (0 if JOURNAL else 1.5), loc="lower right")
    _save(fig, "qdial_physical")


def fig_deep_coupling():
    """Trained deep-coupling study (exp_deep_coupling): heat-maps over (L, alpha) for k=1,2 of
    (a) trained cut entanglement S_A, (b) uncoupled-fusion gap, (c) frozen (cut-free) gap."""
    d = _load("deep_coupling_results.json"); rows = d["rows"]
    Ls = sorted(set(r["L"] for r in rows)); als = sorted(set(r["alpha"] for r in rows)); ks = sorted(set(r["k"] for r in rows))
    panels = [("S_A", "cut entanglement $S_A$ (bits)", "Purples", (0, 1.0)),
              ("gap_uncoupled", "uncoupled-fusion gap", "RdBu_r", (-0.1, 0.1)),
              ("gap_frozen_cutfree", "frozen-fusion gap", "RdBu_r", (-0.1, 0.25))]
    fig, axes = plt.subplots(len(ks), len(panels), figsize=(WP, (1.75 if JOURNAL else 1.55) * len(ks) + 0.4), squeeze=False)
    for i, k in enumerate(ks):
        for j, (key, title, cmap, (vmin, vmax)) in enumerate(panels):
            val = (lambda r: r["acc_coupled"] - r["acc_frozen_cutfree"]) if key == "gap_frozen_cutfree" else (lambda r: r[key])
            M = np.array([[np.mean([val(r) for r in rows if r["k"] == k and r["L"] == L and r["alpha"] == a])
                           for a in als] for L in Ls])
            ax = axes[i, j]
            im = ax.imshow(M, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto", origin="lower")
            for (li, ai), v in np.ndenumerate(M):
                ax.text(ai, li, f"{v:.2f}", ha="center", va="center", fontsize=6.5 if JOURNAL else SMALL,
                        color="white" if (cmap == "Purples" and v > 0.55) else "black")
            ax.set_xticks(range(len(als))); ax.set_xticklabels([f"{a:g}" for a in als])
            ax.set_yticks(range(len(Ls))); ax.set_yticklabels([str(L) for L in Ls])
            if i == len(ks) - 1: ax.set_xlabel("cross-cut dependence $\\alpha$")
            if j == 0: ax.set_ylabel(f"coupling layers $L$ ($k{{=}}{k}$)")
            if i == 0: ax.set_title(title, fontsize=STYLE["fontsize"] - 1)
            fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    fig.tight_layout(w_pad=1.5 if JOURNAL else 1.08)
    _save(fig, "deep_coupling")


def fig_pipeline():
    """Experimental and tool flow (Fig. 1): data -> encoding -> circuit -> readout -> evaluation.
    Column headers name the stage; box colour names the path (grey shared, red reconstruction,
    amber intermediate dial, blue fusion, green evaluation); the italic lines inside each readout
    box name the engines that executed it. Typeface: PIPELINE_FONT (see top of file)."""
    import glob
    import matplotlib.patches as mpatches
    from matplotlib import font_manager as fm
    fam = None
    if PIPELINE_FONT:
        files = glob.glob(os.path.join(HERE, "paper", "fonts", "*.ttf"))
        for f in files:
            fm.fontManager.addfont(f)
        if files:
            fam = PIPELINE_FONT
    rc = {"font.family": fam, "mathtext.fontset": "custom", "mathtext.rm": fam, "mathtext.it": fam + ":italic",
          "mathtext.bf": fam + ":bold", "mathtext.sf": fam, "mathtext.cal": fam} if fam else {}
    f_head, f_title, f_body, f_eng = (6.5, 6.5, 6.0, 5.5) if fam else (6.5, 5.9, 5.8, 5.3)   # DejaVu is wider
    with plt.rc_context(rc):
        fig, ax = plt.subplots(figsize=(WP, WP * 0.66))
        fig.subplots_adjust(0, 0, 1, 1)
        ax.set_xlim(0, 100); ax.set_ylim(0, 66); ax.axis("off")
        C = {"shared": ("#f2f3f4", "#5d6d7e"), "recon": ("#fadbd8", "#c0392b"), "dial": ("#fdebd0", "#b9770e"),
             "fusion": ("#d6eaf8", "#2471a3"), "eval": ("#e3f1e5", "#1e8449")}
        ink, soft, line = "#1b2631", "#515a5a", "#34495e"
        def box(x, y, w, h, title, body, kind, engine=None, dashed=False):
            fc, ec = ("white", "#707b7c") if dashed else C[kind]
            ax.add_patch(mpatches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.0",
                                                 fc=fc, ec=ec, lw=0.9, ls=(0, (3, 2)) if dashed else "-"))
            top = y + h - 1.3
            ax.text(x + w / 2, top, title, ha="center", va="top", fontsize=f_title, fontweight="bold", color=ink)
            ax.text(x + w / 2, top - 2.7, body, ha="center", va="top", fontsize=f_body, linespacing=1.1, color=ink)
            if engine:
                ax.text(x + w / 2, y + 1.0, engine, ha="center", va="bottom", fontsize=f_eng, style="italic", linespacing=1.1, color=soft)
        def head(x0, x1, y):   # arrowhead only
            ax.annotate("", xy=(x1, y), xytext=(x0, y),
                        arrowprops=dict(arrowstyle="-|>", lw=0.8, color=line, shrinkA=0, shrinkB=0, mutation_scale=6))
        def fanout(x0, y0, x1, targets):
            """one trunk from (x0,y0) to the gap midpoint, a vertical bus, one branch per target."""
            xm = (x0 + x1) / 2
            ys = sorted(targets + [y0])
            ax.plot([x0, xm], [y0, y0], color=line, lw=0.8, solid_capstyle="round")
            if len(ys) > 1:
                ax.plot([xm, xm], [ys[0], ys[-1]], color=line, lw=0.8, solid_capstyle="round")
            for yt in targets:
                head(xm, x1, yt)
        def fanin(sources, x0, x1, y1):
            """branches from several (y) sources into one target through a vertical bus."""
            xm = (x0 + x1) / 2
            ys = sorted(sources + [y1])
            for ys0 in sources:
                ax.plot([x0, xm], [ys0, ys0], color=line, lw=0.8, solid_capstyle="round")
            ax.plot([xm, xm], [ys[0], ys[-1]], color=line, lw=0.8, solid_capstyle="round")
            head(xm, x1, y1)
        # columns: x, width (gaps of 4 units carry the connectors)
        cols = [(1, 14.5, "DATA"), (19.5, 14.5, "ENCODING"), (38, 18, "CIRCUIT"), (60, 19.5, "READOUT"), (83.5, 16, "EVALUATION")]
        for x, w, lab in cols:
            ax.text(x + w / 2, 65.5, lab, ha="center", va="top", fontsize=f_head, color="#707b7c", fontweight="bold")
            ax.plot([x, x + w], [62.4, 62.4], color="#d5d8dc", lw=0.8)
        H = 15.5
        box(1, 27.5, 14.5, 12, "datasets", "synthetic ($\\alpha$),\ntabular, image,\nTFIM, entangled", "shared")
        box(19.5, 27.5, 14.5, 12, "angle encoding", "register split\n$x_A \\mid x_B$", "shared")
        box(38, 35.5, 18, 11, "coupled QNN", "$k$ cross gates,\ntrained end-to-end", "shared")
        box(38, 9.25, 18, 11, "uncoupled circuits", "no cross gate,\nshared loss (Alg. 1)", "fusion")
        box(60, 43, 19.5, H, "reconstruction (B1)", "$\\kappa^{2k}$ overhead,\nfixed head", "recon",
            "exact $\\cdot$ Aer $\\cdot$ IBM\nqiskit-addon-cutting")
        box(60, 25, 19.5, H, "dial $Q$ (partial QPD)", "kept QPD terms $+$\nlocal $m_A, m_B$", "dial",
            "exact $\\cdot$ Aer $\\cdot$ IBM\nqiskit-addon-cutting")
        box(60, 7, 19.5, H, "late fusion (B2)", "trained head $f_\\theta$ on\nlocal $\\langle Z\\rangle$ features", "fusion",
            "statevector $\\cdot$ shots\nAer $\\cdot$ IBM")
        box(83.5, 27, 16, 20, "evaluation", "accuracy, F1, AUC,\npaired CIs, TOST,\noverhead, shots,\n$S_A$, knee $Q^\\star$", "eval")
        box(83.5, 6, 16, 14.5, "classical anchors", "tuned SVM / MLP\non $x$; split-feature\nfusion on $x_A \\mid x_B$", "shared", dashed=True)
        # connectors: data -> encoding -> {coupled, uncoupled}; coupled -> {B1, dial}; uncoupled -> B2; readouts -> evaluation
        head(15.5, 19.5, 33.5)
        fanout(34, 33.5, 38, [41, 14.75])
        fanout(56, 41, 60, [50.75, 32.75])
        head(56, 60, 14.75)
        fanin([50.75, 32.75, 14.75], 79.5, 83.5, 37)
        ax.annotate("", xy=(91.5, 27), xytext=(91.5, 20.5), arrowprops=dict(arrowstyle="-|>", lw=0.8, color="#707b7c", shrinkA=0, shrinkB=0, mutation_scale=6))
        # colour key
        kx = 1
        for kind, lab in [("shared", "shared stage"), ("recon", "reconstruction path"), ("dial", "intermediate dial"),
                          ("fusion", "fusion path"), ("eval", "evaluation")]:
            fc, ec = C[kind]
            ax.add_patch(mpatches.Rectangle((kx, 1.2), 2.0, 2.0, fc=fc, ec=ec, lw=0.8))
            ax.text(kx + 2.8, 2.2, lab, ha="left", va="center", fontsize=f_eng, color=soft)
            kx += 2.8 + 0.86 * len(lab) + 3.0
        _save(fig, "pipeline", tight=False)

FIGURES = [fig_qdial_physical, fig_deep_coupling, fig_pipeline, fig_cost_crossover, fig_independent, fig_shot_robustness,
           fig_regime_boundary, fig_trainability, fig_qdial_frontier,
           fig_device_noise, fig_qiskit_overhead, fig_multilayer, fig_scale_qubits,
           fig_overhead_panel, fig_noise_panel, fig_scaling_panel, fig_scaleup, fig_tfim]


def main():
    print(f"Generating figures -> {OUT}")
    for fn in FIGURES:
        try:
            fn()
        except Exception as e:
            print(f"  SKIP {fn.__name__}: {e}")
    print("done.")


if __name__ == "__main__":
    main()
