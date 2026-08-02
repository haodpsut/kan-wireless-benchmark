"""plots.py -- result figures in the Agentra IEEE-Transactions house style via agentra_plot
(plot-figure-dph). Single-column 3.5in, brand palette, two-channel encoding (colour + hatch),
every figure through ap.save() print-ready gate. No ad-hoc matplotlib styling, no Okabe-Ito.
"""
import csv, os, sys
import numpy as np

ASSETS = os.environ.get("AGENTRA_PLOT_ASSETS", os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "my-skills", "ALL-FIGURES",
    "0-SKILLS", "plot-figure-dph", "assets"))
sys.path.insert(0, os.path.abspath(ASSETS))
import agentra_plot as ap  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

ap.use_style()
HERE = os.path.dirname(__file__)
RES = os.path.join(HERE, "results")
FIG = os.path.abspath(os.path.join(HERE, "..", "paper", "figs"))
os.makedirs(FIG, exist_ok=True)
STORE = os.path.join(FIG, ".methods_kan.json")

# one identity per method, held across every figure in the paper
METHODS = ["CNN", "MLP", "KANspline", "KANfourier", "MemPoly"]
LABEL = {"CNN": "CNN", "MLP": "MLP", "KANspline": "KAN (spline)",
         "KANfourier": "KAN (Fourier)", "MemPoly": "mem. poly."}
ap.reset_registry(STORE); ap.register(METHODS, store=STORE)


def load_csv(name):
    with open(os.path.join(RES, name)) as f:
        return list(csv.DictReader(f))


def out(stem):
    return os.path.join(FIG, stem)


def si(x):
    x = float(x)
    for d, s in [(1e6, "M"), (1e3, "k")]:
        if x >= d:
            v = x / d
            return f"{v:.0f}{s}" if v >= 10 or v == int(v) else f"{v:.1f}{s}"
    return f"{x:.0f}"


def si_xticks(ax, xs):
    """Explicit SI x-ticks so a log axis avoids mathtext $10^n$ superscripts (<7pt floor)."""
    from matplotlib.ticker import NullLocator
    ax.set_xticks(list(xs)); ax.set_xticklabels([si(x) for x in xs])
    ax.xaxis.set_minor_locator(NullLocator())


def bar_panel(rows, valcol, order, ylabel, stem, headline_txt, loc="upper right",
              invert=False):
    """Grouped bar over methods with CI whiskers; colour+hatch = two-channel encoding."""
    d = {r["model"]: r for r in rows}
    xs = np.arange(len(order))
    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    for i, m in enumerate(order):
        v = float(d[m][valcol])
        lo, hi = float(d[m]["ci_lo"]), float(d[m]["ci_hi"])
        s = ap.sty(m)
        ax.bar(i, v, width=0.68, facecolor="white", edgecolor=s["color"], linewidth=1.0,
               hatch=ap.hatch_for(m))
        ax.errorbar(i, v, yerr=[[abs(v - lo)], [abs(hi - v)]], fmt="none",
                    ecolor=s["color"], elinewidth=1.0, capsize=2.5)
    ax.set_xticks(xs); ax.set_xticklabels([LABEL[m] for m in order], rotation=20, ha="right")
    ax.set_ylabel(ylabel)
    if invert:
        ax.invert_yaxis()
    ap.headline(ax, headline_txt, loc=loc)
    fig.tight_layout(); ap.save(fig, out(stem))


def fig_dpd():
    rows = load_csv("exp_dpd.csv")
    bar_panel(rows, "nmse_mean", ["MLP", "KANfourier", "KANspline", "MemPoly"],
              "modelling NMSE [dB]", "fig_dpd",
              "matched MLP best;\nKAN worse by 0.8-2.2 dB", loc="lower right", invert=True)


def fig_amc():
    rows = load_csv("exp_amc.csv")
    bar_panel(rows, "acc_mean", ["CNN", "MLP", "KANspline", "KANfourier"],
              "accuracy (low-SNR)", "fig_amc",
              "CNN 0.66 >> KAN;\nmatched MLP > both KAN", loc="upper right")


def fig_interp():
    r2 = [float(x["r2"]) for x in load_csv("edge_r2.csv")]
    meta = {x["metric"]: x["value"] for x in load_csv("exp_interp.csv")}
    deg = float(meta["symbolic_degradation_db"])
    med = float(meta["median_edge_R2_deg3"])
    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    ax.hist(r2, bins=24, range=(0.9, 1.001), color="white", edgecolor=ap.INK,
            linewidth=1.0, hatch="///")
    ax.axvline(med, color=ap.WARM, ls="--", lw=1.2)
    ax.set_xlim(0.9, 1.002)
    ax.set_xlabel("per-edge $R^2$ of cubic fit")
    ax.set_ylabel("edge count")
    ap.headline(ax, f"edges simple (median $R^2$={med:.3f})\n"
                    f"but symbolic form costs {deg:+.2f} dB", loc="upper left")
    fig.tight_layout(); ap.save(fig, out("fig_interp"))


def fig_amc_snr():
    """SNR sweep: the honest crossover. KAN-spline overtakes MLP at higher SNR; CNN dominates."""
    rows = load_csv("abl_amc_snr.csv")
    order = ["CNN", "MLP", "KANspline", "KANfourier"]
    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    for m in order:
        pts = sorted(((int(r["snr_db"]), float(r["acc_mean"])) for r in rows if r["model"] == m))
        xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
        ax.plot(xs, ys, **ap.sty(m, markersize=3), label=LABEL[m])
    ax.axvline(0, color=ap.GRAY, ls=":", lw=0.8)
    ax.set_xlabel("SNR [dB]"); ax.set_ylabel("accuracy")
    ax.legend(fontsize=7, ncol=2, loc="upper left", handlelength=1.5, columnspacing=1.0)
    ap.headline(ax, "KAN-spline overtakes MLP\nabove ~0 dB; CNN leads", loc="lower right")
    fig.tight_layout(); ap.save(fig, out("fig_amc_snr"))


def fig_dpd_budget():
    rows = load_csv("abl_dpd_budget.csv")
    order = ["MLP", "KANfourier", "KANspline"]
    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    for m in order:
        pts = sorted(((int(r["params"]), float(r["nmse_mean"])) for r in rows if r["model"] == m))
        xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
        ax.plot(xs, ys, **ap.sty(m, markersize=3.5), label=LABEL[m])
    ax.set_xscale("log"); ax.set_xlabel("parameter budget"); si_xticks(ax, xs)
    ax.set_ylabel("modelling NMSE [dB]"); ax.invert_yaxis()
    ax.legend(fontsize=7, loc="upper right")
    ap.headline(ax, "gap persists;\nspline-KAN plateaus", loc="lower left")
    fig.tight_layout(); ap.save(fig, out("fig_dpd_budget"))


def fig_dpd_samples():
    rows = load_csv("abl_dpd_samples.csv")
    order = ["MLP", "KANfourier"]
    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    for m in order:
        pts = sorted(((int(r["ntrain"]), float(r["nmse_mean"])) for r in rows if r["model"] == m))
        xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
        ax.plot(xs, ys, **ap.sty(m, markersize=3.5), label=LABEL[m])
    ax.set_xscale("log"); ax.set_xlabel("training examples"); si_xticks(ax, xs)
    ax.set_ylabel("modelling NMSE [dB]"); ax.invert_yaxis()
    ax.legend(fontsize=7, loc="upper right")
    ap.headline(ax, "KAN wins in\nlow-data regime", loc="lower left")
    fig.tight_layout(); ap.save(fig, out("fig_dpd_samples"))


def fig_permod():
    """Per-modulation KAN-spline minus MLP accuracy. KAN's gain concentrates on the
    constellation-geometry modulations, supporting the per-edge-boundary hypothesis."""
    rows = load_csv("exp_amc_permod.csv")
    acc = {}
    for r in rows:
        acc.setdefault(r["model"], {})[r["modulation"]] = float(r["acc_mean"])
    mods = list(acc["MLP"].keys())
    diff = sorted(((m, acc["KANspline"][m] - acc["MLP"][m]) for m in mods), key=lambda t: t[1])
    labels = [m for m, _ in diff]; vals = [v for _, v in diff]
    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    for i, (m, v) in enumerate(diff):
        c = ap.AGREEN if v >= 0 else ap.WARM
        ax.barh(i, v, color="white", edgecolor=c, linewidth=1.0,
                hatch="///" if v >= 0 else "...")
    ax.axvline(0, color=ap.INK, lw=0.8)
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("KAN(spline) $-$ MLP accuracy")
    ap.headline(ax, "KAN gains on phase/amplitude\nmodulations (BPSK, PAM4, 8PSK)", loc="lower right")
    fig.tight_layout(); ap.save(fig, out("fig_permod"))


def fig_edges():
    """Three representative learned KAN edges (solid) with their cubic fits (dashed):
    concrete evidence the edges are simple readable curves."""
    rows = load_csv("exp_edges.csv")
    eids = sorted({int(r["edge"]) for r in rows})
    names = ["best-fit edge", "median edge", "worst-fit edge"]
    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    cols = [ap.INK, ap.AGREEN, ap.WARM]
    for eid, nm, c in zip(eids, names, cols):
        pts = [(float(r["x"]), float(r["f_kan"]), float(r["f_cubic"])) for r in rows
               if int(r["edge"]) == eid]
        x = [p[0] for p in pts]; fk = [p[1] for p in pts]; fc = [p[2] for p in pts]
        ax.plot(x, fk, color=c, lw=1.4, label=nm)
        ax.plot(x, fc, color=c, lw=1.0, ls=(0, (3, 2)))
    ax.set_xlabel("edge input $x$"); ax.set_ylabel("edge function $\\phi(x)$")
    ax.legend(fontsize=7, loc="best")
    ap.headline(ax, "solid: learned edge\ndashed: cubic fit", loc="upper left")
    fig.tight_layout(); ap.save(fig, out("fig_edges"))


def fig_chest():
    """Channel-estimation regime: classical LMMSE is best; spline-KAN ties the matched MLP."""
    rows = {r["model"]: r for r in load_csv("exp_chest.csv")}
    order = ["LMMSE", "MLP", "KANspline", "KANfourier", "LinInterp"]
    lab = {"LMMSE": "LMMSE", "MLP": "MLP", "KANspline": "KAN (spline)",
           "KANfourier": "KAN (Fourier)", "LinInterp": "lin. interp."}
    sty = {"LMMSE": (ap.INK, "xxx"), "MLP": (ap.AGREEN, "///"),
           "KANspline": (ap.WARM, "\\\\\\"), "KANfourier": (ap.GRAY, "..."),
           "LinInterp": (ap.LIME, "++")}
    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    for i, m in enumerate(order):
        v = float(rows[m]["nmse_mean"]); lo = float(rows[m]["ci_lo"]); hi = float(rows[m]["ci_hi"])
        c, hh = sty[m]
        ax.bar(i, v, width=0.68, facecolor="white", edgecolor=c, linewidth=1.0, hatch=hh)
        ax.errorbar(i, v, yerr=[[abs(v - lo)], [abs(hi - v)]], fmt="none", ecolor=c,
                    elinewidth=1.0, capsize=2.5)
    ax.set_xticks(range(len(order))); ax.set_xticklabels([lab[m] for m in order], rotation=20, ha="right")
    ax.set_ylabel("channel NMSE [dB]"); ax.invert_yaxis()
    ap.headline(ax, "LMMSE best; spline-KAN\nties matched MLP", loc="lower right")
    fig.tight_layout(); ap.save(fig, out("fig_chest"))


if __name__ == "__main__":
    fig_chest(); print("fig_chest ok")
    fig_edges(); print("fig_edges ok")
    fig_permod(); print("fig_permod ok")
    fig_dpd(); print("fig_dpd ok")
    fig_amc(); print("fig_amc ok")
    fig_interp(); print("fig_interp ok")
    fig_amc_snr(); print("fig_amc_snr ok")
    fig_dpd_budget(); print("fig_dpd_budget ok")
    fig_dpd_samples(); print("fig_dpd_samples ok")
    print("all figures written to", FIG)
