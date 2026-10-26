"""Sinh bang LaTeX cho muc CO CHE tu CSV

⚠ \\caption la DOI SO DI DONG: \\input trong do phai co \\protect, khong thi
   latexmk chet voi "Argument of \\Hy@tempa has an extra }". Da tra gia mot lan., de bang khong bao gio lech voi so da do.

Luat nha: bang go tay thi chay lai thi nghiem KHONG cap nhat. Nen bang nay sinh ra,
khong go. Ghi vao paper/bang/.
"""
import csv, os

HERE = os.path.dirname(os.path.abspath(__file__))
# MOT nha duy nhat cho ket qua: dung duong ma chinh cac script thi nghiem GHI VAO,
# khong tao thu muc thu hai. Da tung co "results-neucom" song song va do la hai nha.
RES = os.path.join(HERE, "results")
RA = os.path.join(HERE, "..", "paper", "bang")
BIEN_THE = ["KAN-spline", "KAN-Fourier", "KAN-fast", "KAN-cheby"]
NHAN = {"KAN-spline": "KAN (spline)", "KAN-Fourier": "KAN (Fourier)",
        "KAN-fast": "KAN (RBF)", "KAN-cheby": "KAN (Chebyshev)"}


def doc(ten):
    with open(os.path.join(RES, ten), newline="") as f:
        return list(csv.DictReader(f))


def bang_coche():
    rows = doc("exp_mechanism2.csv")
    lams = sorted({r["lam"] for r in rows}, key=float)
    ns = sorted({r["n"] for r in rows}, key=int)
    out = [r"\begin{table}[t]", r"\centering",
           r"\caption{Additivity experiment. Paired NMSE difference (dB) between each KAN"
           r" instantiation and a matched MLP, \protect\input{claims/mech_seeds} seeds;"
           r" negative"
           r" means the KAN is better."
           r" $\lambda$ is the share of signal power carried by cross-coordinate interaction,"
           r" with target difficulty held fixed (see text). Bold marks intervals excluding"
           r" zero in favour of the MLP.}",
           r"\label{tab:coche}",
           r"\small", r"\setlength{\tabcolsep}{4pt}",
           r"\begin{tabular}{l" + "r" * len(lams) + "}", r"\toprule",
           r"$n$ & \multicolumn{%d}{c}{$\lambda$} \\" % len(lams),
           r"\cmidrule(lr){2-%d}" % (len(lams) + 1),
           " & " + " & ".join(f"${float(l):.2f}$" for l in lams) + r" \\", r"\midrule"]
    for n in ns:
        out.append(r"\multicolumn{%d}{l}{\textit{$n=%s$ training samples}} \\" % (len(lams) + 1, n))
        for bt in BIEN_THE:
            o = [NHAN[bt]]
            for l in lams:
                r = [x for x in rows if x["n"] == n and x["lam"] == l
                     and x["model"] == f"PAIRED {bt} - MLP"]
                if not r:
                    o.append("--"); continue
                v, lo, hi = float(r[0]["nmse_db"]), float(r[0]["ci_lo"]), float(r[0]["ci_hi"])
                s = f"${v:+.2f}$"
                if lo > 0:                      # MLP thang, khoang khong chua 0
                    s = r"$\mathbf{%+.2f}$" % v
                o.append(s)
            out.append(" & ".join(o) + r" \\")
        if n != ns[-1]:
            out.append(r"\midrule")
    out += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(out)


def bang_dieu_chinh():
    rows = [r for r in doc("exp_mechanism2_tuning.csv") if r["mlp"].startswith("BEST")]
    eps = sorted({r["epochs"] for r in rows}, key=int)
    lams = sorted({r["lam"] for r in rows}, key=float)
    out = [r"\begin{table}[t]", r"\centering",
           r"\caption{Tuning control at $n=4096$. Paired NMSE difference (dB) between the"
           r" spline-KAN and the \emph{best} swept MLP, with both models given the same"
           r" schedule; negative means the KAN is better.}",
           r"\label{tab:coche-tune}",
           r"\begin{tabular}{l" + "r" * len(lams) + "}", r"\toprule",
           r"Schedule & " + " & ".join(r"$\lambda=%.2f$" % float(l) for l in lams) + r" \\",
           r"\midrule"]
    for e in eps:
        khoa = "mech_ep_ngan" if e == eps[0] else "mech_ep_dai"
        o = [r"$\input{claims/%s}$ epochs" % khoa]
        for l in lams:
            r = [x for x in rows if x["epochs"] == e and x["lam"] == l]
            o.append(f"${float(r[0]['paired']):+.2f}$" if r else "--")
        out.append(" & ".join(o) + r" \\")
    out += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(out)


if __name__ == "__main__":
    os.makedirs(RA, exist_ok=True)
    for ten, noi_dung in (("co-che.tex", bang_coche()),
                          ("co-che-dieu-chinh.tex", bang_dieu_chinh())):
        with open(os.path.join(RA, ten), "w") as f:
            f.write("%% SINH TU code/sinh_bang_coche.py, DUNG SUA TAY\n" + noi_dung + "\n")
        print("ghi:", os.path.relpath(os.path.join(RA, ten), HERE))
