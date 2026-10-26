"""Sinh THAN cac bang ket qua chinh tu CSV, thay vi go tay.

⛔ Vi sao: sau khi nang hai che do hoi quy len 10 hat giong, cong c5 (da vá de nhin
duoc so thap phan trong che do toan) bat 120 so go tay, va mot so trong do DA CU:
MLP -16.27 -> -16.32, KAN spline -14.09 -> -14.02. Bang go tay khong tu cap nhat,
dung lop loi so nha da ghi. Nay bang sinh ra, chay lai thi nghiem la bang doi theo.

Ghi ra paper/bang/<ten>.tex, moi tep chi chua phan THAN (cac dong du lieu), de giu
nguyen caption va label dang nam trong main.tex.
"""
import csv, os

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
RA = os.path.join(HERE, "..", "paper", "bang")


def doc(ten):
    with open(os.path.join(RES, ten), newline="") as f:
        return {r["model"]: r for r in csv.DictReader(f)}


def o(v, sd=None, dam=False, nd=2):
    t = f"{float(v):.{nd}f}" if sd is None else f"{float(v):.{nd}f}\\pm{float(sd):.{nd+0}f}"
    return (r"$\mathbf{%s}$" % t) if dam else f"${t}$"


def than_dpd():
    d = doc("exp_dpd.csv")
    tot = min(d, key=lambda k: float(d[k]["nmse_mean"]))
    ten = [("MLP", "MLP (matched)"), ("MemPoly", "GMP (classical)"),
           ("KANfourier", "KAN, Fourier"), ("KANspline", "KAN, spline (canon.)")]
    ra = []
    for k, nhan in ten:
        r = d[k]
        p = "$%s$" % r["params"] if r.get("params") else "--"
        ra.append(f"{nhan:22s} & {p:9s} & "
                  f"{o(r['nmse_mean'], r['nmse_std'], k == tot)} \\\\")
    return "\n".join(ra)


def than_amc():
    d = doc("exp_amc.csv")
    tot = max(d, key=lambda k: float(d[k]["acc_mean"]))
    ten = [("CNN", "CNN (task-appropriate)"), ("MLP", "MLP (matched)"),
           ("KANspline", "KAN, spline (canon.)"), ("KANfourier", "KAN, Fourier")]
    ra = []
    for k, nhan in ten:
        r = d[k]
        ra.append(f"{nhan:24s} & ${r['params']}$ & "
                  f"{o(r['acc_mean'], r['acc_std'], k == tot, nd=3)} \\\\")
    return "\n".join(ra)


def than_chest():
    d = doc("exp_chest.csv")
    tot = min(d, key=lambda k: float(d[k]["nmse_mean"]))
    ten = [("LMMSE", "LMMSE (classical)"), ("MLP", "MLP (matched)"),
           ("KANspline", "KAN, spline (canon.)"), ("KANfourier", "KAN, Fourier"),
           ("LinInterp", "linear interpolation")]
    ra = []
    for k, nhan in ten:
        if k not in d:
            continue
        r = d[k]
        ra.append(f"{nhan:20s} & {o(r['nmse_mean'], r['nmse_std'], k == tot)} \\\\")
    return "\n".join(ra)


def ghi_vao_bai(nhan, than, duong):
    """Thay cac hang giua \\midrule va \\bottomrule cua bang co nhan <nhan>.

    ⛔ Khong dung \\input{} BEN TRONG tabular: da thu, latex chet voi
    "Misplaced \\noalign" du hang cuoi co hay khong co \\\\. Nen bo sinh GHI THANG vao
    main.tex, giong nep sinh-bang.py cua ke-hoach-cuoi-2026.
    """
    s = open(duong, encoding="utf-8").read()
    i = s.find("\\label{%s}" % nhan)
    assert i > 0, "khong thay nhan %s" % nhan
    a = s.index("\\midrule", i) + len("\\midrule")
    b = s.index("\\bottomrule", a)
    cu = s[a:b]
    s = s[:a] + "\n" + than + "\n" + s[b:]
    open(duong, "w", encoding="utf-8").write(s)
    return cu.strip() != than.strip()


if __name__ == "__main__":
    import sys
    CHI_KIEM = "--kiem" in sys.argv     # kiem ma KHONG sua, de dung lam cong
    BAI = os.path.join(HERE, "..", "paper", "main.tex")
    if CHI_KIEM:
        goc = open(BAI, encoding="utf-8").read()
    doi = 0
    for nhan, f in (("tab:dpd", than_dpd), ("tab:amc", than_amc), ("tab:chest", than_chest)):
        than = f()
        if ghi_vao_bai(nhan, than, BAI):
            doi += 1
            print(f"  ⟳ {nhan}: DA CAP NHAT tu CSV")
            for d in than.split("\n"):
                print("      " + d)
        else:
            print(f"  = {nhan}: khong doi")
    if CHI_KIEM:
        open(BAI, "w", encoding="utf-8").write(goc)
        if doi:
            print(f"\n⛔ DO: {doi}/3 bang trong bai LECH voi CSV")
            sys.exit(1)
        print("\n✅ DAT: 3/3 bang khop CSV")
        sys.exit(0)
    print(f"\n{doi}/3 bang doi gia tri")
