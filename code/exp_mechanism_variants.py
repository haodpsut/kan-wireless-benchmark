"""R1.1: ban do che do co tong quat sang cac ban cai KAN KHAC khong?

Chay DUNG luoi cua exp_mechanism.py (cung ham dich, cung hat giong, cung ngan sach
tham so) cho FastKAN va Chebyshev-KAN. Ghi ra TEP RIENG, khong ghi chung duong dan
voi exp_mechanism.py.

Doi chieu cheo mien phi: cot MLP o day phai TRUNG KHIT voi cot MLP cua exp_mechanism.py
vi cung ham dich, cung hat giong, cung du lieu. Lech la dau hieu mot trong hai sai.
"""
import csv, os, sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from harness import MLPreg, match_width, nparams, train_reg, bootstrap_ci, paired_diff_ci
from variants import KANfast, KANcheby
import exp_mechanism as M

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "results", "exp_mechanism_variants.csv")


def main():
    rng_t = np.random.default_rng(0)
    f_add_raw, f_int_raw = M.muc_tieu(rng_t)
    f_add = M.chuan_hoa(f_add_raw, np.random.default_rng(1))
    f_int = M.chuan_hoa(f_int_raw, np.random.default_rng(2))

    D, B = M.D, M.BUDGET
    hfa = match_width(B, lambda h: KANfast(D, h, 1))
    hch = match_width(B, lambda h: KANcheby(D, h, 1))
    hm = match_width(B, lambda h: MLPreg(D, h, 1))
    cau = {"KAN-fast": (lambda: KANfast(D, hfa, 1)),
           "KAN-cheby": (lambda: KANcheby(D, hch, 1)),
           "MLP": (lambda: MLPreg(D, hm, 1))}
    print("[kiem] ngan sach tham so khop:",
          {k: nparams(v()) for k, v in cau.items()}, "| muc tieu", B)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    rows = []
    for n in M.NTRAIN:
        for lam in M.LAMS:
            acc = {k: [] for k in cau}
            for s in M.SEEDS:
                X, Y, Xt, Yt = M.sinh(lam, n, s, f_add, f_int)
                for k, build in cau.items():
                    torch.manual_seed(s)
                    acc[k].append(train_reg(build(), X, Y, Xt, Yt, seed=s))
            for k in cau:
                m, sd, lo, hi = bootstrap_ci(acc[k])
                rows.append(dict(n=n, lam=lam, model=k, nmse_db=round(m, 4),
                                 sd=round(sd, 4), ci_lo=round(lo, 4), ci_hi=round(hi, 4)))
            for k in ("KAN-fast", "KAN-cheby"):
                d, lo, hi = paired_diff_ci(acc[k], acc["MLP"])
                thang = "KAN" if hi < 0 else ("MLP" if lo > 0 else "khong phan biet duoc")
                rows.append(dict(n=n, lam=lam, model=f"PAIRED {k} - MLP",
                                 nmse_db=round(d, 4), sd="", ci_lo=round(lo, 4),
                                 ci_hi=round(hi, 4), ket_luan=thang))
                print(f"n={n:5d} lam={lam:4.2f} {k:10s} - MLP = {d:+7.3f} dB "
                      f"[{lo:+.3f},{hi:+.3f}] -> {thang}")
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["n", "lam", "model", "nmse_db", "sd",
                                          "ci_lo", "ci_hi", "ket_luan"])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print("\nghi:", OUT)


if __name__ == "__main__":
    main()
