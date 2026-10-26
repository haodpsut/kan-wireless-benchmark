"""Doc ngoai vong 1, muc 3(a): dao dau o n=256 KHONG co doi chung dieu chinh.

Bang 9 quet MLP o n=4096, tuc no bao ve cho cho KAN THANG. Cho KAN THUA la
n=256, lam cao, va o do cau hoi dung la: KAN co bi dieu chinh it hon khong?

Script nay quet PHIA KAN o n=256 tai lam=0,75 va lam=1,0: bac co so Fourier K,
so moc luoi spline, va chieu rong, luon GIU ngan sach 2300 tham so. Lay cau hinh
KAN TOT NHAT roi so ghep cap voi MLP.
"""
import csv, os, sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from harness import (KANreg, KANspline, MLPreg, match_width, nparams, train_reg,
                     bootstrap_ci, paired_diff_ci)
import exp_mechanism2 as M

LAMS = [0.75, 1.0]
N = 256
SEEDS = list(range(10))
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results",
                   "exp_kan_tuning_n256.csv")


def main():
    rng_t = np.random.default_rng(0)
    fa, fi = M.muc_tieu(rng_t)
    f_add = M.chuan_hoa(fa, np.random.default_rng(1))
    f_int = M.chuan_hoa(fi, np.random.default_rng(2))
    D, B = M.D, M.BUDGET

    cau = {}
    for K in (2, 4, 8, 16):
        h = match_width(B, lambda h, K=K: KANreg(D, h, 1, K))
        cau[f"Fourier K={K}"] = (lambda h=h, K=K: KANreg(D, h, 1, K), h)
    for G in (4, 6, 10, 16):
        h = match_width(B, lambda h, G=G: KANspline(D, h, 1, G))
        cau[f"spline G={G}"] = (lambda h=h, G=G: KANspline(D, h, 1, G), h)
    hm = match_width(B, lambda h: MLPreg(D, h, 1))

    print(f"ngan sach {B}; MLP h={hm} ({nparams(MLPreg(D, hm, 1))} tham so)")
    for ten, (f, h) in cau.items():
        print(f"   {ten:12s} h={h:4d}  {nparams(f()):6d} tham so")

    rows = []
    for lam in LAMS:
        data = [M.sinh(lam, N, s, f_add, f_int) for s in SEEDS]
        mlp = []
        for s, (X, Y, Xt, Yt) in zip(SEEDS, data):
            torch.manual_seed(s)
            mlp.append(train_reg(MLPreg(D, hm, 1), X, Y, Xt, Yt, seed=s))
        mm, ms, _, _ = bootstrap_ci(mlp)
        print(f"\n=== lam={lam} | MLP {mm:.3f} +- {ms:.3f} dB ===")
        rows.append(dict(lam=lam, model="MLP", nmse_db=round(mm, 4), sd=round(ms, 4)))
        tot = None
        for ten, (build, h) in cau.items():
            v = []
            for s, (X, Y, Xt, Yt) in zip(SEEDS, data):
                torch.manual_seed(s)
                v.append(train_reg(build(), X, Y, Xt, Yt, seed=s))
            m, sd, _, _ = bootstrap_ci(v)
            d, lo, hi = paired_diff_ci(v, mlp)
            kl = "KAN" if hi < 0 else ("MLP" if lo > 0 else "khong phan biet duoc")
            print(f"   {ten:12s} {m:7.3f} +- {sd:.3f} | KAN-MLP={d:+7.3f} "
                  f"[{lo:+.3f},{hi:+.3f}] -> {kl}")
            rows.append(dict(lam=lam, model=ten, nmse_db=round(m, 4), sd=round(sd, 4),
                             paired=round(d, 4), ci_lo=round(lo, 4), ci_hi=round(hi, 4),
                             ket_luan=kl))
            if tot is None or m < tot[1]:
                tot = (ten, m, v)
        d, lo, hi = paired_diff_ci(tot[2], mlp)
        kl = "KAN" if hi < 0 else ("MLP" if lo > 0 else "khong phan biet duoc")
        print(f"   => KAN TOT NHAT {tot[0]}: KAN-MLP={d:+.3f} [{lo:+.3f},{hi:+.3f}] -> {kl}")
        rows.append(dict(lam=lam, model=f"BEST={tot[0]}", nmse_db=round(tot[1], 4),
                         paired=round(d, 4), ci_lo=round(lo, 4), ci_hi=round(hi, 4),
                         ket_luan=kl))

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["lam", "model", "nmse_db", "sd", "paired",
                                          "ci_lo", "ci_hi", "ket_luan"])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print("\nghi:", OUT)


if __name__ == "__main__":
    main()
