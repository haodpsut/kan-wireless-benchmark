"""Doc ngoai vong 1, muc 2(b): phep so "matched capacity" dang LECH DO SAU.

Do duoc: MLPreg la d->h->h->1, tuc **2 lop an**; KANreg/KANspline la l1->l2, tuc
**1 lop an**. Cong thuc (2) co h^2 con (3)(4) khong, dung nhu phan bien chi ra. Va
bo quet cua chinh bai do do sau dang gia 0,47 dB tren DPD, nen lech do sau la mot
confound THAT, va no chay theo chieu co loi cho ket luan cua bai.

Script nay so o HAI muc do sau, moi muc deu KHOP ngan sach tham so:
    do sau 1 lop an:  MLP(d->h->1)      doi  KAN 2 lop (l1->l2)
    do sau 2 lop an:  MLP(d->h->h->1)   doi  KAN 3 lop (l1->l2->l3)
Neu thu tu GIU NGUYEN o ca hai muc thi ket luan song sot qua viec khop do sau.
Neu doi thi phai sua ket luan, va phai noi ro.
"""
import csv, os, sys
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from harness import (FourierKANLayer, SplineKANLayer, MLPreg, match_width, nparams,
                     train_reg, bootstrap_ci, paired_diff_ci)
import exp_mechanism2 as M

LAMS = [0.0, 0.5, 1.0]
NTRAIN = [256, 4096]
SEEDS = list(range(10))
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results",
                   "exp_depth_matched.csv")


class MLP1(nn.Module):
    """MOT lop an: d -> h -> out."""
    def __init__(self, din, h, dout):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(din, h), nn.ReLU(), nn.Linear(h, dout))
    def forward(self, x): return self.net(x)


class KANspline3(nn.Module):
    """HAI lop an: ba lop KAN noi tiep."""
    def __init__(self, din, h, dout, grid=6, k=3):
        super().__init__()
        self.l1, self.l2, self.l3 = (SplineKANLayer(din, h, grid, k),
                                     SplineKANLayer(h, h, grid, k),
                                     SplineKANLayer(h, dout, grid, k))
    def forward(self, x): return self.l3(self.l2(self.l1(x)))


class KANspline2(nn.Module):
    """MOT lop an, dung cau hinh cua bai."""
    def __init__(self, din, h, dout, grid=6, k=3):
        super().__init__()
        self.l1, self.l2 = SplineKANLayer(din, h, grid, k), SplineKANLayer(h, dout, grid, k)
    def forward(self, x): return self.l2(self.l1(x))


def main():
    rng_t = np.random.default_rng(0)
    fa, fi = M.muc_tieu(rng_t)
    f_add = M.chuan_hoa(fa, np.random.default_rng(1))
    f_int = M.chuan_hoa(fi, np.random.default_rng(2))
    D, B = M.D, M.BUDGET

    cap = {}
    h = match_width(B, lambda h: MLP1(D, h, 1)); cap[("1 lop an", "MLP")] = (lambda: MLP1(D, h, 1), h)
    h2 = match_width(B, lambda h: KANspline2(D, h, 1)); cap[("1 lop an", "KAN-spline")] = (lambda: KANspline2(D, h2, 1), h2)
    h3 = match_width(B, lambda h: MLPreg(D, h, 1)); cap[("2 lop an", "MLP")] = (lambda: MLPreg(D, h3, 1), h3)
    h4 = match_width(B, lambda h: KANspline3(D, h, 1)); cap[("2 lop an", "KAN-spline")] = (lambda: KANspline3(D, h4, 1), h4)

    print(f"ngan sach muc tieu {B} tham so:")
    for (ds, ten), (f, hh) in cap.items():
        print(f"   {ds:10s} {ten:11s} h={hh:4d}  {nparams(f()):6d} tham so")

    rows = []
    for ds in ("1 lop an", "2 lop an"):
        print(f"\n===== DO SAU: {ds} =====")
        for n in NTRAIN:
            for lam in LAMS:
                data = [M.sinh(lam, n, s, f_add, f_int) for s in SEEDS]
                acc = {}
                for ten in ("MLP", "KAN-spline"):
                    build = cap[(ds, ten)][0]
                    v = []
                    for s, (X, Y, Xt, Yt) in zip(SEEDS, data):
                        torch.manual_seed(s)
                        v.append(train_reg(build(), X, Y, Xt, Yt, seed=s))
                    acc[ten] = v
                    m, sd, lo, hi = bootstrap_ci(v)
                    rows.append(dict(do_sau=ds, n=n, lam=lam, model=ten,
                                     nmse_db=round(m, 4), sd=round(sd, 4)))
                d, lo, hi = paired_diff_ci(acc["KAN-spline"], acc["MLP"])
                kl = "KAN" if hi < 0 else ("MLP" if lo > 0 else "khong phan biet duoc")
                rows.append(dict(do_sau=ds, n=n, lam=lam, model="PAIRED KAN - MLP",
                                 nmse_db=round(d, 4), ci_lo=round(lo, 4),
                                 ci_hi=round(hi, 4), ket_luan=kl))
                print(f"   n={n:5d} lam={lam:4.2f}  KAN-MLP = {d:+7.3f} dB"
                      f" [{lo:+.3f},{hi:+.3f}] -> {kl}")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["do_sau", "n", "lam", "model", "nmse_db", "sd",
                                          "ci_lo", "ci_hi", "ket_luan"])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print("\nghi:", OUT)


if __name__ == "__main__":
    main()
