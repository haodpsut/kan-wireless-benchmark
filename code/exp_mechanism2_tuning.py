"""DOI CHUNG dieu chinh cho ham dich BAN 2 (da khu confound do kho), n=4096: co phai KAN thang khap moi lam chi vi MLP
BI DIEU CHINH IT HON?

Luoi co che cho mot ket qua kho tin: o n=256 diem giao hien ro (KAN thang o lam thap,
MLP thang o lam cao), nhung o n=4096 thi KAN thang o MOI lam, ke ca lam=1 (tuong tac
thuan). Nhieu du lieu hon le ra phai co loi cho mo hinh IT CAU TRUC hon, nen ket qua
nay nghi la ARTIFACT cua viec dieu chinh KHONG DOI XUNG, dung lop loi ma chinh bai
nay di canh bao nguoi khac.

Phep thu: o n=4096, cho CA HAI ben cung co hoi:
  - quet do sau x kich hoat cho MLP, lay cau hinh TOT NHAT
  - cho ca hai lich huan luyen dai hon (epochs 60 va 180), DOI XUNG
Neu MLP tot nhat khep lai duoc khoang cach thi phai bao dung the va ha cap ket luan.
"""
import csv, os, sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from harness import KANspline, MLPreg, match_width, nparams, train_reg, paired_diff_ci, bootstrap_ci
from exp_mlp_sweep import MLPflex, match_flex
import exp_mechanism2 as M

LAMS = [0.5, 1.0]
N = 4096
EPOCHS = [60, 180]
CONFIGS = [(2, "relu"), (3, "relu"), (4, "relu")]
SEEDS = list(range(10))
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results",
                   "exp_mechanism2_tuning.csv")


def main():
    rng_t = np.random.default_rng(0)
    fa, fi = M.muc_tieu(rng_t)
    f_add = M.chuan_hoa(fa, np.random.default_rng(1))
    f_int = M.chuan_hoa(fi, np.random.default_rng(2))
    D, B = M.D, M.BUDGET
    hs = match_width(B, lambda h: KANspline(D, h, 1))
    print(f"KAN-spline h={hs} ({nparams(KANspline(D, hs, 1))}p), ngan sach {B}")

    rows = []
    for ep in EPOCHS:
        for lam in LAMS:
            data = [M.sinh(lam, N, s, f_add, f_int) for s in SEEDS]
            kan = []
            for s, (X, Y, Xt, Yt) in zip(SEEDS, data):
                torch.manual_seed(s)
                kan.append(train_reg(KANspline(D, hs, 1), X, Y, Xt, Yt, seed=s, epochs=ep))
            mk, sk, _, _ = bootstrap_ci(kan)
            print(f"\nepochs={ep} lam={lam}: KAN-spline {mk:.3f} +- {sk:.3f} dB")
            tot_nhat = None
            for depth, act in CONFIGS:
                h = match_flex(B, D, 1, depth, act)
                v = []
                for s, (X, Y, Xt, Yt) in zip(SEEDS, data):
                    torch.manual_seed(s)
                    v.append(train_reg(MLPflex(D, h, 1, depth, act), X, Y, Xt, Yt,
                                       seed=s, epochs=ep))
                mm, sm, _, _ = bootstrap_ci(v)
                d, lo, hi = paired_diff_ci(kan, v)
                kl = "KAN" if hi < 0 else ("MLP" if lo > 0 else "khong phan biet duoc")
                print(f"   MLP depth{depth}-{act:4s} h={h:3d}: {mm:7.3f} +- {sm:.3f}"
                      f" | KAN-MLP={d:+7.3f} [{lo:+.3f},{hi:+.3f}] -> {kl}")
                rows.append(dict(epochs=ep, lam=lam, mlp=f"depth{depth}-{act}",
                                 kan_nmse=round(mk, 4), mlp_nmse=round(mm, 4),
                                 paired=round(d, 4), ci_lo=round(lo, 4),
                                 ci_hi=round(hi, 4), ket_luan=kl))
                if tot_nhat is None or mm < tot_nhat[1]:
                    tot_nhat = (f"depth{depth}-{act}", mm, v)
            d, lo, hi = paired_diff_ci(kan, tot_nhat[2])
            kl = "KAN" if hi < 0 else ("MLP" if lo > 0 else "khong phan biet duoc")
            print(f"   => MLP TOT NHAT {tot_nhat[0]}: KAN-MLP={d:+.3f} [{lo:+.3f},{hi:+.3f}]"
                  f" -> {kl}")
            rows.append(dict(epochs=ep, lam=lam, mlp=f"BEST={tot_nhat[0]}",
                             kan_nmse=round(mk, 4), mlp_nmse=round(tot_nhat[1], 4),
                             paired=round(d, 4), ci_lo=round(lo, 4), ci_hi=round(hi, 4),
                             ket_luan=kl))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["epochs", "lam", "mlp", "kan_nmse", "mlp_nmse",
                                          "paired", "ci_lo", "ci_hi", "ket_luan"])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print("\nghi:", OUT)


if __name__ == "__main__":
    main()
