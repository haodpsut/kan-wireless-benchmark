"""R2.7b: tuyen bo 0,19 dB o ngan sach K=8 KHONG co khoang tin cay ghep cap.

Ban cu, muc VIII: Fourier-KAN K=8 dat -17,14 dB o 4634 tham so, MLP ba lop dieu chinh
dat -16,95 dB o cung ngan sach => "regression win survives a symmetrically tuned MLP,
but only by 0,19 dB". Ma do lech chuan theo hat giong cua chinh bo quet la ~0,12-0,17 dB,
voi 5 hat giong, va phep so la KHONG GHEP CAP.

Script nay chay lai dung phep so do, GHEP CAP theo hat giong, 20 hat giong, kem:
  - bootstrap CI cua hieu ghep cap
  - Wilcoxon sign-rank CHINH XAC (scipy, mau nho nen dung exact)
  - dau hieu kiem tra tai lap: in lai trung binh de doi chieu voi -17,142 va -16,95
Neu CI chua 0 thi ket luan dung phai la KHONG PHAN BIET DUOC, va cau trong bai phai ha cap.
"""
import csv, os, sys
import numpy as np
import torch
from scipy.stats import wilcoxon

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from harness import KANreg, nparams, train_reg, bootstrap_ci, paired_diff_ci
from exp_dpd import make_dataset, L
from exp_mlp_sweep import MLPflex, match_flex

SEEDS = list(range(20))
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results",
                   "exp_bigbudget_paired.csv")


def main():
    din = 2 * L
    kan = lambda: KANreg(din, 24, 2, 8)          # K=8, dung cau hinh cua abl_dpd_kanhp
    target = nparams(kan())
    h3 = match_flex(target, din, 2, 3, "relu")   # MLP 3 lop, cau hinh TOT NHAT cua bo quet
    mlp = lambda: MLPflex(din, h3, 2, 3, "relu")
    print(f"ngan sach: Fourier-KAN K=8 = {nparams(kan())} | MLP depth3-relu h={h3} = {nparams(mlp())}")

    a, b = [], []
    for s in SEEDS:
        Xtr, Ytr = make_dataset(s)
        Xte, Yte = make_dataset(100 + s)
        torch.manual_seed(s); a.append(train_reg(kan(), Xtr, Ytr, Xte, Yte, seed=s))
        torch.manual_seed(s); b.append(train_reg(mlp(), Xtr, Ytr, Xte, Yte, seed=s))

    ma, sda, loa, hia = bootstrap_ci(a)
    mb, sdb, lob, hib = bootstrap_ci(b)
    print(f"\nFourier-KAN K=8 : {ma:.3f} +- {sda:.3f} dB   CI[{loa:.3f},{hia:.3f}]"
          f"   (ban cu bao -17,142)")
    print(f"MLP depth3-relu : {mb:.3f} +- {sdb:.3f} dB   CI[{lob:.3f},{hib:.3f}]"
          f"   (ban cu bao -16,95)")

    d, lo, hi = paired_diff_ci(a, b)             # am = KAN tot hon (NMSE thap hon)
    st, p = wilcoxon(a, b, method="exact")
    kl = "KAN tot hon" if hi < 0 else ("MLP tot hon" if lo > 0 else "KHONG PHAN BIET DUOC")
    print(f"\nhieu GHEP CAP (KAN - MLP) = {d:+.3f} dB  CI95[{lo:+.3f},{hi:+.3f}]")
    print(f"Wilcoxon sign-rank chinh xac: W={st:.1f}  p={p:.4f}  (n={len(SEEDS)} hat giong)")
    print(f"=> KET LUAN: {kl}")
    print(f"=> do lon tuyen bo cu: 0,19 dB. Do duoc o day: {abs(d):.3f} dB,"
          f" sd theo hat giong {np.std(np.array(a)-np.array(b)):.3f} dB")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dai_luong", "gia_tri", "sd", "ci_lo", "ci_hi", "ghi_chu"])
        w.writerow(["fourier_kan_K8", round(ma, 4), round(sda, 4), round(loa, 4), round(hia, 4),
                    f"{len(SEEDS)} hat giong; ban cu -17.142 voi 5 hat giong"])
        w.writerow(["mlp_depth3_relu", round(mb, 4), round(sdb, 4), round(lob, 4), round(hib, 4),
                    f"{len(SEEDS)} hat giong; ban cu -16.95"])
        w.writerow(["paired_kan_minus_mlp", round(d, 4), "", round(lo, 4), round(hi, 4), kl])
        w.writerow(["wilcoxon_exact_p", round(float(p), 6), "", "", "", f"W={st}"])
        for s, va, vb in zip(SEEDS, a, b):
            w.writerow([f"seed{s}", round(va, 4), "", "", "", f"mlp={round(vb,4)}"])
    print("\nghi:", OUT)


if __name__ == "__main__":
    main()
