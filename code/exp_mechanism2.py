"""CO CHE, ban 2: sua LOI THIET KE cua ban 1.

Ban 1 (exp_mechanism.py) tron phan cong tinh dang sin(w_i x_i + p_i) voi phan tuong
tac dang tong x_j*x_k. Chuan hoa phuong sai da go confound ve BIEN DO, nhung cot MLP
to ra mot confound CON LAI va nang hon:

    MLP o n=256:  -1.54 dB (lam=0)  ->  -3.01 dB (lam=1)
    MLP o n=4096: -9.22 dB (lam=0)  -> -16.75 dB (lam=1)

MLP TOT HON khi tuong tac MANH hon. Nghia la lam khong chi doi "do tuong tac" ma doi
luon DO KHO noi tai: phan sin la ham dao dong (w den 2,5 tren x in [-2,2]), con phan
tuong tac la dang SONG TUYEN TINH, tron va bac thap, de xap xi hon han.
=> lam dieu khien HAI thu: tinh cong tinh VA do kho. Dung lop loi "mot tham so dieu
   khien hai thu" ma so nha da ghi.

Sua: cho HAI phan CUNG MOT LOP HAM, cung bac 2, chi khac o cho co so hang CHEO khong.
    f_add = sum_i a_i * x_i^2        (bac 2, CONG TINH thuan)
    f_int = sum_{(j,k)} c_jk * x_j*x_k   (bac 2, CHEO thuan)
Mot mo hinh bieu dien duoc da thuc bac 2 thi bieu dien duoc CA HAI chinh xac, nen
khac biet duy nhat con lai la TINH CONG TINH.

Doi chung DO DUOC, khong phai gia dinh: in NMSE cua mot BINH PHUONG TOI THIEU tren
day du dac trung bac 2 (1, x_i, x_i^2, x_j*x_k) o tung lam. Do la mo hinh TRUNG TINH
voi tinh cong tinh; neu no gan nhu khong doi theo lam thi hai dau lam DE NHU NHAU,
va luc do so sanh KAN-MLP moi sach.
"""
import csv, os, sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from harness import (KANspline, KANreg, MLPreg, match_width, nparams, train_reg,
                     bootstrap_ci, paired_diff_ci)
from variants import KANfast, KANcheby

D = 8
BUDGET = 2300
LAMS = [0.0, 0.25, 0.5, 0.75, 1.0]
NTRAIN = [256, 4096]
NTEST = 4000
SEEDS = list(range(10))
NOISE_SD = 0.05
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results",
                   "exp_mechanism2.csv")


def muc_tieu(rng):
    a = rng.uniform(0.5, 1.5, D) * rng.choice([-1.0, 1.0], D)
    pairs = [(i, j) for i in range(D) for j in range(i + 1, D)]
    idx = rng.choice(len(pairs), size=D, replace=False)
    pr = [pairs[k] for k in idx]
    c = rng.uniform(0.5, 1.5, len(pr)) * rng.choice([-1.0, 1.0], len(pr))
    f_add = lambda x: sum(a[i] * x[:, i] ** 2 for i in range(D))
    f_int = lambda x: sum(c[k] * x[:, i] * x[:, j] for k, (i, j) in enumerate(pr))
    return f_add, f_int


def chuan_hoa(f, rng):
    xr = rng.uniform(-2, 2, (20000, D)).astype(np.float32)
    v = f(xr); mu, sd = float(v.mean()), float(v.std())
    return lambda x: ((f(x) - mu) / sd).astype(np.float32)


def sinh(lam, n, seed, f_add, f_int):
    rng = np.random.default_rng(1000 + seed)
    X = rng.uniform(-2, 2, (n, D)).astype(np.float32)
    Xt = rng.uniform(-2, 2, (NTEST, D)).astype(np.float32)
    mix = lambda x: np.sqrt(1 - lam) * f_add(x) + np.sqrt(lam) * f_int(x)
    Y = (mix(X) + rng.normal(0, NOISE_SD, n)).astype(np.float32)[:, None]
    Yt = (mix(Xt) + rng.normal(0, NOISE_SD, NTEST)).astype(np.float32)[:, None]
    return X, Y, Xt, Yt


def dac_trung_bac2(X):
    """1, x_i, x_i^2, x_j*x_k: khong gian ham bac 2 DAY DU."""
    cols = [np.ones((len(X), 1), np.float32), X, X ** 2]
    cross = [(X[:, i] * X[:, j])[:, None] for i in range(D) for j in range(i + 1, D)]
    return np.concatenate(cols + cross, 1)


def moc_trung_tinh(X, Y, Xt, Yt):
    """Binh phuong toi thieu tren day du dac trung bac 2 => NMSE dB. Trung tinh
    voi tinh cong tinh, nen dung lam THUOC DO DO KHO cua tung lam."""
    A, At = dac_trung_bac2(X), dac_trung_bac2(Xt)
    w, *_ = np.linalg.lstsq(A, Y, rcond=None)
    p = At @ w
    return float(10 * np.log10(((p - Yt) ** 2).sum() / (Yt ** 2).sum()))


def main():
    rng_t = np.random.default_rng(0)
    fa, fi = muc_tieu(rng_t)
    f_add = chuan_hoa(fa, np.random.default_rng(1))
    f_int = chuan_hoa(fi, np.random.default_rng(2))

    print("[doi chung DO KHO] NMSE cua binh phuong toi thieu bac 2 (mo hinh TRUNG TINH):")
    for lam in LAMS:
        X, Y, Xt, Yt = sinh(lam, 4096, 0, f_add, f_int)
        print(f"   lam={lam:4.2f}  {moc_trung_tinh(X, Y, Xt, Yt):8.3f} dB")
    print("   => hai dau lam de nhu nhau thi cac so nay gan nhau; lech nhieu la CON confound")

    hs = match_width(BUDGET, lambda h: KANspline(D, h, 1))
    hf = match_width(BUDGET, lambda h: KANreg(D, h, 1))
    hfa = match_width(BUDGET, lambda h: KANfast(D, h, 1))
    hch = match_width(BUDGET, lambda h: KANcheby(D, h, 1))
    hm = match_width(BUDGET, lambda h: MLPreg(D, h, 1))
    cau = {"KAN-spline": lambda: KANspline(D, hs, 1),
           "KAN-Fourier": lambda: KANreg(D, hf, 1),
           "KAN-fast": lambda: KANfast(D, hfa, 1),
           "KAN-cheby": lambda: KANcheby(D, hch, 1),
           "MLP": lambda: MLPreg(D, hm, 1)}
    print("\n[kiem] ngan sach tham so:", {k: nparams(v()) for k, v in cau.items()},
          "| muc tieu", BUDGET)

    rows = []
    for n in NTRAIN:
        for lam in LAMS:
            data = [sinh(lam, n, s, f_add, f_int) for s in SEEDS]
            ls = [moc_trung_tinh(*d) for d in data]
            m_ls, s_ls, _, _ = bootstrap_ci(ls)
            rows.append(dict(n=n, lam=lam, model="LSQ-poly2(moc do kho)",
                             nmse_db=round(m_ls, 4), sd=round(s_ls, 4)))
            acc = {}
            for k, build in cau.items():
                v = []
                for s, (X, Y, Xt, Yt) in zip(SEEDS, data):
                    torch.manual_seed(s)
                    v.append(train_reg(build(), X, Y, Xt, Yt, seed=s))
                acc[k] = v
                m, sd, lo, hi = bootstrap_ci(v)
                rows.append(dict(n=n, lam=lam, model=k, nmse_db=round(m, 4),
                                 sd=round(sd, 4), ci_lo=round(lo, 4), ci_hi=round(hi, 4)))
            print(f"\nn={n} lam={lam}: moc LSQ={m_ls:.2f} dB | MLP={np.mean(acc['MLP']):.2f} dB")
            for k in cau:
                if k == "MLP":
                    continue
                d, lo, hi = paired_diff_ci(acc[k], acc["MLP"])
                kl = "KAN" if hi < 0 else ("MLP" if lo > 0 else "khong phan biet duoc")
                print(f"   {k:12s} - MLP = {d:+7.3f} dB [{lo:+.3f},{hi:+.3f}] -> {kl}")
                rows.append(dict(n=n, lam=lam, model=f"PAIRED {k} - MLP",
                                 nmse_db=round(d, 4), ci_lo=round(lo, 4),
                                 ci_hi=round(hi, 4), ket_luan=kl))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["n", "lam", "model", "nmse_db", "sd",
                                          "ci_lo", "ci_hi", "ket_luan"])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print("\nghi:", OUT)


if __name__ == "__main__":
    main()
