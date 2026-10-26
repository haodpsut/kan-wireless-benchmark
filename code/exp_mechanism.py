"""CƠ CHẾ: vì sao KAN thắng ở chế độ này mà thua ở chế độ kia.

Tra loi cau hoi ma nhieu nguoi doc hoi cung mot kieu: ban truoc chi in RA rang
có điểm giao, không nói VÌ SAO.

Giả thuyết kiểm được:
  MLP đặt phi tuyến ở ĐỈNH, KAN đặt ở CẠNH. Với d đầu vào, KAN có O(d*G) bậc tự do
  MỘT BIẾN, nên lớp hàm CỘNG TÍNH f(x) = sum_i phi_i(x_i) nằm trong không gian biểu
  diễn của KAN gần như chính xác, còn MLP phải xấp xỉ nó bằng tổ hợp của các đỉnh.
  => DỰ ĐOÁN: KAN thắng khi hàm đích gần cộng tính, và mất lợi thế khi TƯƠNG TÁC
     giữa các chiều mạnh lên.

Thiết kế tách biến, mọi thứ khác giữ NGUYÊN:
  y = sqrt(1-lam) * f_add(x) + sqrt(lam) * f_int(x) + nhiễu
  f_add, f_int đều được CHUẨN HOÁ về phương sai 1 trên một mẫu tham chiếu lớn TRƯỚC
  khi trộn, nên lam chỉ điều khiển ĐÚNG MỘT THỨ: tỉ lệ công suất tương tác.
  (Nếu không chuẩn hoá thì lam vừa đổi tương tác vừa đổi SNR -> một tham số điều
   khiển hai thứ, đúng lớp lỗi đã ghi trong sổ nhà.)

Đối chứng âm bắt buộc: lam = 0 (cộng tính thuần) và lam = 1 (tương tác thuần).
Nếu điểm giao KHÔNG dịch theo lam thì giả thuyết BỊ BÁC, và phải báo đúng như thế.
"""
import csv, os, sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from harness import (KANspline, KANreg, MLPreg, match_width, nparams,
                     train_reg, bootstrap_ci, paired_diff_ci)

D = 8                      # so chieu dau vao
BUDGET = 2300              # ngan sach tham so, lay theo thi nghiem PA cua bai
LAMS = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0]
NTRAIN = [256, 4096]       # truc du lieu it / du lieu nhieu
NTEST = 4000
SEEDS = list(range(10))
NOISE_SD = 0.05            # nhieu quan sat, co dinh
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "exp_mechanism.csv")


def muc_tieu(rng_target):
    """Sinh MOT ham dich co dinh: phan cong tinh va phan tuong tac doi mot."""
    a = rng_target.uniform(0.5, 1.5, D)
    w = rng_target.uniform(0.8, 2.5, D)
    p = rng_target.uniform(0, 2 * np.pi, D)
    pairs = [(i, j) for i in range(D) for j in range(i + 1, D)]
    idx = rng_target.choice(len(pairs), size=D, replace=False)
    pr = [pairs[k] for k in idx]
    c = rng_target.uniform(0.5, 1.5, len(pr)) * rng_target.choice([-1.0, 1.0], len(pr))

    def f_add(x):
        return sum(a[i] * np.sin(w[i] * x[:, i] + p[i]) for i in range(D))

    def f_int(x):
        return sum(c[k] * x[:, i] * x[:, j] for k, (i, j) in enumerate(pr))

    return f_add, f_int


def chuan_hoa(f, rng):
    """Chia cho do lech chuan do tren mau tham chieu lon => phuong sai 1."""
    xr = rng.uniform(-2, 2, (20000, D)).astype(np.float32)
    v = f(xr)
    mu, sd = float(v.mean()), float(v.std())
    return lambda x: ((f(x) - mu) / sd).astype(np.float32)


def sinh(lam, n, seed, f_add, f_int):
    rng = np.random.default_rng(1000 + seed)
    X = rng.uniform(-2, 2, (n, D)).astype(np.float32)
    Xt = rng.uniform(-2, 2, (NTEST, D)).astype(np.float32)
    mix = lambda x: (np.sqrt(1 - lam) * f_add(x) + np.sqrt(lam) * f_int(x))
    Y = (mix(X) + rng.normal(0, NOISE_SD, n)).astype(np.float32)[:, None]
    Yt = (mix(Xt) + rng.normal(0, NOISE_SD, NTEST)).astype(np.float32)[:, None]
    return X, Y, Xt, Yt


def main():
    rng_t = np.random.default_rng(0)
    f_add_raw, f_int_raw = muc_tieu(rng_t)
    f_add = chuan_hoa(f_add_raw, np.random.default_rng(1))
    f_int = chuan_hoa(f_int_raw, np.random.default_rng(2))

    # kiem tra chuan hoa that su lam phuong sai khong doi theo lam
    xr = np.random.default_rng(3).uniform(-2, 2, (20000, D)).astype(np.float32)
    print("[kiem] phuong sai tin hieu theo lam (phai gan 1,00 o moi lam):")
    for lam in LAMS:
        v = np.sqrt(1 - lam) * f_add(xr) + np.sqrt(lam) * f_int(xr)
        print(f"   lam={lam:4.2f}  var={v.var():.4f}")

    hs = match_width(BUDGET, lambda h: KANspline(D, h, 1))
    hf = match_width(BUDGET, lambda h: KANreg(D, h, 1))
    hm = match_width(BUDGET, lambda h: MLPreg(D, h, 1))
    cau = {"KAN-spline": (lambda: KANspline(D, hs, 1)),
           "KAN-Fourier": (lambda: KANreg(D, hf, 1)),
           "MLP": (lambda: MLPreg(D, hm, 1))}
    print("[kiem] ngan sach tham so khop:",
          {k: nparams(v()) for k, v in cau.items()}, "| muc tieu", BUDGET)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    rows = []
    for n in NTRAIN:
        for lam in LAMS:
            acc = {k: [] for k in cau}
            for s in SEEDS:
                X, Y, Xt, Yt = sinh(lam, n, s, f_add, f_int)
                for k, build in cau.items():
                    torch.manual_seed(s)
                    acc[k].append(train_reg(build(), X, Y, Xt, Yt, seed=s))
            for k in cau:
                m, sd, lo, hi = bootstrap_ci(acc[k])
                rows.append(dict(n=n, lam=lam, model=k, nmse_db=round(m, 4),
                                 sd=round(sd, 4), ci_lo=round(lo, 4), ci_hi=round(hi, 4)))
            # hieu GHEP CAP: MLP tru KAN, am = KAN tot hon (NMSE thap hon la tot)
            for k in ("KAN-spline", "KAN-Fourier"):
                d, lo, hi = paired_diff_ci(acc[k], acc["MLP"])
                thang = "KAN" if hi < 0 else ("MLP" if lo > 0 else "khong phan biet duoc")
                rows.append(dict(n=n, lam=lam, model=f"PAIRED {k} - MLP",
                                 nmse_db=round(d, 4), sd="", ci_lo=round(lo, 4),
                                 ci_hi=round(hi, 4), ket_luan=thang))
                print(f"n={n:5d} lam={lam:4.2f} {k:12s} - MLP = {d:+7.3f} dB "
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
