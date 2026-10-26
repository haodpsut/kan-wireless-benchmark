"""Doi chung CO THE HONG cho cau hoi do kho, theo yeu cau cua doc ngoai vong 1, muc 2(a).

⛔ Vi sao phai thay doi chung cu: muc 10.2 dung binh phuong toi thieu tren DAY DU dac
trung bac 2 lam "moc trung tinh". Nhung ham dich cua chinh thi nghiem la
    f_add = sum a_i x_i^2        thuoc span{x_i^2}
    f_int = sum c_jk x_j x_k     thuoc span{x_j x_k}
nen voi MOI lam, dich NAM TRON trong khong gian dac trung cua OLS. OLS do do dac ta
DUNG o moi lam: do chech bang 0, phuong sai la sigma^2 * p/n voi p=45 co dinh. NMSE
cua no PHANG THEO CAU TRUC. Mot doi chung khong the hong thi khong phai bang chung.

Script nay lam hai viec:
  A. CHUNG MINH dieu tren: in NMSE cua OLS theo lam, roi lam lai voi mot dich CO phan
     NGOAI span (them sin(3x)) de thay OLS luc do MOI hong. Neu A cho ket qua nhu du
     doan thi loi phan bien la DUNG va phai nhan.
  B. DOI CHUNG MOI, co the hong: mo hinh dac trung ngau nhien (random Fourier features)
     + hoi quy ridge, GIOI HAN NGAN SACH bang dung so tham so cua cac mang (2300), va
     KHONG he biet gi ve tinh cong tinh. Neu NMSE cua no phang theo lam thi cau hoi do
     kho da duoc tra loi bang mot phep thu co the truot; neu no doc thi mot phan muc
     suy giam thuoc ve DO KHO va phai sua lai do lon hieu ung.
"""
import csv, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import exp_mechanism2 as M

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results",
                   "exp_mechanism2_control.csv")
NGAN_SACH = M.BUDGET          # 2300 tham so, dung ngan sach cua cac mang


def nmse_db(p, y):
    return float(10 * np.log10(((p - y) ** 2).sum() / (y ** 2).sum()))


def ols_bac2(X, Y, Xt, Yt):
    A, At = M.dac_trung_bac2(X), M.dac_trung_bac2(Xt)
    w, *_ = np.linalg.lstsq(A, Y, rcond=None)
    return nmse_db(At @ w, Yt)


class DacTrungNgauNhien:
    """z(x) = cos(W x + b), W ~ N(0, s^2). So THAM SO HOC DUOC = so dac trung D.

    Khong co cau truc cong tinh nao duoc cai san: moi dac trung tron TAT CA toa do.
    Day la cho no CO THE hong khi dich kho len.
    """
    def __init__(self, D, d, s=1.0, seed=0):
        r = np.random.default_rng(seed)
        self.W = r.normal(0, s, (d, D)).astype(np.float32)
        self.b = r.uniform(0, 2 * np.pi, D).astype(np.float32)
        self.D = D

    def z(self, X):
        return np.cos(X @ self.W + self.b).astype(np.float32)

    def fit_predict(self, X, Y, Xt, lam_ridge=1e-6):
        Z, Zt = self.z(X), self.z(Xt)
        A = Z.T @ Z + lam_ridge * len(X) * np.eye(self.D, dtype=np.float32)
        w = np.linalg.solve(A, Z.T @ Y)
        return Zt @ w


def main():
    rng_t = np.random.default_rng(0)
    fa, fi = M.muc_tieu(rng_t)
    f_add = M.chuan_hoa(fa, np.random.default_rng(1))
    f_int = M.chuan_hoa(fi, np.random.default_rng(2))
    D = M.D
    rows = []

    print("=" * 78)
    print("[A] CHUNG MINH doi chung CU khong the hong")
    print(f"    so dac trung bac 2 = {M.dac_trung_bac2(np.zeros((2, D), np.float32)).shape[1]}"
          f" (1 + d + d + so cap)")
    for n in M.NTRAIN:
        v = []
        for lam in M.LAMS:
            X, Y, Xt, Yt = M.sinh(lam, n, 0, f_add, f_int)
            v.append(ols_bac2(X, Y, Xt, Yt))
        print(f"    n={n:5d}  OLS bac2 theo lam: " + " ".join(f"{x:7.2f}" for x in v)
              + f"   bien do {max(v)-min(v):.3f} dB")
        rows += [dict(doi_chung="OLS-bac2", n=n, lam=l, nmse_db=round(x, 4))
                 for l, x in zip(M.LAMS, v)]

    # dich CO phan ngoai span: them sin(3 x_i) vao phan cong tinh
    def f_add_ngoai_span(x):
        return (f_add(x) + np.sin(3 * x).sum(1)).astype(np.float32)
    v = []
    for lam in M.LAMS:
        rng = np.random.default_rng(1000)
        X = rng.uniform(-2, 2, (4096, D)).astype(np.float32)
        Xt = rng.uniform(-2, 2, (M.NTEST, D)).astype(np.float32)
        mix = lambda z: np.sqrt(1 - lam) * f_add_ngoai_span(z) + np.sqrt(lam) * f_int(z)
        Y = mix(X)[:, None]; Yt = mix(Xt)[:, None]
        v.append(ols_bac2(X, Y, Xt, Yt))
    print(f"    doi chung NGUOC (dich co sin(3x), NGOAI span): "
          + " ".join(f"{x:7.2f}" for x in v) + f"   bien do {max(v)-min(v):.3f} dB")
    print("    => OLS chi hong khi dich RA KHOI span cua no. Trong thiet ke cua bai thi")
    print("       dich LUON nam trong span, nen moc cu PHANG THEO CAU TRUC. Phan bien DUNG.")
    rows += [dict(doi_chung="OLS-bac2-dich-ngoai-span", n=4096, lam=l, nmse_db=round(x, 4))
             for l, x in zip(M.LAMS, v)]

    print("\n" + "=" * 78)
    print(f"[B] DOI CHUNG MOI co the hong: dac trung ngau nhien + ridge, D={NGAN_SACH}")
    for n in M.NTRAIN:
        tb, sd = [], []
        for lam in M.LAMS:
            v = []
            for s in M.SEEDS:
                X, Y, Xt, Yt = M.sinh(lam, n, s, f_add, f_int)
                rf = DacTrungNgauNhien(NGAN_SACH, D, seed=s)
                v.append(nmse_db(rf.fit_predict(X, Y, Xt), Yt))
            tb.append(float(np.mean(v))); sd.append(float(np.std(v)))
            rows.append(dict(doi_chung="random-features", n=n, lam=lam,
                             nmse_db=round(tb[-1], 4), sd=round(sd[-1], 4)))
        print(f"    n={n:5d}  " + " ".join(f"{x:7.2f}" for x in tb)
              + f"   bien do {max(tb)-min(tb):.3f} dB")
    print("    => bien do NHO thi cau hoi do kho da duoc tra bang phep thu CO THE truot;")
    print("       bien do LON thi mot phan suy giam thuoc ve DO KHO, phai sua do lon hieu ung.")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["doi_chung", "n", "lam", "nmse_db", "sd"])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print("\nghi:", OUT)


if __name__ == "__main__":
    main()
