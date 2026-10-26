"""R2.3: phan bien 2 noi DUNG rang phep so AMC cua ban cu la BAT CONG.

Ban cu: CNN co bo trich dac trung tich chap (khai thac cau truc thoi gian I/Q), con
KAN va MLP nhan dau vao DAN PHANG 256 chieu. Nhu vay chenh lech co the den tu
INDUCTIVE BIAS cua tich chap, khong phai tu chuyen phi tuyen o DINH so voi o CANH.

Sua: dung CHUNG mot backbone tich chap, chi doi DAU phan loai, va khop ngan sach
tham so GIUA CAC DAU. Cau hoi tach sach: voi cung bo dac trung, dau KAN co hon dau
MLP khong?

Dau duoc so: linear (nhu CNN cu, lam moc) · MLP · KAN-Fourier · KAN-spline.
"""
import csv, os, sys
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from harness import (FourierKANLayer, SplineKANLayer, nparams, match_width,
                     train_cls, bootstrap_ci, paired_diff_ci)
from exp_amc import WIN

SHA_BAND = "a0808c1d07f456234b0f9147b68ada8d40f981e307e4d09848dd4317ea11c311"


def load_band():
    """Nap dai SNN da loc neu co, KHONG thi doc pickle goc.

    Ban .npz duoc sinh bang chinh load_rml() cua exp_amc.py (xem prep_rml_band.py),
    nen khong co duong du lieu thu hai. Bam SHA-256 phai khop, neu khong thi DUNG:
    vang mat hien vat dung khong duoc doc thanh "du lieu dung".
    """
    import hashlib
    import numpy as _np
    band = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "rml_band.npz")
    if os.path.exists(band):
        z = _np.load(band, allow_pickle=False)
        X, y, mods = z["X"], z["y"], [str(m) for m in z["mods"]]
        h = hashlib.sha256()
        h.update(_np.ascontiguousarray(X).tobytes())
        h.update(_np.ascontiguousarray(y).tobytes())
        got = h.hexdigest()
        if got != SHA_BAND:
            raise SystemExit(f"⛔ BAM LECH: {got} != {SHA_BAND} => du lieu khong phai ban da do")
        print(f"[kiem] nap rml_band.npz, SHA-256 khop: {got[:16]}...")
        return X, y, mods
    from exp_amc import load_rml
    print("[kiem] khong thay rml_band.npz, doc pickle goc")
    return load_rml()

SEEDS = list(range(10))
NSUB = 40000
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results",
                   "exp_amc_backbone.csv")


class Backbone(nn.Module):
    """DUNG NGUYEN bo trich dac trung cua CNN trong exp_amc.py, de so duoc voi ban cu."""
    def __init__(self):
        super().__init__()
        self.c = nn.Sequential(nn.Conv1d(2, 24, 5, padding=2), nn.ReLU(),
                               nn.Conv1d(24, 24, 5, padding=2), nn.ReLU(),
                               nn.AdaptiveAvgPool1d(8))
        self.dout = 24 * 8

    def forward(self, x):
        b = x.shape[0]
        return self.c(x.view(b, 2, WIN)).reshape(b, -1)


class LinearHead(nn.Module):
    def __init__(self, din, nc):
        super().__init__(); self.f = nn.Linear(din, nc)
    def forward(self, z): return self.f(z)


class MLPHead(nn.Module):
    def __init__(self, din, h, nc):
        super().__init__()
        self.f = nn.Sequential(nn.Linear(din, h), nn.ReLU(), nn.Linear(h, nc))
    def forward(self, z): return self.f(z)


class KANFourierHead(nn.Module):
    def __init__(self, din, h, nc, K=4):
        super().__init__()
        self.l1, self.l2 = FourierKANLayer(din, h, K), FourierKANLayer(h, nc, K)
    def forward(self, z): return self.l2(torch.tanh(self.l1(z)))


class KANSplineHead(nn.Module):
    def __init__(self, din, h, nc):
        super().__init__()
        self.l1, self.l2 = SplineKANLayer(din, h), SplineKANLayer(h, nc)
    def forward(self, z): return self.l2(self.l1(z))


class Net(nn.Module):
    """Backbone CHUNG + mot dau. Nhan dau vao dan phang de dung lai train_cls."""
    def __init__(self, head_fn, nc):
        super().__init__()
        self.bb = Backbone()
        self.head = head_fn(self.bb.dout, nc)
    def forward(self, x): return self.head(self.bb(x))


def main():
    X, y, mods = load_band()
    nc = len(mods)
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(X))[:NSUB]
    Xf = X.reshape(len(X), -1)[idx]; yf = y[idx]
    ntr = int(0.7 * len(Xf))
    Xtr, ytr, Xte, yte = Xf[:ntr], yf[:ntr], Xf[ntr:], yf[ntr:]
    dbb = Backbone().dout
    print(f"N={len(Xf)} lop={nc} dac trung backbone={dbb}")

    # ngan sach dau: lay dau KAN-Fourier h=8 lam muc tieu, khop hai dau kia vao do
    target = nparams(KANFourierHead(dbb, 8, nc))
    h_mlp = match_width(target, lambda h: MLPHead(dbb, h, nc), lo=2, hi=64)
    h_spl = match_width(target, lambda h: KANSplineHead(dbb, h, nc), lo=2, hi=48)
    heads = {
        "linear (moc, nhu CNN cu)": lambda d, c: LinearHead(d, c),
        "MLP head":                 lambda d, c: MLPHead(d, h_mlp, c),
        "KAN-Fourier head":         lambda d, c: KANFourierHead(d, 8, c),
        "KAN-spline head":          lambda d, c: KANSplineHead(d, h_spl, c),
    }
    print("tham so TOAN MANG (backbone dung chung + dau):")
    for k, f in heads.items():
        print(f"   {k:28s} {nparams(Net(f, nc)):7d}  (dau: {nparams(f(dbb, nc)):6d})")

    acc = {}
    for k, f in heads.items():
        acc[k] = [train_cls(Net(f, nc), Xtr, ytr, Xte, yte, seed=s) for s in SEEDS]
        m, sd, lo, hi = bootstrap_ci(acc[k])
        print(f"{k:28s} acc={m:.4f} +- {sd:.4f}  CI[{lo:.4f},{hi:.4f}]")

    rows = []
    for k in heads:
        m, sd, lo, hi = bootstrap_ci(acc[k])
        rows.append(dict(head=k, acc=round(m, 4), sd=round(sd, 4),
                         ci_lo=round(lo, 4), ci_hi=round(hi, 4)))
    print("\nhieu GHEP CAP, duong = dau dau tien tot hon:")
    for a, b in [("KAN-Fourier head", "MLP head"), ("KAN-spline head", "MLP head"),
                 ("MLP head", "linear (moc, nhu CNN cu)"),
                 ("KAN-Fourier head", "linear (moc, nhu CNN cu)")]:
        d, lo, hi = paired_diff_ci(acc[a], acc[b])
        kl = a if lo > 0 else (b if hi < 0 else "khong phan biet duoc")
        print(f"   {a:18s} - {b:24s} = {d:+.4f} [{lo:+.4f},{hi:+.4f}] -> {kl}")
        rows.append(dict(head=f"PAIRED {a} - {b}", acc=round(d, 4), sd="",
                         ci_lo=round(lo, 4), ci_hi=round(hi, 4), ket_luan=kl))

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", newline="") as fo:
        w = csv.DictWriter(fo, fieldnames=["head", "acc", "sd", "ci_lo", "ci_hi", "ket_luan"])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print("\nghi:", OUT)


if __name__ == "__main__":
    main()
