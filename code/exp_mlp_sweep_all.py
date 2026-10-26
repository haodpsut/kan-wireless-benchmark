"""R2.3 (nua sau): bo quet MLP cu CHI chay cho PA. Phan bien noi dung.

Script nay quet DUNG luoi do sau x kich hoat cho HAI tac vu con lai: uoc luong kenh
OFDM va phan loai dieu che. Muc dich khong phai lam MLP thang, ma la tra loi cau
"KAN co thang mot baseline CHUA DIEU CHINH khong".

Ket luan chi duoc viet sau khi co so, va moi phep so deu in hieu GHEP CAP.
"""
import csv, os, sys
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from harness import (KANreg, KANcls, nparams, train_reg, train_cls,
                     bootstrap_ci, paired_diff_ci)
from exp_mlp_sweep import MLPflex, match_flex

CONFIGS = [(1, "relu"), (2, "relu"), (3, "relu"), (2, "gelu"), (2, "tanh"), (4, "relu")]
SEEDS = list(range(10))
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results",
                   "exp_mlp_sweep_all.csv")


class MLPflexCls(nn.Module):
    def __init__(self, din, h, nc, depth=2, act="relu"):
        super().__init__()
        A = {"relu": nn.ReLU, "gelu": nn.GELU, "tanh": nn.Tanh}[act]
        L = [nn.Linear(din, h), A()]
        for _ in range(depth - 1):
            L += [nn.Linear(h, h), A()]
        L += [nn.Linear(h, nc)]
        self.net = nn.Sequential(*L)

    def forward(self, x):
        return self.net(x)


def quet_chest(rows):
    import exp_chest as C
    din, dout = 2 * C.NP, 2 * C.N
    target = nparams(KANreg(din, 24, dout, 4))
    print(f"\n== OFDM channel estimation | ngan sach KAN tham chieu = {target} ==")
    kan = []
    for s in SEEDS:
        Xtr, Ytr = C.make_dataset(s); Xte, Yte = C.make_dataset(200 + s, C.NTEST)
        torch.manual_seed(s)
        kan.append(train_reg(KANreg(din, 24, dout, 4), Xtr, Ytr, Xte, Yte, seed=s))
    m, sd, lo, hi = bootstrap_ci(kan)
    print(f"   Fourier-KAN (moc)      : {m:.3f} +- {sd:.3f} dB")
    rows.append(dict(task="chest", config="fourier-KAN(ref)", params=target,
                     mean=round(m, 4), sd=round(sd, 4)))
    best = None
    for depth, act in CONFIGS:
        h = match_flex(target, din, dout, depth, act)
        p = nparams(MLPflex(din, h, dout, depth, act))
        v = []
        for s in SEEDS:
            Xtr, Ytr = C.make_dataset(s); Xte, Yte = C.make_dataset(200 + s, C.NTEST)
            torch.manual_seed(s)
            v.append(train_reg(MLPflex(din, h, dout, depth, act), Xtr, Ytr, Xte, Yte, seed=s))
        mm, ss, l2, h2 = bootstrap_ci(v)
        d, dl, dh = paired_diff_ci(kan, v)
        kl = "KAN" if dh < 0 else ("MLP" if dl > 0 else "khong phan biet duoc")
        print(f"   MLP depth{depth}-{act:4s} h={h:3d} ({p:5d}p): {mm:.3f} +- {ss:.3f} dB"
              f" | KAN-MLP={d:+.3f} [{dl:+.3f},{dh:+.3f}] -> {kl}")
        rows.append(dict(task="chest", config=f"depth{depth}-{act}", params=p,
                         mean=round(mm, 4), sd=round(ss, 4), paired=round(d, 4),
                         ci_lo=round(dl, 4), ci_hi=round(dh, 4), ket_luan=kl))
        if best is None or mm < best[1]:
            best = (f"depth{depth}-{act}", mm)
    print(f"   => MLP tot nhat: {best[0]} = {best[1]:.3f} dB")


def quet_amc(rows):
    try:
        from exp_amc_backbone import load_band
    except Exception as e:
        print("\n== AMC: bo qua, khong nap duoc du lieu:", e); return
    X, y, mods = load_band()
    nc = len(mods); din = X.shape[1] * X.shape[2]
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(X))[:40000]
    Xf = X.reshape(len(X), -1)[idx]; yf = y[idx]
    ntr = int(0.7 * len(Xf))
    Xtr, ytr, Xte, yte = Xf[:ntr], yf[:ntr], Xf[ntr:], yf[ntr:]
    target = nparams(KANcls(din, 32, nc, 4))
    print(f"\n== AMC | N={len(Xf)} din={din} lop={nc} | ngan sach KAN tham chieu = {target} ==")
    kan = [train_cls(KANcls(din, 32, nc, 4), Xtr, ytr, Xte, yte, seed=s) for s in SEEDS]
    m, sd, lo, hi = bootstrap_ci(kan)
    print(f"   Fourier-KAN (moc)      : acc {m:.4f} +- {sd:.4f}")
    rows.append(dict(task="amc", config="fourier-KAN(ref)", params=target,
                     mean=round(m, 4), sd=round(sd, 4)))
    best = None
    for depth, act in CONFIGS:
        h = match_flex(target, din, nc, depth, act)
        p = nparams(MLPflexCls(din, h, nc, depth, act))
        v = [train_cls(MLPflexCls(din, h, nc, depth, act), Xtr, ytr, Xte, yte, seed=s)
             for s in SEEDS]
        mm, ss, l2, h2 = bootstrap_ci(v)
        d, dl, dh = paired_diff_ci(kan, v)       # duong = KAN tot hon (acc cao hon)
        kl = "KAN" if dl > 0 else ("MLP" if dh < 0 else "khong phan biet duoc")
        print(f"   MLP depth{depth}-{act:4s} h={h:3d} ({p:6d}p): acc {mm:.4f} +- {ss:.4f}"
              f" | KAN-MLP={d:+.4f} [{dl:+.4f},{dh:+.4f}] -> {kl}")
        rows.append(dict(task="amc", config=f"depth{depth}-{act}", params=p,
                         mean=round(mm, 4), sd=round(ss, 4), paired=round(d, 4),
                         ci_lo=round(dl, 4), ci_hi=round(dh, 4), ket_luan=kl))
        if best is None or mm > best[1]:
            best = (f"depth{depth}-{act}", mm)
    print(f"   => MLP tot nhat: {best[0]} = {best[1]:.4f}")


def main():
    rows = []
    quet_chest(rows)
    quet_amc(rows)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["task", "config", "params", "mean", "sd",
                                          "paired", "ci_lo", "ci_hi", "ket_luan"])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print("\nghi:", OUT)


if __name__ == "__main__":
    main()
