"""
Experiment AMC: automatic modulation classification on the REAL RML2016.10a dataset
(the high-dimensional, structured-input classification regime). We deliberately restrict
to the LOW-SNR band [-6, +6] dB where accuracy is far from saturated (headroom exists), so
model differences are resolvable -- the opposite of the saturated-benchmark trap.

Honest matched-capacity comparison:
  - KAN (spline) and KAN (Fourier): the trendy model.
  - MLP: parameter-matched to the Fourier-KAN (the fairness control).
  - CNN: a task-appropriate STRONG baseline that exploits the I/Q time structure (so KAN is
    not allowed to look good only against flat baselines).
Multi-seed, bootstrap CI. Whatever it shows, we report.
"""
import numpy as np, torch, torch.nn as nn, pickle, csv, os
from harness import (KANcls, KANspline, MLPcls, nparams, match_width,
                     train_cls, bootstrap_ci, paired_diff_ci)

DATA = "../data/RML2016.10a/RML2016.10a_dict.pkl"
SNR_LO, SNR_HI = -6, 6            # non-saturated band
WIN = 128


def load_rml(snr_lo=SNR_LO, snr_hi=SNR_HI):
    with open(DATA, "rb") as f:
        d = pickle.load(f, encoding="latin1")
    mods = sorted({k[0] for k in d})
    midx = {m: i for i, m in enumerate(mods)}
    X, y = [], []
    for (mod, snr), arr in d.items():
        if snr_lo <= snr <= snr_hi:
            X.append(arr.astype(np.float32)); y += [midx[mod]] * len(arr)
    X = np.concatenate(X, 0)                       # [N,2,128]
    y = np.array(y, np.int64)
    # per-example power normalization (standard AMC preprocessing)
    X = X / (np.sqrt((X ** 2).sum((1, 2), keepdims=True) / WIN) + 1e-8)
    return X, y, mods


class CNNc(nn.Module):
    def __init__(self, nc):
        super().__init__()
        self.c = nn.Sequential(nn.Conv1d(2, 24, 5, padding=2), nn.ReLU(),
                               nn.Conv1d(24, 24, 5, padding=2), nn.ReLU(),
                               nn.AdaptiveAvgPool1d(8))
        self.f = nn.Linear(24 * 8, nc)

    def forward(self, x):
        b = x.shape[0]; return self.f(self.c(x.view(b, 2, WIN)).view(b, -1))


if __name__ == "__main__":
    X, y, mods = load_rml()
    nc = len(mods); din = 2 * WIN
    rng = np.random.default_rng(0)
    # cap for CPU feasibility (honest subsample; note in paper), stratified enough at random
    idx = rng.permutation(len(X))[:40000]
    Xf = X.reshape(len(X), -1)[idx]; yf = y[idx]
    ntr = int(0.7 * len(Xf))
    Xtr, ytr, Xte, yte = Xf[:ntr], yf[:ntr], Xf[ntr:], yf[ntr:]
    print(f"RML2016.10a low-SNR[{SNR_LO},{SNR_HI}]dB | N={len(Xf)} classes={nc} din={din}")

    H, K = 32, 4
    p_fkan = nparams(KANcls(din, H, nc, K))
    h_skan = match_width(p_fkan, lambda h: KANspline(din, h, nc), lo=4, hi=48)
    h_mlp = match_width(p_fkan, lambda h: MLPcls(din, h, nc))
    p_skan, p_mlp = nparams(KANspline(din, h_skan, nc)), nparams(MLPcls(din, h_mlp, nc))
    p_cnn = nparams(CNNc(nc))
    print(f"Fourier-KAN={p_fkan} | spline-KAN(h={h_skan})={p_skan} | "
          f"MLP(h={h_mlp})={p_mlp} | CNN={p_cnn}")

    res = {"KANspline": [], "KANfourier": [], "MLP": [], "CNN": []}
    for seed in range(10):
        res["KANspline"].append(train_cls(KANspline(din, h_skan, nc), Xtr, ytr, Xte, yte, seed))
        res["KANfourier"].append(train_cls(KANcls(din, H, nc, K), Xtr, ytr, Xte, yte, seed))
        res["MLP"].append(train_cls(MLPcls(din, h_mlp, nc), Xtr, ytr, Xte, yte, seed))
        Xtr3 = Xtr.reshape(-1, 2, WIN); Xte3 = Xte.reshape(-1, 2, WIN)
        res["CNN"].append(train_cls(CNNc(nc), Xtr3.reshape(len(Xtr3), -1), ytr,
                                    Xte3.reshape(len(Xte3), -1), yte, seed))
        print(f"  seed{seed}: sKAN={res['KANspline'][-1]:.4f} fKAN={res['KANfourier'][-1]:.4f} "
              f"MLP={res['MLP'][-1]:.4f} CNN={res['CNN'][-1]:.4f}")

    os.makedirs("results", exist_ok=True)
    with open("results/exp_amc.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["model", "params", "acc_mean", "acc_std", "ci_lo", "ci_hi"])
        for name, p in [("KANspline", p_skan), ("KANfourier", p_fkan), ("MLP", p_mlp), ("CNN", p_cnn)]:
            mean, std, lo, hi = bootstrap_ci(res[name])
            w.writerow([name, p, round(mean, 4), round(std, 4), round(lo, 4), round(hi, 4)])
            print(f"{name}: acc {mean:.4f}±{std:.4f} CI[{lo:.4f},{hi:.4f}]")
    dk, lo, hi = paired_diff_ci(res["KANfourier"], res["MLP"])
    print(f"KANfourier-MLP paired acc diff: {dk:+.4f} CI[{lo:+.4f},{hi:+.4f}]")
    dc, lo2, hi2 = paired_diff_ci(res["CNN"], res["KANfourier"])
    print(f"CNN-KANfourier paired acc diff: {dc:+.4f} CI[{lo2:+.4f},{hi2:+.4f}]")
    print("wrote results/exp_amc.csv")
