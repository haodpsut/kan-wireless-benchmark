"""
AMC ablations on real RML2016.10a:
  A. SNR SWEEP  -- per-SNR test accuracy for every model across the full -20..18 dB range.
     This directly shows the saturated-benchmark trap: at high SNR the models converge, so a
     benchmark reported only there cannot separate architectures; the low-SNR band is where a
     real difference lives (and is where our main result is measured).
  B. BUDGET SWEEP (flat models) -- does MLP>KAN persist as the budget grows?
"""
import numpy as np, torch, csv, os
from harness import KANcls, KANspline, MLPcls, nparams, match_width, train_cls
from exp_amc import CNNc, WIN
import pickle

DATA = "../data/RML2016.10a/RML2016.10a_dict.pkl"
SEEDS = range(3)


def load_full():
    with open(DATA, "rb") as f:
        d = pickle.load(f, encoding="latin1")
    mods = sorted({k[0] for k in d}); midx = {m: i for i, m in enumerate(mods)}
    X, y, snr = [], [], []
    for (mod, s), arr in d.items():
        X.append(arr.astype(np.float32)); y += [midx[mod]] * len(arr); snr += [s] * len(arr)
    X = np.concatenate(X, 0)
    X = X / (np.sqrt((X ** 2).sum((1, 2), keepdims=True) / WIN) + 1e-8)
    return X, np.array(y), np.array(snr), mods


def fit_predict(mk, Xtr, ytr, Xte, seed, conv=False):
    import torch.nn as nn
    torch.manual_seed(seed); model = mk()
    opt = torch.optim.Adam(model.parameters(), 2e-3, weight_decay=1e-5)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 30); lf = nn.CrossEntropyLoss()
    ds = torch.utils.data.TensorDataset(torch.tensor(Xtr), torch.tensor(ytr))
    dl = torch.utils.data.DataLoader(ds, batch_size=256, shuffle=True)
    for _ in range(30):
        for xb, yb in dl:
            opt.zero_grad(); lf(model(xb), yb).backward(); opt.step()
        sch.step()
    model.eval()
    with torch.no_grad():
        return model(torch.tensor(Xte)).argmax(1).numpy()


if __name__ == "__main__":
    X, y, snr, mods = load_full(); nc = len(mods); din = 2 * WIN
    rng = np.random.default_rng(0); idx = rng.permutation(len(X))[:60000]
    Xa, ya, sa = X.reshape(len(X), -1)[idx], y[idx], snr[idx]
    ntr = int(0.7 * len(Xa)); tr, te = slice(0, ntr), slice(ntr, len(Xa))
    os.makedirs("results", exist_ok=True)

    H, K = 32, 4
    p = nparams(KANcls(din, H, nc, K))
    hs = match_width(p, lambda h: KANspline(din, h, nc), 4, 48)
    hm = match_width(p, lambda h: MLPcls(din, h, nc))
    builders = {"CNN": lambda: CNNc(nc), "MLP": lambda: MLPcls(din, hm, nc),
                "KANspline": lambda: KANspline(din, hs, nc), "KANfourier": lambda: KANcls(din, H, nc, K)}

    # ---- A. SNR sweep ----
    snr_vals = sorted(set(sa.tolist()))
    acc = {k: {s: [] for s in snr_vals} for k in builders}
    for seed in SEEDS:
        for k, mk in builders.items():
            pred = fit_predict(mk, Xa[tr], ya[tr], Xa[te], seed)
            for s in snr_vals:
                m = sa[te] == s
                if m.sum(): acc[k][s].append((pred[m] == ya[te][m]).mean())
        print(f"seed{seed} done")
    with open("results/abl_amc_snr.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["model", "snr_db", "acc_mean", "acc_std"])
        for k in builders:
            for s in snr_vals:
                a = np.array(acc[k][s]); w.writerow([k, s, round(a.mean(), 4), round(a.std(), 4)])
    hi = [s for s in snr_vals if s >= 12]
    print("high-SNR(>=12) mean acc:", {k: round(np.mean([np.mean(acc[k][s]) for s in hi]), 3) for k in builders})

    # ---- B. budget sweep (flat models) ----
    with open("results/abl_amc_budget.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["budget_H", "params", "model", "acc_mean", "acc_std"])
        for Hb in [16, 32, 64]:
            pb = nparams(KANcls(din, Hb, nc, K)); hmb = match_width(pb, lambda h: MLPcls(din, h, nc))
            hsb = match_width(pb, lambda h: KANspline(din, h, nc), 4, 64)
            fb = {"MLP": lambda: MLPcls(din, hmb, nc), "KANfourier": lambda: KANcls(din, Hb, nc, K),
                  "KANspline": lambda: KANspline(din, hsb, nc)}
            for k, mk in fb.items():
                vs = [(fit_predict(mk, Xa[tr], ya[tr], Xa[te], s) == ya[te]).mean() for s in SEEDS]
                w.writerow([Hb, pb, k, round(np.mean(vs), 4), round(np.std(vs), 4)])
            print(f"budget H={Hb} (~{pb}p) done")
    print("wrote abl_amc_{snr,budget}.csv")
