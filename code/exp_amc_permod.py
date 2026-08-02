"""
Per-modulation analysis on RML2016.10a: does the KAN vs MLP vs CNN ordering hold uniformly
across modulation types, or is it concentrated in a few classes? We train each model on the
mid-band and report per-modulation accuracy, averaged over seeds. This turns the aggregate
number into a resolved picture and shows WHERE each model's competence lies.
"""
import numpy as np, torch, csv, os
from harness import KANcls, KANspline, MLPcls, nparams, match_width
from exp_amc import CNNc, WIN
import pickle, torch.nn as nn
from sklearn.metrics import accuracy_score

DATA = "../data/RML2016.10a/RML2016.10a_dict.pkl"
SNR_LO, SNR_HI = 0, 12          # clean-ish band where models separate


def load(snr_lo=SNR_LO, snr_hi=SNR_HI):
    with open(DATA, "rb") as f:
        d = pickle.load(f, encoding="latin1")
    mods = sorted({k[0] for k in d}); midx = {m: i for i, m in enumerate(mods)}
    X, y = [], []
    for (mod, s), arr in d.items():
        if snr_lo <= s <= snr_hi:
            X.append(arr.astype(np.float32)); y += [midx[mod]] * len(arr)
    X = np.concatenate(X, 0); X = X / (np.sqrt((X ** 2).sum((1, 2), keepdims=True) / WIN) + 1e-8)
    return X.reshape(len(X), -1), np.array(y), mods


def train(mk, Xtr, ytr, seed):
    torch.manual_seed(seed); m = mk()
    opt = torch.optim.Adam(m.parameters(), 2e-3, weight_decay=1e-5)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 30); lf = nn.CrossEntropyLoss()
    dl = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(
        torch.tensor(Xtr), torch.tensor(ytr)), batch_size=256, shuffle=True)
    for _ in range(30):
        for xb, yb in dl:
            opt.zero_grad(); lf(m(xb), yb).backward(); opt.step()
        sch.step()
    m.eval(); return m


if __name__ == "__main__":
    X, y, mods = load(); nc = len(mods); din = 2 * WIN
    rng = np.random.default_rng(0); idx = rng.permutation(len(X))[:45000]
    Xf, yf = X[idx], y[idx]; ntr = int(0.7 * len(Xf))
    Xtr, ytr, Xte, yte = Xf[:ntr], yf[:ntr], Xf[ntr:], yf[ntr:]
    H, K = 32, 4
    p = nparams(KANcls(din, H, nc, K))
    hs = match_width(p, lambda h: KANspline(din, h, nc), 4, 48)
    hm = match_width(p, lambda h: MLPcls(din, h, nc))
    builders = {"CNN": lambda: CNNc(nc), "MLP": lambda: MLPcls(din, hm, nc),
                "KANspline": lambda: KANspline(din, hs, nc)}
    permod = {k: {m: [] for m in mods} for k in builders}
    for seed in range(3):
        for k, mk in builders.items():
            model = train(mk, Xtr, ytr, seed)
            with torch.no_grad():
                pred = model(torch.tensor(Xte)).argmax(1).numpy()
            for mi, mod in enumerate(mods):
                mask = yte == mi
                if mask.sum(): permod[k][mod].append((pred[mask] == yte[mask]).mean())
        print(f"seed{seed} done")
    os.makedirs("results", exist_ok=True)
    with open("results/exp_amc_permod.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["model", "modulation", "acc_mean", "acc_std"])
        for k in builders:
            for mod in mods:
                a = np.array(permod[k][mod]); w.writerow([k, mod, round(a.mean(), 4), round(a.std(), 4)])
    # headline: spread of KAN-vs-MLP advantage across modulations
    diff = {mod: np.mean(permod["KANspline"][mod]) - np.mean(permod["MLP"][mod]) for mod in mods}
    best = max(diff, key=diff.get); worst = min(diff, key=diff.get)
    print(f"KAN-spline vs MLP per-mod: best +{diff[best]:.2f} ({best}), worst {diff[worst]:+.2f} ({worst})")
    print("wrote results/exp_amc_permod.csv")
